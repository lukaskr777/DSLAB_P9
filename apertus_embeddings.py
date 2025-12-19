"""
Conversation-level embeddings for swiss-ai/apertus-sft-mixture.

Pipeline (single model, multiple parquet files):
1. Load each Parquet file (only needed columns: conversation_id, messages).
2. Parse/flatten `messages` into a single `conversation_text` string per row.
3. Compute L2-normalized sentence embeddings for each conversation using
   "sentence-transformers/paraphrase-multilingual-mpnet-base-v2".
4. Save, for each input file:
   - one .npy file with embeddings (same row order as the resulting DataFrame),
   - one Parquet file with (conversation_id, conversation_text).

To process the full train set, split `train.parquet` into several smaller parquet files 
(e.g. train_part_00.parquet, train_part_01.parquet, ...) and list them in INPUT_PATHS.
The script keeps one model in memory and processesone DataFrame per iteration to avoid cumulative memory growth.
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from utility_scripts.clustering_utils import ensure_list_of_dicts, flatten_messages_to_text


# ---------- CONFIG ----------

DATA_DIR = Path("data/swiss-ai_apertus-sft-mixture")
OUT_DIR = DATA_DIR

# List of parquet files to process.
# INPUT_PATHS = [DATA_DIR / "split_train" / f"train_part_{i:02d}.parquet" for i in range(55)]
# INPUT_PATHS = [DATA_DIR / "train_sampled.parquet"]
INPUT_PATHS = [DATA_DIR / f"small_train_part_{i}.parquet" for i in range(10)]

# Single model used for all embeddings
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Default batch size for encoding
DEFAULT_BATCH_SIZE = 32


# ---------- Helpers ----------

def add_conversation_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse `messages` column into a single `conversation_text` string.

    - `messages` is converted to a list[dict] per row via ensure_list_of_dicts.
    - For each row, we flatten all messages into a single text string
      (with roles prefixed where available) using flatten_messages_to_text.
    - Rows with empty/whitespace-only conversation_text are dropped.
    """
    if "messages" not in df.columns:
        raise KeyError("Expected a 'messages' column in the input DataFrame.")

    parsed_messages = [ensure_list_of_dicts(msgs) for msgs in df["messages"]]

    n_rows = len(parsed_messages)
    empty_ratio = sum(len(m) == 0 for m in parsed_messages) / max(n_rows, 1)
    print(f"Fraction of conversations with 0 parsed messages: {empty_ratio:.3f}")

    df = df.copy()

    conversation_texts: list[str] = []
    for msgs in parsed_messages:
        conversation_texts.append(flatten_messages_to_text(msgs))

    df["conversation_text"] = conversation_texts

    # Keep only rows with non-empty text
    mask_nonempty = df["conversation_text"].str.strip().astype(bool)
    df = df.loc[mask_nonempty].reset_index(drop=True)

    return df


def compute_embeddings_for_texts(
    texts: list[str], model: SentenceTransformer, batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """Compute L2-normalized sentence embeddings for each conversation text."""
    if not texts:
        raise ValueError("No texts provided for embedding computation.")

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # produce L2-normalized embeddings
    )
    return emb


def _model_name_to_suffix(model_name: str) -> str:
    """Turn a model name into a safe filename suffix."""
    return model_name.replace("/", "__").replace(":", "_")


def process_file(
    data_path: Path, out_dir: Path, model: SentenceTransformer, batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """
    Process a single parquet file:
    - Load (conversation_id, messages),
    - add conversation_text,
    - compute embeddings,
    - save embeddings .npy and text-only parquet.
    """
    print(f"\n=== Processing file: {data_path} ===")
    if not data_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {data_path}")

    # Load only the columns we actually need
    df = pd.read_parquet(data_path, columns=["conversation_id", "messages"])
    print(f"Loaded shape: {df.shape}")

    # Add conversation_text and drop empty conversations
    print("Parsing `messages` into `conversation_text`...")
    df = add_conversation_text(df)
    print(f"After dropping empty conversations: {df.shape}")

    texts = df["conversation_text"].fillna("").astype(str).tolist()
    n = len(texts)
    print(f"Number of conversations with text: {n}")
    if n == 0:
        print("No non-empty conversation texts found after parsing; skipping file.")
        return

    input_stem = data_path.stem
    suffix = _model_name_to_suffix(MODEL_NAME)

    emb_path = out_dir / f"{input_stem}_conversation_embeddings__{suffix}.npy"
    conv_path = out_dir / f"{input_stem}_conversations_text_only.parquet"

    if emb_path.exists():
        print(f"Loading precomputed embeddings from: {emb_path}")
        emb = np.load(emb_path)
    else:
        print(f"Computing embeddings for conversations with model {MODEL_NAME!r}...")
        emb = compute_embeddings_for_texts(texts, model=model, batch_size=batch_size)
        np.save(emb_path, emb)
        print(f"Saved embeddings to: {emb_path}")

    print(f"Embeddings shape: {emb.shape}")

    # Save text-only parquet (id + text)
    df[["conversation_id", "conversation_text"]].to_parquet(conv_path, index=False)
    print(f"Saved conversation texts to: {conv_path}")

    # Explicitly drop large objects and run GC to minimize cumulative memory
    del df, texts, emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- Main ----------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading sentence-transformer model {MODEL_NAME!r} on device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    for data_path in INPUT_PATHS:
        process_file(data_path=data_path, out_dir=OUT_DIR, model=model, batch_size=DEFAULT_BATCH_SIZE)

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
