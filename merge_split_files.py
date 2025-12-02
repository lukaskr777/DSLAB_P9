from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd


# ---------- CONFIG ----------

DATA_DIR = Path("data/swiss-ai_apertus-sft-mixture")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
N_PARTS = 10  # small_train_part_0 .. small_train_part_9

MERGE_TEXTS = True  # set False if you only care about the embeddings


# ---------- Helpers ----------

def _model_name_to_suffix(model_name: str) -> str:
    """Turn a model name into a safe filename suffix."""
    return model_name.replace("/", "__").replace(":", "_")


def build_paths(data_dir: Path, n_parts: int, model_name: str):
    suffix = _model_name_to_suffix(model_name)

    emb_files = [
        data_dir / f"small_train_part_{i}_conversation_embeddings__{suffix}.npy"
        for i in range(n_parts)
    ]
    text_files = [
        data_dir / f"small_train_part_{i}_conversations_text_only.parquet"
        for i in range(n_parts)
    ]

    merged_emb_path = data_dir / f"small_train_conversation_embeddings__{suffix}.npy"
    merged_text_path = data_dir / "small_train_conversations_text_only.parquet"

    return emb_files, text_files, merged_emb_path, merged_text_path


# ---------- Merge embeddings (.npy) ----------

def merge_embeddings(emb_files: list[Path], merged_emb_path: Path) -> None:
    # First pass: check files, compute total rows and embedding dimension
    total_rows = 0
    emb_dim = None
    dtype = None

    for f in emb_files:
        if not f.exists():
            raise FileNotFoundError(f"Embedding file not found: {f}")
        arr = np.load(f, mmap_mode="r")
        if emb_dim is None:
            emb_dim = arr.shape[1]
            dtype = arr.dtype
        else:
            if arr.shape[1] != emb_dim:
                raise ValueError(f"Inconsistent embedding dim in {f}: {arr.shape[1]} vs {emb_dim}")
        total_rows += arr.shape[0]

    if emb_dim is None or total_rows == 0:
        raise ValueError("No embeddings found to merge.")

    print(f"Total rows: {total_rows}, embedding dim: {emb_dim}, dtype: {dtype}")

    # Create a proper .npy file with header using open_memmap
    print(f"Creating merged memmap at: {merged_emb_path}")
    merged = open_memmap(
        merged_emb_path,
        mode="w+",
        dtype=dtype,
        shape=(total_rows, emb_dim),
    )

    # Second pass: copy each part into the correct slice
    offset = 0
    for f in emb_files:
        arr = np.load(f, mmap_mode="r")
        n = arr.shape[0]
        print(f"Copying {n} rows from {f} into [{offset}:{offset + n})")
        merged[offset : offset + n] = arr
        offset += n

    # Ensure data is written to disk
    merged.flush()
    del merged

    print(f"Merged embeddings saved to: {merged_emb_path}")


# ---------- Merge text Parquets (optional) ----------

def merge_text_parquets(text_files: list[Path], merged_text_path: Path) -> None:
    dfs = []
    for f in text_files:
        if not f.exists():
            raise FileNotFoundError(f"Text parquet not found: {f}")
        print(f"Reading: {f}")
        df = pd.read_parquet(f)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged text shape: {merged_df.shape}")
    merged_df.to_parquet(merged_text_path, index=False)
    print(f"Merged conversations saved to: {merged_text_path}")


# ---------- Main ----------

def main() -> None:
    emb_files, text_files, merged_emb_path, merged_text_path = build_paths(
        DATA_DIR, N_PARTS, MODEL_NAME
    )

    merge_embeddings(emb_files, merged_emb_path)

    if MERGE_TEXTS:
        merge_text_parquets(text_files, merged_text_path)


if __name__ == "__main__":
    main()
