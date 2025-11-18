"""
Conversation-level embeddings for swiss-ai/apertus-sft-mixture.

Pipeline:
1. Load sampled Parquet.
2. Parse/flatten `messages` into a single `conversation_text` string.
3. Compute L2-normalized sentence embeddings for each conversation, once per model.
4. Save one .npy file per model (same conversation order for all models).
"""

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ---------- CONFIG ----------

DATA_PATH = Path("data/swiss-ai_apertus-sft-mixture/train_sampled_enriched.parquet")
OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")

# Models for which we want one embedding matrix each
MODEL_NAMES: list[str] = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/stsb-xlm-r-multilingual",
    "sentence-transformers/gtr-t5-base",
]

# Base name for saved embedding arrays; the model name will be appended
EMBEDDINGS_BASENAME = "train_sampled_conversation_embeddings"

# Whether to split the data into parts to avoid OOM; if True, set N_PARTS and PART
USE_PARTS = False 
N_PARTS = 8  # total number of splits
PART = 1


# ---------- Helpers for messages parsing ----------

def _ensure_list_of_dicts(obj: Any) -> list[dict[str, Any]]:
    """Best-effort conversion of `obj` to a list of message dicts."""
    # numpy array of dicts
    if isinstance(obj, np.ndarray):
        return [m for m in obj.tolist() if isinstance(m, dict)]

    # plain list
    if isinstance(obj, list):
        return [m for m in obj if isinstance(m, dict)]

    # single dict
    if isinstance(obj, dict):
        return [obj]

    # JSON string (if ever needed)
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return _ensure_list_of_dicts(parsed)
        except Exception:
            return []

    # generic iterable fallback
    try:
        it = list(obj)
        return [m for m in it if isinstance(m, dict)]
    except Exception:
        return []


def _extract_text_from_content(content: Any) -> str:
    """
    Extract human-readable text from the nested `content` structure.

    Priority:
    - content["text"] if non-empty
    - join all parts[i]["text"] in content["parts"]
    - join all blocks[i]["text"] in content["blocks"]
    """
    if content is None:
        return ""

    if not isinstance(content, dict):
        return str(content)

    texts: list[str] = []

    # 1) direct text field
    txt = content.get("text")
    if isinstance(txt, str) and txt.strip():
        texts.append(txt.strip())

    # 2) parts -> [{'text': ..., 'type': ...}, ...]
    parts = content.get("parts")
    if parts is not None:
        try:
            parts_iter = list(parts)
        except TypeError:
            parts_iter = [parts]
        for p in parts_iter:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())

    # 3) blocks -> [{'text': ..., 'type': 'response', ...}, ...]
    blocks = content.get("blocks")
    if blocks is not None:
        try:
            blocks_iter = list(blocks)
        except TypeError:
            blocks_iter = [blocks]
        for b in blocks_iter:
            if isinstance(b, dict):
                t = b.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())

    return "\n".join(texts)


def _flatten_messages_to_text(messages: Iterable[dict[str, Any]]) -> str:
    """Join a list of message dicts into a single conversation string."""
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip()
        content_text = _extract_text_from_content(m.get("content"))

        if not content_text:
            continue

        if role:
            parts.append(f"{role}: {content_text}")
        else:
            parts.append(content_text)

    return "\n".join(parts)


def add_conversation_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse `messages` column into a single `conversation_text` string.

    - `messages` is converted to a list[dict] per row.
    - For each message, we extract human-readable text from `content`.
    - We prefix with the role (user/assistant/system) when available.
    - Rows with empty conversation_text are dropped.
    """
    parsed_messages: list[list[dict[str, Any]]] = []
    for msgs in df["messages"]:
        parsed_messages.append(_ensure_list_of_dicts(msgs))

    df = df.copy()
    df["messages_parsed"] = parsed_messages

    empty_ratio = (df["messages_parsed"].str.len() == 0).mean()
    print(f"Fraction of conversations with 0 parsed messages: {empty_ratio:.3f}")

    conversation_texts: list[str] = []
    for msgs in parsed_messages:
        conversation_texts.append(_flatten_messages_to_text(msgs))

    df["conversation_text"] = conversation_texts

    # Keep only rows with non-empty text
    df = df[df["conversation_text"].str.strip().astype(bool)].reset_index(drop=True)

    # Drop heavy intermediate column
    df = df.drop(columns=["messages_parsed"])

    return df


# ---------- Embeddings ----------

def compute_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute L2-normalized sentence embeddings for each conversation text."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading sentence-transformer model {model_name!r} on device: {device}")
    model = SentenceTransformer(model_name, device=device)
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


# ---------- Main ----------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading sampled data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded shape: {df.shape}")

    # Step 1: add conversation_text
    print("Parsing `messages` into `conversation_text`...")
    df = add_conversation_text(df)
    print(f"After dropping empty conversations: {df.shape}")

    texts = df["conversation_text"].fillna("").astype(str).tolist()
    n = len(texts)
    print(f"Number of conversations with text: {n}")

    if USE_PARTS and not (1 <= PART <= N_PARTS):
        raise ValueError(f"PART must be in [1, {N_PARTS}], got {PART}")
    
    if USE_PARTS:
        chunk_size = (n + N_PARTS - 1) // N_PARTS  # ceiling division
        start = (PART - 1) * chunk_size
        end = min(start + chunk_size, n)
        print(f"Processing PART {PART}/{N_PARTS}: indices [{start}, {end})")
        part_texts = texts[start:end]
    else:
        part_texts = texts

    # Step 2: compute and save embeddings once per model
    for model_name in MODEL_NAMES:
        suffix = _model_name_to_suffix(model_name)
        second_suffix = f"__part{PART}" if USE_PARTS else ""
        emb_path = OUT_DIR / f"{EMBEDDINGS_BASENAME}__{suffix}{second_suffix}.npy"

        if emb_path.exists():
            print(f"[{model_name}] Loading precomputed embeddings from: {emb_path}")
            emb = np.load(emb_path)
        else:
            print(f"[{model_name}] Computing embeddings for conversations...")
            emb = compute_embeddings(part_texts, model_name=model_name)
            np.save(emb_path, emb)
            print(f"[{model_name}] Saved embeddings to: {emb_path}")

        print(f"[{model_name}] Embeddings shape: {emb.shape}")

    conversations_path = OUT_DIR / "train_sampled_conversations_text_only.parquet"
    df[["conversation_id", "conversation_text"]].to_parquet(conversations_path, index=False)
    print(f"Saved conversation texts to: {conversations_path}")


if __name__ == "__main__":
    main()
