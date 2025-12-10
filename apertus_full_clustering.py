"""
Assign KMeans cluster labels to the full apertus-sft-mixture embedding dataset, using models fitted on a 5% sample.

This script:
- Memory-maps the full conversation embeddings (stored as a .dat file) to avoid loading everything into RAM at once.
- Loads the UMAP model, StandardScaler, and KMeans model previously fitted on a sampled subset of the data.
- Processes the full dataset in batches:
    * L2-normalizes each batch of embeddings,
    * applies the fitted UMAP transform,
    * applies the fitted StandardScaler,
    * predicts KMeans cluster labels.
- Writes the resulting labels into a memory-mapped int32 array on disk and additionally saves a compact `.npy` copy.

It assumes that:
- The full embeddings `.dat` file, and the UMAP/StandardScaler/KMeans artifacts (trained on the sample) 
  already exist in `OUT_DIR`.
- The constants `N_SAMPLES_FULL`, `EMBEDDING_DIM`, and `EMBEDDINGS_DTYPE` 
  correctly describe the full embedding matrix layout.
"""

from pathlib import Path

import joblib
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap


# ---------- CONFIG (full dataset) ----------

OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")

# Full embeddings stored as .dat (memory-mapped)
EMBEDDINGS_DAT_PATH = OUT_DIR / (
    "train_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.dat"
)

# Full dataset parameters
N_SAMPLES_FULL = 3_942_208
EMBEDDING_DIM = 768
EMBEDDINGS_DTYPE = np.float32

# Artifacts fitted on the 1% sample
TAG = "train_sampled_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2"

UMAP_MODEL_PATH = OUT_DIR / f"{TAG}_umap_model.joblib"
SCALER_PATH = OUT_DIR / f"{TAG}_umap_scaler.joblib"
KMEANS_MODEL_PATH = OUT_DIR / f"{TAG}_kmeans_model.joblib"

# Output cluster labels for the full dataset
FULL_LABELS_DAT_PATH = OUT_DIR / f"train_full_{TAG}_kmeans_labels.dat"
FULL_LABELS_NPY_PATH = OUT_DIR / f"train_full_{TAG}_kmeans_labels.npy"

# Chunking settings
BATCH_SIZE = 100_000


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EMBEDDINGS_DAT_PATH.exists():
        raise FileNotFoundError(f"Full embeddings .dat not found: {EMBEDDINGS_DAT_PATH}")

    print(f"Loading UMAP model from: {UMAP_MODEL_PATH}")
    umap_model: umap.UMAP = joblib.load(UMAP_MODEL_PATH)

    print(f"Loading StandardScaler from: {SCALER_PATH}")
    scaler: StandardScaler = joblib.load(SCALER_PATH)

    print(f"Loading KMeans model from: {KMEANS_MODEL_PATH}")
    kmeans_model: KMeans = joblib.load(KMEANS_MODEL_PATH)
    n_clusters = kmeans_model.n_clusters
    print(f"KMeans has {n_clusters} clusters.")

    print(f"Memory-mapping full embeddings from: {EMBEDDINGS_DAT_PATH}")
    emb_full = np.memmap(
        EMBEDDINGS_DAT_PATH,
        mode="r",
        dtype=EMBEDDINGS_DTYPE,
        shape=(N_SAMPLES_FULL, EMBEDDING_DIM),
    )

    print(f"Creating memmap for full labels at: {FULL_LABELS_DAT_PATH}")
    labels_mem = np.memmap(
        FULL_LABELS_DAT_PATH,
        mode="w+",
        dtype=np.int32,
        shape=(N_SAMPLES_FULL,),
    )

    n_batches = (N_SAMPLES_FULL + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total samples: {N_SAMPLES_FULL}, batch size: {BATCH_SIZE}, n_batches: {n_batches}")

    for batch_idx in tqdm(range(n_batches), desc="Assigning clusters to full dataset"):
        start = batch_idx * BATCH_SIZE
        end = min(N_SAMPLES_FULL, (batch_idx + 1) * BATCH_SIZE)
        if start >= end:
            break

        # Slice embeddings [start:end, :]
        emb_chunk = np.asarray(emb_full[start:end], dtype=np.float32)

        # L2-normalize
        norms = np.linalg.norm(emb_chunk, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        emb_chunk_norm = emb_chunk / norms

        # UMAP transform (uses fitted model from sample)
        X_umap_chunk = umap_model.transform(emb_chunk_norm)

        # Scale with fitted StandardScaler
        X_red_chunk = scaler.transform(X_umap_chunk)

        # Predict KMeans clusters
        labels_chunk = kmeans_model.predict(X_red_chunk).astype(np.int32)

        labels_mem[start:end] = labels_chunk

    # Flush to disk
    labels_mem.flush()
    del labels_mem

    # Optional: create a .npy wrapper (labels are much smaller than embeddings)
    labels_mem_ro = np.memmap(
        FULL_LABELS_DAT_PATH,
        mode="r",
        dtype=np.int32,
        shape=(N_SAMPLES_FULL,),
    )
    labels_array = np.asarray(labels_mem_ro)
    np.save(FULL_LABELS_NPY_PATH, labels_array)
    print(f"Saved full labels as memmap (.dat) to: {FULL_LABELS_DAT_PATH}")
    print(f"Saved full labels as .npy to: {FULL_LABELS_NPY_PATH}")
    print("Done assigning clusters to full dataset.")


if __name__ == "__main__":
    main()
