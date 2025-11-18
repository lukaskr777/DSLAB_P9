import numpy as np
from pathlib import Path

OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")
N_PARTS = 8
suffix = "sentence-transformers__gtr-t5-base"

parts = []
for part in range(1, N_PARTS + 1):
    path = OUT_DIR / f"train_sampled_conversation_embeddings__{suffix}__part{part}.npy"
    print(f"Loading {path}")
    arr = np.load(path)
    parts.append(arr)

full = np.concatenate(parts, axis=0)
print("Full embeddings shape:", full.shape)

full_path = OUT_DIR / f"train_sampled_conversation_embeddings__{suffix}.npy"
np.save(full_path, full)
print("Saved combined embeddings to:", full_path)
