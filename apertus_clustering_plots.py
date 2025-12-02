"""
Post-clustering visualization script for swiss-ai/apertus-sft-mixture.

This script:
- Loads the clustered DataFrame produced by the clustering script.
- Loads the corresponding UMAP-reduced embeddings.
- Recomputes a standardized reduced space (X_red) for plotting.
- Regenerates:
    * Cluster scatter plots (Dim1 vs Dim2)
    * Cluster size bar plots
    * Per-cluster feature mean heatmap (z-scored)
    * Per-feature bar plots of per-cluster means
    * Cluster-wise word clouds
    * Top-5 words per cluster tables

Existing figures directory is cleared with `ensure_empty_dir` before plotting.
"""

import sys
from pathlib import Path
from typing import Any
import string
import re
from collections import Counter
import warnings

# Silence stopwordsiso/pkg_resources deprecation warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import stopwordsiso
from wordcloud import WordCloud

from utility_scripts.file_utils import ensure_empty_dir


# ---------- CONFIG (must match the clustering script) ----------

OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")

EMBEDDINGS_PATH = OUT_DIR / (
    "small_train_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.npy"
)
TAG = EMBEDDINGS_PATH.stem
UMAP_EMBEDDINGS_PATH = OUT_DIR / f"{TAG}_umap.npy"
CLUSTERED_DF_PATH = OUT_DIR / f"{TAG}_clustered.parquet"

# Root directory for figures; a subdirectory will be created per embeddings file
FIGS_ROOT = Path("figs/apertus_clustering")
FIGS_DIR = FIGS_ROOT / TAG
if sys.platform == "win32":
    FONT_PATH = r"C:\Windows\Fonts\NotoSans-Regular.ttf"
else:
    FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

# Feature columns used in feature-summary plots
FEATURE_COLUMNS = ["text_length"]


# ---------- Multilingual stopwords for word clouds ----------

EXTRA_STOPWORDS = {"user", "assistant", "system", "```", "###"}

STOPWORDS_PER_LANG: dict[str, set[str]] = {}
GLOBAL_STOPWORDS: set[str] = set()

for lang in stopwordsiso.langs():
    words = stopwordsiso.stopwords(lang)
    STOPWORDS_PER_LANG[lang] = set(words)
    GLOBAL_STOPWORDS |= STOPWORDS_PER_LANG[lang]

GLOBAL_STOPWORDS |= EXTRA_STOPWORDS


# ---------- Plotting helpers ----------

def make_cluster_plots(
    df: pd.DataFrame,
    X_red: np.ndarray,
    label_col: str,
    figs_dir: Path,
    *,
    use_robust_limits: bool = False,
) -> None:
    """
    Diagnostic cluster plots:
    - 2D scatter of reduced space (Dim 1 vs Dim 2)
    - Cluster size bar chart
    - Heatmap of per-cluster feature means (z-scored)
    - Per-feature bar charts of per-cluster raw means

    If use_robust_limits=True:
        - The scatter plot uses 1%-99% quantile axis limits
        - Output filename has "_robust" appended

    For HDBSCAN, points labeled -1 (noise) are dropped from all per-cluster plots.
    """
    figs_dir.mkdir(parents=True, exist_ok=True)
    labels = df[label_col].to_numpy()

    # Drop HDBSCAN noise for per-cluster plots
    if label_col == "cluster_hdbscan":
        mask = labels != -1
        df = df.loc[mask].reset_index(drop=True)
        labels = labels[mask]
        X_red = X_red[mask]

    if df.empty:
        print(f"[{label_col}] No data after filtering, skipping plots.")
        return

    # -------------------------------
    # Scatter plot (Dim 1 vs Dim 2)
    # -------------------------------
    if X_red.shape[1] >= 2:
        x = X_red[:, 0]
        y = X_red[:, 1]

        suffix = "_robust" if use_robust_limits else ""

        # Map cluster labels -> 0..K-1 and use a discrete colormap
        unique_labels = np.unique(labels)
        n_clusters = unique_labels.shape[0]
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        idx_colors = np.array([label_to_idx[lab] for lab in labels])

        # Discrete colormap with one entry per cluster
        cmap = colormaps.get_cmap("tab20").resampled(n_clusters)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            x,
            y,
            c=idx_colors,
            s=5,
            alpha=0.7,
            cmap=cmap,
        )
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title(f"Reduced-space scatter (colored by {label_col})")

        cbar = plt.colorbar(scatter, ticks=np.arange(n_clusters))
        # Show actual cluster labels on the colorbar
        cbar.ax.set_yticklabels([str(lab) for lab in unique_labels])
        cbar.set_label(label_col)

        if use_robust_limits:
            x_lo, x_hi = np.quantile(x, [0.01, 0.99])
            y_lo, y_hi = np.quantile(y, [0.01, 0.99])
            x_m = 0.05 * (x_hi - x_lo)
            y_m = 0.05 * (y_hi - y_lo)
            plt.xlim(x_lo - x_m, x_hi + x_m)
            plt.ylim(y_lo - y_m, y_hi + y_m)

        plt.tight_layout()
        out_path = figs_dir / f"scatter_dim1_dim2_{label_col}{suffix}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # -------------------------------
    # Cluster size bar chart
    # -------------------------------
    counts = df[label_col].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.xlabel("Cluster")
    plt.ylabel("Number of items")
    plt.title(f"Cluster sizes ({label_col})")
    plt.tight_layout()
    out_path = figs_dir / f"cluster_sizes_{label_col}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    # -------------------------------
    # Feature summary heatmap + bar plots
    # -------------------------------
    numeric_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not numeric_features:
        print(f"[{label_col}] No numeric feature columns found, skipping feature plots.")
        return

    cluster_means = df.groupby(label_col, observed=True)[numeric_features].mean().sort_index()
    if cluster_means.empty:
        print(f"[{label_col}] Empty cluster_means, skipping feature plots.")
        return

    scaler = StandardScaler()
    cluster_means_z = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns,
    )

    # Heatmap
    plt.figure(figsize=(1.5 * len(numeric_features) + 2, 0.4 * len(cluster_means_z) + 2))
    im = plt.imshow(cluster_means_z.values, aspect="auto")
    plt.colorbar(im, label="Mean z-score")
    plt.xticks(
        ticks=np.arange(len(numeric_features)),
        labels=numeric_features,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        ticks=np.arange(len(cluster_means_z)),
        labels=cluster_means_z.index.astype(str).tolist(),
    )
    plt.xlabel("Feature")
    plt.ylabel("Cluster")
    plt.title(f"Per-cluster feature z-score means ({label_col})")
    plt.tight_layout()
    out_path = figs_dir / f"cluster_feature_means_heatmap_{label_col}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Per-feature bar plots
    for feat in numeric_features:
        plt.figure(figsize=(8, 4))
        cluster_means[feat].plot(kind="bar")
        plt.xlabel("Cluster")
        plt.ylabel(f"Mean {feat}")
        plt.title(f"Mean {feat} per cluster ({label_col})")
        plt.tight_layout()
        out_path = figs_dir / f"cluster_mean_{feat}_{label_col}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


# ---------- Word clouds ----------

def preprocess_text_for_wordcloud(text: str) -> str:
    """
    Clean a conversation text for word cloud generation.

    - Remove role prefixes like 'user:', 'assistant:', 'system:'
    - Lowercase
    - Tokenize on whitespace
    - Strip leading/trailing punctuation
    - Drop tokens that are:
        * LaTeX / command-like (start with '\')
        * pure punctuation
        * multinlingual stopwords
        * very short ASCII tokens (len < 3)
    """
    if not text:
        return ""

    # Remove role prefixes (case-insensitive)
    text = re.sub(r"\b(user|assistant|system)\s*:", " ", text, flags=re.IGNORECASE)

    # Normalize case
    text = text.lower()

    # Simple whitespace tokenization
    raw_tokens = text.split()

    # Extra punctuation chars beyond string.punctuation (en-dash, quotes, etc.)
    extra_punct = "“”„»«…–—"

    filtered: list[str] = []
    for tok in raw_tokens:
        # Strip leading/trailing punctuation and quotes
        t = tok.strip(string.punctuation + extra_punct)

        # Skip empty after stripping
        if not t:
            continue

        # Skip LaTeX / command-like tokens such as \), \text{, \times
        if t.startswith("\\"):
            continue

        # After stripping, e.g. '"the' -> 'the', 'is:' -> 'is', skip stopwords
        if t in GLOBAL_STOPWORDS:
            continue

        # For ASCII tokens, require length >= 3
        if t.isascii() and len(t) < 3:
            continue

        filtered.append(t)

    return " ".join(filtered)


def make_cluster_wordclouds(
    df: pd.DataFrame, label_col: str, text_col: str, figs_dir: Path, font_path: str | None = None
) -> None:
    """
    Compute and save a word cloud image for each cluster, based on `text_col`.

    - Uses multilingual stopwords via stopwordsiso.
    - Detects language per conversation via langid and uses language-specific
      stopwords when available.
    - Removes role prefixes and very short tokens.

    Saves PNG files into `figs_dir / "wordclouds" / label_col`.
    """
    if text_col not in df.columns:
        print(f"[{label_col}] Column {text_col!r} not found, skipping word clouds.")
        return

    wc_dir = figs_dir / "wordclouds" / label_col
    wc_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(label_col, observed=True)[text_col]

    for cluster_label, texts in tqdm(grouped, desc=f"Generating word clouds ({label_col})"):
        pieces: list[str] = []

        for raw in texts.astype(str):
            if not raw.strip():
                continue

            cleaned = preprocess_text_for_wordcloud(raw)
            if not cleaned:
                continue

            if len(cleaned.split()) < 3:
                # Skip extremely sparse texts for wordcloud purposes
                continue

            pieces.append(cleaned)

        combined = "\n".join(pieces)
        if not combined.strip():
            print(f"Skipping cluster {cluster_label} (no usable text for wordcloud).")
            continue

        wc_kwargs: dict[str, Any] = {
            "width": 1600,
            "height": 900,
            "background_color": "white",
            "stopwords": set(),  # already applied our own
            "max_words": 200,
        }
        if font_path is not None:
            wc_kwargs["font_path"] = font_path

        wc = WordCloud(**wc_kwargs).generate(combined)

        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Cluster {cluster_label}")
        plt.tight_layout()

        out_path = wc_dir / f"cluster_{cluster_label}_wordcloud.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def compute_top_words_per_cluster(df: pd.DataFrame, label_col: str, text_col: str, out_path: Path) -> None:
    """
    For each cluster in `label_col`, compute the five most frequent words (after preprocessing and stopword removal) 
    in `text_col`, and write them to a TXT file as a tab-separated table.

    If a cluster has fewer than 5 distinct words, remaining slots are left empty.
    """
    if text_col not in df.columns:
        print(f"[{label_col}] Column {text_col!r} not found, skipping top-words table.")
        return

    grouped = df.groupby(label_col, observed=True)[text_col]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("cluster\tword1\tcount1\tword2\tcount2\tword3\tcount3\tword4\tcount4\tword5\tcount5\n")

        for cluster_label, texts in grouped:
            counter: Counter[str] = Counter()

            for raw in texts.astype(str):
                raw = raw.strip()
                if not raw:
                    continue

                cleaned = preprocess_text_for_wordcloud(raw)
                if not cleaned:
                    continue

                tokens = cleaned.split()
                if not tokens:
                    continue

                counter.update(tokens)

            if not counter:
                # No usable tokens for this cluster
                f.write(f"{cluster_label}\t\t\t\t\t\t\n")
                continue

            top5 = counter.most_common(5)
            # Pad to exactly five entries
            while len(top5) < 5:
                top5.append(("", 0))

            f.write(str(cluster_label))
            for word, count in top5:
                f.write(f"\t{word}\t{count}")
            f.write("\n")

    print(f"[{label_col}] Top-words table written to: {out_path}")


# ---------- Main ----------

def main() -> None:
    if not CLUSTERED_DF_PATH.exists():
        raise FileNotFoundError(f"Clustered DataFrame not found: {CLUSTERED_DF_PATH}")
    if not UMAP_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"UMAP embeddings not found: {UMAP_EMBEDDINGS_PATH}")

    print(f"Loading clustered DataFrame from: {CLUSTERED_DF_PATH}")
    df = pd.read_parquet(CLUSTERED_DF_PATH)
    print(f"Clustered DF shape: {df.shape}")

    print(f"Loading UMAP embeddings from: {UMAP_EMBEDDINGS_PATH}")
    X_umap = np.load(UMAP_EMBEDDINGS_PATH)
    print(f"UMAP shape: {X_umap.shape}")

    if len(df) != X_umap.shape[0]:
        raise ValueError(
            f"Mismatch between DataFrame rows ({len(df)}) and UMAP rows ({X_umap.shape[0]}). "
            "Ensure both were produced by the same clustering run."
        )

    # Standardize reduced space for plotting (same as clustering script)
    scaler = StandardScaler()
    X_red = scaler.fit_transform(X_umap)

    # Recreate figures directory (clear existing plots)
    ensure_empty_dir(FIGS_DIR)
    print(f"Figures will be written to: {FIGS_DIR}")

    # 1) Cluster plots (HDBSCAN, Leiden, and KMeans, with and without robust limits)
    print("Generating cluster plots...")
    for label_col in ("cluster_hdbscan", "cluster_leiden", "cluster_kmeans"):
        if label_col not in df.columns:
            print(f"Column {label_col!r} not in DataFrame, skipping plots for it.")
            continue

        make_cluster_plots(df, X_red, label_col=label_col, figs_dir=FIGS_DIR, use_robust_limits=False)
        make_cluster_plots(df, X_red, label_col=label_col, figs_dir=FIGS_DIR, use_robust_limits=True)

    # 2) Word clouds
    print("Generating word clouds per cluster...")
    if "conversation_text" not in df.columns:
        print("Column 'conversation_text' not found; cannot build word clouds.")
    else:
        for label_col in ("cluster_hdbscan", "cluster_leiden", "cluster_kmeans"):
            if label_col not in df.columns:
                continue
            make_cluster_wordclouds(
                df=df, label_col=label_col, text_col="conversation_text", figs_dir=FIGS_DIR, font_path=FONT_PATH
            )

    # 3) Top-5 words per cluster (per clustering method)
    print("Computing top-5 words per cluster...")
    if "conversation_text" not in df.columns:
        print("Column 'conversation_text' not found; cannot compute top words.")
    else:
        for label_col in ("cluster_hdbscan", "cluster_leiden", "cluster_kmeans"):
            if label_col not in df.columns:
                continue
            out_path = FIGS_DIR / f"top_words_{label_col}.txt"
            compute_top_words_per_cluster(df=df, label_col=label_col, text_col="conversation_text", out_path=out_path)

    print("Done.")


if __name__ == "__main__":
    main()
