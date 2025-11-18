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

Existing figures directory is cleared with `ensure_empty_dir` before plotting.
"""

from pathlib import Path
from typing import Any
import warnings

# Silence stopwordsiso/pkg_resources deprecation warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tqdm
import re
import stopwordsiso
import langid
from wordcloud import WordCloud

from utility_scripts.file_utils import ensure_empty_dir


# ---------- CONFIG (must match the clustering script) ----------

OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")

# Embeddings choice only matters via TAG; keep in sync with clustering script
EMBEDDINGS_PATH = OUT_DIR / (
    "train_sampled_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.npy"
)

# Root directory for figures; a subdirectory will be created per embeddings file
FIGS_ROOT = Path("figs/apertus_clustering")

TAG = EMBEDDINGS_PATH.stem  # e.g. "train_sampled_conversation_embeddings__..."
UMAP_EMBEDDINGS_PATH = OUT_DIR / f"{TAG}_umap.npy"
CLUSTERED_DF_PATH = OUT_DIR / f"{TAG}_clustered.parquet"

FIGS_DIR = FIGS_ROOT / TAG

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

# Feature columns used in feature-summary plots
FEATURE_COLUMNS = [
    "n_turns",
    "user_msg_len",
    "assistant_msg_len",
    "text_length",
]


# ---------- Multilingual stopwords for word clouds ----------

EXTRA_STOPWORDS = {"user", "assistant", "system"}

STOPWORDS_PER_LANG: dict[str, set[str]] = {}
GLOBAL_STOPWORDS: set[str] = set()

for _lang in stopwordsiso.langs():
    words = stopwordsiso.stopwords(_lang)
    STOPWORDS_PER_LANG[_lang] = set(words)
    GLOBAL_STOPWORDS |= STOPWORDS_PER_LANG[_lang]

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
        - The scatter plot uses 1%–99% quantile axis limits
        - Output filename has "_robust" appended
    """
    figs_dir.mkdir(parents=True, exist_ok=True)
    labels = df[label_col].to_numpy()

    # Drop HDBSCAN noise for per-cluster summaries/2D scatter
    if label_col == "cluster_hdbscan":
        mask = labels != -1
        df = df[mask].reset_index(drop=True)
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

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            x,
            y,
            c=labels,
            s=5,
            alpha=0.7,
            cmap="tab20",
        )
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title(f"Reduced-space scatter (colored by {label_col})")
        cbar = plt.colorbar(scatter)
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

    cluster_means = (
        df.groupby(label_col, observed=True)[numeric_features]
        .mean()
        .sort_index()
    )
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
    plt.figure(
        figsize=(
            1.5 * len(numeric_features) + 2,
            0.4 * len(cluster_means_z) + 2,
        )
    )
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

def preprocess_text_for_wordcloud(text: str, lang: str | None = None) -> str:
    """
    Clean a conversation text for word cloud generation.

    - Remove role prefixes like 'user:', 'assistant:', 'system:'
    - Lowercase
    - Tokenize on whitespace
    - Drop very short tokens (len < 3 for ASCII tokens)
    - Remove multilingual stopwords (per-language if available, else global)
    """
    if not text:
        return ""

    # Remove role prefixes (case-insensitive)
    text = re.sub(r"\b(user|assistant|system)\s*:", " ", text, flags=re.IGNORECASE)

    # Normalize case
    text = text.lower()

    # Simple whitespace tokenization
    tokens = text.split()

    # Choose stopword set
    if lang and lang in STOPWORDS_PER_LANG:
        sw = STOPWORDS_PER_LANG[lang] | EXTRA_STOPWORDS
    else:
        sw = GLOBAL_STOPWORDS

    # Filter tokens:
    # - For ASCII tokens, require length >= 3
    # - For non-ASCII (e.g. CJK), allow shorter tokens
    filtered: list[str] = []
    for tok in tokens:
        if tok in sw:
            continue
        if tok.isascii() and len(tok) < 3:
            continue
        filtered.append(tok)

    return " ".join(filtered)


def make_cluster_wordclouds(
    df: pd.DataFrame,
    label_col: str,
    text_col: str,
    figs_dir: Path,
    font_path: str | None = None,
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

    for cluster_label, texts in tqdm.tqdm(grouped, desc=f"Generating word clouds ({label_col})"):
        pieces: list[str] = []

        for raw in texts.astype(str):
            if not raw.strip():
                continue

            lang, _ = langid.classify(raw)
            cleaned = preprocess_text_for_wordcloud(raw, lang=lang)
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

        make_cluster_plots(
            df,
            X_red,
            label_col=label_col,
            figs_dir=FIGS_DIR,
            use_robust_limits=False,
        )
        make_cluster_plots(
            df,
            X_red,
            label_col=label_col,
            figs_dir=FIGS_DIR,
            use_robust_limits=True,
        )

    # 2) Word clouds
    print("Generating word clouds per cluster...")
    if "conversation_text" not in df.columns:
        print("Column 'conversation_text' not found; cannot build word clouds.")
    else:
        for label_col in ("cluster_hdbscan", "cluster_leiden", "cluster_kmeans"):
            if label_col not in df.columns:
                continue
            make_cluster_wordclouds(
                df=df,
                label_col=label_col,
                text_col="conversation_text",
                figs_dir=FIGS_DIR,
                font_path=FONT_PATH,
            )

    print("Done.")


if __name__ == "__main__":
    main()
