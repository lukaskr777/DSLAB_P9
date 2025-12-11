"""
Post-clustering visualization script (language-wise) for swiss-ai/apertus-sft-mixture.

This script works on the outputs of the language-wise clustering script
(which uses the RUN_SUFFIX = "_langwise" convention).

It:
- Loads the clustered DataFrame produced by the language-wise clustering script.
- Loads the corresponding UMAP-reduced embeddings (RUN_TAG-based).
- Recomputes a standardized reduced space (X_red) for plotting.
- Regenerates:
    * Scatter plots in reduced space (Dim1 vs Dim2), for:
        - language labels ('lang')
        - language-wise HDBSCAN clusters (language-aware display labels)
        - final aggregated clusters ('cluster_langwise_final', filtered as in global metrics)
    * Cluster size bar plots
    * Per-cluster feature mean heatmap (z-scored)
    * Per-feature bar plots of per-cluster raw means
    * Cluster-wise word clouds (for 'cluster_langwise_final')
    * Top-5 words per final cluster tables (for 'cluster_langwise_final')
    * Representative prompts per final cluster (top 3 per cluster, ordered from shorter to longer;
      both all languages and English-only)

Figures are written under figs/apertus_clustering/RUN_TAG, and that directory is cleared with `ensure_empty_dir`
before plotting, so that previous *non-langwise* figures (under TAG) are preserved.

Additionally, English-only plots and summaries are written under figs/apertus_clustering/RUN_TAG/english.
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
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import stopwordsiso
from wordcloud import WordCloud

from utility_scripts.file_utils import ensure_empty_dir


# ---------- CONFIG (must match the language-wise clustering script) ----------

OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")

EMBEDDINGS_PATH = OUT_DIR / (
    "small_train_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.npy"
)
TAG = EMBEDDINGS_PATH.stem

# Must match RUN_SUFFIX / RUN_TAG in the clustering script
RUN_SUFFIX = "_langwise"
RUN_TAG = f"{TAG}{RUN_SUFFIX}"

UMAP_EMBEDDINGS_PATH = OUT_DIR / f"{RUN_TAG}_umap.npy"
CLUSTERED_DF_PATH = OUT_DIR / f"{RUN_TAG}_clustered.parquet"

# Root directory for figures; use RUN_TAG to avoid overwriting old (non-langwise) figures
FIGS_ROOT = Path("figs/apertus_clustering")
FIGS_DIR = FIGS_ROOT / RUN_TAG

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

for _lang in stopwordsiso.langs():
    words = stopwordsiso.stopwords(_lang)
    STOPWORDS_PER_LANG[_lang] = set(words)
    GLOBAL_STOPWORDS |= STOPWORDS_PER_LANG[_lang]

GLOBAL_STOPWORDS |= EXTRA_STOPWORDS


# ---------- Color map helper (exactly n_clusters colors) ----------

def get_colormap(n_clusters: int) -> mcolors.ListedColormap:
    """
    Get a discrete colormap with exactly n_clusters distinct colors.

    Uses tab20 / tab20b / tab20c as a pool (up to ~60 colors).
    If more colors are requested, falls back to sampling from a continuous colormap.
    """
    if n_clusters <= 0:
        # Fallback: at least one color
        n_clusters = 1

    base1 = colormaps.get_cmap("tab20")
    base2 = colormaps.get_cmap("tab20b")
    base3 = colormaps.get_cmap("tab20c")

    # These are ListedColormap, so they have .colors
    colors1 = np.array(base1.colors)
    colors2 = np.array(base2.colors)
    colors3 = np.array(base3.colors)

    all_colors = np.vstack([colors1, colors2, colors3])
    total_available = all_colors.shape[0]

    if n_clusters <= total_available:
        return mcolors.ListedColormap(all_colors[:n_clusters])

    # More clusters than our pool: sample from a continuous colormap
    continuous = colormaps.get_cmap("viridis")
    sampled = continuous(np.linspace(0.0, 1.0, n_clusters))
    return mcolors.ListedColormap(sampled)


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

    Noise filtering and any language-based filtering are expected to be handled
    by the caller (df / X_red should already be filtered).
    """
    figs_dir.mkdir(parents=True, exist_ok=True)
    labels = df[label_col].to_numpy()

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

        # Discrete colormap with exactly one entry per cluster
        cmap = get_colormap(n_clusters)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x, y, c=idx_colors, cmap=cmap, s=5, alpha=0.7, linewidths=0)
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
        plt.savefig(out_path, dpi=300)
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
    plt.savefig(out_path, dpi=300)
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
        scaler.fit_transform(cluster_means), index=cluster_means.index, columns=cluster_means.columns
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
    plt.savefig(out_path, dpi=300)
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
        plt.savefig(out_path, dpi=300)
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
        * LaTeX / command-like (start with '\\')
        * pure punctuation
        * multilingual stopwords
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

        # Skip stopwords (we have already built GLOBAL_STOPWORDS)
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

    - Uses the GLOBAL_STOPWORDS built from stopwordsiso.
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
        plt.savefig(out_path, dpi=300)
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


# ---------- Representative prompts per cluster ----------

def compute_cluster_representatives(
    df: pd.DataFrame, X_red: np.ndarray, label_col: str, text_col: str, out_path: Path
) -> None:
    """
    For each cluster (in label_col), pick the top 3 representative conversations:

    - Compute centroid of X_red for the cluster.
    - Compute Euclidean distance to centroid for each sample in the cluster.
    - Select the 3 samples with smallest distance.
    - Order these 3 by conversation length (shorter to longer).
    - Use the corresponding conversation_texts as "standard prompts".

    Writes a TSV with columns:
        cluster    prompt1    prompt2    prompt3
    """
    if text_col not in df.columns:
        print(f"[{label_col}] Column {text_col!r} not found, skipping representatives.")
        return

    labels = df[label_col].to_numpy()
    unique_labels = pd.unique(labels)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("cluster\tprompt1\tprompt2\tprompt3\n")

        for lab in unique_labels:
            # Skip NaN labels (if any)
            if pd.isna(lab):
                continue

            mask = labels == lab
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue

            X_c = X_red[idx]
            if X_c.size == 0:
                continue

            # Distances to centroid
            centroid = X_c.mean(axis=0)
            dists = np.sum((X_c - centroid) ** 2, axis=1)
            order = np.argsort(dists)

            # Take up to 3 closest
            top_k = min(3, len(order))
            chosen_indices = idx[order[:top_k]]

            # Collect (length, text) pairs
            texts_with_len: list[tuple[int, str]] = []
            for g_idx in chosen_indices:
                row = df.iloc[g_idx]
                raw_text = str(row[text_col]).replace("\r", " ").replace("\n", " ")
                raw_text = re.sub(r"\s+", " ", raw_text).strip()
                if not raw_text:
                    continue
                texts_with_len.append((len(raw_text), raw_text))

            if not texts_with_len:
                continue

            # Sort by length: shorter to longer
            texts_with_len.sort(key=lambda t: t[0])
            prompts = [t[1] for t in texts_with_len]

            # Pad to exactly 3 prompts
            while len(prompts) < 3:
                prompts.append("")

            f.write(str(lab))
            for p in prompts[:3]:
                f.write("\t" + p)
            f.write("\n")

    print(f"[{label_col}] Cluster representatives (top 3) written to: {out_path}")


# ---------- Filtering helper to match global metrics logic ----------

def filter_final_for_global_plots(df: pd.DataFrame, X_red: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Filter (df, X_red) for global plots on 'cluster_langwise_final' so that:

    - We drop noise labels ('*_noise').
    - We drop pure-language labels (e.g., 'en', 'de').
    - We keep only languages that have >= 2 distinct clusters (e.g., 'en_c0', 'en_c1', ...).

    Returns the filtered DataFrame (with reset index) and the filtered X_red.
    """
    label_col = "cluster_langwise_final"
    if label_col not in df.columns:
        return df, X_red

    labels_all = df[label_col].astype(str)

    # Cluster-like labels: '{lang}_c{cluster_id}'
    mask_cluster_like = labels_all.str.contains("_c")
    # Noise labels: '{lang}_noise'
    mask_noise = labels_all.str.endswith("_noise")

    # Extract language for cluster-like labels
    lang_for_cluster = labels_all[mask_cluster_like].str.split("_c", n=1, expand=True)[0]

    if lang_for_cluster.empty:
        return df.iloc[0:0].copy(), X_red[[]]

    # Count distinct clusters per language
    tmp = pd.DataFrame(
        {
            "lang": lang_for_cluster.to_numpy(),
            "label": labels_all[mask_cluster_like].to_numpy(),
        }
    )
    cluster_counts = tmp.groupby("lang")["label"].nunique()
    langs_with_multi = set(cluster_counts[cluster_counts >= 2].index)

    if not langs_with_multi:
        # No language has >= 2 clusters
        return df.iloc[0:0].copy(), X_red[[]]

    # For all rows, recover language part for cluster-like labels
    all_lang_for_cluster = labels_all.where(mask_cluster_like).str.split("_c", n=1, expand=True)[0]
    mask_lang_has_multi = all_lang_for_cluster.isin(langs_with_multi)

    # Final mask: true clusters, non-noise, language with >= 2 clusters
    mask_keep = mask_cluster_like & (~mask_noise) & mask_lang_has_multi

    df_filt = df.loc[mask_keep].reset_index(drop=True)
    X_filt = X_red[mask_keep.to_numpy()]

    return df_filt, X_filt


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
            "Ensure both were produced by the same language-wise clustering run."
        )

    # Standardize reduced space for plotting (same shape as clustering script)
    scaler = StandardScaler()
    X_red = scaler.fit_transform(X_umap)

    # Build a language-aware display column for HDBSCAN labels
    if "cluster_langwise_hdbscan" in df.columns and "lang" in df.columns:
        cl = df["cluster_langwise_hdbscan"].to_numpy()
        lang = df["lang"].astype(str).to_numpy()

        display_labels = []
        for l, c in zip(lang, cl):
            if c == -1:
                display_labels.append(f"{l}_noise")
            else:
                display_labels.append(f"{l}_c{int(c)}")

        df["cluster_langwise_hdbscan_display"] = pd.Series(display_labels, index=df.index, dtype="string")
        print("Created 'cluster_langwise_hdbscan_display' column.")
    else:
        print("Cannot create 'cluster_langwise_hdbscan_display' (missing 'cluster_langwise_hdbscan' or 'lang').")

    # Recreate figures directory (clear existing plots for this RUN_TAG only)
    ensure_empty_dir(FIGS_DIR)
    print(f"Figures will be written to: {FIGS_DIR}")

    # ---------------------------------------------------------------------
    # 1) Cluster plots (all languages)
    # ---------------------------------------------------------------------
    print("Generating cluster plots (all languages)...")

    # 1.1) Language labels
    if "lang" in df.columns:
        make_cluster_plots(df, X_red, label_col="lang", figs_dir=FIGS_DIR, use_robust_limits=False)
        make_cluster_plots(df, X_red, label_col="lang", figs_dir=FIGS_DIR, use_robust_limits=True)
    else:
        print("Column 'lang' not in DataFrame, skipping language plots.")

    # 1.2) Final aggregated clusters (filtered to match global metrics logic)
    if "cluster_langwise_final" in df.columns:
        df_final, X_final = filter_final_for_global_plots(df, X_red)
        if df_final.empty:
            print("[cluster_langwise_final] No data after global-filtering; skipping final-cluster plots.")
        else:
            make_cluster_plots(
                df_final,
                X_final,
                label_col="cluster_langwise_final",
                figs_dir=FIGS_DIR,
                use_robust_limits=False,
            )
            make_cluster_plots(
                df_final,
                X_final,
                label_col="cluster_langwise_final",
                figs_dir=FIGS_DIR,
                use_robust_limits=True,
            )
    else:
        print("Column 'cluster_langwise_final' not in DataFrame, skipping final cluster plots.")

    # 1.3) Per-language HDBSCAN clusters (language-aware display labels, noise excluded)
    if "cluster_langwise_hdbscan_display" in df.columns:
        mask_non_noise = ~df["cluster_langwise_hdbscan_display"].str.endswith("_noise")
        if mask_non_noise.any():
            df_hdb = df.loc[mask_non_noise].reset_index(drop=True)
            X_hdb = X_red[mask_non_noise.to_numpy()]

            make_cluster_plots(
                df_hdb,
                X_hdb,
                label_col="cluster_langwise_hdbscan_display",
                figs_dir=FIGS_DIR,
                use_robust_limits=False,
            )
            make_cluster_plots(
                df_hdb,
                X_hdb,
                label_col="cluster_langwise_hdbscan_display",
                figs_dir=FIGS_DIR,
                use_robust_limits=True,
            )
        else:
            print("[cluster_langwise_hdbscan_display] All labels are noise; skipping HDBSCAN plots.")
    else:
        print("Column 'cluster_langwise_hdbscan_display' not in DataFrame, skipping HDBSCAN plots.")

    # ---------------------------------------------------------------------
    # 2) Word clouds (all languages, final clusters)
    # ---------------------------------------------------------------------
    print("Generating word clouds per final cluster (all languages)...")
    if "conversation_text" not in df.columns:
        print("Column 'conversation_text' not found; cannot build word clouds.")
    else:
        label_col = "cluster_langwise_final"
        if label_col in df.columns:
            make_cluster_wordclouds(
                df=df,
                label_col=label_col,
                text_col="conversation_text",
                figs_dir=FIGS_DIR,
                font_path=FONT_PATH,
            )
        else:
            print(f"Column {label_col!r} not found; skipping word clouds.")

    # ---------------------------------------------------------------------
    # 3) Top-5 words per final cluster (all languages)
    # ---------------------------------------------------------------------
    print("Computing top-5 words per final cluster (all languages)...")
    if "conversation_text" not in df.columns:
        print("Column 'conversation_text' not found; cannot compute top words.")
    else:
        label_col = "cluster_langwise_final"
        if label_col in df.columns:
            out_path = FIGS_DIR / f"top_words_{label_col}.txt"
            compute_top_words_per_cluster(
                df=df, label_col=label_col, text_col="conversation_text", out_path=out_path
            )
        else:
            print(f"Column {label_col!r} not found; skipping top-words table.")

    # ---------------------------------------------------------------------
    # 4) Representative prompts per final cluster (all languages)
    # ---------------------------------------------------------------------
    print("Computing representative prompts per final cluster (all languages)...")
    if "conversation_text" in df.columns and "cluster_langwise_final" in df.columns:
        out_rep_all = FIGS_DIR / "cluster_representatives_cluster_langwise_final.txt"
        compute_cluster_representatives(
            df=df,
            X_red=X_red,
            label_col="cluster_langwise_final",
            text_col="conversation_text",
            out_path=out_rep_all,
        )
    else:
        print("Missing columns for representatives on all languages; skipping.")

    # ---------------------------------------------------------------------
    # 5) ENGLISH-FOCUSED SECTION
    # ---------------------------------------------------------------------
    if "lang" in df.columns:
        mask_en = df["lang"] == "en"
        if mask_en.any():
            print("Generating English-only plots and summaries...")
            df_en = df.loc[mask_en].reset_index(drop=True)
            X_red_en = X_red[mask_en.to_numpy()]

            figs_dir_en = FIGS_DIR / "english"
            ensure_empty_dir(figs_dir_en)
            print(f"English-only figures will be written to: {figs_dir_en}")

            # 5.1) English-only cluster plots for final and HDBSCAN-display labels
            if "cluster_langwise_final" in df_en.columns:
                make_cluster_plots(
                    df_en,
                    X_red_en,
                    label_col="cluster_langwise_final",
                    figs_dir=figs_dir_en,
                    use_robust_limits=False,
                )
                make_cluster_plots(
                    df_en,
                    X_red_en,
                    label_col="cluster_langwise_final",
                    figs_dir=figs_dir_en,
                    use_robust_limits=True,
                )
            else:
                print("[EN] 'cluster_langwise_final' not in English subset, skipping plots.")

            if "cluster_langwise_hdbscan_display" in df_en.columns:
                mask_en_non_noise = ~df_en["cluster_langwise_hdbscan_display"].str.endswith("_noise")
                if mask_en_non_noise.any():
                    df_en_hdb = df_en.loc[mask_en_non_noise].reset_index(drop=True)
                    X_en_hdb = X_red_en[mask_en_non_noise.to_numpy()]

                    make_cluster_plots(
                        df_en_hdb,
                        X_en_hdb,
                        label_col="cluster_langwise_hdbscan_display",
                        figs_dir=figs_dir_en,
                        use_robust_limits=False,
                    )
                    make_cluster_plots(
                        df_en_hdb,
                        X_en_hdb,
                        label_col="cluster_langwise_hdbscan_display",
                        figs_dir=figs_dir_en,
                        use_robust_limits=True,
                    )
                else:
                    print("[EN] All English HDBSCAN-display labels are noise; skipping HDBSCAN plots.")
            else:
                print("[EN] 'cluster_langwise_hdbscan_display' not in English subset, skipping HDBSCAN plots.")

            # 5.2) English-only word clouds and top words
            if "conversation_text" in df_en.columns:
                # Word clouds
                if "cluster_langwise_final" in df_en.columns:
                    make_cluster_wordclouds(
                        df=df_en,
                        label_col="cluster_langwise_final",
                        text_col="conversation_text",
                        figs_dir=figs_dir_en,
                        font_path=FONT_PATH,
                    )
                else:
                    print("[EN] 'cluster_langwise_final' missing in English subset; skipping word clouds.")

                # Top words
                if "cluster_langwise_final" in df_en.columns:
                    out_top_en = figs_dir_en / "top_words_cluster_langwise_final.txt"
                    compute_top_words_per_cluster(
                        df=df_en,
                        label_col="cluster_langwise_final",
                        text_col="conversation_text",
                        out_path=out_top_en,
                    )
                else:
                    print("[EN] 'cluster_langwise_final' missing in English subset; skipping top-words table.")

                # 5.3) English-only representatives (top 3, shorter→longer)
                if "cluster_langwise_final" in df_en.columns:
                    out_rep_en = figs_dir_en / "cluster_representatives_cluster_langwise_final_en.txt"
                    compute_cluster_representatives(
                        df=df_en,
                        X_red=X_red_en,
                        label_col="cluster_langwise_final",
                        text_col="conversation_text",
                        out_path=out_rep_en,
                    )
                else:
                    print("[EN] 'cluster_langwise_final' missing in English subset; skipping representatives.")
            else:
                print("[EN] 'conversation_text' not found in English subset; skipping English word-based plots.")
        else:
            print("No English conversations found (lang == 'en'); skipping English-only plots.")
    else:
        print("Column 'lang' not found; cannot generate English-only summaries.")

    print("Done.")


if __name__ == "__main__":
    main()
