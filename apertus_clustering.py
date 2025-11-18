"""
Clustering script for swiss-ai/apertus-sft-mixture.

This script assumes that conversation-level embeddings have already been
computed and stored in a .npy file (one row per conversation, in the same
order as the DataFrame after parsing messages).

Pipeline:
1. Load original Parquet and parse/flatten `messages` into:
   - `conversation_text`
   - simple conversation-level features (n_turns, lengths, etc.).
2. Load precomputed L2-normalized embeddings from EMBEDDINGS_PATH.
3. Clean embeddings: drop rows with non-finite values.
4. L2-normalize embeddings again (cosine geometry).
5. Run UMAP (metric='cosine') to reduce to a low-dimensional space.
6. Standardize UMAP coordinates.
7. Run HDBSCAN on the reduced space.
8. Run KMeans sweep over k on the same reduced space, select best k,
   then run final KMeans.
9. Save:
   - UMAP embeddings and UMAP model
   - clustered DataFrame
   - HDBSCAN and KMeans models
   - diagnostic plots
   - a TXT file summarizing clustering scores/statistics.

To use different embeddings (e.g., from different models), just change the
global EMBEDDINGS_PATH and rerun the script.
"""

import json
from pathlib import Path
from typing import Any, Iterable
import warnings

# Silence stopwordsiso/pkg_resources deprecation warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

# Silence sklearn 'force_all_finite' deprecation warning used inside HDBSCAN
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning,
)

import tqdm
import hdbscan
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

import re
import stopwordsiso
import langid
from wordcloud import WordCloud
import umap


# ---------- CONFIG ----------

DATA_PATH = Path("data/swiss-ai_apertus-sft-mixture/train_sampled_enriched.parquet")
OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")

# Root directory for figures; a subdirectory will be created per embeddings file
FIGS_ROOT = Path("figs/apertus_clustering")

# ---- Embeddings to use for this run ----
# Change this path to point to the embeddings file you want to cluster.
# EMBEDDINGS_PATH = OUT_DIR / (
#     "train_sampled_conversation_embeddings__sentence-transformers__all-MiniLM-L6-v2.npy"
# )
EMBEDDINGS_PATH = OUT_DIR / (
    "train_sampled_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.npy"
)
# EMBEDDINGS_PATH = OUT_DIR / (
#     "train_sampled_conversation_embeddings__sentence-transformers__stsb-xlm-r-multilingual.npy"
# )
# EMBEDDINGS_PATH = OUT_DIR / (
#     "train_sampled_conversation_embeddings__sentence-transformers__gtr-t5-base.npy"
# )

# UMAP settings
UMAP_N_COMPONENTS = 15
UMAP_N_NEIGHBORS = 70
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 0

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 250
HDBSCAN_MIN_SAMPLES = 50  # None -> default (min_cluster_size)
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0

# KMeans
KMEANS_K_VALUES = list(range(2, 42))
KMEANS_RANDOM_STATE = 0

# Silhouette
MAX_SILHOUETTE_SAMPLES: int | None = None  # None -> use all samples

# Feature columns used in cluster summaries / plots
FEATURE_COLUMNS = [
    "n_turns",
    "user_msg_len",
    "assistant_msg_len",
    "text_length",
]


# ---------- Derived paths (per-embeddings tag) ----------

TAG = EMBEDDINGS_PATH.stem  # e.g. "train_sampled_conversation_embeddings__..."

UMAP_EMBEDDINGS_PATH = OUT_DIR / f"{TAG}_umap.npy"
UMAP_MODEL_PATH = OUT_DIR / f"{TAG}_umap_model.joblib"
CLUSTERED_DF_PATH = OUT_DIR / f"{TAG}_clustered.parquet"
HDBSCAN_MODEL_PATH = OUT_DIR / f"{TAG}_hdbscan_model.joblib"
KMEANS_MODEL_PATH = OUT_DIR / f"{TAG}_kmeans_model.joblib"
SCORES_TXT_PATH = OUT_DIR / f"{TAG}_cluster_scores.txt"

FIGS_DIR = FIGS_ROOT / TAG

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


# ---------- Multilingual stopwords for word clouds ----------

EXTRA_STOPWORDS = {"user", "assistant", "system"}

# Per-language and global stopwords from stopwordsiso
STOPWORDS_PER_LANG: dict[str, set[str]] = {}
GLOBAL_STOPWORDS: set[str] = set()

for _lang in stopwordsiso.langs():
    words = stopwordsiso.stopwords(_lang)
    STOPWORDS_PER_LANG[_lang] = set(words)
    GLOBAL_STOPWORDS |= STOPWORDS_PER_LANG[_lang]

GLOBAL_STOPWORDS |= EXTRA_STOPWORDS


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


def add_conversation_text_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse `messages` column into:
    - conversation_text
    - n_turns
    - user_msg_len
    - assistant_msg_len
    - text_length
    """
    parsed_messages: list[list[dict[str, Any]]] = []
    for msgs in df["messages"]:
        parsed_messages.append(_ensure_list_of_dicts(msgs))

    df = df.copy()
    df["messages_parsed"] = parsed_messages

    empty_ratio = (df["messages_parsed"].str.len() == 0).mean()
    print(f"Fraction of conversations with 0 parsed messages: {empty_ratio:.3f}")

    conversation_texts: list[str] = []
    n_turns: list[int] = []
    user_msg_len: list[int] = []
    assistant_msg_len: list[int] = []

    for msgs in parsed_messages:
        conversation_text = _flatten_messages_to_text(msgs)
        conversation_texts.append(conversation_text)
        n_turns.append(len(msgs))

        u_len = 0
        a_len = 0
        for m in msgs:
            content = m.get("content")
            text = _extract_text_from_content(content)
            role = m.get("role", "")
            if role == "user":
                u_len += len(text)
            elif role == "assistant":
                a_len += len(text)
        user_msg_len.append(u_len)
        assistant_msg_len.append(a_len)

    df["conversation_text"] = conversation_texts
    df["n_turns"] = n_turns
    df["user_msg_len"] = user_msg_len
    df["assistant_msg_len"] = assistant_msg_len
    df["text_length"] = df["conversation_text"].str.len().fillna(0).astype(int)

    df = df[df["conversation_text"].str.strip().astype(bool)].reset_index(drop=True)

    # Drop heavy intermediate column if not needed later
    df = df.drop(columns=["messages_parsed"])

    return df


# ---------- Metrics helpers ----------

def evaluate_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    max_silhouette_samples: int | None = MAX_SILHOUETTE_SAMPLES,
) -> dict[str, float]:
    """
    Compute internal clustering metrics on X for the given labels.

    Returns a dict with keys:
        - "silhouette"
        - "davies_bouldin"
        - "calinski_harabasz"
    """
    # Ensure 1D labels and consistent length
    labels = np.asarray(labels)
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of samples.")

    unique_labels = np.unique(labels)
    # If all points are in one cluster, metrics are undefined
    if unique_labels.size <= 1:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    n_samples = X.shape[0]

    # Silhouette (optionally subsampled)
    try:
        if max_silhouette_samples is not None and n_samples > max_silhouette_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(n_samples, size=max_silhouette_samples, replace=False)
            sil = float(silhouette_score(X[idx], labels[idx]))
        else:
            sil = float(silhouette_score(X, labels))
    except Exception:
        sil = float("nan")

    # Davies–Bouldin and Calinski–Harabasz on full data
    try:
        db = float(davies_bouldin_score(X, labels))
    except Exception:
        db = float("nan")

    try:
        ch = float(calinski_harabasz_score(X, labels))
    except Exception:
        ch = float("nan")

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
    }


def sweep_kmeans(
    X: np.ndarray,
    k_values: list[int],
    random_state: int = KMEANS_RANDOM_STATE,
    max_silhouette_samples: int | None = MAX_SILHOUETTE_SAMPLES,
) -> tuple[int, dict[int, dict[str, float]]]:
    """
    Run KMeans for a range of k, compute inertia and internal indices,
    and return the selected k and a dict of metrics for each k.

    Metrics per k:
        - inertia
        - silhouette
        - davies_bouldin
        - calinski_harabasz

    Selection:
        1. Max silhouette (if any finite value exists)
        2. Else, min inertia.
    """
    n_samples = X.shape[0]
    results: dict[int, dict[str, float]] = {}

    for k in tqdm.tqdm(k_values, desc="Sweeping k for KMeans"):
        k_eff = min(k, n_samples)  # guard against k > n_samples
        if k_eff <= 1:
            results[k] = {
                "inertia": float("nan"),
                "silhouette": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"),
            }
            continue

        km = KMeans(n_clusters=k_eff, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)

        metrics = evaluate_clustering_metrics(X, labels, max_silhouette_samples)
        results[k] = {
            "inertia": inertia,
            **metrics,
        }

    # Selection: prefer max silhouette if any is finite, else min inertia
    valid_sil = {
        k: r["silhouette"]
        for k, r in results.items()
        if not np.isnan(r["silhouette"])
    }

    if valid_sil:
        best_k = max(valid_sil, key=lambda kv: valid_sil[kv])
    else:
        valid_inertia = {
            k: r["inertia"]
            for k, r in results.items()
            if not np.isnan(r["inertia"])
        }
        if not valid_inertia:
            best_k = min(k_values)
        else:
            best_k = min(valid_inertia, key=lambda kv: valid_inertia[kv])

    return best_k, results


def cluster_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = KMEANS_RANDOM_STATE,
) -> tuple[np.ndarray, KMeans]:
    """Cluster with KMeans for a given k. Returns label array and fitted KMeans model."""
    n_samples = X.shape[0]
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=int), KMeans(
            n_clusters=1, random_state=random_state, n_init="auto"
        )

    n_clusters = min(n_clusters, n_samples)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return labels, km


# ---------- Plotting ----------

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

    # Drop HDBSCAN noise for per-cluster summaries
    if label_col == "cluster_hdbscan":
        mask = labels != -1
        df = df[mask].reset_index(drop=True)
        labels = labels[mask]
        X_red = X_red[mask]

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
            # Focus on the central mass of points
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
    FEATURE_COLUMNS = ["n_turns", "user_msg_len", "assistant_msg_len", "text_length"]
    numeric_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not numeric_features:
        return

    cluster_means = (
        df.groupby(label_col, observed=True)[numeric_features]
        .mean()
        .sort_index()
    )
    if cluster_means.empty:
        return

    scaler = StandardScaler()
    cluster_means_z = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns,
    )

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

    - Remove role prefixes like 'user:', 'assistant:', 'system:'.
    - Lowercase.
    - Tokenize on whitespace.
    - Drop very short tokens (len < 3 for ASCII tokens).
    - Remove multilingual stopwords (per-language if available, else global).
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
    filtered = []
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
        return

    wc_dir = figs_dir / "wordclouds" / label_col
    wc_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(label_col, observed=True)[text_col]

    for cluster_label, texts in tqdm.tqdm(grouped, desc="Generating word clouds"):
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


# ---------- Scores file ----------

def write_scores_file(
    df: pd.DataFrame,
    X_red: np.ndarray,
    kmeans_results: dict[int, dict[str, float]],
    best_k: int,
    scores_path: Path,
) -> None:
    """Write clustering statistics and scores to a TXT file."""
    counts_hdb = df["cluster_hdbscan"].value_counts(dropna=False).sort_index()
    counts_km = df["cluster_kmeans"].value_counts(dropna=False).sort_index()

    n_samples = len(df)
    n_noise = int(counts_hdb.get(-1, 0))
    noise_frac = n_noise / n_samples if n_samples else float("nan")
    n_clusters_hdb = int((counts_hdb.index != -1).sum())

    # HDBSCAN metrics on non-noise points
    labels_all_hdb = df["cluster_hdbscan"].to_numpy()
    mask_hdb = labels_all_hdb != -1
    if mask_hdb.any():
        X_hdb = X_red[mask_hdb]
        labels_hdb = labels_all_hdb[mask_hdb].astype(int)
        metrics_hdb = evaluate_clustering_metrics(X_hdb, labels_hdb)
    else:
        metrics_hdb = {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    # Final KMeans metrics
    labels_k_best = df["cluster_kmeans"].to_numpy().astype(int)
    metrics_k_best = evaluate_clustering_metrics(X_red, labels_k_best)

    with scores_path.open("w", encoding="utf-8") as f:
        f.write(f"Embeddings file: {EMBEDDINGS_PATH}\n")
        f.write(f"TAG: {TAG}\n")
        f.write(f"n_samples: {n_samples}\n")
        f.write(f"Reduced shape (UMAP dims): {X_red.shape}\n")
        f.write("\n")

        # HDBSCAN summary
        f.write("=== HDBSCAN ===\n")
        f.write(f"min_cluster_size: {HDBSCAN_MIN_CLUSTER_SIZE}\n")
        f.write(f"min_samples: {HDBSCAN_MIN_SAMPLES}\n")
        f.write(f"metric: {HDBSCAN_METRIC}\n")
        f.write(f"cluster_selection_method: {HDBSCAN_CLUSTER_SELECTION_METHOD}\n")
        f.write(f"cluster_selection_epsilon: {HDBSCAN_CLUSTER_SELECTION_EPSILON}\n")
        f.write(f"num_clusters_excl_noise: {n_clusters_hdb}\n")
        f.write(f"n_noise: {n_noise}\n")
        f.write(f"noise_fraction: {noise_frac:.4f}\n")
        f.write(
            "internal_metrics_on_non_noise: "
            f"silhouette={metrics_hdb['silhouette']:.6f}, "
            f"davies_bouldin={metrics_hdb['davies_bouldin']:.6f}, "
            f"calinski_harabasz={metrics_hdb['calinski_harabasz']:.6f}\n"
        )
        f.write("cluster_sizes (including noise):\n")
        for label, cnt in counts_hdb.items():
            f.write(f"  label={label}: count={cnt}\n")
        f.write("\n")

        # KMeans sweep
        f.write("=== KMeans sweep ===\n")
        f.write("k, inertia, silhouette, davies_bouldin, calinski_harabasz\n")
        for k in sorted(kmeans_results):
            r = kmeans_results[k]
            inertia = r["inertia"]
            sil = r["silhouette"]
            db = r["davies_bouldin"]
            ch = r["calinski_harabasz"]
            sil_str = f"{sil:.6f}" if not np.isnan(sil) else "nan"
            db_str = f"{db:.6f}" if not np.isnan(db) else "nan"
            ch_str = f"{ch:.6f}" if not np.isnan(ch) else "nan"
            f.write(f"{k}, {inertia:.6e}, {sil_str}, {db_str}, {ch_str}\n")
        f.write(f"\nSelected k (best_k): {best_k}\n")
        f.write(
            "Final KMeans internal_metrics: "
            f"silhouette={metrics_k_best['silhouette']:.6f}, "
            f"davies_bouldin={metrics_k_best['davies_bouldin']:.6f}, "
            f"calinski_harabasz={metrics_k_best['calinski_harabasz']:.6f}\n\n"
        )

        # Final KMeans cluster sizes
        f.write("=== KMeans (k = best_k) cluster sizes ===\n")
        for label, cnt in counts_km.items():
            f.write(f"  label={label}: count={cnt}\n")


# ---------- Main pipeline ----------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using embeddings from: {EMBEDDINGS_PATH}")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")

    emb = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings shape: {emb.shape}")

    # Load data and compute conversation_text + features
    print(f"Loading sampled data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded shape: {df.shape}")

    print("Parsing `messages` and computing conversation-level features...")
    df = add_conversation_text_and_features(df)
    print(f"After dropping empty conversations: {df.shape}")

    if len(df) != emb.shape[0]:
        raise ValueError(
            f"Mismatch between DataFrame rows ({len(df)}) and embeddings rows ({emb.shape[0]}). "
            "Ensure you are using embeddings computed on this exact filtered dataset and in the same order."
        )

    # Drop rows with any NaN/inf in embeddings
    mask_finite_rows = np.isfinite(emb).all(axis=1)
    n_bad_rows = int((~mask_finite_rows).sum())
    if n_bad_rows > 0:
        bad_idx = np.where(~mask_finite_rows)[0]
        print(
            f"Dropping {n_bad_rows} rows with non-finite embeddings "
            f"(out of {emb.shape[0]} total rows)."
        )
        bad_idx_path = OUT_DIR / f"{TAG}_nonfinite_embedding_rows.txt"
        np.savetxt(bad_idx_path, bad_idx, fmt="%d")
        print(f"Saved indices of dropped rows to: {bad_idx_path}")

        emb = emb[mask_finite_rows]
        df = df.loc[mask_finite_rows].reset_index(drop=True)

    # L2-normalize embeddings (cosine geometry)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    emb_norm = emb / norms

        # UMAP reduction
    X_umap: np.ndarray
    if UMAP_EMBEDDINGS_PATH.exists() and UMAP_MODEL_PATH.exists():
        print(f"Loading precomputed UMAP embeddings from: {UMAP_EMBEDDINGS_PATH}")
        X_umap = np.load(UMAP_EMBEDDINGS_PATH)
        print(f"Loading UMAP model from: {UMAP_MODEL_PATH}")
        umap_model = joblib.load(UMAP_MODEL_PATH)
    else:
        print("Running UMAP on embeddings...")
        umap_model = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            n_components=UMAP_N_COMPONENTS,
            metric=UMAP_METRIC,
            random_state=UMAP_RANDOM_STATE,
        )
        # Force to numpy array so Pylance knows the type
        X_umap = np.asarray(umap_model.fit_transform(emb_norm), dtype=float)
        np.save(UMAP_EMBEDDINGS_PATH, X_umap)
        joblib.dump(umap_model, UMAP_MODEL_PATH)
        print(f"Saved UMAP-reduced embeddings to: {UMAP_EMBEDDINGS_PATH}")
        print(f"Saved UMAP model to: {UMAP_MODEL_PATH}")
    print(f"UMAP shape: {X_umap.shape}")

    # Standardize reduced space for clustering
    scaler = StandardScaler()
    X_red = scaler.fit_transform(X_umap)

    # HDBSCAN on reduced space
    print("Clustering with HDBSCAN on UMAP-reduced embeddings...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
    )
    labels_hdbscan = clusterer.fit_predict(X_red)
    df["cluster_hdbscan"] = labels_hdbscan
    joblib.dump(clusterer, HDBSCAN_MODEL_PATH)
    print(f"Saved HDBSCAN model to: {HDBSCAN_MODEL_PATH}")

    # KMeans sweep + final clustering on reduced space
    print("Running KMeans sweep over k values...")
    best_k, kmeans_results = sweep_kmeans(X_red, KMEANS_K_VALUES)
    print(f"Selected k for KMeans: {best_k}")
    print(f"Clustering with KMeans using k={best_k} on UMAP-reduced embeddings...")
    labels_kmeans, kmeans_model = cluster_kmeans(X_red, n_clusters=best_k)
    df["cluster_kmeans"] = labels_kmeans
    joblib.dump(kmeans_model, KMEANS_MODEL_PATH)
    print(f"Saved KMeans model to: {KMEANS_MODEL_PATH}")

    # Save clustered DataFrame
    df.to_parquet(CLUSTERED_DF_PATH, index=False)
    print(f"Saved clustered conversations to: {CLUSTERED_DF_PATH}")

    # Quick cluster summary to stdout
    print("\nCluster counts for HDBSCAN:")
    print(df["cluster_hdbscan"].value_counts(dropna=False).sort_index())
    print("\nCluster counts for KMeans:")
    print(df["cluster_kmeans"].value_counts(dropna=False).sort_index())

    # Plots
    print("\nGenerating cluster plots...")
    make_cluster_plots(df, X_red, label_col="cluster_hdbscan", figs_dir=FIGS_DIR)
    make_cluster_plots(df, X_red, label_col="cluster_kmeans", figs_dir=FIGS_DIR)
    make_cluster_plots(df, X_red, label_col="cluster_hdbscan", figs_dir=FIGS_DIR, use_robust_limits=True)
    make_cluster_plots(df, X_red, label_col="cluster_kmeans", figs_dir=FIGS_DIR, use_robust_limits=True)

    # Word clouds
    print("\nGenerating word clouds per cluster...")
    make_cluster_wordclouds(
        df=df,
        label_col="cluster_hdbscan",
        text_col="conversation_text",
        figs_dir=FIGS_DIR,
        font_path=FONT_PATH,
    )
    make_cluster_wordclouds(
        df=df,
        label_col="cluster_kmeans",
        text_col="conversation_text",
        figs_dir=FIGS_DIR,
        font_path=FONT_PATH,
    )

    # Scores file
    print(f"\nWriting clustering scores to: {SCORES_TXT_PATH}")
    write_scores_file(
        df=df,
        X_red=X_red,
        kmeans_results=kmeans_results,
        best_k=best_k,
        scores_path=SCORES_TXT_PATH,
    )
    print("Done.")


if __name__ == "__main__":
    main()
