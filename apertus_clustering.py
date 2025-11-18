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
3. Run PCA on embeddings.
4. Run HDBSCAN on PCA-reduced embeddings.
5. Run KMeans sweep over k, select best k, then run final KMeans.
6. Save:
   - PCA embeddings and PCA model
   - clustered DataFrame
   - HDBSCAN and KMeans models
   - diagnostic plots
   - a TXT file summarizing clustering scores/statistics.

To use different embeddings (e.g., from different models), just change the global EMBEDDINGS_PATH and rerun the script.
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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

import re
import stopwordsiso
import langid
from wordcloud import WordCloud


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
# EMBEDDINGS_PATH = OUT_DIR / (
#     "train_sampled_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.npy"
# )
# EMBEDDINGS_PATH = OUT_DIR / (
#     "train_sampled_conversation_embeddings__sentence-transformers__stsb-xlm-r-multilingual.npy"
# )
EMBEDDINGS_PATH = OUT_DIR / (
    "train_sampled_conversation_embeddings__sentence-transformers__gtr-t5-base.npy"
)

# PCA
N_COMPONENTS_PCA = 50
PCA_RANDOM_STATE = 0

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES: int | None = None  # None -> default (min_cluster_size)
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0

# KMeans
KMEANS_K_VALUES = list(range(2, 32))
KMEANS_RANDOM_STATE = 0

# Silhouette
MAX_SILHOUETTE_SAMPLES: int | None = None  # None -> full dataset

# Feature columns used in cluster summaries / plots
FEATURE_COLUMNS = [
    "n_turns",
    "user_msg_len",
    "assistant_msg_len",
    "text_length",
]


# ---------- Derived paths (per-embeddings tag) ----------

TAG = EMBEDDINGS_PATH.stem  # e.g. "train_sampled_conversation_embeddings__..."

PCA_EMBEDDINGS_PATH = OUT_DIR / f"{TAG}_pca.npy"
PCA_MODEL_PATH = OUT_DIR / f"{TAG}_pca_model.joblib"
CLUSTERED_DF_PATH = OUT_DIR / f"{TAG}_clustered.parquet"
HDBSCAN_MODEL_PATH = OUT_DIR / f"{TAG}_hdbscan_model.joblib"
KMEANS_MODEL_PATH = OUT_DIR / f"{TAG}_kmeans_model.joblib"
SCORES_TXT_PATH = OUT_DIR / f"{TAG}_cluster_scores.txt"

FIGS_DIR = FIGS_ROOT / TAG

FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS system font


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


# ---------- PCA + clustering ----------

def reduce_dimensionality(
    emb: np.ndarray,
    n_components: int = N_COMPONENTS_PCA,
    random_state: int = PCA_RANDOM_STATE,
) -> tuple[np.ndarray, PCA]:
    """
    PCA reduction on embedding matrix (n_samples, dim) -> (n_samples, n_components_eff),
    and return the fitted PCA object.
    """
    n_samples, dim = emb.shape
    if n_samples <= 1:
        return emb.copy(), PCA(n_components=min(dim, 1), random_state=random_state)

    n_components_eff = min(n_components, dim, n_samples - 1)
    pca = PCA(n_components=n_components_eff, random_state=random_state)
    X_pca = pca.fit_transform(emb)
    return X_pca, pca


def cluster_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    cluster_selection_method: str = HDBSCAN_CLUSTER_SELECTION_METHOD,
    cluster_selection_epsilon: float = HDBSCAN_CLUSTER_SELECTION_EPSILON,
) -> tuple[np.ndarray, hdbscan.HDBSCAN]:
    """Cluster with HDBSCAN. Returns label array (noise labeled as -1) and the fitted model."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer


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
    X_pca: np.ndarray,
    label_col: str,
    figs_dir: Path,
) -> None:
    """
    Produce basic diagnostic plots:
    - PCA scatter (PC1 vs PC2) colored by cluster
    - Cluster size bar chart
    - Heatmap of per-cluster feature means (z-scored)
    - Per-feature bar charts of per-cluster raw means
    """
    figs_dir.mkdir(parents=True, exist_ok=True)
    labels = df[label_col].to_numpy()

    # Drop label -1 for HDBSCAN noise in summaries/plots
    if label_col == "cluster_hdbscan":
        mask = labels != -1
        df = df[mask].reset_index(drop=True)
        labels = labels[mask]
        X_pca = X_pca[mask]

    # --- PCA scatter: PC1 vs PC2 colored by cluster ---
    if X_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            s=5,
            alpha=0.7,
            cmap="tab20",
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA scatter (colored by {label_col})")
        cbar = plt.colorbar(scatter)
        cbar.set_label(label_col)
        plt.tight_layout()
        out_path = figs_dir / f"pca_scatter_pc1_pc2_{label_col}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # --- Cluster size bar chart ---
    counts = df[label_col].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.xlabel("Cluster")
    plt.ylabel("Number of conversations")
    plt.title(f"Cluster sizes ({label_col})")
    plt.tight_layout()
    out_path = figs_dir / f"cluster_sizes_{label_col}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    # --- Per-cluster feature means ---
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

    # Z-score scaling across clusters so features are comparable in the heatmap
    scaler = StandardScaler()
    cluster_means_z = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns,
    )

    # Heatmap of z-scored feature means
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

    # Individual bar plots per feature (raw means)
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
    - Drop very short tokens (len < 3).
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

    # Filter tokens: length >= 3 and not in stopwords
    filtered = [
        tok
        for tok in tokens
        if len(tok) >= 3 and tok not in sw
    ]

    return " ".join(filtered)


def make_cluster_wordclouds(
    df: pd.DataFrame,
    label_col: str,
    text_col: str,
    figs_dir: Path,
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

            # Language detection
            lang, _ = langid.classify(raw)

            cleaned = preprocess_text_for_wordcloud(raw, lang=lang)
            if cleaned:
                pieces.append(cleaned)

        combined = "\n".join(pieces)
        if not combined.strip():
            continue

        # We already applied our own stopwords, so pass an empty set here.
        wc = WordCloud(
            width=1600,
            height=900,
            background_color="white",
            stopwords=set(),
            max_words=200,
            font_path=FONT_PATH,
        ).generate(combined)

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
    X_pca: np.ndarray,
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

    # --- HDBSCAN metrics on non-noise points ---
    # Get all HDBSCAN labels as a numpy array once
    labels_all_hdb = df["cluster_hdbscan"].to_numpy()
    mask_hdb = labels_all_hdb != -1

    if mask_hdb.any():
        X_hdb = X_pca[mask_hdb]
        # Index the numpy array directly and cast to int to make Pylance happy
        labels_hdb = labels_all_hdb[mask_hdb].astype(int)
        metrics_hdb = evaluate_clustering_metrics(X_hdb, labels_hdb)
    else:
        metrics_hdb = {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    # --- Final KMeans metrics ---
    labels_k_best = df["cluster_kmeans"].to_numpy().astype(int)
    metrics_k_best = evaluate_clustering_metrics(X_pca, labels_k_best)

    with scores_path.open("w", encoding="utf-8") as f:
        f.write(f"Embeddings file: {EMBEDDINGS_PATH}\n")
        f.write(f"TAG: {TAG}\n")
        f.write(f"n_samples: {n_samples}\n")
        f.write(f"PCA shape: {X_pca.shape}\n")
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

    # Sanity check: clean NaN / inf in embeddings before PCA
    if not np.isfinite(emb).all():
        row_mask = np.isfinite(emb).all(axis=1)
        n_bad_rows = (~row_mask).sum()
        print(f"Dropping {n_bad_rows} rows with NaN/inf embeddings before PCA.")
        bad_idx_path = OUT_DIR / f"{TAG}_dropped_embedding_rows.txt"
        np.savetxt(bad_idx_path, np.where(~row_mask)[0], fmt="%d")
        print(f"Saved indices of dropped rows to: {bad_idx_path}")
        emb = emb[row_mask]
        df = df.loc[row_mask].reset_index(drop=True)

    # PCA
    if PCA_EMBEDDINGS_PATH.exists() and PCA_MODEL_PATH.exists():
        print(f"Loading precomputed PCA embeddings from: {PCA_EMBEDDINGS_PATH}")
        X_pca = np.load(PCA_EMBEDDINGS_PATH)
        print(f"Loading PCA model from: {PCA_MODEL_PATH}")
        pca = joblib.load(PCA_MODEL_PATH)
    else:
        print("Running PCA on embeddings...")
        X_pca, pca = reduce_dimensionality(emb, n_components=N_COMPONENTS_PCA)
        np.save(PCA_EMBEDDINGS_PATH, X_pca)
        joblib.dump(pca, PCA_MODEL_PATH)
        print(f"Saved PCA-reduced embeddings to: {PCA_EMBEDDINGS_PATH}")
        print(f"Saved PCA model to: {PCA_MODEL_PATH}")

    # HDBSCAN
    print("Clustering with HDBSCAN on PCA-reduced embeddings...")
    labels_hdbscan, hdbscan_model = cluster_hdbscan(X_pca)
    df["cluster_hdbscan"] = labels_hdbscan
    joblib.dump(hdbscan_model, HDBSCAN_MODEL_PATH)
    print(f"Saved HDBSCAN model to: {HDBSCAN_MODEL_PATH}")

    # KMeans sweep + final clustering
    print("Running KMeans sweep over k values...")
    best_k, kmeans_results = sweep_kmeans(X_pca, KMEANS_K_VALUES)
    print(f"Selected k for KMeans: {best_k}")
    print(f"Clustering with KMeans using k={best_k} on PCA-reduced embeddings...")
    labels_kmeans, kmeans_model = cluster_kmeans(X_pca, n_clusters=best_k)
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
    make_cluster_plots(df, X_pca, label_col="cluster_hdbscan", figs_dir=FIGS_DIR)
    make_cluster_plots(df, X_pca, label_col="cluster_kmeans", figs_dir=FIGS_DIR)

    # Word clouds
    print("\nGenerating word clouds per cluster...")
    make_cluster_wordclouds(
        df=df,
        label_col="cluster_hdbscan",
        text_col="conversation_text",
        figs_dir=FIGS_DIR,
    )
    make_cluster_wordclouds(
        df=df,
        label_col="cluster_kmeans",
        text_col="conversation_text",
        figs_dir=FIGS_DIR,
    )

    # Scores file
    print(f"\nWriting clustering scores to: {SCORES_TXT_PATH}")
    write_scores_file(
        df=df,
        X_pca=X_pca,
        kmeans_results=kmeans_results,
        best_k=best_k,
        scores_path=SCORES_TXT_PATH,
    )
    print("Done.")


if __name__ == "__main__":
    main()
