"""
Clustering script for swiss-ai/apertus-sft-mixture.

This script assumes that conversation-level embeddings have already been computed and stored in a .npy file
(one row per conversation, in the same order as the DataFrame after parsing messages).

Pipeline:
1. Load original Parquet and parse/flatten `messages` into:
   - `conversation_text`
   - simple conversation-level features (n_turns, lengths, etc.).
2. Load precomputed L2-normalized embeddings from EMBEDDINGS_PATH.
3. Clean embeddings: drop rows with non-finite values.
4. L2-normalize embeddings (cosine geometry).
5. Run UMAP (metric='cosine') to reduce to a low-dimensional space; cache:
   - UMAP-reduced embeddings (X_umap)
   - UMAP model (for graph-based clustering such as Leiden).
6. Standardize UMAP coordinates (X_red).
7. Run HDBSCAN on X_red.
8. Run Leiden clustering on the UMAP k-NN graph.
9. Run KMeans sweep over k on X_red, select best k, then run final KMeans.
10. Save:
    - UMAP embeddings
    - clustered DataFrame
    - a TXT file summarizing clustering scores/statistics.

To use different embeddings (e.g., from different models), just change the global EMBEDDINGS_PATH and rerun the script.
"""

from pathlib import Path
from typing import cast
import warnings

# Silence sklearn 'force_all_finite' deprecation warning used inside HDBSCAN
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning,
)

# Silence UMAP warning about fixed random state
warnings.filterwarnings(
    "ignore",
    message="n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.",
    category=UserWarning,
)

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import hdbscan
import igraph as ig
import leidenalg as la
import umap

from utility_scripts.clustering_utils import ensure_list_of_dicts, extract_text_from_content, flatten_messages_to_text


# ---------- CONFIG ----------

DATA_PATH = Path("data/swiss-ai_apertus-sft-mixture/train_sampled.parquet")
OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")

# ---- Embeddings to use for this run ----
# Change this path to point to the desired embeddings file.
EMBEDDINGS_PATH = OUT_DIR / (
    "train_sampled_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.npy"
)

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

# Leiden
LEIDEN_RESOLUTION = 1.0
LEIDEN_RANDOM_STATE = 0

# Silhouette
MAX_SILHOUETTE_SAMPLES: int | None = None  # None -> use all samples


# ---------- Derived paths (per-embeddings tag) ----------

TAG = EMBEDDINGS_PATH.stem

UMAP_EMBEDDINGS_PATH = OUT_DIR / f"{TAG}_umap.npy"
UMAP_MODEL_PATH = OUT_DIR / f"{TAG}_umap_model.joblib"
SCALER_PATH = OUT_DIR / f"{TAG}_umap_scaler.joblib"
KMEANS_MODEL_PATH = OUT_DIR / f"{TAG}_kmeans_model.joblib"
CLUSTERED_DF_PATH = OUT_DIR / f"{TAG}_clustered.parquet"
SCORES_TXT_PATH = OUT_DIR / f"{TAG}_cluster_scores.txt"


# ---------- Small helpers ----------

EMPTY_METRICS = {
    "silhouette": float("nan"),
    "davies_bouldin": float("nan"),
    "calinski_harabasz": float("nan"),
}


def _format_metric(value: float) -> str:
    """Format a metric as '%.6f' or 'nan' if not finite."""
    return f"{value:.6f}" if not np.isnan(value) else "nan"


# ---------- Data preparation ----------

def add_conversation_text_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse `messages` into conversation-level text and simple features.

    Adds:
        - conversation_text
        - n_turns
        - user_msg_len
        - assistant_msg_len
        - text_length

    Drops conversations where conversation_text is empty/whitespace.
    """
    parsed_messages = [ensure_list_of_dicts(msgs) for msgs in df["messages"]]

    empty_ratio = sum(len(m) == 0 for m in parsed_messages) / max(len(parsed_messages), 1)
    print(f"Fraction of conversations with 0 parsed messages: {empty_ratio:.3f}")

    df = df.copy()

    conversation_texts: list[str] = []
    n_turns: list[int] = []
    user_msg_len: list[int] = []
    assistant_msg_len: list[int] = []

    for msgs in parsed_messages:
        conversation_text = flatten_messages_to_text(msgs)
        conversation_texts.append(conversation_text)
        n_turns.append(len(msgs))

        u_len = 0
        a_len = 0
        for m in msgs:
            content = m.get("content")
            text = extract_text_from_content(content)
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

    mask_nonempty = df["conversation_text"].str.strip().astype(bool)
    df = df.loc[mask_nonempty].reset_index(drop=True)

    return df


# ---------- Metrics helpers ----------

def evaluate_clustering_metrics(
    X: np.ndarray, labels: np.ndarray, max_silhouette_samples: int | None = MAX_SILHOUETTE_SAMPLES,
) -> dict[str, float]:
    """
    Compute internal clustering metrics on X for the given labels.

    Returns a dict with keys:
        - "silhouette"
        - "davies_bouldin"
        - "calinski_harabasz"
    """
    labels = np.asarray(labels)
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of samples.")

    unique_labels = np.unique(labels)
    if unique_labels.size <= 1:
        return EMPTY_METRICS.copy()

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

    for k in tqdm(k_values, desc="Sweeping k for KMeans"):
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
        results[k] = {"inertia": inertia, **metrics}

    # Selection: prefer max silhouette if any is finite, else min inertia
    valid_sil = {k: r["silhouette"] for k, r in results.items() if not np.isnan(r["silhouette"])}

    if valid_sil:
        best_k = max(valid_sil.items(), key=lambda kv: kv[1])[0]
    else:
        valid_inertia = {k: r["inertia"] for k, r in results.items() if not np.isnan(r["inertia"])}
        if valid_inertia:
            best_k = min(valid_inertia.items(), key=lambda kv: kv[1])[0]
        else:
            best_k = min(k_values)

    return best_k, results


def cluster_kmeans(X: np.ndarray, n_clusters: int, random_state: int = KMEANS_RANDOM_STATE) -> KMeans:
    """Cluster with KMeans for a given k and return the fitted KMeans object."""
    n_samples = X.shape[0]
    if n_samples <= 1:
        km = KMeans(n_clusters=1, random_state=random_state, n_init="auto")
        km.fit(X)
        return km

    n_clusters = min(n_clusters, n_samples)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(X)
    return km


def cluster_leiden_from_umap_graph(
    umap_graph: csr_matrix,
    resolution: float = LEIDEN_RESOLUTION,
    *,
    seed: int = LEIDEN_RANDOM_STATE,
) -> np.ndarray:
    """
    Run Leiden clustering on the UMAP k-NN graph and return cluster labels.

    Uses igraph + leidenalg on the fuzzy simplicial set graph produced by UMAP.
    """
    graph_csr = umap_graph

    sources, targets = graph_csr.nonzero()
    weights = graph_csr.data

    shape = graph_csr.shape
    assert shape is not None
    n_vertices = shape[0]

    g = ig.Graph(
        n=n_vertices,
        edges=list(zip(sources, targets)),
        directed=False,
    )
    g.es["weight"] = weights.tolist()

    partition = la.find_partition(
        g, la.RBConfigurationVertexPartition, weights=g.es["weight"], resolution_parameter=resolution, seed=seed
    )

    labels = np.asarray(partition.membership, dtype=int)
    return labels


# ---------- Scores file ----------

def write_scores_file(
    df: pd.DataFrame, X_red: np.ndarray, kmeans_results: dict[int, dict[str, float]], best_k: int, scores_path: Path
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
        metrics_hdb = EMPTY_METRICS.copy()

    # Final KMeans metrics
    labels_k_best = df["cluster_kmeans"].to_numpy().astype(int)
    metrics_k_best = evaluate_clustering_metrics(X_red, labels_k_best)

    # Leiden metrics (if available)
    has_leiden = "cluster_leiden" in df.columns
    if has_leiden:
        labels_leiden = df["cluster_leiden"].to_numpy().astype(int)
        metrics_leiden = evaluate_clustering_metrics(X_red, labels_leiden)
        counts_leiden = df["cluster_leiden"].value_counts(dropna=False).sort_index()
    else:
        metrics_leiden = EMPTY_METRICS.copy()
        counts_leiden = pd.Series(dtype=int)

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
            f"silhouette={_format_metric(metrics_hdb['silhouette'])}, "
            f"davies_bouldin={_format_metric(metrics_hdb['davies_bouldin'])}, "
            f"calinski_harabasz={_format_metric(metrics_hdb['calinski_harabasz'])}\n"
        )
        f.write("cluster_sizes (including noise):\n")
        for label, cnt in counts_hdb.items():
            f.write(f"  label={label}: count={cnt}\n")
        f.write("\n")

        # Leiden summary (if present)
        if has_leiden:
            f.write("=== Leiden ===\n")
            f.write(f"resolution: {LEIDEN_RESOLUTION}\n")
            f.write(
                "internal_metrics: "
                f"silhouette={_format_metric(metrics_leiden['silhouette'])}, "
                f"davies_bouldin={_format_metric(metrics_leiden['davies_bouldin'])}, "
                f"calinski_harabasz={_format_metric(metrics_leiden['calinski_harabasz'])}\n"
            )
            f.write("cluster_sizes:\n")
            for label, cnt in counts_leiden.items():
                f.write(f"  label={label}: count={cnt}\n")
            f.write("\n")

        # KMeans sweep
        f.write("=== KMeans sweep ===\n")
        f.write("k, inertia, silhouette, davies_bouldin, calinski_harabasz\n")
        for k in sorted(kmeans_results):
            r = kmeans_results[k]
            inertia = r["inertia"]
            sil_str = _format_metric(r["silhouette"])
            db_str = _format_metric(r["davies_bouldin"])
            ch_str = _format_metric(r["calinski_harabasz"])
            f.write(f"{k}, {inertia:.6e}, {sil_str}, {db_str}, {ch_str}\n")
        f.write(f"\nSelected k (best_k): {best_k}\n")
        f.write(
            "Final KMeans internal_metrics: "
            f"silhouette={_format_metric(metrics_k_best['silhouette'])}, "
            f"davies_bouldin={_format_metric(metrics_k_best['davies_bouldin'])}, "
            f"calinski_harabasz={_format_metric(metrics_k_best['calinski_harabasz'])}\n\n"
        )

        # Final KMeans cluster sizes
        f.write("=== KMeans (k = best_k) cluster sizes ===\n")
        for label, cnt in counts_km.items():
            f.write(f"  label={label}: count={cnt}\n")


# ---------- Main pipeline ----------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        print(f"Dropping {n_bad_rows} rows with non-finite embeddings (out of {emb.shape[0]} total rows).")
        bad_idx_path = OUT_DIR / f"{TAG}_nonfinite_embedding_rows.txt"
        np.savetxt(bad_idx_path, bad_idx, fmt="%d")
        print(f"Saved indices of dropped rows to: {bad_idx_path}")

        emb = emb[mask_finite_rows]
        df = df.loc[mask_finite_rows].reset_index(drop=True)
        print(f"After dropping non-finite embeddings: emb={emb.shape}, df={df.shape}")

    # L2-normalize embeddings (cosine geometry)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    emb_norm = emb / norms

    # UMAP reduction (embeddings + model, so we can use the graph for Leiden)
    if UMAP_EMBEDDINGS_PATH.exists() and UMAP_MODEL_PATH.exists():
        print(f"Loading precomputed UMAP embeddings from: {UMAP_EMBEDDINGS_PATH}")
        X_umap = np.load(UMAP_EMBEDDINGS_PATH)
        print(f"Loading UMAP model from: {UMAP_MODEL_PATH}")
        umap_model: umap.UMAP = joblib.load(UMAP_MODEL_PATH)
    else:
        print("Running UMAP on embeddings...")
        umap_model = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            n_components=UMAP_N_COMPONENTS,
            metric=UMAP_METRIC,
            random_state=UMAP_RANDOM_STATE,
        )
        X_umap = np.asarray(umap_model.fit_transform(emb_norm), dtype=float)
        np.save(UMAP_EMBEDDINGS_PATH, X_umap)
        joblib.dump(umap_model, UMAP_MODEL_PATH)
        print(f"Saved UMAP-reduced embeddings to: {UMAP_EMBEDDINGS_PATH}")
        print(f"Saved UMAP model to: {UMAP_MODEL_PATH}")

    print(f"UMAP shape: {X_umap.shape}")

    # Standardize reduced space for clustering
    scaler = StandardScaler()
    X_red = scaler.fit_transform(X_umap)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved StandardScaler to: {SCALER_PATH}")

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

    # Leiden on UMAP graph
    print("Clustering with Leiden on UMAP k-NN graph...")
    graph_attr = getattr(umap_model, "graph_", None)
    if not isinstance(graph_attr, csr_matrix):
        raise TypeError("umap_model.graph_ is not a scipy.sparse.csr_matrix as expected.")
    graph_csr = cast(csr_matrix, graph_attr)
    labels_leiden = cluster_leiden_from_umap_graph(graph_csr)
    df["cluster_leiden"] = labels_leiden

    # KMeans sweep + final clustering on reduced space
    print("Running KMeans sweep over k values...")
    best_k, kmeans_results = sweep_kmeans(X_red, KMEANS_K_VALUES)
    print(f"Selected k for KMeans: {best_k}")

    print(f"Fitting final KMeans with k={best_k} on sample...")
    kmeans_model = cluster_kmeans(X_red, n_clusters=best_k)
    labels_kmeans = kmeans_model.labels_.astype(int)
    df["cluster_kmeans"] = labels_kmeans

    joblib.dump(kmeans_model, KMEANS_MODEL_PATH)
    print(f"Saved KMeans model to: {KMEANS_MODEL_PATH}")

    # Save clustered DataFrame
    df.to_parquet(CLUSTERED_DF_PATH, index=False)
    print(f"Saved clustered conversations to: {CLUSTERED_DF_PATH}")

    # Quick cluster summary to stdout
    print("\nCluster counts for HDBSCAN:")
    print(df["cluster_hdbscan"].value_counts(dropna=False).sort_index())
    print("\nCluster counts for Leiden:")
    print(df["cluster_leiden"].value_counts(dropna=False).sort_index())
    print("\nCluster counts for KMeans:")
    print(df["cluster_kmeans"].value_counts(dropna=False).sort_index())

    # Scores file
    print(f"\nWriting clustering scores to: {SCORES_TXT_PATH}")
    write_scores_file( df=df, X_red=X_red, kmeans_results=kmeans_results, best_k=best_k, scores_path=SCORES_TXT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
