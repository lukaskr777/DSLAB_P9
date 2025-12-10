"""
Language-wise clustering script for swiss-ai/apertus-sft-mixture.

This script assumes that conversation-level embeddings have already been computed and stored in a .npy file 
(one row per conversation, in the same order as the text-only Parquet).

Pipeline:
1. Load text-only Parquet:
   - columns: conversation_id, conversation_text.
   - add simple conversation-level features (e.g., text_length).
2. Load precomputed embeddings from EMBEDDINGS_PATH.
3. Clean embeddings: drop rows with non-finite values.
4. L2-normalize embeddings (cosine geometry).
5. Run UMAP (metric='cosine') to reduce to a low-dimensional space; cache:
   - UMAP-reduced embeddings (X_umap)
   - UMAP model
   - UMAP metadata (hyperparameters, n_rows, etc.)
6. Standardize UMAP coordinates (X_red); cache the StandardScaler.
7. Detect language of each conversation (cached with metadata) and build:
   - df["lang"]
   - language → [conversation_id] mapping (JSON file).
8. For each language separately:
   - Let n_lang = number of conversations in that language.
   - If n_lang ≥ MIN_LANG_SAMPLES_FOR_CLUSTERING:
       * set min_cluster_size = max(HDBSCAN_MIN_CLUSTER_SIZE_FLOOR, int(HDBSCAN_MIN_CLUSTER_SIZE_FRACTION * n_lang)).
       * set min_samples = max(1, min(min_cluster_size, int(HDBSCAN_MIN_SAMPLES_FRACTION * min_cluster_size))).
       * run HDBSCAN on X_red restricted to that language.
       * let noise_frac = (#points with label -1) / n_lang.
       * if HDBSCAN puts all points in noise OR noise_frac > MAX_NOISE_FRACTION:
             - treat language as a single cluster (final label = language code).
             - cluster_langwise_hdbscan set to -1 for all points.
         else:
             - assign per-language HDBSCAN cluster ids.
             - build aggregated final labels of the form "{lang}_c{cluster_id}" or "{lang}_noise".
     Else:
       * no HDBSCAN; aggregated final label is simply the language code.
9. Save:
   - clustered DataFrame with language-wise clusters
   - language → [conversation_id] JSON
   - cached language series (+ metadata)
   - a TXT file summarizing language-wise clustering statistics.

Outputs use a suffix "_langwise" in filenames so that previous runs are not overwritten.
"""

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

from pathlib import Path
from dataclasses import dataclass
import json

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from langdetect import detect, DetectorFactory, LangDetectException
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap


# ---------- CONFIG ----------

# Use the merged text-only file that corresponds to the merged embeddings
DATA_PATH = Path("data/swiss-ai_apertus-sft-mixture/small_train_conversations_text_only.parquet")
OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")

# ---- Embeddings to use for this run ----
# Change this path to point to the desired embeddings file.
EMBEDDINGS_PATH = OUT_DIR / (
    "small_train_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.npy"
)

# UMAP settings
UMAP_N_COMPONENTS = 15
UMAP_N_NEIGHBORS = 70
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 0

# HDBSCAN
# Absolute minimum cluster size for any language
HDBSCAN_MIN_CLUSTER_SIZE_FLOOR = 200
# Fraction of language size used to set min_cluster_size (before applying floor)
HDBSCAN_MIN_CLUSTER_SIZE_FRACTION = 0.005  # min_cluster_size ~ 0.5% of n_lang, with a hard floor
# min_samples ~= fraction * min_cluster_size, but clamped into [1, min_cluster_size]
HDBSCAN_MIN_SAMPLES_FRACTION = 0.25
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0

# If the fraction of points labeled as noise by HDBSCAN exceeds this threshold, 
# the language is treated as a single cluster.
MAX_NOISE_FRACTION = 0.6

# langdetect determinism
DetectorFactory.seed = 0

# Language-wise clustering
# Languages with fewer conversations than this threshold will not be clustered;
# all their conversations will share a single cluster label equal to the language code.
MIN_LANG_SAMPLES_FOR_CLUSTERING = 300

# Silhouette
MAX_SILHOUETTE_SAMPLES: int | None = None  # None -> use all samples

# Suffix for this run so that we don't overwrite previous outputs
RUN_SUFFIX = "_langwise"


# ---------- Derived paths (per-embeddings tag and run suffix) ----------

TAG = EMBEDDINGS_PATH.stem
RUN_TAG = f"{TAG}{RUN_SUFFIX}"

UMAP_EMBEDDINGS_PATH = OUT_DIR / f"{RUN_TAG}_umap.npy"
UMAP_MODEL_PATH = OUT_DIR / f"{RUN_TAG}_umap_model.joblib"
UMAP_META_PATH = OUT_DIR / f"{RUN_TAG}_umap_meta.json"

SCALER_PATH = OUT_DIR / f"{RUN_TAG}_umap_scaler.joblib"
CLUSTERED_DF_PATH = OUT_DIR / f"{RUN_TAG}_clustered.parquet"
LANG_MAP_PATH = OUT_DIR / f"{RUN_TAG}_lang_to_conversation_ids.json"
LANG_SERIES_PATH = OUT_DIR / f"{RUN_TAG}_lang_series.parquet"
LANG_META_PATH = OUT_DIR / f"{RUN_TAG}_lang_meta.json"

SCORES_TXT_PATH = OUT_DIR / f"{RUN_TAG}_cluster_scores.txt"
NONFINITE_IDX_PATH = OUT_DIR / f"{RUN_TAG}_nonfinite_embedding_rows.txt"


# ---------- Small helpers ----------

EMPTY_METRICS: dict[str, float] = {
    "silhouette": float("nan"),
    "davies_bouldin": float("nan"),
    "calinski_harabasz": float("nan"),
}


def _format_metric(value: float) -> str:
    """Format a metric as '%.6f' or 'nan' if not finite."""
    return f"{value:.6f}" if not np.isnan(value) else "nan"


# ---------- Data preparation (text-only) ----------

def add_basic_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For text-only input (conversation_id, conversation_text), add simple features.

    Adds:
        - text_length: length of conversation_text in characters

    Assumes that no empty conversations were dropped in the embedding pipeline,
    so we do not drop any rows here.
    """
    if "conversation_text" not in df.columns:
        raise KeyError("Expected a 'conversation_text' column in the input DataFrame.")

    df = df.copy()
    df["conversation_text"] = df["conversation_text"].fillna("").astype(str)
    df["text_length"] = df["conversation_text"].str.len().astype(int)
    return df


def detect_languages_for_df(df: pd.DataFrame, text_col: str = "conversation_text", min_chars: int = 20) -> pd.Series:
    """
    Detect language for each row in df[text_col].

    Returns a pandas Series of ISO-like language codes (e.g., 'en', 'de')
    or 'unknown' when detection fails or the text is too short.
    """
    texts = df[text_col].fillna("").astype(str).tolist()
    langs: list[str] = []

    for t in tqdm(texts, desc="Detecting languages"):
        t = t.strip()
        if len(t) < min_chars:
            langs.append("unknown")
            continue
        try:
            lang = detect(t)
        except LangDetectException:
            lang = "unknown"
        langs.append(lang)

    return pd.Series(langs, index=df.index, dtype="string")


# ---------- Metrics helpers ----------

def evaluate_clustering_metrics(
    X: np.ndarray, labels: np.ndarray, max_silhouette_samples: int | None = MAX_SILHOUETTE_SAMPLES
) -> dict[str, float]:
    """
    Compute internal clustering metrics on X for the given labels.

    Returns a dict with keys:
        - "silhouette"
        - "davies_bouldin"
        - "calinski_harabasz"

    The input labels may contain a "noise" label (e.g., -1); the caller should
    filter out noise before calling this function if needed.
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


# ---------- Language-wise clustering helper ----------

@dataclass
class LangClusterInfo:
    n_lang: int
    min_cluster_size: int
    min_samples: int
    noise_fraction: float
    used_hdbscan: bool
    collapsed_to_single_cluster: bool


def cluster_language(X_lang: np.ndarray, lang: str, n_lang: int) -> tuple[np.ndarray, np.ndarray, LangClusterInfo]:
    """
    Cluster a single language subset in standardized UMAP space.

    Returns:
        labels_hdb: np.ndarray[int] of length n_lang (HDBSCAN labels, -1 = noise or not clustered).
        labels_final: np.ndarray[object] of length n_lang (final labels, e.g. {lang}, {lang}_c{ID}, {lang}_noise).
        info: LangClusterInfo with metadata about the clustering.
    """
    # Default: everything “unclustered” and assigned to the language itself.
    labels_hdb = np.full(n_lang, -1, dtype=np.int32)
    labels_final = np.full(n_lang, lang, dtype=object)

    # Small languages: do not run HDBSCAN at all
    if n_lang < MIN_LANG_SAMPLES_FOR_CLUSTERING:
        info = LangClusterInfo(
            n_lang=n_lang,
            min_cluster_size=0,
            min_samples=0,
            noise_fraction=float("nan"),
            used_hdbscan=False,
            collapsed_to_single_cluster=True,
        )
        return labels_hdb, labels_final, info

    # Per-language HDBSCAN hyperparameters
    min_cluster_size = max(HDBSCAN_MIN_CLUSTER_SIZE_FLOOR, int(HDBSCAN_MIN_CLUSTER_SIZE_FRACTION * n_lang))
    min_cluster_size = max(2, min_cluster_size)

    min_samples = int(HDBSCAN_MIN_SAMPLES_FRACTION * min_cluster_size)
    min_samples = max(1, min(min_cluster_size, min_samples))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
    )
    labels_h = clusterer.fit_predict(X_lang)

    noise_mask = labels_h == -1
    noise_fraction = float(noise_mask.mean()) if n_lang > 0 else 0.0

    # All-noise or high-noise languages: collapse to a single cluster (the language itself)
    if np.all(noise_mask) or noise_fraction > MAX_NOISE_FRACTION:
        info = LangClusterInfo(
            n_lang=n_lang,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            noise_fraction=noise_fraction,
            used_hdbscan=True,
            collapsed_to_single_cluster=True,
        )
        return labels_hdb, labels_final, info

    # Otherwise, keep HDBSCAN clusters
    labels_hdb = labels_h.astype(np.int32, copy=False)
    labels_final = np.array(
        [f"{lang}_noise" if lab == -1 else f"{lang}_c{int(lab)}" for lab in labels_hdb],
        dtype=object,
    )

    info = LangClusterInfo(
        n_lang=n_lang,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        noise_fraction=noise_fraction,
        used_hdbscan=True,
        collapsed_to_single_cluster=False,
    )
    return labels_hdb, labels_final, info


# ---------- Scores file ----------

def write_scores_file(df: pd.DataFrame, X_red: np.ndarray, scores_path: Path) -> None:
    """
    Write language-wise clustering statistics and scores to a TXT file.

    Uses columns:
        - 'lang'
        - 'cluster_langwise_hdbscan'
        - 'cluster_langwise_final'
    and the standardized UMAP coordinates X_red.
    """
    n_samples = len(df)

    with scores_path.open("w", encoding="utf-8") as f:
        f.write(f"Embeddings file: {EMBEDDINGS_PATH}\n")
        f.write(f"TAG: {TAG}\n")
        f.write(f"RUN_TAG: {RUN_TAG}\n")
        f.write(f"n_samples: {n_samples}\n")
        f.write(f"Reduced shape (UMAP dims): {X_red.shape}\n")
        f.write("\n")

        f.write("=== Languages summary ===\n")
        lang_counts = df["lang"].value_counts(dropna=False).sort_index()
        for lang, cnt in lang_counts.items():
            f.write(f"  lang={lang!r}: count={cnt}\n")
        f.write("\n")

        f.write("=== Language-wise HDBSCAN statistics ===\n")

        # Iterate over languages in a stable order
        unique_langs = sorted(df["lang"].dropna().unique())
        lang_array = df["lang"].to_numpy()

        for lang in unique_langs:
            mask_lang = lang_array == lang
            idx_lang = np.where(mask_lang)[0]
            n_lang = idx_lang.size

            f.write(f"\nLanguage {lang!r}\n")
            f.write(f"  n_conversations: {n_lang}\n")

            # Cluster counts for final labels (includes small languages collapsed,
            # languages where HDBSCAN returned only noise, and high-noise languages)
            counts_final = (
                df.loc[mask_lang, "cluster_langwise_final"]
                .value_counts(dropna=False)
                .sort_index()
            )
            f.write("  cluster_langwise_final counts:\n")
            for label, cnt in counts_final.items():
                f.write(f"    {label!r}: count={cnt}\n")

            # For languages below threshold, HDBSCAN was not run at all
            if n_lang < MIN_LANG_SAMPLES_FOR_CLUSTERING:
                f.write(
                    f"  (n_conversations < MIN_LANG_SAMPLES_FOR_CLUSTERING={MIN_LANG_SAMPLES_FOR_CLUSTERING}, "
                    "no HDBSCAN run; cluster is the language itself.)\n"
                )
                continue

            labels_hdb = df.loc[mask_lang, "cluster_langwise_hdbscan"].to_numpy()

            # If HDBSCAN was not used (all -1) due to all-noise or high-noise,
            # we treated language as a single cluster
            if np.all(labels_hdb == -1):
                f.write(
                    "  HDBSCAN: either all points labeled as noise, or noise fraction above threshold; "
                    "language was treated as a single cluster in 'cluster_langwise_final'. "
                    "No internal metrics computed.\n"
                )
                continue

            # HDBSCAN labels with potential noise; metrics on non-noise points
            mask_non_noise = labels_hdb != -1
            if not mask_non_noise.any():
                f.write("  HDBSCAN: no non-noise points; no internal metrics.\n")
                continue

            labels_non_noise = labels_hdb[mask_non_noise]
            unique_non_noise = np.unique(labels_non_noise)
            if unique_non_noise.size <= 1:
                f.write(
                    "  HDBSCAN: only one non-noise cluster; "
                    "internal metrics (silhouette/DB/CH) would be degenerate.\n"
                )
                continue

            X_lang = X_red[idx_lang]
            X_lang_non_noise = X_lang[mask_non_noise]

            metrics = evaluate_clustering_metrics(X_lang_non_noise, labels_non_noise)
            noise_fraction = float((labels_hdb == -1).mean()) if n_lang > 0 else 0.0

            f.write(
                "  HDBSCAN internal_metrics (non-noise points): "
                f" noise_fraction={noise_fraction:.6f}, "
                f"silhouette={_format_metric(metrics['silhouette'])}, "
                f"davies_bouldin={_format_metric(metrics['davies_bouldin'])}, "
                f"calinski_harabasz={_format_metric(metrics['calinski_harabasz'])}\n"
            )


# ---------- Main pipeline ----------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using embeddings from: {EMBEDDINGS_PATH}")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")

    emb = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings shape: {emb.shape}")

    # Load text-only data and compute basic features
    print(f"Loading text-only data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded shape: {df.shape}")

    print("Computing basic text features from `conversation_text`...")
    df = add_basic_text_features(df)
    print(f"After feature computation: {df.shape}")

    if len(df) != emb.shape[0]:
        raise ValueError(
            f"Mismatch between DataFrame rows ({len(df)}) and embeddings rows ({emb.shape[0]}). "
            "Ensure you are using embeddings computed on this exact dataset and in the same order."
        )

    # Drop rows with any NaN/inf in embeddings
    mask_finite_rows = np.isfinite(emb).all(axis=1)
    n_bad_rows = int((~mask_finite_rows).sum())
    if n_bad_rows > 0:
        bad_idx = np.where(~mask_finite_rows)[0]
        print(f"Dropping {n_bad_rows} rows with non-finite embeddings (out of {emb.shape[0]} total rows).")
        np.savetxt(NONFINITE_IDX_PATH, bad_idx, fmt="%d")
        print(f"Saved indices of dropped rows to: {NONFINITE_IDX_PATH}")

        emb = emb[mask_finite_rows]
        df = df.loc[mask_finite_rows].reset_index(drop=True)
        print(f"After dropping non-finite embeddings: emb={emb.shape}, df={df.shape}")

    # L2-normalize embeddings (cosine geometry)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    emb_norm = emb / norms

    # ---------- UMAP reduction (embeddings + model + meta) ----------

    def build_umap_meta(n_rows: int) -> dict:
        return {
            "n_components": UMAP_N_COMPONENTS,
            "n_neighbors": UMAP_N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": UMAP_METRIC,
            "random_state": UMAP_RANDOM_STATE,
            "n_rows": int(n_rows),
            "embeddings_path": str(EMBEDDINGS_PATH),
        }

    X_umap: np.ndarray
    umap_model: umap.UMAP

    use_umap_cache = UMAP_EMBEDDINGS_PATH.exists() and UMAP_MODEL_PATH.exists() and UMAP_META_PATH.exists()

    if use_umap_cache:
        try:
            umap_meta_cached = json.loads(UMAP_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            print("Could not read UMAP metadata; recomputing UMAP.")
            use_umap_cache = False
        else:
            umap_meta_current = build_umap_meta(emb_norm.shape[0])
            if umap_meta_cached != umap_meta_current:
                print("UMAP metadata does not match current configuration; recomputing UMAP.")
                use_umap_cache = False

    if use_umap_cache:
        print(f"Loading precomputed UMAP embeddings from: {UMAP_EMBEDDINGS_PATH}")
        X_umap_loaded = np.load(UMAP_EMBEDDINGS_PATH)
        print(f"Loading UMAP model from: {UMAP_MODEL_PATH}")
        umap_model_loaded: umap.UMAP = joblib.load(UMAP_MODEL_PATH)

        if X_umap_loaded.shape[0] == emb_norm.shape[0] and X_umap_loaded.shape[1] == UMAP_N_COMPONENTS:
            print("UMAP cache matches current data; reusing.")
            X_umap = X_umap_loaded
            umap_model = umap_model_loaded
        else:
            print(
                "UMAP cache shape does not match current data shape "
                f"(cached={X_umap_loaded.shape}, current={(emb_norm.shape[0], UMAP_N_COMPONENTS)}). "
                "Recomputing UMAP and overwriting cache."
            )
            use_umap_cache = False

    if not use_umap_cache:
        print("Running UMAP on embeddings...")
        umap_model = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            n_components=UMAP_N_COMPONENTS,
            metric=UMAP_METRIC,
            random_state=UMAP_RANDOM_STATE,
            verbose=True,
        )
        X_umap = np.asarray(umap_model.fit_transform(emb_norm), dtype=float)
        np.save(UMAP_EMBEDDINGS_PATH, X_umap)
        joblib.dump(umap_model, UMAP_MODEL_PATH)
        umap_meta_current = build_umap_meta(emb_norm.shape[0])
        UMAP_META_PATH.write_text(json.dumps(umap_meta_current, indent=2), encoding="utf-8")
        print(f"Saved UMAP-reduced embeddings to: {UMAP_EMBEDDINGS_PATH}")
        print(f"Saved UMAP model to: {UMAP_MODEL_PATH}")
        print(f"Saved UMAP metadata to: {UMAP_META_PATH}")

    print(f"UMAP shape: {X_umap.shape}")

    # Standardize reduced space for clustering
    scaler = StandardScaler()
    X_red = scaler.fit_transform(X_umap)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved StandardScaler to: {SCALER_PATH}")

    # ---------- Language detection with caching (series + meta) ----------

    def build_lang_meta(n_rows: int) -> dict:
        return {
            "n_rows": int(n_rows),
            "data_path": str(DATA_PATH),
        }

    use_lang_cache = LANG_SERIES_PATH.exists() and LANG_META_PATH.exists()

    if use_lang_cache:
        try:
            lang_meta_cached = json.loads(LANG_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            print("Could not read language metadata; recomputing language detection.")
            use_lang_cache = False
        else:
            lang_meta_current = build_lang_meta(len(df))
            if lang_meta_cached != lang_meta_current:
                print("Language metadata does not match current data; recomputing language detection.")
                use_lang_cache = False

    if use_lang_cache:
        print(f"Loading cached language detections from: {LANG_SERIES_PATH}")
        lang_df = pd.read_parquet(LANG_SERIES_PATH)
        lang_series = lang_df["lang"]
        if len(lang_series) != len(df):
            print(
                "Cached language series length does not match current DataFrame; "
                "recomputing language detection."
            )
            lang_series = detect_languages_for_df(df)
            pd.DataFrame({"lang": lang_series}).to_parquet(LANG_SERIES_PATH, index=False)
            LANG_META_PATH.write_text(json.dumps(build_lang_meta(len(df)), indent=2), encoding="utf-8")
            print(f"Saved language detections to: {LANG_SERIES_PATH}")
            print(f"Saved language metadata to: {LANG_META_PATH}")
    else:
        print("Detecting language for each conversation...")
        lang_series = detect_languages_for_df(df)
        pd.DataFrame({"lang": lang_series}).to_parquet(LANG_SERIES_PATH, index=False)
        LANG_META_PATH.write_text(json.dumps(build_lang_meta(len(df)), indent=2), encoding="utf-8")
        print(f"Saved language detections to: {LANG_SERIES_PATH}")
        print(f"Saved language metadata to: {LANG_META_PATH}")

    df["lang"] = lang_series.astype("string")
    print("Language counts:")
    print(df["lang"].value_counts())

    # Build lang -> [conversation_id] mapping and save to JSON
    print("\nBuilding language → conversation_id mapping...")
    lang_to_conv_ids: dict[str, list[int]] = {}
    for lang, group in df.groupby("lang", dropna=False):
        lang_str = str(lang)
        conv_ids = group["conversation_id"].tolist()
        lang_to_conv_ids[lang_str] = conv_ids
        print(f"  {lang_str!r}: {len(conv_ids)} conversations")

    with LANG_MAP_PATH.open("w", encoding="utf-8") as f:
        json.dump(lang_to_conv_ids, f, ensure_ascii=False, indent=2)
    print(f"Saved language mapping to: {LANG_MAP_PATH}")

    # ---------- Language-wise HDBSCAN clustering on standardized UMAP space ----------

    print("\nRunning language-wise HDBSCAN clustering on standardized UMAP space...")

    # Initialize language-wise HDBSCAN cluster id and final labels
    df["cluster_langwise_hdbscan"] = -1  # -1 = noise or not clustered (for small/high-noise languages)
    df["cluster_langwise_final"] = pd.Series(pd.NA, index=df.index, dtype="string")

    lang_array = df["lang"].to_numpy()
    unique_langs = sorted(df["lang"].dropna().unique())

    for lang in unique_langs:
        mask_lang = lang_array == lang
        idx_lang = np.where(mask_lang)[0]
        n_lang = idx_lang.size

        print(f"\nLanguage {lang!r}: {n_lang} conversations")

        if n_lang == 0:
            continue

        X_lang = X_red[idx_lang]

        labels_hdb, labels_final, info = cluster_language(X_lang, lang, n_lang)

        if not info.used_hdbscan:
            print(f"  Too few samples (<{MIN_LANG_SAMPLES_FOR_CLUSTERING}); treating language as a single cluster.")
        else:
            print(
                f"  HDBSCAN parameters for this language: "
                f"min_cluster_size={info.min_cluster_size}, min_samples={info.min_samples}"
            )
            if info.collapsed_to_single_cluster:
                print(
                    "  HDBSCAN either returned only noise or noise fraction "
                    f"{info.noise_fraction:.3f} exceeded MAX_NOISE_FRACTION={MAX_NOISE_FRACTION:.3f}; "
                    "treating language as a single cluster instead."
                )

        df.loc[mask_lang, "cluster_langwise_hdbscan"] = labels_hdb
        df.loc[mask_lang, "cluster_langwise_final"] = labels_final

        if not info.collapsed_to_single_cluster:
            counts_lang = df.loc[mask_lang, "cluster_langwise_final"].value_counts(dropna=False).sort_index()
            print("  cluster_langwise_final counts:")
            print(counts_lang)

    print("\nLanguage-wise HDBSCAN clustering completed.")

    # Save clustered DataFrame
    df.to_parquet(CLUSTERED_DF_PATH, index=False)
    print(f"Saved clustered conversations to: {CLUSTERED_DF_PATH}")

    # Scores file (language-wise metrics)
    print(f"\nWriting language-wise clustering scores to: {SCORES_TXT_PATH}")
    write_scores_file(df=df, X_red=X_red, scores_path=SCORES_TXT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
