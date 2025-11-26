"""
User-level clustering for swiss-ai/apertus-70b-instruct usage.

This script:
- Restricts LiteLLM_SpendLogs to model_group == "swiss-ai/apertus-70b-instruct".
- Removes "extreme power users" that would form singleton clusters.
- Builds per-user feature vectors capturing:
    * volume & activity (requests, active days, span, density),
    * efficiency (tokens/request, spend per token),
    * diversity (model/provider entropy, dominant model share),
    * temporal behavior (mean inter-request time, most active hour, weekday fraction),
    * performance & reliability proxies (latency, cache hit rate, success rate),
    * stability across days (CV of daily spend/requests).
- Runs KMeans with automatic k-selection via silhouette, while rejecting
  degenerate solutions where a cluster is too small.
- Runs PCA for visualization (if possible).

Outputs in figs/apertus70b_user_clusters:
    - apertus70b_user_stats.csv
    - apertus70b_user_clusters.csv
    - apertus70b_cluster_profiles.csv
    - apertus70b_cluster_summary.json
    - removed_extreme_users.csv (if any user was removed)
    - users_pca_clusters.png
    - cluster_sizes.png
    - cluster_profiles_bars.png
    - cluster_profile_<feature>.png
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utility_scripts.file_utils import (
    PathLike,
    ensure_empty_dir,
    read_table,
    save_csv,
    save_text,
)
from utility_scripts.plot_utils import bar
from utility_scripts.df_reading_utils import (
    require,
    safe_div,
    clean_table,
)


TARGET_MODEL_GROUP = "swiss-ai/apertus-70b-instruct"

# Base and extended feature lists (user-level)
FEATURE_COLS_BASE = [
    "total_spend",
    "request_count",
    "avg_spend_per_req",
    "n_unique_models",
    "requests_per_active_day",
    "completion_ratio_mean",
    "mean_inter_req_s",
    "n_unique_model_groups",
    "n_unique_providers",
    "active_days",
]

FEATURE_COLS_EXTRA = [
    # Temporal activity
    "activity_span_days",
    "activity_density",
    "most_active_hour",
    "weekday_fraction",

    # Token- and cost-efficiency
    "avg_tokens_per_req",
    "avg_completion_tokens",
    "prompt_to_completion_ratio_mean",
    "spend_per_token_mean",

    # Diversity / specialization
    "model_entropy",
    "provider_entropy",
    "dominant_model_share",

    # Performance / latency
    "mean_latency_s",
    "latency_std_s",
    "mean_completion_latency_s",

    # Cache and session behavior
    "cache_hit_rate",
    "n_unique_sessions",
    "avg_requests_per_session",

    # Quality / reliability proxies
    "success_rate",
    "median_spend_per_active_day",

    # Aggregate stability metrics
    "spend_cv",
    "requests_cv",
]

FEATURE_COLS_GLOBAL = FEATURE_COLS_BASE + FEATURE_COLS_EXTRA

# Heavy-tailed features to log1p-transform before clustering
LOG_COLS = (
    "total_spend",
    "request_count",
    "avg_spend_per_req",
    "requests_per_active_day",
    "mean_inter_req_s",
    "avg_tokens_per_req",
    "spend_per_token_mean",
)


# --------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------


def _mean_inter_request_seconds_from_series(s: pd.Series) -> float:
    """Mean time gap in seconds between consecutive timestamps in `s` (NaN if < 2 valid entries)."""
    t = pd.to_datetime(s, utc=True, errors="coerce").dropna().sort_values()
    if t.size <= 1:
        return float("nan")
    d = t.diff().dropna().dt.total_seconds()
    return float(d.mean()) if len(d) else float("nan")


def _shannon_entropy_from_counts(counts: pd.Series) -> float:
    """Shannon entropy in nats from a vector of nonnegative counts."""
    c = pd.to_numeric(counts, errors="coerce").fillna(0).astype(float)
    tot = c.sum()
    if tot <= 0:
        return 0.0
    p = c / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _as_int(x: Any) -> int:
    """Convert arbitrary scalar-like input to int; return -1 if conversion fails or NaN."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return int(np.asarray(x).item())
    except Exception:
        s = pd.to_numeric(pd.Series([x]), errors="coerce")
        v = s.iloc[0]
        return int(v) if pd.notna(v) else -1


def _build_features(df_stats: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Build numeric, finite feature matrix:
    - select given columns,
    - log1p-transform heavy-tailed ones,
    - coerce to float, median-impute, drop constant columns.
    """
    feats = df_stats.reindex(columns=feature_cols).copy()

    # Log heavy tails safely
    for c in LOG_COLS:
        if c in feats.columns:
            feats[c] = np.log1p(pd.to_numeric(feats[c], errors="coerce").clip(lower=0))

    # Coerce numeric and impute medians
    feats = feats.apply(pd.to_numeric, errors="coerce").astype("float64")
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0.0)

    # Drop all-NaN or constant columns
    if feats.shape[1] > 0:
        feats = feats.loc[:, ~feats.isna().all(axis=0)]
        feats = feats.loc[:, feats.nunique(dropna=False) > 1]

    return feats


def _cluster_and_plot(
    feats: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    min_cluster_fraction: float = 0.01,
    min_cluster_size_abs: int = 5,
) -> tuple[np.ndarray, str, str]:
    """
    Fit KMeans with k in [2..min(5, n)], select by silhouette, add PCA for viz,
    make cluster size + profile plots, write clustered CSVs.

    Degenerate solutions with very small clusters (e.g. 1 heavy outlier)
    are rejected: for a candidate k, we require

        min_cluster_size >= max(min_cluster_size_abs,
                                min_cluster_fraction * n_samples)

    Returns (labels, k_str, sil_str).
    """
    n_samples = feats.shape[0]

    # Degenerate guards
    if n_samples < 2 or feats.shape[1] == 0:
        labels = np.full(len(stats_df), -1, dtype=int)
        stats_df["cluster_kmeans"] = labels
        save_csv(stats_df, "apertus70b_user_clusters.csv", out_dir)
        save_text(
            json.dumps(
                [{"cluster": -1, "n_users": int(len(stats_df)), "reason": "insufficient data"}],
                indent=2,
            ),
            "apertus70b_cluster_summary.json",
            out_dir,
        )
        return labels, "NA", "nan"

    # Standardize and PCA
    X = feats.values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    n_comp = int(min(3, X_std.shape[0], X_std.shape[1]))
    X_pca = None
    explained = np.array([])
    if n_comp >= 1:
        pca = PCA(n_components=n_comp, random_state=0)
        X_pca = pca.fit_transform(X_std)
        explained = pca.explained_variance_ratio_
        for i in range(n_comp):
            stats_df[f"pca{i+1}"] = X_pca[:, i]

    # KMeans model selection with degeneracy check
    best_k, best_score, best_labels = None, -np.inf, None
    k_max = min(5, n_samples)
    min_allowed_size = max(min_cluster_size_abs, int(np.ceil(min_cluster_fraction * n_samples)))

    for k in range(2, max(3, k_max + 1)):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labs = km.fit_predict(X_std)
            unique, counts = np.unique(labs, return_counts=True)
            if unique.size < 2:
                continue
            # Reject solutions with too-small clusters
            if counts.min() < min_allowed_size:
                continue
            score = silhouette_score(X_std, labs)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labs
        except Exception:
            continue

    if best_labels is None:
        # Fall back to a single cluster if no acceptable partition is found
        labels = np.zeros(len(stats_df), dtype=int)
        k_str, sil_str = "1", "nan"
    else:
        labels = np.asarray(best_labels, dtype=int)
        k_str, sil_str = str(best_k), f"{best_score:.3f}"

    stats_df["cluster_kmeans"] = labels

    # Save clustered table (keep PCA columns if present)
    keep_pca = [c for c in ("pca1", "pca2", "pca3") if c in stats_df.columns]
    cols_to_save = [*stats_df.columns.intersection(["end_user"]), "cluster_kmeans", *keep_pca]
    cols_to_save += [c for c in FEATURE_COLS_GLOBAL if c in stats_df.columns]
    save_csv(stats_df[cols_to_save], "apertus70b_user_clusters.csv", out_dir)

    # PCA scatter if ≥ 2 PCs
    if {"pca1", "pca2"}.issubset(stats_df.columns):
        plt.figure()
        plt.scatter(stats_df["pca1"], stats_df["pca2"], c=labels, s=10, alpha=0.8)
        e1 = (explained[0] * 100) if explained.size >= 1 else 0.0
        e2 = (explained[1] * 100) if explained.size >= 2 else 0.0
        plt.xlabel(f"PCA1 ({e1:.1f}% var)")
        plt.ylabel(f"PCA2 ({e2:.1f}% var)")
        plt.title(f"{title_prefix} — PCA by KMeans clusters\n(k={k_str}, silhouette={sil_str})")
        plt.tight_layout()
        plt.savefig(out_dir / "users_pca_clusters.png", dpi=150)
        plt.close()

    # Cluster sizes bar
    size_series = pd.Series(labels).value_counts().sort_index()
    bar(
        series=size_series,
        title=f"{title_prefix} — users per cluster",
        xlabel="cluster",
        ylabel="users",
        fname="cluster_sizes.png",
        outdir=out_dir,
        order=sorted(size_series.index.tolist()),
        sort_values=False,
    )

    # Cluster profiles: z-scored feature means (numeric-only)
    numeric_cols = [
        c for c in FEATURE_COLS_GLOBAL
        if c in stats_df.columns and pd.api.types.is_numeric_dtype(stats_df[c])
    ]
    if numeric_cols:
        raw_means = stats_df.groupby("cluster_kmeans", observed=True)[numeric_cols].mean(numeric_only=True)
        feat_means = stats_df[numeric_cols].mean(numeric_only=True)
        feat_stds = stats_df[numeric_cols].std(numeric_only=True).replace(0, np.nan)
        cluster_means_z = (raw_means - feat_means) / feat_stds
        cluster_means_z = cluster_means_z.apply(pd.to_numeric, errors="coerce").astype("float64")

        if not cluster_means_z.dropna(how="all").empty:
            plot_df = cluster_means_z.fillna(0.0)
            plt.figure(figsize=(max(10, 0.6 * len(plot_df.columns)), 6))
            plot_df.T.plot(kind="bar")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("mean feature (z-score)")
            plt.title(f"{title_prefix} — cluster profiles (z-scored means)")
            plt.tight_layout()
            plt.savefig(out_dir / "cluster_profiles_bars.png", dpi=150)
            plt.close()
        else:
            save_text(
                "Skipped z-score bar plot: no numeric data or all-NaN after filtering.",
                "cluster_profiles_bars.SKIPPED.txt",
                out_dir,
            )
    else:
        save_text(
            "Skipped z-score bar plot: no numeric feature columns available.",
            "cluster_profiles_bars.SKIPPED.txt",
            out_dir,
        )

    # Cluster profile table with mean/median of raw features
    cols_for_table = [
        c for c in FEATURE_COLS_GLOBAL if c in stats_df.columns and pd.api.types.is_numeric_dtype(stats_df[c])
    ]
    if cols_for_table:
        cluster_profiles = stats_df.groupby("cluster_kmeans", observed=True)[cols_for_table].agg(["mean", "median"])
        save_csv(cluster_profiles, "apertus70b_cluster_profiles.csv", out_dir)
    else:
        save_text(
            "No numeric feature columns available for cluster_profiles.",
            "apertus70b_cluster_profiles.SKIPPED.txt",
            out_dir,
        )

    # Compact JSON summary (volume-oriented)
    def _safe_median(df_in: pd.DataFrame, col: str) -> float:
        if col not in df_in:
            return 0.0
        s = pd.to_numeric(df_in[col], errors="coerce")
        if s.empty:
            return 0.0
        val = s.median(skipna=True)
        return float(val) if pd.notna(val) else 0.0

    rows: list[dict[str, float | int]] = []
    for cid, subc in stats_df.groupby("cluster_kmeans", observed=True):
        cid_int = _as_int(cid)
        rows.append(
            {
                "cluster": cid_int,
                "n_users": int(len(subc)),
                "median_requests": _safe_median(subc, "request_count"),
                "median_total_spend": _safe_median(subc, "total_spend"),
                "median_req_per_active_day": _safe_median(subc, "requests_per_active_day"),
            }
        )

    save_text(json.dumps(rows, indent=2), "apertus70b_cluster_summary.json", out_dir)

    return labels, k_str, sil_str


# --------------------------------------------------------------------
# Extreme-user filter (reusable)
# --------------------------------------------------------------------


def remove_extreme_apertus70b_users(
    df: pd.DataFrame,
    min_requests_abs: int = 10_000,
    min_ratio_vs_second: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect and remove "extreme power users" for Apertus-70B before clustering.

    Heuristic:
    - Compute request_count per end_user (within the already restricted df).
    - Let c1 = max request_count, c2 = second largest.
    - If c1 >= max(min_requests_abs, min_ratio_vs_second * c2), drop that user.
    - If the condition is not met, return df unchanged.

    Returns:
        filtered_df, removed_users_df

    You can reuse this function in other scripts that do user-level
    statistics or clustering on Apertus-70B.
    """
    if "end_user" not in df.columns or "request_id" not in df.columns:
        return df, pd.DataFrame(columns=["end_user", "request_count"])

    counts = (
        df.groupby("end_user", dropna=False)["request_id"]
        .count()
        .sort_values(ascending=False)
    )

    if counts.empty or counts.size < 2:
        return df, pd.DataFrame(columns=["end_user", "request_count"])

    c1 = counts.iloc[0]
    c2 = counts.iloc[1]

    threshold = max(min_requests_abs, min_ratio_vs_second * c2)
    if c1 < threshold:
        # No extreme outlier
        return df, pd.DataFrame(columns=["end_user", "request_count"])

    extreme_user = counts.index[0]
    removed = pd.DataFrame(
        {"end_user": [extreme_user], "request_count": [int(c1)]}
    )

    filtered = df[df["end_user"] != extreme_user].copy()
    return filtered, removed


# --------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------


def analyze_apertus70b_user_clusters(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/apertus70b_user_clusters",
) -> None:
    """
    Compute per-user usage features for swiss-ai/apertus-70b-instruct,
    remove extreme power users, then cluster users and visualize
    cluster structure.
    """
    out = ensure_empty_dir(outdir)

    df = read_table("LiteLLM_SpendLogs", dataset=dataset, dir_name=dir_name)
    if df.empty:
        raise SystemExit("No data loaded from LiteLLM_SpendLogs.")

    needed = {
        "request_id",
        "spend",
        "startTime",
        "endTime",
        "completionStartTime",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "model",
        "model_group",
        "custom_llm_provider",
        "api_key",
        "end_user",
        "call_type",
        "status",
        "session_id",
        "cache_hit",
    }
    miss = require(df, needed, strict=False)
    if "model_group" in miss:
        raise SystemExit("Missing required column 'model_group'.")

    # Clean + enrich (adds durations, ratios, date)
    df = clean_table(
        df,
        add_date_key=True,
        date_key="date",
        date_freq="D",
        compute_durations=True,
        compute_ratios=True,
    )

    # Restrict to the target model group
    df = df[df["model_group"] == TARGET_MODEL_GROUP].copy()
    if df.empty:
        raise SystemExit(f"No rows for model_group={TARGET_MODEL_GROUP!r}.")

    # Remove extreme power users for Apertus-70B
    df, removed_users = remove_extreme_apertus70b_users(df)
    if not removed_users.empty:
        save_csv(removed_users, "removed_extreme_users.csv", out)

    # ----------------------------------------------------------------
    # Per-user aggregates (within Apertus 70B only)
    # ----------------------------------------------------------------

    # Model concentration (still useful if multiple underlying models exist in the group)
    user_model_counts = (
        df.groupby(["end_user", "model"], observed=True)["request_id"]
        .count()
        .rename("model_req_count")
        .reset_index()
    )
    top_model_share = (
        user_model_counts.sort_values(["end_user", "model_req_count"], ascending=[True, False])
        .groupby("end_user", observed=True)["model_req_count"]
        .agg(["sum", "max"])
        .assign(dominant_model_share=lambda x: safe_div(x["max"], x["sum"]).fillna(0.0))
        [["dominant_model_share"]]
        .reset_index()
    )

    # Model entropy
    model_entropy = (
        user_model_counts.groupby("end_user", observed=True)["model_req_count"]
        .apply(_shannon_entropy_from_counts)
        .rename("model_entropy")
        .reset_index()
    )

    # Provider entropy
    user_provider_counts = (
        df.groupby(["end_user", "custom_llm_provider"], observed=True)["request_id"]
        .count()
        .rename("prov_req_count")
        .reset_index()
    )
    provider_entropy = (
        user_provider_counts.groupby("end_user", observed=True)["prov_req_count"]
        .apply(_shannon_entropy_from_counts)
        .rename("provider_entropy")
        .reset_index()
    )

    # Temporal gaps
    temporal = (
        df.groupby("end_user", observed=True)["startTime"]
        .apply(_mean_inter_request_seconds_from_series)
        .rename("mean_inter_req_s")
        .reset_index()
    )

    # Active days
    active_days = (
        df.dropna(subset=["date"])
        .groupby("end_user", observed=True)["date"]
        .nunique()
        .rename("active_days")
        .reset_index()
    )

    # First/last seen
    user_first_last = (
        df.groupby("end_user", observed=True)["startTime"]
        .agg(first_seen="min", last_seen="max")
        .reset_index()
    )

    # Latency metrics
    lat = pd.DataFrame({"end_user": df["end_user"]})
    lat["latency_s"] = (df["endTime"] - df["startTime"]).dt.total_seconds()  # type: ignore[attr-defined]
    lat["completion_latency_s"] = (df["endTime"] - df["completionStartTime"]).dt.total_seconds()  # type: ignore[attr-defined]
    latency_agg = (
        lat.groupby("end_user", observed=True)
        .agg(
            mean_latency_s=("latency_s", "mean"),
            latency_std_s=("latency_s", "std"),
            mean_completion_latency_s=("completion_latency_s", "mean"),
        )
        .reset_index()
    )

    # Cache hit rate (robust truthy parsing)
    cache_hit_norm = df["cache_hit"].astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    cache_bool = df.assign(cache_hit_bool=cache_hit_norm.isin(truthy))
    cache_hit_rate = (
        cache_bool.groupby("end_user", observed=True)["cache_hit_bool"]
        .mean()
        .rename("cache_hit_rate")
        .reset_index()
    )

    # Session stats
    sess_agg = (
        df.groupby("end_user", observed=True)
        .agg(n_unique_sessions=("session_id", "nunique"))
        .reset_index()
    )

    # Success rate
    status_norm = df["status"].astype(str).str.strip().str.lower()
    success_rate = (
        df.assign(_ok=status_norm.eq("success"))
        .groupby("end_user", observed=True)["_ok"]
        .mean()
        .rename("success_rate")
        .reset_index()
    )

    # Weekday fraction (share of requests on Mon–Fri)
    weekday_frac = (
        df.assign(_wd=df["startTime"].dt.weekday < 5)  # type: ignore[attr-defined]
        .groupby("end_user", observed=True)["_wd"]
        .mean()
        .rename("weekday_fraction")
        .reset_index()
    )

    # Most active hour (0–23)
    hours = df.assign(h=df["startTime"].dt.hour)  # type: ignore[attr-defined]
    most_active_hour = (
        hours.groupby(["end_user", "h"], observed=True)["request_id"]
        .count()
        .rename("cnt")
        .reset_index()
    )
    most_active_hour = (
        most_active_hour.sort_values(["end_user", "cnt"], ascending=[True, False])
        .groupby("end_user", observed=True)
        .first()["h"]
        .rename("most_active_hour")
        .reset_index()
    )

    # Per-user base aggregate
    base_agg = (
        df.groupby("end_user", observed=True)
        .agg(
            request_count=("request_id", "count"),
            total_spend=("spend", "sum"),
            avg_spend_per_req=("spend", "mean"),
            total_tokens_sum=("total_tokens", "sum"),
            total_tokens_mean=("total_tokens", "mean"),
            total_tokens_std=("total_tokens", "std"),
            prompt_tokens_mean=("prompt_tokens", "mean"),
            completion_tokens_mean=("completion_tokens", "mean"),
            completion_ratio_mean=("completion_ratio", "mean"),
            n_unique_models=("model", "nunique"),
            n_unique_model_groups=("model_group", "nunique"),
            n_unique_providers=("custom_llm_provider", "nunique"),
        )
        .reset_index()
    )

    # Prompt-to-completion ratio (request level, then user mean)
    pr_co_ratio = safe_div(df["prompt_tokens"], df["completion_tokens"], allow_zero=False)
    prc = df[["end_user"]].copy()
    prc["pr_co_ratio"] = pr_co_ratio
    prc_agg = (
        prc.groupby("end_user", observed=True)["pr_co_ratio"]
        .mean()
        .rename("prompt_to_completion_ratio_mean")
        .reset_index()
    )

    # Daily aggregates for stability metrics
    daily = (
        df.dropna(subset=["date"])
        .groupby(["end_user", "date"], observed=True)
        .agg(
            day_spend=("spend", "sum"),
            day_requests=("request_id", "count"),
        )
        .reset_index()
    )
    daily_stats = (
        daily.groupby("end_user", observed=True)
        .agg(
            day_spend_mean=("day_spend", "mean"),
            day_spend_std=("day_spend", "std"),
            day_req_mean=("day_requests", "mean"),
            day_req_std=("day_requests", "std"),
            median_spend_per_active_day=("day_spend", "median"),
        )
        .reset_index()
    )
    daily_stats["spend_cv"] = safe_div(daily_stats["day_spend_std"], daily_stats["day_spend_mean"]).fillna(0.0)
    daily_stats["requests_cv"] = safe_div(daily_stats["day_req_std"], daily_stats["day_req_mean"]).fillna(0.0)

    # Assemble user-level table for Apertus 70B
    user_stats = (
        base_agg
        .merge(top_model_share, on="end_user", how="left")
        .merge(model_entropy, on="end_user", how="left")
        .merge(provider_entropy, on="end_user", how="left")
        .merge(active_days, on="end_user", how="left")
        .merge(temporal, on="end_user", how="left")
        .merge(user_first_last, on="end_user", how="left")
        .merge(latency_agg, on="end_user", how="left")
        .merge(cache_hit_rate, on="end_user", how="left")
        .merge(sess_agg, on="end_user", how="left")
        .merge(success_rate, on="end_user", how="left")
        .merge(weekday_frac, on="end_user", how="left")
        .merge(most_active_hour, on="end_user", how="left")
        .merge(prc_agg, on="end_user", how="left")
        .merge(
            daily_stats[["end_user", "median_spend_per_active_day", "spend_cv", "requests_cv"]],
            on="end_user",
            how="left",
        )
    )

    # Derived features
    user_stats["active_days"] = pd.to_numeric(user_stats["active_days"], errors="coerce").fillna(0).astype(int)
    user_stats["requests_per_active_day"] = safe_div(
        user_stats["request_count"],
        user_stats["active_days"],
        allow_zero=False,
    ).fillna(0.0)

    # Activity span and density
    user_stats["activity_span_days"] = (
        (pd.to_datetime(user_stats["last_seen"], utc=True) - pd.to_datetime(user_stats["first_seen"], utc=True))
        .dt.total_seconds()
        / (3600 * 24)
    )
    user_stats["activity_span_days"] = user_stats["activity_span_days"].clip(lower=0).fillna(0.0)
    user_stats["activity_density"] = safe_div(
        user_stats["active_days"],
        user_stats["activity_span_days"],
    ).fillna(0.0)

    # Efficiency metrics
    user_stats["avg_tokens_per_req"] = safe_div(
        user_stats["total_tokens_sum"],
        user_stats["request_count"],
    ).fillna(0.0)
    user_stats["avg_completion_tokens"] = pd.to_numeric(
        user_stats["completion_tokens_mean"],
        errors="coerce",
    ).fillna(0.0)
    user_stats["spend_per_token_mean"] = safe_div(
        user_stats["total_spend"],
        user_stats["total_tokens_sum"],
    ).fillna(0.0)

    # Ensure concentration feature is present
    if "dominant_model_share" not in user_stats.columns and "top_model_share" in user_stats.columns:
        user_stats["dominant_model_share"] = user_stats["top_model_share"]

    # Avg requests per session
    user_stats["n_unique_sessions"] = pd.to_numeric(
        user_stats["n_unique_sessions"],
        errors="coerce",
    ).fillna(0).astype(int)
    user_stats["avg_requests_per_session"] = safe_div(
        user_stats["request_count"],
        user_stats["n_unique_sessions"],
    ).fillna(0.0)

    # Ensure numeric for all feature columns we care about
    for c in FEATURE_COLS_GLOBAL + [
        "first_seen",
        "last_seen",
        "total_tokens_sum",
        "completion_tokens_mean",
        "dominant_model_share",
    ]:
        if c in user_stats.columns:
            user_stats[c] = pd.to_numeric(user_stats[c], errors="coerce").fillna(0.0)

    # Save raw user feature table
    save_csv(user_stats, "apertus70b_user_stats.csv", out)

    # ----------------------------------------------------------------
    # Clustering on global feature set
    # ----------------------------------------------------------------
    feats_global = _build_features(user_stats, FEATURE_COLS_GLOBAL)
    labels_global, k_str_global, sil_str_global = _cluster_and_plot(
        feats=feats_global,
        stats_df=user_stats.copy(),
        out_dir=Path(out),
        title_prefix="Apertus-70B users",
    )

    # Additional raw-mean per-cluster bar plots (one per feature)
    if len(np.unique(labels_global)) > 1:
        numeric_cols = [
            c for c in FEATURE_COLS_GLOBAL
            if c in user_stats.columns and pd.api.types.is_numeric_dtype(user_stats[c])
        ]
        raw_cluster_means = (
            user_stats.assign(cluster_kmeans=labels_global)
            .groupby("cluster_kmeans", observed=True)[numeric_cols]
            .mean(numeric_only=True)
        )
        cats = sorted(pd.unique(labels_global).astype(int))
        for feat in numeric_cols:
            series = raw_cluster_means[feat]
            bar(
                series=series,
                title=f"{feat} — mean per cluster (Apertus-70B users)",
                xlabel="cluster",
                ylabel=f"mean({feat})",
                fname=f"cluster_profile_{feat}.png",
                outdir=out,
                order=cats,
                sort_values=False,
            )

    # Compact JSON summary (add k / silhouette)
    summary_rows: list[dict[str, Any]] = []
    user_stats_with_labels = user_stats.copy()
    user_stats_with_labels["cluster_kmeans"] = labels_global if labels_global is not None else -1
    for c_id, subc in user_stats_with_labels.groupby("cluster_kmeans", observed=True):
        cluster_id = _as_int(c_id)
        summary_rows.append(
            {
                "cluster": cluster_id,
                "n_users": int(len(subc)),
                "median_requests": float(np.asarray(subc["request_count"].median()).item()) if len(subc) else 0.0,
                "median_total_spend": float(np.asarray(subc["total_spend"].median()).item()) if len(subc) else 0.0,
                "median_req_per_active_day": float(
                    np.asarray(subc["requests_per_active_day"].median()).item()
                )
                if len(subc)
                else 0.0,
            }
        )

    summary_meta = {
        "k_selected": k_str_global,
        "silhouette": sil_str_global,
        "clusters": summary_rows,
    }
    save_text(json.dumps(summary_meta, indent=2), "apertus70b_cluster_summary.json", out)


if __name__ == "__main__":
    DATA_DIR = "data"
    FIGS_DIR = "figs"
    analyze_apertus70b_user_clusters(
        dir_name=DATA_DIR,
        dataset="litellm",
        outdir=f"{FIGS_DIR}/apertus70b_user_clusters",
    )
