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

from utility_scripts.file_utils import PathLike, ensure_empty_dir, read_table, save_csv, sanitize_fname, save_text
from utility_scripts.plot_utils import line, bar, hist, grouped_hist
from utility_scripts.df_reading_utils import require, safe_div, clean_table, cumulative_share


# ---------- Small utilities ----------

def _mean_inter_request_seconds_from_series(s: pd.Series) -> float:
    t = pd.to_datetime(s, utc=True, errors="coerce").dropna().sort_values()
    if t.size <= 1:
        return float("nan")
    d = t.diff().dropna().dt.total_seconds()
    return float(d.mean()) if len(d) else float("nan")


FEATURE_COLS = [
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

LOG_COLS = (
    "total_spend",
    "request_count",
    "avg_spend_per_req",
    "requests_per_active_day",
    "mean_inter_req_s",
)


def _build_features(df_stats: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return a numeric, finite feature matrix with log1p on heavy-tailed columns, median imputation, and constant-column drop."""
    feats = df_stats[feature_cols].copy()

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
    name: str,
    feats: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
) -> tuple[np.ndarray, str, str]:
    """
    Fit KMeans with k in [2..min(5, n)], select by silhouette, add PCA for viz, make plots, write CSV/JSON.
    Returns (labels, k_str, sil_str). Uses filenames identical to the original script.
    """
    # Degenerate guards
    if feats.shape[0] < 2 or feats.shape[1] == 0:
        labels = np.full(len(stats_df), -1, dtype=int)
        stats_df["cluster_kmeans"] = labels
        save_csv(stats_df, "user_clusters.csv", out_dir)
        save_text(
            json.dumps([{"cluster": -1, "n_users": int(len(stats_df)), "reason": "insufficient data"}], indent=2),
            "cluster_summary.json",
            out_dir,
        )
        return labels, "NA", "nan"

    # Standardize and optional PCA for visualization
    X = feats.values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # PCA components limited by samples and features
    n_comp = int(min(3, X_std.shape[0], X_std.shape[1]))
    pca = None
    X_pca = None
    explained = np.array([])
    if n_comp >= 1:
        pca = PCA(n_components=n_comp, random_state=0)
        X_pca = pca.fit_transform(X_std)
        explained = pca.explained_variance_ratio_
        for i in range(n_comp):
            stats_df[f"pca{i+1}"] = X_pca[:, i]

    # KMeans model selection
    best_k, best_score, best_labels = None, -np.inf, None
    k_max = min(6, X_std.shape[0] + 1)
    for k in range(2, k_max):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labs = km.fit_predict(X_std)
            if np.unique(labs).size < 2:
                continue
            score = silhouette_score(X_std, labs)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labs
        except Exception:
            # Keep trying other k
            continue

    if best_labels is None:
        labels = np.full(len(stats_df), -1, dtype=int)
        k_str, sil_str = "NA", "nan"
    else:
        labels = np.asarray(best_labels, dtype=int)
        k_str, sil_str = str(best_k), f"{best_score:.3f}"

    stats_df["cluster_kmeans"] = labels

    # Save clustered table (keep PCA columns if present)
    keep_pca = [c for c in (f"pca1", f"pca2", f"pca3") if c in stats_df.columns]
    save_csv(stats_df[[*stats_df.columns.intersection(["end_user"]), "cluster_kmeans", *keep_pca, *FEATURE_COLS]], "user_clusters.csv", out_dir)

    # PCA scatter if ≥2 PCs
    if {"pca1", "pca2"}.issubset(stats_df.columns):
        plt.figure()
        plt.scatter(stats_df["pca1"], stats_df["pca2"], c=labels, s=10, alpha=0.8)
        e1 = (explained[0] * 100) if explained.size >= 1 else 0.0
        e2 = (explained[1] * 100) if explained.size >= 2 else 0.0
        plt.xlabel(f"PCA1 ({e1:.1f}% var)")
        plt.ylabel(f"PCA2 ({e2:.1f}% var)")
        plt.title(f"{title_prefix} — PCA by KMeans clusters (k={k_str}, silhouette={sil_str})")
        plt.tight_layout()
        plt.savefig(out_dir / "users_pca_clusters.png", dpi=150)
        plt.close()

    # Clustered histograms on active_days (if present)
    if "active_days" in stats_df.columns:
        grouped_hist(
            values=stats_df["active_days"],
            groups=stats_df["cluster_kmeans"],
            title=f"{title_prefix} — active days per end_user",
            xlabel="Active days",
            ylabel="Users",
            fname="user_active_days_hist_clustered.png",
            outdir=out_dir,
            bins=30,
            stacked=True,
            log_scale=False,
        )
        grouped_hist(
            values=stats_df["active_days"],
            groups=stats_df["cluster_kmeans"],
            title=f"{title_prefix} — active days per end_user (log scale)",
            xlabel="Active days",
            ylabel="Users (log scale)",
            fname="user_active_days_hist_clustered_log.png",
            outdir=out_dir,
            bins=30,
            stacked=True,
            log_scale=True,
        )

    # Cluster profiles: z-scored feature means
    raw_means = stats_df.groupby("cluster_kmeans", observed=True)[FEATURE_COLS].mean(numeric_only=True)
    feat_means = feats.mean()
    feat_stds = feats.std().replace(0, np.nan)
    cluster_means_z = (raw_means[feats.columns] - feat_means) / feat_stds
    cluster_means_z = cluster_means_z.replace([np.inf, -np.inf], np.nan)

    if not cluster_means_z.dropna(how="all").empty:
        plot_df = cluster_means_z.astype("float64").fillna(0.0)
        plt.figure(figsize=(max(8, 0.6 * len(plot_df.columns)), 5))
        plot_df.T.plot(kind="bar")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean feature (z-score)")
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

    # Cluster profile table with mean/median of raw features actually present
    cluster_profiles = stats_df.groupby("cluster_kmeans", observed=True)[FEATURE_COLS].agg(["mean", "median"])
    save_csv(cluster_profiles, "cluster_profiles.csv", out_dir)

    # Compact JSON summary

    def _safe_median(df: pd.DataFrame, col: str) -> float:
        if col not in df:
            return 0.0
        s = pd.to_numeric(df[col], errors="coerce")
        if s.empty:
            return 0.0
        val = s.median(skipna=True)
        return float(val) if pd.notna(val) else 0.0
    
    def _as_int(x: Any) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        try:
            return int(np.asarray(x).item())
        except Exception:
            s = pd.to_numeric(pd.Series([x]), errors="coerce")
            v = s.iloc[0]
            return int(v) if pd.notna(v) else -1

    rows: list[dict[str, float | int]] = []
    for cid, subc in stats_df.groupby("cluster_kmeans", observed=True):
        cid_int = _as_int(cid)
        rows.append({
            "cluster": cid_int,
            "n_users": int(len(subc)),
            "median_requests": _safe_median(subc, "request_count"),
            "median_total_spend": _safe_median(subc, "total_spend"),
            "median_req_per_active_day": _safe_median(subc, "requests_per_active_day"),
        })

    save_text(json.dumps(rows, indent=2), "cluster_summary.json", out_dir)

    return labels, k_str, sil_str


# ---------- Main analysis ----------

def analyze_users_usage(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/users_usage",
) -> None:
    out = ensure_empty_dir(outdir)

    df = read_table("LiteLLM_SpendLogs", dataset=dataset, dir_name=dir_name)
    if df.empty:
        raise SystemExit("No data loaded from LiteLLM_SpendLogs.")

    needed = {
        "request_id", "spend", "startTime", "endTime", "completionStartTime",
        "total_tokens", "prompt_tokens", "completion_tokens",
        "model", "model_group", "custom_llm_provider",
        "api_key", "end_user", "call_type", "status",
    }
    miss = require(df, needed, strict=False)
    if miss:
        raise SystemExit(f"Missing required columns: {sorted(miss)}")

    # Clean + enrich
    df = clean_table(df)

    # ---------- Per-user aggregates ----------
    user_model_counts = (
        df.groupby(["end_user", "model"], observed=True)["request_id"]
        .count().rename("model_req_count").reset_index()
    )

    top_model_share = (
        user_model_counts.sort_values(["end_user", "model_req_count"], ascending=[True, False])
        .groupby("end_user", observed=True)["model_req_count"]
        .agg(["sum", "max"])
        .assign(top_model_share=lambda x: x["max"] / x["sum"])
        [["top_model_share"]]
        .reset_index()
    )

    temporal = (
        df.groupby("end_user", observed=True)["startTime"]
        .apply(_mean_inter_request_seconds_from_series)
        .rename("mean_inter_req_s")
        .reset_index()
    )

    active_days = (
        df.dropna(subset=["date"])
        .groupby("end_user", observed=True)["date"]
        .nunique()
        .rename("active_days")
        .reset_index()
    )

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

    user_first_last = (
        df.groupby("end_user", observed=True)["startTime"].agg(first_seen="min", last_seen="max").reset_index()
    )

    user_stats = (
        base_agg.merge(top_model_share, on="end_user", how="left")
                .merge(active_days, on="end_user", how="left")
                .merge(temporal, on="end_user", how="left")
                .merge(user_first_last, on="end_user", how="left")
    )

    # Robust rates
    user_stats["active_days"] = pd.to_numeric(user_stats["active_days"], errors="coerce").fillna(0).astype(int)
    user_stats["requests_per_active_day"] = safe_div(
        user_stats["request_count"], user_stats["active_days"], allow_zero=False
    ).fillna(0.0)

    save_csv(user_stats, "user_stats.csv", out)

    # ---------- Distributions ----------
    hist(
        series=user_stats["active_days"],
        title="Distribution of active days per end_user",
        xlabel="Active days",
        ylabel="Users",
        fname="user_active_days_hist.png",
        outdir=out,
        bins=30,
    )
    hist(
        series=user_stats["active_days"],
        title="Distribution of active days per end_user (log scale)",
        xlabel="Active days",
        ylabel="Users (log scale)",
        fname="user_active_days_hist_log.png",
        outdir=out,
        bins=30,
        log_scale=True,
    )

    spend_log1p = np.log1p(pd.to_numeric(user_stats["total_spend"], errors="coerce").clip(lower=0))
    hist(
        series=pd.Series(spend_log1p),
        title="Distribution of total spend per end_user (log1p scale)",
        xlabel="log1p(total_spend)",
        ylabel="Users",
        fname="user_total_spend_hist.png",
        outdir=out,
        bins=60,
    )

    # ---------- Whale curve ----------
    whale = cumulative_share(
        df=user_stats.rename(columns={"end_user": "key_tmp"}),
        key="key_tmp",
        value_col="total_spend",
        positive_only=True,
        normalize_rank=True,
        dropna_key=True,
    ).rename(columns={"rank": "user_rank_frac", "cum_share": "cum_share"})
    save_csv(whale, "whale_curve.csv", out)

    line(
        x="user_rank_frac",
        y="cum_share",
        data=whale,
        title="Whale curve of spend by end_user",
        xlabel="User rank (fraction)",
        ylabel="Cumulative spend share",
        fname="user_spend_whale_curve.png",
        outdir=out,
    )

    # ---------- Global clustering ----------
    feats_global = _build_features(user_stats, FEATURE_COLS)
    labels_global, k_str_global, sil_str_global = _cluster_and_plot(
        name="global",
        feats=feats_global,
        stats_df=user_stats.copy(),  # copy for safe column adds
        out_dir=Path(out),
        title_prefix="All models",
    )

    # ---------- Per-feature raw means per cluster (global) ----------
    if len(np.unique(labels_global)) > 1:
        raw_cluster_means = user_stats.assign(cluster_kmeans=labels_global).groupby("cluster_kmeans", observed=True)[FEATURE_COLS].mean()
        cats = sorted(pd.unique(labels_global).astype(int))
        for feat in FEATURE_COLS:
            series = raw_cluster_means[feat]
            bar(
                series=series,
                title=f"{feat} — mean per cluster (raw scale)",
                xlabel="Cluster",
                ylabel=f"mean({feat})",
                fname=f"cluster_profile_{sanitize_fname(feat)}.png",
                outdir=out,
                order=cats,
                sort_values=False,
            )

    # ---------- Per-model user clustering (top-10 by request volume) ----------
    top_10_models = df["model"].value_counts(dropna=True).head(10).index.tolist()
    per_model_index: list[dict[str, str | int]] = []

    for model_name in top_10_models:
        sub = df[df["model"] == model_name].copy()
        model_slug = sanitize_fname(str(model_name))
        out_model = ensure_empty_dir(Path(out) / "by_model" / model_slug)

        # Per-user stats within this model
        sub_user_temporal = (
            sub.groupby("end_user", observed=True)["startTime"]
            .apply(_mean_inter_request_seconds_from_series)
            .rename("mean_inter_req_s")
            .reset_index()
        )
        sub_active_days = (
            sub.dropna(subset=["date"])
            .groupby("end_user", observed=True)["date"]
            .nunique()
            .rename("active_days")
            .reset_index()
        )
        sub_base_agg = (
            sub.groupby("end_user", observed=True)
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
                n_unique_models=("model", "nunique"),          # ~1 here
                n_unique_model_groups=("model_group", "nunique"),
                n_unique_providers=("custom_llm_provider", "nunique"),
            )
            .reset_index()
        )

        sub_user_stats = (
            sub_base_agg.merge(sub_active_days, on="end_user", how="left")
                        .merge(sub_user_temporal, on="end_user", how="left")
        )
        sub_user_stats["active_days"] = pd.to_numeric(sub_user_stats["active_days"], errors="coerce").fillna(0).astype(int)
        sub_user_stats["requests_per_active_day"] = safe_div(
            sub_user_stats["request_count"], sub_user_stats["active_days"], allow_zero=False
        ).fillna(0.0)

        save_csv(sub_user_stats, "user_stats.csv", out_model)

        # Build features and cluster via the shared helper
        feats_m = _build_features(sub_user_stats, FEATURE_COLS)
        labels_m, k_str_m, sil_str_m = _cluster_and_plot(
            name=str(model_name),
            feats=feats_m,
            stats_df=sub_user_stats.copy(),
            out_dir=Path(out_model),
            title_prefix=str(model_name),
        )

        # Per-model index row
        per_model_index.append({
            "model": str(model_name),
            "k": k_str_m,
            "silhouette": sil_str_m,
            "n_users": int(len(sub_user_stats)),
            "out_dir": str(Path(out_model)),
        })

    # Master index of per-model runs
    save_text(json.dumps(per_model_index, indent=2), "by_model_index.json", Path(out))

    # ---------- Compact JSON summary per global cluster ----------
    summary_rows = []
    # Use user_stats + global labels for the membership
    user_stats_with_labels = user_stats.copy()
    user_stats_with_labels["cluster_kmeans"] = labels_global if labels_global is not None else -1
    for c_id, subc in user_stats_with_labels.groupby("cluster_kmeans", observed=True):
        cluster_id = int(np.asarray(c_id).item())
        top_3_models = (
            df[df["end_user"].isin(subc["end_user"])]
            .groupby("model")["request_id"]
            .count()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        summary_rows.append(
            {
                "cluster": cluster_id,
                "n_users": int(len(subc)),
                "median_requests": float(np.asarray(subc["request_count"].median()).item()) if len(subc) else 0.0,
                "median_total_spend": float(np.asarray(subc["total_spend"].median()).item()) if len(subc) else 0.0,
                "median_req_per_active_day": float(np.asarray(subc["requests_per_active_day"].median()).item()) if len(subc) else 0.0,
                "top_3_models": top_3_models,
            }
        )
    save_text(json.dumps(summary_rows, indent=2), "cluster_summary.json", out)


if __name__ == "__main__":
    DATA_DIR = "data"
    FIGS_DIR = "figs"
    analyze_users_usage(dir_name=DATA_DIR, dataset="litellm", outdir=f"{FIGS_DIR}/users_usage")
