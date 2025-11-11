import json
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


def analyze_users_usage(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/users_usage",
) -> None:
    out = ensure_empty_dir(outdir)

    df = read_table("LiteLLM_SpendLogs", dataset=dataset, dir_name=dir_name)
    if df.empty:
        raise SystemExit("No data loaded from LiteLLM_SpendLogs.")

    # Minimal schema presence. `clean_table` handles typing and derived columns.
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
    }
    miss = require(df, needed, strict=False)
    if miss:
        raise SystemExit(f"Missing required columns: {sorted(miss)}")

    # Clean and enrich. Keep defaults: clamp negative spend, add date, durations, ratio.
    df = clean_table(df)

    # ---------- Per-user aggregates ----------
    # Requests per model per user
    user_model_counts = (
        df.groupby(["end_user", "model"], observed=True)["request_id"]
        .count()
        .rename("model_req_count")
        .reset_index()
    )

    # Share of top model per user
    top_model_share = (
        user_model_counts.sort_values(["end_user", "model_req_count"], ascending=[True, False])
        .groupby("end_user", observed=True)["model_req_count"]
        .agg(["sum", "max"])
        .assign(top_model_share=lambda x: x["max"] / x["sum"])
        [["top_model_share"]]
        .reset_index()
    )

    # Mean inter-request seconds per user
    def _mean_inter_request_seconds_from_series(s: pd.Series) -> float:
        t = pd.to_datetime(s, utc=True, errors="coerce").dropna().sort_values()
        if t.size <= 1:
            return float("nan")
        d = t.diff().dropna().dt.total_seconds()
        return float(d.mean()) if len(d) else float("nan")

    temporal = (
        df.groupby("end_user", observed=True)["startTime"]
        .apply(_mean_inter_request_seconds_from_series)
        .rename("mean_inter_req_s")
        .reset_index()
    )

    # Active days per user (uses date added by clean_table)
    active_days = (
        df.dropna(subset=["date"])
        .groupby("end_user", observed=True)["date"]
        .nunique()
        .rename("active_days")
        .reset_index()
    )

    # Base numeric aggregates
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

    # Save aggregated table
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

    # Total spend per user, log1p scale
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

    # ---------- Clustering ----------
    feature_cols = [
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
    features = user_stats[feature_cols].copy()

    # Log-transform heavy-tailed features
    for c in ["total_spend", "request_count", "avg_spend_per_req", "requests_per_active_day", "mean_inter_req_s"]:
        if c in features.columns:
            features[c] = np.log1p(pd.to_numeric(features[c], errors="coerce").clip(lower=0))

    # Median impute numerics
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True))

    # Guard against degenerate inputs
    if features.shape[0] < 2:
        # Not enough users to cluster. Save what we have and exit gracefully.
        user_stats["cluster_kmeans"] = -1
        save_csv(user_stats, "user_clusters.csv", out)
        save_text(json.dumps([{"cluster": -1, "n_users": int(len(user_stats))}], indent=2), "cluster_summary.json", out)
        return

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    # PCA with safe component count
    n_comp = int(min(3, X.shape[1], X.shape[0]))
    pca = PCA(n_components=n_comp, random_state=0)
    X_pca = pca.fit_transform(X)
    for i in range(n_comp):
        user_stats[f"pca{i+1}"] = X_pca[:, i]
    explained = pca.explained_variance_ratio_

    # KMeans model selection with guards
    best_k = None
    best_score = -np.inf
    best_labels = None
    for k in range(2, min(6, X.shape[0] + 1)):  # k cannot exceed number of samples
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels
        except Exception:
            continue

    if best_labels is None:
        # Fallback to single-cluster assignment
        user_stats["cluster_kmeans"] = -1
        sil_str = "nan"
        k_str = "NA"
    else:
        user_stats["cluster_kmeans"] = np.asarray(best_labels, dtype=int)
        sil_str = f"{best_score:.3f}"
        k_str = str(best_k)

    # Clustered histograms of active days
    require(user_stats, ["cluster_kmeans", "active_days"], strict=True)

    grouped_hist(
        values=user_stats["active_days"],
        groups=user_stats["cluster_kmeans"],
        title="Distribution of active days per end_user",
        xlabel="Active days",
        ylabel="Users",
        fname="user_active_days_hist_clustered.png",
        outdir=outdir,
        bins=30,
        stacked=True,
        log_scale=False,
    )

    grouped_hist(
        values=user_stats["active_days"],
        groups=user_stats["cluster_kmeans"],
        title="Distribution of active days per end_user (log scale)",
        xlabel="Active days",
        ylabel="Users (log scale)",
        fname="user_active_days_hist_clustered_log.png",
        outdir=outdir,
        bins=30,
        stacked=True,
        log_scale=True,
    )


    # Cluster profile table (mean + median of raw features)
    cluster_profiles = (
        pd.DataFrame(features, columns=feature_cols)
        .assign(cluster=user_stats["cluster_kmeans"].values)
        .groupby("cluster")
        .agg(["mean", "median"])
    )
    save_csv(cluster_profiles, "cluster_profiles.csv", out)

    # Save clustered users
    keep_pca = [c for c in [f"pca{i}" for i in range(1, 4)] if c in user_stats.columns]
    save_csv(
        user_stats[["end_user", "cluster_kmeans", *keep_pca] + feature_cols],
        "user_clusters.csv",
        out,
    )

    # PCA scatter with cluster colors (if at least 2 PCs)
    if {"pca1", "pca2"}.issubset(user_stats.columns):
        plt.figure()
        sc = plt.scatter(user_stats["pca1"], user_stats["pca2"], c=user_stats["cluster_kmeans"], s=10, alpha=0.8)
        exp1 = explained[0] * 100 if explained.size >= 1 else 0.0
        exp2 = explained[1] * 100 if explained.size >= 2 else 0.0
        plt.xlabel(f"PCA1 ({exp1:.1f}% var)")
        plt.ylabel(f"PCA2 ({exp2:.1f}% var)")
        plt.title(f"PCA of users colored by KMeans clusters (k={k_str}, silhouette={sil_str})")
        plt.tight_layout()
        plt.savefig(Path(out) / "users_pca_clusters.png", dpi=150)
        plt.close()

    # Cluster normalized feature profiles (z-scored means)
    raw_means = (
        pd.DataFrame(features, columns=feature_cols)
        .assign(cluster=user_stats["cluster_kmeans"].values)
        .groupby("cluster")[feature_cols]
        .mean()
    )
    feat_means = pd.DataFrame(features, columns=feature_cols).mean()
    feat_stds = pd.DataFrame(features, columns=feature_cols).std().replace(0, np.nan)
    cluster_means_z = (raw_means - feat_means) / feat_stds

    plt.figure(figsize=(max(8, 0.6 * len(feature_cols)), 5))
    cluster_means_z.T.plot(kind="bar")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean feature (z-score)")
    plt.title("Cluster profiles (z-scored feature means)")
    plt.tight_layout()
    plt.savefig(Path(out) / "cluster_profiles_bars.png", dpi=150)
    plt.close()

    # Per-feature raw means by cluster
    raw_cluster_means = user_stats.groupby("cluster_kmeans", observed=True)[feature_cols].mean()
    cats = sorted(pd.unique(user_stats["cluster_kmeans"]).astype(int))
    for feat in feature_cols:
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

    # Compact JSON summary per cluster
    summary_rows = []
    for c_id, sub in user_stats.groupby("cluster_kmeans", observed=True):
        cluster_id = int(np.asarray(c_id).item())
        # medians
        median_requests = float(np.asarray(sub["request_count"].median()).item()) if len(sub) else 0.0
        median_total_spend = float(np.asarray(sub["total_spend"].median()).item()) if len(sub) else 0.0
        median_req_per_active_day = float(np.asarray(sub["requests_per_active_day"].median()).item()) if len(sub) else 0.0
        # top-3 models among users in the cluster
        top_3_models = (
            df[df["end_user"].isin(sub["end_user"])]
            .groupby("model")["request_id"]
            .count()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        summary_rows.append(
            {
                "cluster": cluster_id,
                "n_users": int(len(sub)),
                "median_requests": median_requests,
                "median_total_spend": median_total_spend,
                "median_req_per_active_day": median_req_per_active_day,
                "top_3_models": top_3_models,
            }
        )

    save_text(json.dumps(summary_rows, indent=2), "cluster_summary.json", out)


if __name__ == "__main__":
    DATA_DIR = "data"
    FIGS_DIR = "figs"

    analyze_users_usage(dir_name=DATA_DIR, dataset="litellm", outdir=f"{FIGS_DIR}/users_usage")
