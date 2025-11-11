import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utility_scripts.file_utils import PathLike, to_dt, ensure_empty_dir, read_table, save_csv, sanitize_fname, save_text
from utility_scripts.plot_utils import line, bar, hist, scatter


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning for LiteLLM_SpendLogs.

    - Coerce numerics for spend and token counts; clamp negative spend to 0.
    - Parse datetimes (UTC) for start/completion/end.
    - Compute latency metrics and a daily date column.
    - Drop rows without a valid end_user.
    """
    out = df.copy()

    # Spend
    out["spend"] = pd.to_numeric(out["spend"], errors="coerce").fillna(0.0)
    out.loc[out["spend"] < 0, "spend"] = 0.0

    # Tokens
    for col in ("total_tokens", "prompt_tokens", "completion_tokens"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Datetimes
    out["startTime"] = to_dt(out["startTime"])
    out["completionStartTime"] = to_dt(out["completionStartTime"])
    out["endTime"] = to_dt(out["endTime"])

    # Durations (seconds)
    out["latency_s"] = (out["endTime"] - out["startTime"]).dt.total_seconds()
    out["ttfb_s"] = (out["completionStartTime"] - out["startTime"]).dt.total_seconds()
    out["gen_s"] = (out["endTime"] - out["completionStartTime"]).dt.total_seconds()

    # Daily key
    out["date"] = out["startTime"].dt.floor("D")

    # Drop invalid end_user
    end_user_str = out["end_user"].astype(str).str.strip()
    out = out[end_user_str.ne("") & end_user_str.ne("nan") & end_user_str.notna()]
    out = out[out["end_user"].notna() & (out["end_user"].astype(str).str.len() > 0)]

    # Completion ratio
    out["completion_ratio"] = (out["completion_tokens"] / out["total_tokens"].replace(0, np.nan)).fillna(0.0)

    # Final sanity
    out = out[out["status"] == "success"]
    out = out.dropna(subset=["startTime", "date"])  # must have a timestamp
    return out


def plot_clustered_histograms(user_stats: pd.DataFrame, out: PathLike) -> None:
    out = Path(out)

    required = {"cluster_kmeans", "active_days"}
    missing = required - set(user_stats.columns)
    if missing:
        raise KeyError(f"user_stats missing columns: {sorted(missing)}")

    # numeric series for consistent bins
    active_days_all = pd.to_numeric(user_stats["active_days"], errors="coerce")
    clusters: list[int] = sorted(map(int, pd.unique(user_stats["cluster_kmeans"])))

    # build per-cluster arrays
    data: list[np.ndarray] = []
    for c in clusters:
        s = pd.to_numeric(
            user_stats.loc[user_stats["cluster_kmeans"] == c, "active_days"],
            errors="coerce",
        ).dropna()
        data.append(s.to_numpy())

    # common bins across clusters
    if active_days_all.dropna().empty:
        bins: int | list[float] = 30
    else:
        bins = np.histogram_bin_edges(active_days_all.dropna().to_numpy(), bins=30).astype(float).tolist()

    # colors
    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(clusters))]

    # linear
    plt.figure()
    plt.hist(
        data,
        bins=bins,
        stacked=True,
        color=colors,
        label=[f"Cluster {c}" for c in clusters],
        edgecolor="none",
    )
    plt.xlabel("Active days")
    plt.ylabel("Users")
    plt.title("Distribution of active days per end_user")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "user_active_days_hist_clustered.png", dpi=150)
    plt.close()

    # log-y
    plt.figure()
    plt.hist(
        data,
        bins=bins,
        stacked=True,
        color=colors,
        label=[f"Cluster {c}" for c in clusters],
        edgecolor="none",
        log=True,
    )
    plt.xlabel("Active days")
    plt.ylabel("Users (log scale)")
    plt.title("Distribution of active days per end_user (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "user_active_days_hist_clustered_log.png", dpi=150)
    plt.close()


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
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = _clean(df)

    # Per-user aggregates
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
        .assign(top_model_share=lambda x: x["max"] / x["sum"])
        [["top_model_share"]]
        .reset_index()
    )

    def _mean_inter_request_seconds_from_series(s: pd.Series) -> float:
        t = s.sort_values()
        if t.size <= 1:
            return np.nan
        d = t.diff().dropna().dt.total_seconds()
        return float(d.mean()) if len(d) else np.nan

    temporal = (
        df.groupby("end_user", observed=True)["startTime"]
        .apply(_mean_inter_request_seconds_from_series)
        .rename("mean_inter_req_s")
        .reset_index()
    )

    active_days = (df
        .dropna(subset=["date"])
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

    user_stats["active_days"] = user_stats["active_days"].fillna(0).astype(int)
    user_stats["requests_per_active_day"] = (
        user_stats["request_count"] / user_stats["active_days"].replace(0, np.nan)
    ).fillna(0.0)

    # Save aggregated table
    save_csv(user_stats, "user_stats.csv", out)

    # Histogram: active days per user
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

    # Hist: total spend per user (log1p to use utils.hist)
    spend_log1p = np.log1p(user_stats["total_spend"].clip(lower=0))
    hist(
        pd.Series(spend_log1p),
        title="Distribution of total spend per end_user (log1p scale)",
        xlabel="log1p(total_spend)",
        ylabel="Users",
        fname="user_total_spend_hist.png",
        outdir=out,
        bins=60,
    )

    # Scatter: requests vs total spend (log1p on both axes, uses utils.scatter)
    req_log = np.log1p(user_stats["request_count"].astype(float))
    spend_log = np.log1p(user_stats["total_spend"].astype(float))
    scatter(
        x=pd.Series(req_log),
        y=pd.Series(spend_log),
        title="Spend vs requests per end_user (log1p–log1p)",
        xlabel="log1p(requests)",
        ylabel="log1p(total_spend)",
        fname="user_spend_vs_requests_loglog.png",
        outdir=out,
    )

    # Whale curve using utils.line
    whale = user_stats.loc[:, ["end_user", "total_spend"]].copy()
    whale["total_spend"] = whale["total_spend"].fillna(0.0)
    whale = whale.sort_values("total_spend", ascending=False, ignore_index=True)
    whale["user_rank_frac"] = (np.arange(1, len(whale) + 1, dtype=float)) / max(len(whale), 1)
    tot = whale["total_spend"].sum()
    whale["cum_share"] = whale["total_spend"].cumsum() / tot if tot > 0 else 0.0

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

    # Clustering features
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

    for c in ["total_spend", "request_count", "avg_spend_per_req", "requests_per_active_day", "mean_inter_req_s"]:
        if c in features:
            features[c] = np.log1p(features[c].clip(lower=0))

    features = features.fillna(features.median(numeric_only=True))

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    # PCA
    pca = PCA(n_components=3, random_state=0)
    X_pca = pca.fit_transform(X)
    user_stats["pca1"] = X_pca[:, 0]
    user_stats["pca2"] = X_pca[:, 1]
    user_stats["pca3"] = X_pca[:, 2]
    explained = pca.explained_variance_ratio_

    # KMeans model selection
    best = {"k": None, "score": -np.inf, "labels": None, "model": None}
    for k in range(2, 6):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -np.inf
            if score > best["score"]:
                best.update({"k": k, "score": score, "labels": labels, "model": km})
        except Exception:
            continue

    user_stats["cluster_kmeans"] = best["labels"].astype(int)

    plot_clustered_histograms(user_stats, out)

    # Cluster profile table
    cluster_profiles = (
        pd.DataFrame(features, columns=feature_cols)
        .assign(cluster=user_stats["cluster_kmeans"].values)
        .groupby("cluster")
        .agg(["mean", "median"])
    )
    save_csv(cluster_profiles, "cluster_profiles.csv", out)

    # Save clustered users
    save_csv(
        user_stats[["end_user", "cluster_kmeans", "pca1", "pca2", "pca3"] + feature_cols],
        "user_clusters.csv",
        out,
    )

    # PCA scatter with cluster colors (matplotlib, utils.scatter lacks color)
    plt.figure()
    sc = plt.scatter(user_stats["pca1"], user_stats["pca2"], c=user_stats["cluster_kmeans"], s=10, alpha=0.8)
    plt.xlabel(f"PCA1 ({explained[0]*100:.1f}% var)")
    plt.ylabel(f"PCA2 ({explained[1]*100:.1f}% var)")
    sil = best["score"]
    sil_str = f"{sil:.3f}" if np.isfinite(sil) else "nan"
    plt.title(f"PCA of users colored by KMeans clusters (k={best['k']}, silhouette={sil_str})")
    plt.tight_layout()
    plt.savefig(Path(out) / "users_pca_clusters.png", dpi=150)
    plt.close()

    # Cluster normalized feature profiles (z-scored means)
    cluster_means = (
        pd.DataFrame(features, columns=feature_cols)
        .assign(cluster=user_stats["cluster_kmeans"].values)
        .groupby("cluster")[feature_cols]
        .mean()
    )
    feat_means = pd.DataFrame(features, columns=feature_cols).mean()
    feat_stds = pd.DataFrame(features, columns=feature_cols).std().replace(0, np.nan)
    cluster_means_z = (cluster_means - feat_means) / feat_stds

    plt.figure(figsize=(max(8, 0.6 * len(feature_cols)), 5))
    cluster_means_z.T.plot(kind="bar")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean feature (z-score)")
    plt.title("Cluster profiles (z-scored feature means)")
    plt.tight_layout()
    plt.savefig(Path(out) / "cluster_profiles_bars.png", dpi=150)
    plt.close()

    # Per-feature raw means by cluster (one plot per feature, no normalization)
    raw_cluster_means = user_stats.groupby("cluster_kmeans", observed=True)[feature_cols].mean()

    for feat in feature_cols:
        cats = sorted(pd.unique(user_stats["cluster_kmeans"]).astype(int))
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
        # normalize numpy scalars to Python scalars
        cluster_id = int(np.asarray(c_id).item())

        median_requests = float(np.asarray(sub["request_count"].median()).item()) if len(sub) else 0.0
        median_total_spend = float(np.asarray(sub["total_spend"].median()).item()) if len(sub) else 0.0
        median_req_per_active_day = float(np.asarray(sub["requests_per_active_day"].median()).item()) if len(sub) else 0.0

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
    analyze_users_usage()
