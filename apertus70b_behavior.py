"""
Model behavior & stability analysis for swiss-ai/apertus-70b-instruct
based on LiteLLM_SpendLogs.

Focuses on behavioral proxies (verbosity, token composition, variability,
retry patterns, and stability across workflows).

Produces plots in figs/apertus70b_behavior, including:

- Completion length / verbosity distributions and daily quantiles
- Prompt vs completion length scatter
- Token-composition (prompt/completion ratios) distributions + drift over time
- Variability (CV) of completion lengths across users
- Inter-arrival / retry behavior within sessions or per user
- Outliers and z-score drift in completion length
- Correlations between completion length and latency (with sign)

Requires helper utilities:
  - utility_scripts.file_utils
  - utility_scripts.plot_utils
  - utility_scripts.df_reading_utils
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utility_scripts.file_utils import (
    PathLike,
    ensure_empty_dir,
    read_table,
    sanitize_fname,
    save_csv,
)
from utility_scripts.plot_utils import (
    line,
    bar,
    hist,
    scatter,
    lines_quantiles,
)
from utility_scripts.df_reading_utils import (
    require,
    clean_table,
    to_utc,
    ensure_date_column,
    safe_div,
    safe_quantile_cut,
)

TARGET_MODEL_GROUP = "swiss-ai/apertus-70b-instruct"


def plot_apertus70b_behavior(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/apertus70b_behavior",
    top: int = 20,
) -> None:
    out = ensure_empty_dir(outdir)

    # ------------------------------------------------------------------
    # Load and restrict to the target model_group
    # ------------------------------------------------------------------
    df = read_table("LiteLLM_SpendLogs", dataset=dataset, dir_name=dir_name)
    if df.empty:
        raise SystemExit("No data loaded from LiteLLM_SpendLogs.")

    needed = {
        "request_id",
        "startTime",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "model_group",
        "end_user",
        "session_id",
        "status",
    }
    miss = require(df, needed, strict=False)
    if "model_group" in miss:
        raise SystemExit("Missing required column 'model_group'.")

    df = clean_table(df)  # adds latency_s, ttft_s, gen_s, completion_ratio, date etc.

    df["__success__"] = (
        df["status"].astype("string").str.lower().eq("success").fillna(False)
        if "status" in df.columns
        else pd.Series(False, index=df.index, dtype="boolean")
    )

    # Restrict to Apertus 70B instruct
    df = df[df["model_group"] == TARGET_MODEL_GROUP].copy()
    if df.empty:
        raise SystemExit(f"No rows for model_group={TARGET_MODEL_GROUP!r}.")

    # ------------------------------------------------------------------
    # Canonical time columns
    # ------------------------------------------------------------------
    ts = to_utc(df["startTime"])
    df["start_ts"] = ts
    df["date"] = ts.dt.floor("D")  # type: ignore[attr-defined]
    df = df.dropna(subset=["date"])
    df = ensure_date_column(df, time_col="startTime", out_col="date", floor="D", dropna=True, sort=True)

    # Token ratios (if not already present)
    if "prompt_ratio" not in df.columns and {"prompt_tokens", "total_tokens"} <= set(df.columns):
        df["prompt_ratio"] = safe_div(df["prompt_tokens"], df["total_tokens"]).clip(lower=0, upper=1)
    if "completion_ratio" not in df.columns and {"completion_tokens", "total_tokens"} <= set(df.columns):
        df["completion_ratio"] = safe_div(df["completion_tokens"], df["total_tokens"]).clip(lower=0, upper=1)

    # ------------------------------------------------------------------
    # 1. Verbosity & completion-length behavior
    # ------------------------------------------------------------------
    # Distributions of prompt / completion lengths
    for col, title, fname in [
        ("prompt_tokens", "Prompt tokens per request (Apertus 70B)", "hist_prompt_tokens.png"),
        ("completion_tokens", "Completion tokens per request (Apertus 70B)", "hist_completion_tokens.png"),
    ]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            hist(s, title, col, "count", fname, out, bins=50)

    # Trimmed completion-token distribution to highlight typical region
    comp_trimmed = safe_quantile_cut(df["completion_tokens"], 0.999)
    if not comp_trimmed.empty:
        hist(
            comp_trimmed,
            "Completion tokens (trimmed 99.9%, Apertus 70B)",
            "completion_tokens",
            "count",
            "hist_completion_tokens_trimmed.png",
            out,
            bins=50,
        )

    # Daily quantiles of completion length
    lines_quantiles(
        "date",
        "completion_tokens",
        df,
        "Completion tokens per request per day: Q1/Median/Q3/Mean (Apertus 70B)",
        "date",
        "completion tokens",
        "daily_completion_tokens_quantiles.png",
        out,
    )
    lines_quantiles(
        "date",
        "completion_tokens",
        df,
        "Completion tokens per request per day: Q1/Median/Q3/Mean/Min/Max (Apertus 70B)",
        "date",
        "completion tokens",
        "daily_completion_tokens_quantiles_with_max.png",
        out,
        plot_extrema=True,
    )

    # Prompt vs completion scatter (verbosity vs prompt size)
    prompt_clean = safe_quantile_cut(df["prompt_tokens"], 0.999)
    comp_clean = df.loc[prompt_clean.index, "completion_tokens"]
    if not prompt_clean.empty:
        scatter(
            prompt_clean,
            comp_clean,
            "Completion tokens vs prompt tokens (trimmed 99.9%, Apertus 70B)",
            "prompt tokens",
            "completion tokens",
            "scatter_prompt_vs_completion.png",
            out,
        )

    # ------------------------------------------------------------------
    # 2. Token-composition behavior & drift
    # ------------------------------------------------------------------
    for col, title, fname in [
        ("prompt_ratio", "Prompt ratio = prompt/total (Apertus 70B)", "hist_prompt_ratio.png"),
        ("completion_ratio", "Completion ratio = completion/total (Apertus 70B)", "hist_completion_ratio.png"),
    ]:
        if col in df.columns:
            s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if not s.empty:
                hist(s.clip(lower=0, upper=1), title, col, "count", fname, out, bins=50)

    # Daily averages and quantiles of ratios
    if "prompt_ratio" in df.columns or "completion_ratio" in df.columns:
        mean_aggs: dict[str, tuple[str, str]] = {}
        if "prompt_ratio" in df.columns:
            mean_aggs["prompt_ratio"] = ("prompt_ratio", "mean")
        if "completion_ratio" in df.columns:
            mean_aggs["completion_ratio"] = ("completion_ratio", "mean")

        if mean_aggs:
            daily_ratios = df.groupby("date", as_index=False).agg(**mean_aggs).sort_values("date", kind="stable")
            if "prompt_ratio" in daily_ratios.columns:
                line(
                    "date",
                    "prompt_ratio",
                    daily_ratios,
                    "Avg prompt ratio per day (Apertus 70B)",
                    "date",
                    "ratio",
                    "daily_avg_prompt_ratio.png",
                    out,
                )
            if "completion_ratio" in daily_ratios.columns:
                line(
                    "date",
                    "completion_ratio",
                    daily_ratios,
                    "Avg completion ratio per day (Apertus 70B)",
                    "date",
                    "ratio",
                    "daily_avg_completion_ratio.png",
                    out,
                )

        if "prompt_ratio" in df.columns:
            lines_quantiles(
                "date",
                "prompt_ratio",
                df,
                "Prompt ratio per day: Q1/Median/Q3/Mean (Apertus 70B)",
                "date",
                "ratio",
                "daily_prompt_ratio_quantiles.png",
                out,
            )
        if "completion_ratio" in df.columns:
            lines_quantiles(
                "date",
                "completion_ratio",
                df,
                "Completion ratio per day: Q1/Median/Q3/Mean (Apertus 70B)",
                "date",
                "ratio",
                "daily_completion_ratio_quantiles.png",
                out,
            )

    # ------------------------------------------------------------------
    # 3. Variability across users
    # ------------------------------------------------------------------
    def _variability_by_user(min_reqs: int = 30) -> None:
        grp = (
            df.groupby("end_user", dropna=False)
            .agg(
                requests=("request_id", "count"),
                mean_completion=("completion_tokens", "mean"),
                std_completion=("completion_tokens", "std"),
            )
            .reset_index()
        )
        grp = grp[grp["requests"] >= min_reqs]
        grp["cv_completion"] = safe_div(grp["std_completion"], grp["mean_completion"])
        if grp.empty:
            return

        save_csv(grp, "user_completion_variability.csv", out)

        top_by_reqs = grp.sort_values("requests", ascending=False).head(top)
        s_mean = top_by_reqs.set_index("end_user")["mean_completion"]
        s_cv = top_by_reqs.set_index("end_user")["cv_completion"]

        if not s_mean.empty:
            bar(
                s_mean,
                f"Mean completion tokens by end_user (Top {len(s_mean)} by requests)",
                "end_user",
                "mean completion tokens",
                "user_mean_completion_top.png",
                out,
                top=len(s_mean),
            )
        if not s_cv.empty:
            bar(
                s_cv,
                f"CV of completion tokens by end_user (Top {len(s_cv)} by requests)",
                "end_user",
                "CV(completion length)",
                "user_cv_completion_top.png",
                out,
                top=len(s_cv),
            )

    _variability_by_user(min_reqs=50)

    # ------------------------------------------------------------------
    # 4. Retry / inter-arrival behavior (stability in interaction)
    # ------------------------------------------------------------------
    # Use session_id if available, otherwise fall back to end_user
    group_key = "session_id" if "session_id" in df.columns else "end_user"

    df = df.sort_values([group_key, "start_ts"], kind="stable")
    df["inter_arrival_s"] = (
        df.groupby(group_key)["start_ts"].diff().dt.total_seconds()  # type: ignore[attr-defined]
    )

    inter = df["inter_arrival_s"].dropna()
    if not inter.empty:
        inter_trim = safe_quantile_cut(inter, 0.999)
        hist(
            inter_trim,
            f"Inter-arrival time within {group_key} (trimmed 99.9%, Apertus 70B)",
            "inter_arrival_s",
            "count",
            "hist_inter_arrival_s.png",
            out,
            bins=50,
        )

    # Rapid retries: inter-arrival <= 30s
    df["is_rapid_retry"] = df["inter_arrival_s"].le(30.0).fillna(False)
    daily_retry = (
        df.groupby("date", as_index=False)["is_rapid_retry"]
        .mean()
        .rename(columns={"is_rapid_retry": "rapid_retry_rate"})  # type: ignore[attr-defined]
        .sort_values("date", kind="stable")
    )
    if not daily_retry.empty:
        line(
            "date",
            "rapid_retry_rate",
            daily_retry,
            f"Rapid retry rate (inter-arrival ≤ 30s) per day (Apertus 70B, key={group_key})",
            "date",
            "rapid retry rate",
            "daily_rapid_retry_rate.png",
            out,
        )
        save_csv(daily_retry, "daily_rapid_retry_rate.csv", out)

    # ------------------------------------------------------------------
    # 5. Outliers and anomalies in completion length
    # ------------------------------------------------------------------
    # Top 0.1% largest completions
    n = max(1, int(len(df) * 0.001))
    top_comp = df.nlargest(n, "completion_tokens")
    save_csv(top_comp, "top_0p1pct_completion_tokens_rows.csv", out)

    # Daily z-score of avg completion length
    daily_comp = (
        df.groupby("date", as_index=False)["completion_tokens"]
        .mean()
        .rename(columns={"completion_tokens": "avg_completion_tokens"})  # type: ignore[attr-defined]
        .sort_values("date", kind="stable")
    )
    daily_comp["rolling_mean"] = daily_comp["avg_completion_tokens"].rolling(14, min_periods=7).mean()
    daily_comp["rolling_std"] = daily_comp["avg_completion_tokens"].rolling(14, min_periods=7).std()
    daily_comp["z"] = safe_div(
        daily_comp["avg_completion_tokens"] - daily_comp["rolling_mean"],
        daily_comp["rolling_std"],
    )
    line(
        "date",
        "z",
        daily_comp,
        "Daily z-score of avg completion tokens (window=14d, Apertus 70B)",
        "date",
        "z-score",
        "daily_avg_completion_tokens_zscore.png",
        out,
    )
    save_csv(daily_comp, "daily_avg_completion_tokens_zscore.csv", out)

    # ------------------------------------------------------------------
    # 6. Correlations involving completion behavior (signed)
    # ------------------------------------------------------------------
    corr_cols = [
        "completion_tokens",
        "prompt_tokens",
        "total_tokens",
        "latency_s",
        "ttft_s",
        "gen_s",
    ]
    corr_df = df[corr_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if not corr_df.empty:
        corr = corr_df.corr(method="pearson")
        save_csv(corr, "correlations_completion_behavior.csv", out)

        # Signed correlations with completion_tokens
        v = corr["completion_tokens"].drop("completion_tokens", errors="ignore").sort_values(ascending=False)
        if not v.empty:
            bar(
                v,
                "Correlation with completion_tokens (Apertus 70B)",
                "feature",
                "correlation",
                "corr_with_completion_tokens.png",
                out,
                top=len(v),
            )

    # Small summary table
    summary = {
        "n_rows": len(df),
        "distinct_users": int(df["end_user"].nunique()),
        "mean_completion_tokens": float(df["completion_tokens"].mean()),
        "p95_completion_tokens": float(df["completion_tokens"].quantile(0.95)),
        "rapid_retry_rate_overall": float(df["is_rapid_retry"].mean()),
    }
    save_csv(pd.DataFrame([summary]), "summary_behavior_overall.csv", out)


if __name__ == "__main__":
    plot_apertus70b_behavior()
