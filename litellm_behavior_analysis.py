"""
Model behavior & stability analysis for swiss-ai/apertus-70b-instruct based on LiteLLM_SpendLogs.

Focuses on behavioral proxies 
(verbosity, token composition, variability, retry patterns, and stability across workflows).

Produces plots in figs/litellm_apertus70b_behavior:

- Completion length / verbosity distributions and daily quantiles
- Prompt vs completion length scatter
- Token-composition (prompt/completion ratios) distributions + drift over time
- Variability (CV) of completion lengths across users (aggregated)
- Inter-arrival / retry behavior within sessions or per user
- Correlations between completion length, tokens, and latency (with sign)

Requires helper utilities:
  - utility_scripts.file_utils
  - utility_scripts.plot_utils
  - utility_scripts.df_reading_utils
"""

import numpy as np
import pandas as pd

from utility_scripts.file_utils import PathLike, ensure_empty_dir, read_table
from utility_scripts.plot_utils import line, bar, hist, scatter, lines_quantiles
from utility_scripts.df_reading_utils import (
    require, clean_table, to_utc, ensure_date_column, safe_div, safe_quantile_cut
)

TARGET_MODEL_GROUP = "swiss-ai/apertus-70b-instruct"


def _add_log_suffix(fname: str) -> str:
    if fname.lower().endswith(".png"):
        return fname[:-4] + "_log.png"
    return fname + "_log"


def plot_apertus70b_behavior(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/litellm_apertus70b_behavior",
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
    df["date"] = ts.dt.floor("D")
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
    # Distributions of prompt / completion lengths (linear + log-scale variants)
    for col, title, fname in [
        ("prompt_tokens", "Prompt tokens per request (Apertus 70B)", "hist_prompt_tokens.png"),
        ("completion_tokens", "Completion tokens per request (Apertus 70B)", "hist_completion_tokens.png"),
    ]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            # Linear scale
            hist(
                s,
                title,
                col,
                "count",
                fname,
                out,
                bins=50,
                log_scale=False,
            )
            # Log scale
            hist(
                s,
                title + " (log scale)",
                col,
                "count",
                _add_log_suffix(fname),
                out,
                bins=50,
                log_scale=True,
            )

    # Trimmed completion-token distribution to highlight typical region (linear only)
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
    # 2. Token-composition behavior & drift (prompt/completion ratios)
    # ------------------------------------------------------------------
    for col, title, fname in [
        ("prompt_ratio", "Prompt ratio = prompt/total (Apertus 70B)", "hist_prompt_ratio.png"),
        ("completion_ratio", "Completion ratio = completion/total (Apertus 70B)", "hist_completion_ratio.png"),
    ]:
        if col in df.columns:
            s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if not s.empty:
                hist(
                    s.clip(lower=0, upper=1),
                    title,
                    col,
                    "count",
                    fname,
                    out,
                    bins=50,
                )

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
    def _variability_by_user(min_reqs: int = 30) -> dict[str, float] | None:
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
            return None

        return {
            "n_users_ge_min_reqs": int(len(grp)),
            "mean_cv_completion": float(grp["cv_completion"].mean()),
            "p90_cv_completion": float(grp["cv_completion"].quantile(0.9)),
        }

    variability_summary = _variability_by_user(min_reqs=50)

    # ------------------------------------------------------------------
    # 4. Retry / inter-arrival behavior (stability in interaction)
    # ------------------------------------------------------------------
    # Use session_id if available, otherwise fall back to end_user
    group_key = "session_id" if "session_id" in df.columns else "end_user"

    df = df.sort_values([group_key, "start_ts"], kind="stable")
    df["inter_arrival_s"] = (
        df.groupby(group_key)["start_ts"].diff().dt.total_seconds()
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


    # ------------------------------------------------------------------
    # 5. Correlations involving completion behavior (signed) + extra plots
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
    corr_matrix = None
    if not corr_df.empty:
        corr_matrix = corr_df.corr(method="pearson")

        # Signed correlations with completion_tokens as a bar plot
        v = corr_matrix["completion_tokens"].drop("completion_tokens", errors="ignore").sort_values(ascending=False)
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

        # Additional correlation scatter plots (trimmed to reduce outliers)
        def _scatter_corr_pair(
            x_col: str,
            y_col: str,
            title: str,
            fname: str,
            quantile: float = 0.99,
        ) -> None:
            x = safe_quantile_cut(corr_df[x_col], quantile)
            y = safe_quantile_cut(corr_df[y_col], quantile)
            idx = x.index.intersection(y.index)
            if len(idx) == 0:
                return
            scatter(
                corr_df.loc[idx, x_col],
                corr_df.loc[idx, y_col],
                title,
                x_col.replace("_", " "),
                y_col.replace("_", " "),
                fname,
                out,
            )

        _scatter_corr_pair(
            "completion_tokens",
            "latency_s",
            "Completion tokens vs latency (trimmed, Apertus 70B)",
            "scatter_completion_vs_latency.png",
        )
        _scatter_corr_pair(
            "completion_tokens",
            "ttft_s",
            "Completion tokens vs TTFT (trimmed, Apertus 70B)",
            "scatter_completion_vs_ttft.png",
        )
        _scatter_corr_pair(
            "completion_tokens",
            "gen_s",
            "Completion tokens vs generation time (trimmed, Apertus 70B)",
            "scatter_completion_vs_gen.png",
        )
        _scatter_corr_pair(
            "prompt_tokens",
            "completion_tokens",
            "Prompt tokens vs completion tokens (trimmed, Apertus 70B)",
            "scatter_prompt_vs_completion_corrview.png",
        )


if __name__ == "__main__":
    plot_apertus70b_behavior()
