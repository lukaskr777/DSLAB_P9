"""
Performance & efficiency analysis for swiss-ai/apertus-70b-instruct
based on LiteLLM_SpendLogs.

Produces plots in figs/apertus70b_performance:

- Daily performance metrics (latency, TTFT, gen time, success rate)
- Daily quantile curves (Q1/median/Q3/mean) for latency, TTFT, gen time, tokens per request, and latency shares
- Token-efficiency metrics (prompt/completion ratios)
- Latency / TTFT / gen distributions (trimmed)
- Latency vs tokens (binned averages and quantiles)
- Success vs failure latency comparison
- Throughput vs latency (requests per minute)

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
    save_csv,
)
from utility_scripts.plot_utils import (
    line,
    hist,
    scatter,
    lines_quantiles,
    bin_and_quantiles,
)
from utility_scripts.df_reading_utils import (
    require,
    clean_table,
    to_utc,
    ensure_date_column,
    aggregate_by_time,
    add_rates,
    safe_div,
    safe_quantile_cut,
)

TARGET_MODEL_GROUP = "swiss-ai/apertus-70b-instruct"


def plot_apertus70b_performance(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/apertus70b_performance",
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
        "endTime",
        "completionStartTime",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "model_group",
        "status",
    }
    miss = require(df, needed, strict=False)
    if "model_group" in miss:
        raise SystemExit("Missing required column 'model_group'.")

    # Clean: adds latency_s, ttft_s, gen_s, date, etc.
    df = clean_table(df)

    # Success helper
    df["__success__"] = (
        df["status"].astype("string").str.lower().eq("success").fillna(False)
        if "status" in df.columns
        else pd.Series(False, index=df.index, dtype="boolean")
    )
    df["__count__"] = 1

    # Restrict to Apertus 70B instruct
    df = df[df["model_group"] == TARGET_MODEL_GROUP].copy()
    if df.empty:
        raise SystemExit(f"No rows for model_group={TARGET_MODEL_GROUP!r}.")

    # ------------------------------------------------------------------
    # Time features and daily aggregates
    # ------------------------------------------------------------------
    ts = to_utc(df["startTime"])
    df["date"] = ts.dt.floor("D")  # type: ignore[attr-defined]
    df = df.dropna(subset=["date"])

    df = ensure_date_column(df, time_col="startTime", out_col="date", floor="D", dropna=True, sort=True)

    daily = aggregate_by_time(
        df,
        time_col="date",
        freq="D",
        sums=("total_tokens", "prompt_tokens", "completion_tokens", "__success__", "__count__"),
        means=("latency_s", "ttft_s", "gen_s"),
        include_count=False,
        custom={
            "requests": ("__count__", "sum"),
            "success": ("__success__", "sum"),
            "total_tokens": ("total_tokens", "sum"),
            "prompt_tokens": ("prompt_tokens", "sum"),
            "completion_tokens": ("completion_tokens", "sum"),
            "avg_latency_s": ("latency_s", "mean"),
            "avg_ttft_s": ("ttft_s", "mean"),
            "avg_gen_s": ("gen_s", "mean"),
        },
    )
    daily = add_rates(
        daily,
        ratios=(
            ("success", "requests", "success_rate"),
            ("total_tokens", "requests", "avg_tokens_per_req"),
            ("avg_ttft_s", "avg_latency_s", "ttft_share"),
            ("avg_gen_s", "avg_latency_s", "gen_share"),
        ),
    )
    daily = daily.sort_values("date", kind="stable")

    # ------------------------------------------------------------------
    # Daily performance lines (means)
    # ------------------------------------------------------------------
    line(
        "date",
        "avg_latency_s",
        daily,
        "Avg latency per day (Apertus 70B)",
        "date",
        "latency (s)",
        "daily_avg_latency.png",
        out,
    )
    line(
        "date",
        "avg_ttft_s",
        daily,
        "Avg time-to-first-token per day (Apertus 70B)",
        "date",
        "ttft (s)",
        "daily_avg_ttft.png",
        out,
    )
    line(
        "date",
        "avg_gen_s",
        daily,
        "Avg generation time per day (Apertus 70B)",
        "date",
        "gen time (s)",
        "daily_avg_gen.png",
        out,
    )
    line(
        "date",
        "avg_tokens_per_req",
        daily,
        "Avg tokens per request per day (Apertus 70B)",
        "date",
        "avg tokens/req",
        "daily_avg_tokens_per_req.png",
        out,
    )
    line(
        "date",
        "success_rate",
        daily,
        "Success rate per day (Apertus 70B)",
        "date",
        "success rate",
        "daily_success_rate.png",
        out,
    )
    line(
        "date",
        "ttft_share",
        daily,
        "TTFT share of total latency (daily, Apertus 70B)",
        "date",
        "share",
        "daily_ttft_share.png",
        out,
    )
    line(
        "date",
        "gen_share",
        daily,
        "Generation share of total latency (daily, Apertus 70B)",
        "date",
        "share",
        "daily_gen_share.png",
        out,
    )

    # ------------------------------------------------------------------
    # Daily quantile curves (Q1/median/Q3/mean)
    # ------------------------------------------------------------------
    # Tokens / request: interpret total_tokens per row
    lines_quantiles(
        "date",
        "total_tokens",
        df,
        "Tokens per request per day: Q1/Median/Q3/Mean (Apertus 70B)",
        "date",
        "tokens/request",
        "daily_tokens_quantiles.png",
        out,
    )
    # Latency
    lines_quantiles(
        "date",
        "latency_s",
        df,
        "Latency per day: Q1/Median/Q3/Mean (Apertus 70B)",
        "date",
        "latency (s)",
        "daily_latency_quantiles.png",
        out,
    )
    lines_quantiles(
        "date",
        "ttft_s",
        df,
        "TTFT per day: Q1/Median/Q3/Mean (Apertus 70B)",
        "date",
        "ttft (s)",
        "daily_ttft_quantiles.png",
        out,
    )
    lines_quantiles(
        "date",
        "gen_s",
        df,
        "Generation time per day: Q1/Median/Q3/Mean (Apertus 70B)",
        "date",
        "gen time (s)",
        "daily_gen_quantiles.png",
        out,
    )

    save_csv(daily, "daily_performance_stats.csv", out)

    # ------------------------------------------------------------------
    # Row-level latency and TTFT/gen shares
    # ------------------------------------------------------------------
    df["ttft_share_row"] = safe_div(df["ttft_s"], df["latency_s"]).clip(lower=0, upper=1)
    df["gen_share_row"] = safe_div(df["gen_s"], df["latency_s"]).clip(lower=0, upper=1)

    # Histograms of latency metrics (trimmed to 99.9% to reduce extreme tails)
    for col, title, fname in [
        ("latency_s", "Latency per request (s) — trimmed 99.9%", "hist_latency_s_trimmed.png"),
        ("ttft_s", "Time to first token (s) — trimmed 99.9%", "hist_ttft_s_trimmed.png"),
        ("gen_s", "Generation time (s) — trimmed 99.9%", "hist_gen_s_trimmed.png"),
    ]:
        series = safe_quantile_cut(df[col], 0.999)
        if not series.empty:
            hist(series, title, col, "count", fname, out, bins=50)

    # Histograms of latency shares
    for col, title, fname in [
        ("ttft_share_row", "TTFT share of latency", "hist_ttft_share_row.png"),
        ("gen_share_row", "Generation share of latency", "hist_gen_share_row.png"),
    ]:
        s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not s.empty:
            hist(s.clip(lower=0, upper=1), title, col, "count", fname, out, bins=50)

    # Daily quantiles for latency shares
    lines_quantiles(
        "date",
        "ttft_share_row",
        df,
        "TTFT share per day: Q1/Median/Q3/Mean (Apertus 70B)",
        "date",
        "share",
        "daily_ttft_share_quantiles.png",
        out,
    )
    lines_quantiles(
        "date",
        "gen_share_row",
        df,
        "Generation share per day: Q1/Median/Q3/Mean (Apertus 70B)",
        "date",
        "share",
        "daily_gen_share_quantiles.png",
        out,
    )

    # ------------------------------------------------------------------
    # Token-efficiency metrics (prompt/completion ratios)
    # ------------------------------------------------------------------
    if "prompt_ratio" not in df.columns and {"prompt_tokens", "total_tokens"} <= set(df.columns):
        df["prompt_ratio"] = safe_div(df["prompt_tokens"], df["total_tokens"]).clip(lower=0, upper=1)
    if "completion_ratio" not in df.columns and {"completion_tokens", "total_tokens"} <= set(df.columns):
        df["completion_ratio"] = safe_div(df["completion_tokens"], df["total_tokens"]).clip(lower=0, upper=1)

    for col, title, fname in [
        ("prompt_ratio", "Prompt ratio = prompt/total", "hist_prompt_ratio.png"),
        ("completion_ratio", "Completion ratio = completion/total", "hist_completion_ratio.png"),
    ]:
        if col in df.columns:
            s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if not s.empty:
                hist(s.clip(lower=0, upper=1), title, col, "count", fname, out, bins=50)

    # Daily quantiles for ratios
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
    # Latency vs tokens (binned averages and quantiles)
    # ------------------------------------------------------------------
    def _bin_and_avg(x: pd.Series, y: pd.Series, bins: int, xlabel: str, fname: str) -> None:
        d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if d.empty:
            return
        d = d[d["x"] >= 0]
        q = min(bins, max(2, d["x"].nunique()))
        d["bin"] = pd.qcut(d["x"], q=q, duplicates="drop")
        g = (
            d.groupby("bin", observed=True, as_index=False)
            .agg(avg_x=("x", "mean"), avg_y=("y", "mean"))
            .sort_values("avg_x", kind="stable")
        )
        line(
            "avg_x",
            "avg_y",
            g,
            f"Avg latency vs {xlabel} (binned, Apertus 70B)",
            xlabel,
            "avg latency (s)",
            fname,
            out,
        )

    _bin_and_avg(
        df["total_tokens"],
        df["latency_s"],
        bins=10,
        xlabel="total tokens",
        fname="latency_vs_tokens_binned.png",
    )

    bin_and_quantiles(
        df["total_tokens"],
        df["latency_s"],
        bins=10,
        xlabel="Latency vs total tokens (binned, Apertus 70B)",
        ylabel="latency (s)",
        fname="latency_vs_tokens_binned_quantiles.png",
        out=out,
    )

    # Simple scatter to see dispersion
    tokens_trim = safe_quantile_cut(df["total_tokens"], 0.99)
    latency_trim = safe_quantile_cut(df["latency_s"], 0.99)
    idx = tokens_trim.index.intersection(latency_trim.index)
    if len(idx) > 0:
        scatter(
            df.loc[idx, "total_tokens"],
            df.loc[idx, "latency_s"],
            "Latency vs total tokens (trimmed 99%, Apertus 70B)",
            "total tokens",
            "latency (s)",
            "scatter_latency_vs_tokens_trimmed.png",
            out,
        )

    # ------------------------------------------------------------------
    # Success vs failure latency comparison
    # ------------------------------------------------------------------
    failures = df[~df["__success__"]]
    successes = df[df["__success__"]]

    def _hist_if_any(series: pd.Series, title: str, fname: str) -> None:
        s = safe_quantile_cut(series, 0.999)
        if not s.empty:
            hist(s, title, "latency_s", "count", fname, out, bins=50)

    if not successes.empty:
        _hist_if_any(
            successes["latency_s"],
            "Latency for successful requests (trimmed 99.9%, Apertus 70B)",
            "hist_latency_success.png",
        )
    if not failures.empty:
        _hist_if_any(
            failures["latency_s"],
            "Latency for failed requests (trimmed 99.9%, Apertus 70B)",
            "hist_latency_failure.png",
        )

    # ------------------------------------------------------------------
    # Throughput vs latency
    # ------------------------------------------------------------------
    per_min = (
        df.set_index("startTime")
        .resample("min")["request_id"]
        .count()
        .reset_index()
        .rename(columns={"request_id": "rpm"})
    )
    if not per_min.empty:
        line(
            "startTime",
            "rpm",
            per_min,
            "Requests per minute (Apertus 70B)",
            "time",
            "req/min",
            "throughput_rpm.png",
            out,
        )

        # Merge approximate latency per minute
        lat_per_min = (
            df.set_index("startTime")
            .resample("min")["latency_s"]
            .mean()
            .reset_index()
            .rename(columns={"latency_s": "avg_latency_s"})
        )
        merged = per_min.merge(lat_per_min, on="startTime", how="left")
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["rpm", "avg_latency_s"])
        if not merged.empty:
            _bin_and_avg(
                merged["rpm"],
                merged["avg_latency_s"],
                bins=10,
                xlabel="requests per minute",
                fname="latency_vs_rpm_binned.png",
            )
            scatter(
                merged["rpm"],
                merged["avg_latency_s"],
                "Latency vs requests per minute (Apertus 70B)",
                "requests per minute",
                "avg latency (s)",
                "scatter_latency_vs_rpm.png",
                out,
            )

    # ------------------------------------------------------------------
    # Overall summary (no spend)
    # ------------------------------------------------------------------
    summary = {
        "n_rows": len(df),
        "n_requests": int(df["request_id"].nunique()),
        "total_tokens": float(df["total_tokens"].sum()),
        "mean_tokens_per_request": float(df["total_tokens"].mean()),
        "mean_latency_s": float(df["latency_s"].mean()),
        "p95_latency_s": float(df["latency_s"].quantile(0.95)),
        "mean_success_rate": float(daily["success_rate"].mean()),
    }
    save_csv(pd.DataFrame([summary]), "summary_performance_overall.csv", out)


if __name__ == "__main__":
    plot_apertus70b_performance()
