"""
Performance & efficiency analysis for swiss-ai/apertus-70b-instruct
based on LiteLLM_SpendLogs.

Produces plots in figs/apertus70b_performance:

- Daily performance metrics (latency, TTFT, gen time, success rate)
- Token efficiency (tokens/request, prompt vs completion ratios)
- Latency and TTFT/gen distributions (trimmed)
- Cost efficiency: row-level and grouped cost per 1k tokens
- Latency vs tokens / spend (binned curves + quantiles)
- Success vs failure latency comparison
- Throughput vs latency (requests per minute/hour)

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
    bar,
    hist,
    scatter,
    scatter_with_fit,
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

from apertus70b_user_clusters import remove_extreme_apertus70b_users

TARGET_MODEL_GROUP = "swiss-ai/apertus-70b-instruct"


def plot_apertus70b_performance(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/apertus70b_performance",
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
        "endTime",
        "completionStartTime",
        "spend",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "model_group",
        "custom_llm_provider",
        "status",
    }
    miss = require(df, needed, strict=False)
    if "model_group" in miss:
        raise SystemExit("Missing required column 'model_group'.")

    df = clean_table(df)  # adds latency_s, ttft_s, gen_s, completion_ratio, date, etc.

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
    
    # Remove extreme power users
    df, removed_users = remove_extreme_apertus70b_users(df)
    if not removed_users.empty:
        save_csv(removed_users, "removed_extreme_users.csv", out)

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
        sums=("spend", "total_tokens", "prompt_tokens", "completion_tokens", "__success__", "__count__"),
        means=("latency_s", "ttft_s", "gen_s"),
        include_count=False,
        custom={
            "spend": ("spend", "sum"),
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
    # Daily performance lines
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

    # ------------------------------------------------------------------
    # Token and cost efficiency
    # ------------------------------------------------------------------
    # Token composition ratios
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

    # Row-level cost per 1k tokens
    df_tokens_pos = df[df["total_tokens"] > 0].copy()
    df_tokens_pos["row_cost_per_1k"] = 1000.0 * df_tokens_pos["spend"] / df_tokens_pos["total_tokens"]
    if not df_tokens_pos["row_cost_per_1k"].empty:
        hist(
            df_tokens_pos["row_cost_per_1k"],
            "Row-level cost per 1k tokens (Apertus 70B)",
            "cost per 1k tokens",
            "count",
            "hist_row_cost_per_1k.png",
            out,
            bins=50,
        )

    # Group-level cost per 1k tokens, by provider and by end_user (top by spend)
    def _cost_per_1k(group_key: str, fname: str, title: str) -> None:
        g = df_tokens_pos.groupby(group_key, dropna=False).agg(
            spend=("spend", "sum"),
            tokens=("total_tokens", "sum"),
        )
        g = g[g["tokens"] > 0]
        if g.empty:
            return
        g["cost_per_1k"] = 1000.0 * g["spend"] / g["tokens"]
        g = g.reset_index()
        s = g.sort_values("spend", ascending=False).head(top).set_index(group_key)["cost_per_1k"]
        if not s.empty:
            bar(
                s,
                f"{title} (Top {len(s)})",
                group_key,
                "cost per 1k tokens",
                fname,
                out,
                top=len(s),
            )

    if "custom_llm_provider" in df.columns:
        _cost_per_1k("custom_llm_provider", "cost_per_1k_by_provider.png", "Cost per 1k tokens by provider")
    _cost_per_1k("end_user", "cost_per_1k_by_user.png", "Cost per 1k tokens by end_user")

    # ------------------------------------------------------------------
    # Latency vs tokens / spend (binned curves)
    # ------------------------------------------------------------------
    def _bin_and_avg(x: pd.Series, y: pd.Series, bins: int, xlabel: str, fname: str) -> None:
        d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if d.empty:
            return

        d = d[d["x"] >= 0]
        if d.empty:
            return

        # Need at least 2 distinct x values to bin
        nunique_x = d["x"].nunique()
        if nunique_x < 2:
            return

        q = min(bins, max(2, nunique_x))

        # qcut can still drop everything (all x equal, etc.)
        try:
            d["bin"] = pd.qcut(d["x"], q=q, duplicates="drop")
        except Exception:
            return

        d = d.dropna(subset=["bin"])
        if d.empty:
            return

        g = (
            d.groupby("bin", observed=True, as_index=False)
            .agg(avg_x=("x", "mean"), avg_y=("y", "mean"))
            .sort_values("avg_x", kind="stable")
        )
        if g.empty:
            return

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
        df["total_tokens"], df["latency_s"], bins=10,
        xlabel="total tokens", fname="latency_vs_tokens_binned.png"
    )
    _bin_and_avg(
        df["spend"], df["latency_s"], bins=10,
        xlabel="row spend", fname="latency_vs_spend_binned.png"
    )

    # Quantile curves for the same relationships
    bin_and_quantiles(
        df["total_tokens"],
        df["latency_s"],
        bins=10,
        xlabel="Latency vs total tokens (binned, Apertus 70B)",
        ylabel="latency (s)",
        fname="latency_vs_tokens_binned_quantiles.png",
        out=out,
    )
    bin_and_quantiles(
        df["spend"],
        df["latency_s"],
        bins=10,
        xlabel="Latency vs row spend (binned, Apertus 70B)",
        ylabel="latency (s)",
        fname="latency_vs_spend_binned_quantiles.png",
        out=out,
    )

    # Scatter and elasticity (daily spend vs tokens)
    agg_daily = df.groupby("date", as_index=False).agg(spend=("spend", "sum"), tokens=("total_tokens", "sum"))
    agg_daily = agg_daily.replace([np.inf, -np.inf], np.nan).dropna()
    if not agg_daily.empty and agg_daily["tokens"].gt(0).any():
        scatter_with_fit(
            x=agg_daily["tokens"].to_numpy(dtype=float),
            y=agg_daily["spend"].to_numpy(dtype=float),
            title="Spend vs tokens (daily, Apertus 70B) — least-squares slope",
            xlabel="tokens",
            ylabel="spend",
            fname="elasticity_spend_vs_tokens.png",
            outdir=out,
            write_params_path="elasticity_spend_vs_tokens.txt",
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
        _hist_if_any(successes["latency_s"], "Latency for successful requests (trimmed 99.9%)", "hist_latency_success.png")
    if not failures.empty:
        _hist_if_any(failures["latency_s"], "Latency for failed requests (trimmed 99.9%)", "hist_latency_failure.png")

    # Failure rate by provider (if available)
    if "custom_llm_provider" in df.columns:
        fail_rate_prov = df.groupby("custom_llm_provider", as_index=False).agg(
            requests=("request_id", "count"),
            failures=("__success__", lambda s: (~s).sum()),
            spend=("spend", "sum"),
        )
        fail_rate_prov["fail_rate"] = safe_div(fail_rate_prov["failures"], fail_rate_prov["requests"])
        s_fail = (
            fail_rate_prov.sort_values("spend", ascending=False)
            .head(top)
            .set_index("custom_llm_provider")["fail_rate"]
        )
        if not s_fail.empty:
            bar(
                s_fail,
                f"Failure rate by provider (Top {len(s_fail)} by spend, Apertus 70B)",
                "provider",
                "failure rate",
                "fail_rate_by_provider.png",
                out,
                top=len(s_fail),
            )

    # ------------------------------------------------------------------
    # Throughput vs latency
    # ------------------------------------------------------------------
    # Requests per minute / hour
    per_min = (
        df.set_index("startTime")
        .resample("min")["request_id"]
        .count()
        .reset_index()
        .rename(columns={"request_id": "rpm"})
    )
    per_hr = (
        df.set_index("startTime")
        .resample("h")["request_id"]
        .count()
        .reset_index()
        .rename(columns={"request_id": "rph"})
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
    if not per_hr.empty:
        line(
            "startTime",
            "rph",
            per_hr,
            "Requests per hour (Apertus 70B)",
            "time",
            "req/hour",
            "throughput_rph.png",
            out,
        )

    # Average latency vs per-minute throughput (binned)
    if not per_min.empty:
        # Merge back approximate latency per minute
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

    # Save a small summary of key aggregates
    summary = {
        "n_rows": len(df),
        "n_requests": int(df["request_id"].nunique()),
        "total_spend": float(df["spend"].sum()),
        "total_tokens": float(df["total_tokens"].sum()),
        "mean_latency_s": float(df["latency_s"].mean()),
        "p95_latency_s": float(df["latency_s"].quantile(0.95)),
        "mean_success_rate": float(daily["success_rate"].mean()),
    }
    save_csv(pd.DataFrame([summary]), "summary_performance_overall.csv", out)


if __name__ == "__main__":
    plot_apertus70b_performance()
