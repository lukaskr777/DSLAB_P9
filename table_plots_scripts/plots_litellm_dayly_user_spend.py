"""
Produce a set of exploratory plots for the LiteLLM_DailyUserSpend table.
Writes figures to figs/litellm_daily_user_spend.
"""

import pandas as pd

from utility_scripts.file_utils import PathLike, read_table, ensure_empty_dir, sanitize_fname
from utility_scripts.plot_utils import line, bar, hist
from utility_scripts.df_reading_utils import require, to_utc, aggregate_by_time, add_rates, top_by_value


def plot_all_litellm_daily_user_spend(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/litellm_daily_user_spend",
    top: int = 10,
) -> None:
    out = ensure_empty_dir(outdir)

    df = read_table("LiteLLM_DailyUserSpend", dataset=dataset, dir_name=dir_name)
    if df.empty:
        raise SystemExit("No data loaded from LiteLLM_DailyUserSpend.")

    needed = {
        "date",
        "spend",
        "api_requests",
        "successful_requests",
        "failed_requests",
        "prompt_tokens",
        "completion_tokens",
        "cache_read_input_tokens",
        "user_id",
        "model",
        "model_group",
        "api_key",
    }
    miss = require(df, needed, strict=False)
    if miss:
        raise SystemExit(f"Missing required columns: {sorted(miss)}")

    # Normalize keys and coerce numerics
    df = df.copy()
    df["user_id"] = df["user_id"].fillna("").replace("", "unknown_user")
    df["model_group"] = df["model_group"].fillna("").replace("", "unknown_group")
    df["api_key"] = df["api_key"].fillna("").replace("", "unknown_key")

    df["date"] = to_utc(df["date"]).dt.floor("D")
    df = df.dropna(subset=["date"]).sort_values("date", kind="stable")

    for c in (
        "spend",
        "api_requests",
        "successful_requests",
        "failed_requests",
        "prompt_tokens",
        "completion_tokens",
        "cache_read_input_tokens",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Daily aggregates with explicit names
    daily = aggregate_by_time(
        df,
        time_col="date",
        freq="D",
        custom={
            "spend": ("spend", "sum"),
            "api_requests": ("api_requests", "sum"),
            "successful_requests": ("successful_requests", "sum"),
            "failed_requests": ("failed_requests", "sum"),
            "prompt_tokens": ("prompt_tokens", "sum"),
            "completion_tokens": ("completion_tokens", "sum"),
            "cache_read_input_tokens": ("cache_read_input_tokens", "sum"),
        },
        sort=True,
    )

    # Rates and averages
    daily = add_rates(
        daily,
        ratios=(
            ("successful_requests", "api_requests", "success_rate"),
        ),
        sums_to_avg_per_count=(
            (("prompt_tokens", "completion_tokens"), "api_requests", "avg_tokens_per_req"),
        ),
    )

    # ---- Time series ----
    if not daily.empty:
        if "spend" in daily:
            line(
                "date", "spend", daily, "Total spend per day", 
                "date", "spend", "daily_total_spend.png", out
            )
        if "api_requests" in daily:
            line(
                "date", "api_requests", daily, "API requests per day", 
                "date", "requests", "daily_api_requests.png", out
            )
        if "success_rate" in daily:
            line(
                "date", "success_rate", daily, "Success rate per day", 
                "date", "success rate", "daily_success_rate.png", out
            )
        if "avg_tokens_per_req" in daily:
            line(
                "date", "avg_tokens_per_req", daily, "Average tokens per request per day", 
                "date", "avg tokens / req", "daily_avg_tokens_per_request.png", out
            )
        if "cache_read_input_tokens" in daily:
            line(
                "date", "cache_read_input_tokens", daily, "Cache read input tokens per day", 
                "date", "cache read input tokens", "daily_cache_read_input_tokens.png", out
            )

    # ---- Bars: Top-N by spend ----
    for key, fname, title in [
        ("user_id", f"top{top}_users_spend.png", f"Top {top} users by total spend"),
        ("model", f"top{top}_models_spend.png", f"Top {top} models by total spend"),
        ("model_group", f"top{top}_model_groups_spend.png", f"Top {top} model groups by total spend"),
        ("api_key", f"top{top}_apikeys_spend.png", f"Top {top} API keys by total spend"),
    ]:
        s = top_by_value(df, keys=key, value_col="spend", top=top)
        if not s.empty:
            bar(s, title=title, xlabel=key, ylabel="total spend", fname=fname, outdir=out, top=len(s))

    # ---- Histograms ----
    s_row_spend = pd.to_numeric(df["spend"], errors="coerce").dropna()
    if not s_row_spend.empty:
        hist(
            s_row_spend,
            title="Row-level spend distribution",
            xlabel="spend (row)",
            ylabel="count",
            fname="hist_row_spend.png",
            outdir=out,
            bins=50,
        )

    if "spend" in daily:
        daily_spend = pd.to_numeric(daily["spend"], errors="coerce").dropna()
        if not daily_spend.empty:
            hist(
                daily_spend,
                title="Daily total spend distribution",
                xlabel="spend (per day)",
                ylabel="count of days",
                fname="hist_daily_spend.png",
                outdir=out,
                bins=min(50, max(10, int(daily_spend.nunique()))),
            )

    # ---- Per-user daily spend lines (Top-N users) ----
    user_spend = (
        df.groupby("user_id", dropna=False)["spend"]
        .sum()
        .sort_values(ascending=False)
        .loc[lambda s: s > 0]
    )
    top_users = user_spend.head(top).index.tolist()

    if top_users:
        per_user_daily = (
            df[df["user_id"].isin(top_users)]
            .groupby(["date", "user_id"], as_index=False)
            .agg(spend=("spend", "sum"))
            .sort_values(["user_id", "date"], kind="stable")
        )
        for user in top_users:
            sub = per_user_daily[per_user_daily["user_id"] == user]
            safe = sanitize_fname(f"daily_spend_user_{user}.png")
            line("date", "spend", sub, f"Daily spend for user {user}", "date", "spend", safe, out)
