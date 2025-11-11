"""
Produce a set of exploratory plots for the LiteLLM_DailyTeamSpend table.
Reads data via utils.read_table and writes figures to figs/litellm_daily_team_spend.

Outputs:
- Time series: total spend per day
- Time series: total API requests per day
- Time series: success rate per day
- Time series: average tokens per request per day
- Bars (Top-N): teams by total spend, models by total spend, model_groups by total spend, api_keys by total spend
- Histograms: row-level spend distribution, daily total spend distribution
- Per-team daily spend lines for Top-N teams
"""

import pandas as pd
from scripts.utils import PathLike, read_table, ensure_empty_dir
from scripts.plotting_functions import line, bar, hist


def _to_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Ensure a YYYY-MM-DD date column is datetime64[ns] and sorted."""
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.date
    out = out.dropna(subset=[col]).sort_values(col)
    return out


def _daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per day totals and simple rates."""
    g = df.groupby("date", as_index=False).agg(
        spend=("spend", "sum"),
        api_requests=("api_requests", "sum"),
        successful_requests=("successful_requests", "sum"),
        failed_requests=("failed_requests", "sum"),
        prompt_tokens=("prompt_tokens", "sum"),
        completion_tokens=("completion_tokens", "sum"),
        cache_read_input_tokens=("cache_read_input_tokens", "sum"),
    )
    g["success_rate"] = g["successful_requests"] / g["api_requests"].where(g["api_requests"] > 0, pd.NA)
    total_tokens = g["prompt_tokens"] + g["completion_tokens"]
    g["avg_tokens_per_req"] = total_tokens / g["api_requests"].where(g["api_requests"] > 0, pd.NA)
    return g


def _top_by_spend(df: pd.DataFrame, key: str, top: int) -> pd.Series:
    """Return a Series of total spend by key, sorted desc, limited to top."""
    s = df.groupby(key)["spend"].sum().sort_values(ascending=False)
    return s.head(top)


def plot_all_litellm_daily_team_spend(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/litellm_daily_team_spend",
    top: int = 10
) -> None:
    """Create all plots for LiteLLM_DailyTeamSpend."""
    out = ensure_empty_dir(outdir)
    
    df = read_table("LiteLLM_DailyTeamSpend", dataset=dataset, dir_name=dir_name)
    if df.empty:
        raise SystemExit("No data loaded from LiteLLM_DailyTeamSpend.")

    needed = {
        "date",
        "spend",
        "api_requests",
        "successful_requests",
        "failed_requests",
        "prompt_tokens",
        "completion_tokens",
        "cache_read_input_tokens",
        "team_id",
        "model",
        "model_group",
        "api_key",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df["team_id"] = df["team_id"].fillna("").replace("", "unknown_team")
    df["model_group"] = df["model_group"].fillna("").replace("", "unknown_group")
    df["api_key"] = df["api_key"].fillna("").replace("", "unknown_key")

    df = _to_date(df, "date")
    daily = _daily_agg(df)

    line(
        x="date",
        y="spend",
        data=daily,
        title="Total spend per day",
        xlabel="date",
        ylabel="spend",
        fname="daily_total_spend.png",
        outdir=out
    )

    line(
        x="date",
        y="api_requests",
        data=daily,
        title="API requests per day",
        xlabel="date",
        ylabel="requests",
        fname="daily_api_requests.png",
        outdir=out
    )

    line(
        x="date",
        y="success_rate",
        data=daily,
        title="Success rate per day",
        xlabel="date",
        ylabel="success rate",
        fname="daily_success_rate.png",
        outdir=out
    )

    line(
        x="date",
        y="avg_tokens_per_req",
        data=daily,
        title="Average tokens per request per day",
        xlabel="date",
        ylabel="avg tokens / req",
        fname="daily_avg_tokens_per_request.png",
        outdir=out
    )

    line(
        x="date",
        y="cache_read_input_tokens",
        data=daily,
        title="Cache read input tokens per day",
        xlabel="date",
        ylabel="cache read input tokens",
        fname="daily_cache_read_input_tokens.png",
        outdir=out
    )

    bar(
        _top_by_spend(df, "team_id", top),
        title=f"Top {top} teams by total spend",
        xlabel="team_id",
        ylabel="total spend",
        fname=f"top{top}_teams_spend.png",
        outdir=out,
        top=top
    )

    bar(
        _top_by_spend(df, "model", top),
        title=f"Top {top} models by total spend",
        xlabel="model",
        ylabel="total spend",
        fname=f"top{top}_models_spend.png",
        outdir=out,
        top=top
    )

    bar(
        _top_by_spend(df, "model_group", top),
        title=f"Top {top} model groups by total spend",
        xlabel="model_group",
        ylabel="total spend",
        fname=f"top{top}_model_groups_spend.png",
        outdir=out,
        top=top
    )

    bar(
        _top_by_spend(df, "api_key", top),
        title=f"Top {top} API keys by total spend",
        xlabel="api_key",
        ylabel="total spend",
        fname=f"top{top}_apikeys_spend.png",
        outdir=out,
        top=top
    )

    hist(
        df["spend"],
        title="Row-level spend distribution",
        xlabel="spend (row)",
        ylabel="count",
        fname="hist_row_spend.png",
        outdir=out,
        bins=50
    )

    daily_spend = daily["spend"]
    hist(
        daily_spend,
        title="Daily total spend distribution",
        xlabel="spend (per day)",
        ylabel="count of days",
        fname="hist_daily_spend.png",
        outdir=out,
        bins=min(50, max(10, daily_spend.nunique()))
    )

    team_spend = df.groupby("team_id")["spend"].sum()
    team_spend = team_spend[team_spend > 0].sort_values(ascending=False)
    top_teams = team_spend.head(top).index.tolist()

    if not top_teams:
        return

    df_top = df[df["team_id"].isin(top_teams)]
    grouped = df_top.groupby(["date", "team_id"], as_index=False).agg(spend=("spend", "sum"))
    per_team_daily = grouped.sort_values(by=["team_id", "date"])

    for team in top_teams:
        sub = per_team_daily[per_team_daily["team_id"] == team]
        line(
            x="date",
            y="spend",
            data=sub,
            title=f"Daily spend for team {team}",
            xlabel="date",
            ylabel="spend",
            fname=f"daily_spend_team_{team}.png",
            outdir=out
        )
