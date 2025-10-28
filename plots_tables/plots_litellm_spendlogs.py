"""
Exploratory plots for LiteLLM_SpendLogs.
Writes figures to figs/litellm_spendlogs.

Outputs:
- Daily lines: total spend, request count, success rate, cache-hit rate,
  avg tokens/request, avg latency, avg time-to-first-token, avg generation time
- Bars (Top-N by spend): end_user, model, model_group, custom_llm_provider, api_key, call_type
- Histograms: row spend, total/prompt/completion tokens, latency, ttfb, gen time
- Whale curve: cumulative spend share vs user rank (end_user, spend > 0)
- Per-top-user daily spend lines
"""

import numpy as np
import pandas as pd
from utils import PathLike, read_table, ensure_empty_dir, line, bar, hist


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["spend"] = pd.to_numeric(out["spend"], errors="coerce").fillna(0.0)
    out.loc[out["spend"] < 0, "spend"] = 0.0

    for col, fallback in [
        ("end_user", "unknown_end_user"),
        ("model", "unknown_model"),
        ("model_group", "unknown_group"),
        ("custom_llm_provider", "unknown_provider"),
        ("api_key", "unknown_key"),
        ("call_type", "unknown_call_type"),
        ("status", "unknown_status"),
    ]:
        out[col] = out[col].fillna("").replace("", fallback)

    out["cache_hit"] = out["cache_hit"].astype(str).str.lower().map({"true": True, "false": False, "none": False}).fillna(False)

    out["startTime"] = _to_dt(out["startTime"])
    out["completionStartTime"] = _to_dt(out["completionStartTime"])
    out["endTime"] = _to_dt(out["endTime"])

    out["latency_s"] = (out["endTime"] - out["startTime"]).dt.total_seconds()
    out["ttfb_s"] = (out["completionStartTime"] - out["startTime"]).dt.total_seconds()
    out["gen_s"] = (out["endTime"] - out["completionStartTime"]).dt.total_seconds()

    for col in ["total_tokens", "prompt_tokens", "completion_tokens"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["date"] = out["startTime"].dt.date
    out = out.dropna(subset=["date"])

    return out


def _daily(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date", as_index=False).agg(
        spend=("spend", "sum"),
        requests=("request_id", "count"),
        success=("status", lambda s: (s == "success").sum()),
        cache_hits=("cache_hit", "sum"),
        total_tokens=("total_tokens", "sum"),
        prompt_tokens=("prompt_tokens", "sum"),
        completion_tokens=("completion_tokens", "sum"),
        avg_latency_s=("latency_s", "mean"),
        avg_ttfb_s=("ttfb_s", "mean"),
        avg_gen_s=("gen_s", "mean"),
    )
    g["success_rate"] = g["success"] / g["requests"].where(g["requests"] > 0, np.nan)
    g["cache_hit_rate"] = g["cache_hits"] / g["requests"].where(g["requests"] > 0, np.nan)
    g["avg_tokens_per_req"] = g["total_tokens"] / g["requests"].where(g["requests"] > 0, np.nan)
    return g.sort_values("date")


def _top_by_spend(df: pd.DataFrame, key: str, top: int) -> pd.Series:
    s = df.groupby(key)["spend"].sum().sort_values(ascending=False)
    return s[s > 0].head(top)


def _cumulative_share(series_by_user: pd.Series) -> pd.DataFrame:
    s = series_by_user[series_by_user > 0].sort_values(ascending=False)
    if s.empty:
        return pd.DataFrame({"rank": [], "cum_share": []})
    cum = s.cumsum() / s.sum()
    rank = np.arange(1, len(s) + 1) / len(s)
    return pd.DataFrame({"rank": rank, "cum_share": cum.values})


def plot_all_litellm_spendlogs(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/litellm_spendlogs",
    top: int = 10,
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
        "cache_hit",
        "status",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = _clean(df)
    daily = _daily(df)

    line("date", "spend", daily, "Total spend per day", "date", "spend", "daily_total_spend.png", out)
    line("date", "requests", daily, "Requests per day", "date", "requests", "daily_requests.png", out)
    line("date", "success_rate", daily, "Success rate per day", "date", "success rate", "daily_success_rate.png", out)
    line("date", "cache_hit_rate", daily, "Cache-hit rate per day", "date", "cache-hit rate", "daily_cache_hit_rate.png", out)
    line("date", "avg_tokens_per_req", daily, "Avg tokens per request per day", "date", "avg tokens/req", "daily_avg_tokens_per_req.png", out)
    line("date", "avg_latency_s", daily, "Avg latency per day", "date", "latency (s)", "daily_avg_latency.png", out)
    line("date", "avg_ttfb_s", daily, "Avg time-to-first-token per day", "date", "ttfb (s)", "daily_avg_ttfb.png", out)
    line("date", "avg_gen_s", daily, "Avg generation time per day", "date", "gen time (s)", "daily_avg_gen.png", out)

    for key, fname, title in [
        ("end_user", "top_users_spend.png", "Top end_users by total spend"),
        ("model", "top_models_spend.png", "Top models by total spend"),
        ("model_group", "top_model_groups_spend.png", "Top model_groups by total spend"),
        ("custom_llm_provider", "top_providers_spend.png", "Top providers by total spend"),
        ("api_key", "top_apikeys_spend.png", "Top API keys by total spend"),
        ("call_type", "top_call_types_spend.png", "Top call types by total spend"),
    ]:
        s = _top_by_spend(df, key, top)
        if not s.empty:
            bar(s, title=f"{title} (Top {len(s)})", xlabel=key, ylabel="total spend", fname=fname, outdir=out, top=len(s))

    hist(df["spend"], "Row-level spend distribution", "spend (row)", "count", "hist_row_spend.png", out, bins=50)

    nz_spend = df.loc[df["spend"] > 0, "spend"]
    if not nz_spend.empty:
        hist(
            pd.Series(np.log10(nz_spend)), 
            "Log10 row spend (spend > 0)", 
            "log10(spend)", 
            "count", 
            "hist_row_spend_log10.png", 
            out,
            bins=min(50, max(10, nz_spend.nunique()))
        )

    for col, title, fname in [
        ("total_tokens", "Total tokens per request", "hist_total_tokens.png"),
        ("prompt_tokens", "Prompt tokens per request", "hist_prompt_tokens.png"),
        ("completion_tokens", "Completion tokens per request", "hist_completion_tokens.png"),
    ]:
        series = df[col].dropna()
        if not series.empty:
            hist(series, title, col, "count", fname, out, bins=50)

    for col, title, fname in [
        ("latency_s", "Latency per request (s)", "hist_latency_s.png"),
        ("ttfb_s", "Time to first token (s)", "hist_ttfb_s.png"),
        ("gen_s", "Generation time (s)", "hist_gen_s.png"),
    ]:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        series = series[(series >= 0) & (series <= series.quantile(0.999, interpolation="linear"))]
        if not series.empty:
            hist(series, title, col, "count", fname, out, bins=50)

    spend_by_user = df.groupby("end_user")["spend"].sum().sort_values(ascending=False)
    cum = _cumulative_share(spend_by_user)
    if not cum.empty:
        line(
            "rank", 
            "cum_share", 
            cum, 
            "Cumulative spend share vs user rank", 
            "user rank fraction", 
            "cumulative share",
            "cumulative_spend_share.png", 
            out
        )

    top_users = spend_by_user[spend_by_user > 0].head(top).index.tolist()
    if top_users:
        per_user_daily = (
            df[df["end_user"].isin(top_users)]
            .groupby(["date", "end_user"], as_index=False)
            .agg(spend=("spend", "sum"))
            .sort_values(["end_user", "date"])
        )
        for u in top_users:
            sub = per_user_daily[per_user_daily["end_user"] == u]
            line("date", "spend", sub, f"Daily spend for {u}", "date", "spend", f"daily_spend_{u}.png", out)
