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
- Hourly patterns: spend, requests, latency by hour; weekday patterns; hour×weekday heatmap
- Model/provider evolution: daily spend share for top-K model_groups and providers
- Spend anomalies: daily spend z-score and rolling deviations
- Latency breakdown shares: TTFB share and GEN share of total latency
- Throughput: requests per minute and per hour time series
- Cost efficiency: spend per 1k tokens by model_group and provider
- Token composition: prompt/completion ratios; scatter vs spend
- Correlations: matrix CSV and bars of |corr| with spend and latency
- User analytics: retention (active days), frequency, tokens; top model popularity by distinct users
- Cache effectiveness: hit vs miss spend and hit rate over time
- Provider mix over time: daily spend share
- Outliers: top 0.1% by spend/tokens/latency CSVs and histograms
- Request type mix: bars and daily lines of call_type share
- Failures: failure counts and latency distributions
- Team splits: spend share per model_group per team_id; end_user diversity per team
- Elasticity: slope of spend vs tokens; scatter with fitted line
- Pareto front: cost_per_1k vs avg_latency across model_groups with frontier highlighted
"""

import numpy as np
import pandas as pd

from scripts.utils import PathLike, to_dt, ensure_empty_dir, read_table, sanitize_fname, save_csv, save_text
from scripts.plotting_functions import line, bar, hist, heatmap, scatter, scatter_with_fit, pareto_frontier_plot


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

    out["cache_hit"] = (
        out["cache_hit"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "none": False})
        .fillna(False)
    )

    out["startTime"] = to_dt(out["startTime"])
    out["completionStartTime"] = to_dt(out["completionStartTime"])
    out["endTime"] = to_dt(out["endTime"])

    out["latency_s"] = (out["endTime"] - out["startTime"]).dt.total_seconds()
    out["ttfb_s"] = (out["completionStartTime"] - out["startTime"]).dt.total_seconds()
    out["gen_s"] = (out["endTime"] - out["completionStartTime"]).dt.total_seconds()

    for col in ["total_tokens", "prompt_tokens", "completion_tokens"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["date"] = out["startTime"].dt.date
    out = out.dropna(subset=["date"])

    # time parts
    out["dow"] = out["startTime"].dt.dayofweek  # 0=Mon
    out["hour"] = out["startTime"].dt.hour
    out["datehour"] = out["startTime"].dt.floor("h")
    out["week"] = out["startTime"].dt.isocalendar().week.astype(int)
    out["month"] = out["startTime"].dt.tz_localize(None).dt.to_period("M").astype(str)

    # numeric for corr
    out["cache_hit_num"] = out["cache_hit"].astype(int)
    out["success_num"] = (out["status"] == "success").astype(int)

    # token composition ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        out["prompt_ratio"] = out["prompt_tokens"] / out["total_tokens"]
        out["completion_ratio"] = out["completion_tokens"] / out["total_tokens"]
    for c in ["prompt_ratio", "completion_ratio"]:
        out.loc[~np.isfinite(out[c]), c] = np.nan

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
    # shares of latency components
    g["ttfb_share"] = g["avg_ttfb_s"] / g["avg_latency_s"]
    g["gen_share"] = g["avg_gen_s"] / g["avg_latency_s"]
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


def _safe_quantile_cut(s: pd.Series, q: float) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s
    return s[s <= s.quantile(q, interpolation="linear")]


# --- Main entry --------------------------------------------------------------

def plot_all_litellm_spendlogs(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/litellm_spendlogs",
    top: int = 10,
    top_share_k: int = 6,  # for evolution plots
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
    line("date", "ttfb_share", daily, "TTFB share of total latency (daily)", "date", "share", "daily_ttfb_share.png", out)
    line("date", "gen_share", daily, "Generation share of total latency (daily)", "date", "share", "daily_gen_share.png", out)

    # --- Bars: Top by spend --------------------------------------------------
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
            bar(
                s,
                title=f"{title} (Top {len(s)})",
                xlabel=key,
                ylabel="total spend",
                fname=fname,
                outdir=out,
                top=len(s),
            )

    # --- Histograms ----------------------------------------------
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
            bins=min(50, max(10, nz_spend.nunique())),
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
        series = _safe_quantile_cut(df[col], 0.999)
        if not series.empty:
            hist(series, title, col, "count", fname, out, bins=50)

    # --- Whale curve ---------------------------------------------------------
    spend_by_user = df.groupby("end_user")["spend"].sum().sort_values(ascending=False)
    cum = _cumulative_share(spend_by_user)
    if not cum.empty:
        line("rank", "cum_share", cum, "Cumulative spend share vs user rank", "user rank fraction", "cumulative share", "cumulative_spend_share.png", out)

    # Per-top-user daily lines
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
            safe = sanitize_fname(f"daily_spend_{u}.png")
            line("date", "spend", sub, f"Daily spend for {u}", "date", "spend", safe, out)

    # --- Temporal dynamics: hourly and weekday patterns ----------------------
    hourly = df.groupby("hour", as_index=False).agg(
        spend=("spend", "sum"),
        requests=("request_id", "count"),
        avg_latency_s=("latency_s", "mean"),
    ).sort_values("hour")
    order = list(range(0, 24))
    bar(
        hourly.set_index("hour")["spend"], 
        "Spend by hour of day", 
        "hour", 
        "spend", 
        "byhour_spend.png", 
        out, 
        top=24, 
        order=order, 
        sort_values=False
    )
    bar(
        hourly.set_index("hour")["requests"], 
        "Requests by hour of day", 
        "hour", 
        "requests", 
        "byhour_requests.png", 
        out, 
        top=24, 
        order=order, 
        sort_values=False
    )
    line("hour", "avg_latency_s", hourly, "Avg latency by hour of day", "hour", "latency (s)", "byhour_latency.png", out)

    weekday = df.groupby("dow", as_index=False).agg(
        spend=("spend", "sum"),
        requests=("request_id", "count"),
        avg_latency_s=("latency_s", "mean"),
    ).sort_values("dow")
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday["dow_label"] = weekday["dow"].map({i: l for i, l in enumerate(dow_labels)})
    order = dow_labels
    bar(
        weekday.set_index("dow_label")["spend"], 
        "Spend by weekday", 
        "weekday", 
        "spend", 
        "byweekday_spend.png", 
        out, 
        top=7,
        order=order,
        sort_values=False,
    )
    bar(
        weekday.set_index("dow_label")["requests"], 
        "Requests by weekday", 
        "weekday", 
        "requests", 
        "byweekday_requests.png", 
        out, 
        top=7,
        order=order,
        sort_values=False,
    )
    line("dow", "avg_latency_s", weekday, "Avg latency by weekday", "weekday (0=Mon)", "latency (s)", "byweekday_latency.png", out)

    # hour × weekday heatmap of spend
    pivot_hw = df.pivot_table(index="dow", columns="hour", values="spend", aggfunc="sum").fillna(0.0)
    pivot_hw = pivot_hw.reindex(index=range(0, 7), columns=range(0, 24), fill_value=0.0)
    pivot_hw = pivot_hw.set_axis(pd.Index(dow_labels, dtype=object, name=pivot_hw.index.name), axis="index")
    heatmap(pivot_hw, "Spend heatmap by weekday × hour", "hour", "weekday", "heatmap_weekday_hour_spend.png", out)

    # --- Model/provider evolution: daily spend share -------------------------
    def _daily_share(df_in: pd.DataFrame, key: str, fname_prefix: str, title_prefix: str) -> None:
        top_keys = df_in.groupby(key)["spend"].sum().sort_values(ascending=False).head(top_share_k).index
        sub = df_in[df_in[key].isin(top_keys)]
        daily_k = sub.groupby(["date", key], as_index=False).agg(spend=("spend", "sum"))
        totals = df_in.groupby("date", as_index=False).agg(total=("spend", "sum"))
        merged = daily_k.merge(totals, on="date", how="left")
        merged["share"] = merged["spend"] / merged["total"].where(merged["total"] > 0, np.nan)
        for k in top_keys:
            kk = merged[merged[key] == k]
            safe = sanitize_fname(f"{fname_prefix}_{k}.png")
            line("date", "share", kk, f"{title_prefix}: {k}", "date", "spend share", safe, out)

    _daily_share(df, "model_group", "evol_share_modelgroup", "Daily spend share for model_group")
    _daily_share(df, "custom_llm_provider", "evol_share_provider", "Daily spend share for provider")

    # --- Spend anomalies: z-score of daily spend -----------------------------
    daily_anom = daily.copy()
    daily_anom["spend_rolling_mean"] = daily_anom["spend"].rolling(14, min_periods=7).mean()
    daily_anom["spend_rolling_std"] = daily_anom["spend"].rolling(14, min_periods=7).std()
    daily_anom["spend_z"] = (daily_anom["spend"] - daily_anom["spend_rolling_mean"]) / daily_anom["spend_rolling_std"]
    line("date", "spend_z", daily_anom, "Daily spend z-score (window=14d)", "date", "z-score", "daily_spend_zscore.png", out)

    # --- Throughput: per-minute and per-hour request rates -------------------
    per_min = (
        df.set_index("startTime")
        .resample("min")["request_id"].count()
        .reset_index().rename(columns={"request_id": "rpm"})
    )
    per_hr = (
        df.set_index("startTime")
        .resample("h")["request_id"].count()
        .reset_index().rename(columns={"request_id": "rph"})
    )
    if not per_min.empty:
        line("startTime", "rpm", per_min, "Requests per minute", "time", "req/min", "throughput_rpm.png", out)
    if not per_hr.empty:
        line("startTime", "rph", per_hr, "Requests per hour", "time", "req/hour", "throughput_rph.png", out)

    # --- Cost efficiency: spend per 1k tokens --------------------------------
    def _spend_per_1k_tokens(group_key: str, fname: str, title: str) -> None:
        g = df.groupby(group_key, as_index=False).agg(
            spend=("spend", "sum"),
            tokens=("total_tokens", "sum"),
            requests=("request_id", "count"),
        )
        g = g[g["tokens"] > 0]
        if g.empty:
            return
        g["cost_per_1k"] = 1000.0 * g["spend"] / g["tokens"]
        s = g.sort_values("spend", ascending=False).head(top).set_index(group_key)["cost_per_1k"]
        bar(s, f"{title} (Top {len(s)})", group_key, "cost per 1k tokens", fname, out, top=len(s))

    _spend_per_1k_tokens("model_group", "cost_per_1k_by_modelgroup.png", "Cost per 1k tokens by model_group")
    _spend_per_1k_tokens("custom_llm_provider", "cost_per_1k_by_provider.png", "Cost per 1k tokens by provider")

    # --- Token composition and relation to spend -----------------------------
    tokens_clean = _safe_quantile_cut(df["total_tokens"], 0.999)
    if not tokens_clean.empty:
        idx = tokens_clean.index
        scatter(
            df.loc[idx, "total_tokens"],
            df.loc[idx, "spend"],
            "Spend vs total tokens (trimmed 99.9%)",
            "total tokens",
            "spend",
            "scatter_spend_vs_tokens.png",
            out,
        )

    # --- Correlations --------------------------------------------------------
    corr_cols = [
        "spend",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "latency_s",
        "ttfb_s",
        "gen_s",
        "cache_hit_num",
        "success_num",
    ]
    corr_df = df[corr_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if not corr_df.empty:
        corr = corr_df.corr(method="pearson")
        save_csv(corr, "correlations.csv", out)
        # |corr| with spend and latency
        for target, fname in [("spend", "corr_abs_with_spend.png"), ("latency_s", "corr_abs_with_latency.png")]:
            v = corr[target].drop(target).abs().sort_values(ascending=False)
            bar(v, f"|corr| with {target}", "feature", "|corr|", fname, out, top=len(v))

    # --- User analytics ------------------------------------------------------
    user_stats = df.groupby("end_user", as_index=False).agg(
        spend=("spend", "sum"),
        requests=("request_id", "count"),
        tokens=("total_tokens", "sum"),
        first=("date", "min"),
        last=("date", "max"),
        active_days=("date", pd.Series.nunique),
    )
    # Retention: active days distribution
    hist(user_stats["active_days"], "Active days per user", "active days", "count", "hist_user_active_days.png", out, bins=30)
    # Top users by active days
    s_active = user_stats.sort_values("active_days", ascending=False).head(top).set_index("end_user")["active_days"]
    bar(s_active, f"Top {len(s_active)} users by active days", "end_user", "active days", "top_users_active_days.png", out, top=len(s_active))

    # Model popularity: distinct users per model_group
    pop = df.groupby("model_group")["end_user"].nunique().sort_values(ascending=False).head(top)
    bar(pop, f"Distinct users per model_group (Top {len(pop)})", "model_group", "distinct users", "modelgroup_distinct_users.png", out, top=len(pop))

    # --- Cache effectiveness -------------------------------------------------
    cache_stats = df.groupby("cache_hit", as_index=False).agg(
        spend=("spend", "sum"),
        requests=("request_id", "count"),
        avg_spend=("spend", "mean"),
    )
    if not cache_stats.empty:
        s_cache = cache_stats.set_index("cache_hit")["avg_spend"]
        bar(s_cache, "Avg row spend: cache_hit vs miss", "cache_hit", "avg spend", "cache_avg_spend.png", out, top=len(s_cache))
    # Hit rate over time
    cache_daily = df.groupby("date", as_index=False).agg(hit=("cache_hit", "sum"), req=("request_id", "count"))
    cache_daily["hit_rate"] = cache_daily["hit"] / cache_daily["req"].where(cache_daily["req"] > 0, np.nan)
    line("date", "hit_rate", cache_daily, "Cache hit rate over time", "date", "hit rate", "cache_hit_rate_over_time.png", out)

    # --- Provider mix over time ----------------------------------------------
    _daily_share(df, "custom_llm_provider", "provider_mix_share", "Provider spend share")

    # --- Outliers -------------------------------------------------------------
    def _write_top_csv(col: str, frac: float, fname: str) -> None:
        n = max(1, int(len(df) * frac))
        top_rows = df.nlargest(n, col)
        save_csv(top_rows, fname, out)

    _write_top_csv("spend", 0.001, "top_0p1pct_spend_rows.csv")
    _write_top_csv("total_tokens", 0.001, "top_0p1pct_tokens_rows.csv")
    _write_top_csv("latency_s", 0.001, "top_0p1pct_latency_rows.csv")

    # Outlier histograms trimmed
    for col, fname in [
        ("spend", "hist_spend_trimmed.png"),
        ("total_tokens", "hist_tokens_trimmed.png"),
        ("latency_s", "hist_latency_trimmed.png"),
    ]:
        series = _safe_quantile_cut(df[col], 0.999)
        if not series.empty:
            hist(series, f"{col} distribution (trimmed 99.9%)", col, "count", fname, out, bins=50)

    # --- Request type mix ----------------------------------------------------
    call_mix = df["call_type"].value_counts()
    bar(call_mix, "Request type mix", "call_type", "count", "call_type_counts.png", out, top=len(call_mix))
    call_daily = df.groupby(["date", "call_type"], as_index=False).agg(req=("request_id", "count"))
    total_daily = df.groupby("date", as_index=False)["request_id"].count().rename({"request_id": "total"}, axis=1)
    call_daily = call_daily.merge(total_daily, on="date", how="left")
    call_daily["share"] = call_daily["req"] / call_daily["total"].where(call_daily["total"] > 0, np.nan)
    for ct in call_daily["call_type"].unique():
        sub = call_daily[call_daily["call_type"] == ct]
        safe = sanitize_fname(f"daily_share_calltype_{ct}.png")
        line("date", "share", sub, f"Daily share for call_type={ct}", "date", "share", safe, out)

    # --- Failures ------------------------------------------------------------
    failures = df[df["status"] != "success"]
    if not failures.empty:
        fail_daily = failures.groupby("date", as_index=False)["request_id"].count().rename({"request_id": "failures"}, axis=1)
        line("date", "failures", pd.DataFrame(fail_daily), "Failures per day", "date", "count", "failures_per_day.png", out)
        series = _safe_quantile_cut(failures["latency_s"], 0.999)
        if not series.empty:
            hist(series, "Failure latency (trimmed 99.9%)", "latency_s", "count", "fail_latency_hist.png", out, bins=50)

    # --- Team splits ---------------------------------------------------------
    if "team_id" in df.columns:
        top_teams = (
            df.groupby("team_id")["spend"]
            .sum()
            .sort_values(ascending=False)
            .head(min(top, df["team_id"].nunique()))
            .index
        )
        for t in top_teams:
            sub = df[df["team_id"] == t]
            s = sub.groupby("model_group")["spend"].sum().sort_values(ascending=False).head(top)
            if not s.empty:
                safe_team = sanitize_fname(f"team_{t}_spend_by_modelgroup.png")
                bar(s, f"Team {t}: spend by model_group (Top {len(s)})", "model_group", "spend", safe_team, out, top=len(s))
            du = sub.groupby("end_user")["request_id"].count()
            if not du.empty:
                k = min(top, len(du))
                safe_team2 = sanitize_fname(f"team_{t}_enduser_requests.png")
                bar(du.sort_values(ascending=False).head(k), f"Team {t}: requests by end_user (Top {k})", "end_user", "requests", safe_team2, out, top=k)

    # --- Elasticity: spend vs tokens slope ----------------------------------
    agg_daily = df.groupby("date", as_index=False).agg(spend=("spend", "sum"), tokens=("total_tokens", "sum"))
    agg_daily = agg_daily.replace([np.inf, -np.inf], np.nan).dropna()
    if not agg_daily.empty and agg_daily["tokens"].gt(0).any():
        scatter_with_fit(
            x=agg_daily["tokens"].to_numpy(dtype=float),
            y=agg_daily["spend"].to_numpy(dtype=float),
            title="Spend vs tokens (daily) — least-squares slope in title",
            xlabel="tokens",
            ylabel="spend",
            fname="elasticity_spend_vs_tokens.png",
            outdir=out,
            write_params_path="elasticity_spend_vs_tokens.txt",
        )

    # --- Pareto front: cost_per_1k vs avg_latency by model_group -------------
    mg_agg = df.groupby("model_group", as_index=False).agg(
        spend=("spend", "sum"),
        tokens=("total_tokens", "sum"),
        avg_latency=("latency_s", "mean"),
        requests=("request_id", "count"),
    )
    mg_agg = mg_agg[(mg_agg["tokens"] > 0) & (mg_agg["requests"] > 100)]
    if not mg_agg.empty:
        mg_agg["cost_per_1k"] = 1000.0 * mg_agg["spend"] / mg_agg["tokens"]
        pareto_frontier_plot(
            x=mg_agg["cost_per_1k"].to_numpy(dtype=float),
            y=mg_agg["avg_latency"].to_numpy(dtype=float),
            labels=mg_agg["model_group"].astype(str).tolist(),
            title="Pareto: cost vs latency by model_group",
            xlabel="cost per 1k tokens",
            ylabel="avg latency (s)",
            fname="pareto_cost_vs_latency_modelgroup.png",
            outdir=out,
        )

    # --- Avg latency by (model × provider) heatmap ---------------------------
    lat_pivot = df.pivot_table(index="model_group", columns="custom_llm_provider", values="latency_s", aggfunc="mean")
    top_mg = df.groupby("model_group")["spend"].sum().sort_values(ascending=False).head(top_share_k).index
    top_prov = df.groupby("custom_llm_provider")["spend"].sum().sort_values(ascending=False).head(top_share_k).index
    lat_pivot = lat_pivot.loc[lat_pivot.index.intersection(top_mg), lat_pivot.columns.intersection(top_prov)].fillna(0.0)
    if lat_pivot.shape[0] and lat_pivot.shape[1]:
        heatmap(lat_pivot, "Avg latency heatmap: model_group × provider", "provider", "model_group", "heatmap_latency_modelgroup_provider.png", out)

    # --- Save selected tables for external analysis --------------------------
    save_csv(user_stats, "user_stats.csv", out)
    save_csv(mg_agg, "modelgroup_agg.csv", out)
    save_csv(hourly, "byhour_stats.csv", out)
    save_csv(weekday, "byweekday_stats.csv", out)

    # Latency ~ tokens and ~ spend: bin by quantiles and plot avg latency
    def _bin_and_avg(x: pd.Series, y: pd.Series, bins: int, label: str, fname: str) -> None:
        d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if d.empty:
            return
        d = d[d["x"] >= 0]
        q = min(bins, max(2, d["x"].nunique()))
        d["bin"] = pd.qcut(d["x"], q=q, duplicates="drop")
        g = (
            d.groupby("bin", observed=True, as_index=False)
            .agg(avg_x=("x", "mean"), avg_y=("y", "mean"))
            .sort_values("avg_x")
        )
        line("avg_x", "avg_y", g, f"Avg latency vs {label} (binned)", label, "avg latency (s)", fname, out)

    _bin_and_avg(df["total_tokens"], df["latency_s"], bins=10, label="total tokens", fname="latency_vs_tokens_binned.png")
    _bin_and_avg(df["spend"], df["latency_s"], bins=10, label="row spend", fname="latency_vs_spend_binned.png")

    # Throughput vs token volume
    per_min_tokens = (
        df.set_index("startTime")
        .resample("min")["total_tokens"].sum()
        .reset_index().rename(columns={"total_tokens": "tpm"})
    )
    if not per_min_tokens.empty:
        line("startTime", "tpm", per_min_tokens, "Tokens per minute", "time", "tokens/min", "throughput_tokens_per_min.png", out)

    # Cache effectiveness: estimated savings from hits
    miss = df[~df["cache_hit"]]
    hits = df[df["cache_hit"]]
    miss_tokens = miss["total_tokens"].fillna(0).sum()
    if miss_tokens > 0 and not hits.empty:
        miss_cpt = miss["spend"].sum() / miss_tokens if miss_tokens > 0 else np.nan
        est_spend_hits_counterfactual = miss_cpt * hits["total_tokens"].fillna(0).sum() if np.isfinite(miss_cpt) else np.nan
        observed_spend_hits = hits["spend"].sum()
        est_savings = (
            max(0.0, float(est_spend_hits_counterfactual - observed_spend_hits))
            if np.isfinite(miss_cpt)
            else np.nan
        )
        save_text(
            "\n".join(
                [
                    f"miss_cost_per_token={miss_cpt}",
                    f"hits_tokens={hits['total_tokens'].fillna(0).sum()}",
                    f"observed_spend_hits={observed_spend_hits}",
                    f"estimated_counterfactual_spend_hits={est_spend_hits_counterfactual}",
                    f"estimated_savings={est_savings}",
                ]
            ),
            "cache_savings_estimate.txt",
            out,
        )

        miss_daily = miss.groupby("date", as_index=False).agg(
            miss_spend=("spend", "sum"),
            miss_tokens=("total_tokens", "sum"),
        )
        hits_daily = hits.groupby("date", as_index=False).agg(
            hit_spend=("spend", "sum"),
            hit_tokens=("total_tokens", "sum"),
        )
        daily_cache = miss_daily.merge(hits_daily, on="date", how="outer").fillna(0.0)
        daily_cache["miss_cpt"] = np.where(
            daily_cache["miss_tokens"] > 0,
            daily_cache["miss_spend"] / daily_cache["miss_tokens"],
            np.nan,
        )
        daily_cache["est_savings"] = np.where(
            np.isfinite(daily_cache["miss_cpt"]),
            daily_cache["miss_cpt"] * daily_cache["hit_tokens"] - daily_cache["hit_spend"],
            np.nan,
        )
        line("date", "est_savings", daily_cache, "Estimated daily cache savings", "date", "estimated savings", "cache_estimated_savings_daily.png", out)

    # Token composition histograms and daily ratios
    for c, title, fname in [
        ("prompt_ratio", "Prompt ratio = prompt/total", "hist_prompt_ratio.png"),
        ("completion_ratio", "Completion ratio = completion/total", "hist_completion_ratio.png"),
    ]:
        s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        if not s.empty:
            hist(s.clip(lower=0, upper=1), title, c, "count", fname, out, bins=50)

    daily_ratios = df.groupby("date", as_index=False).agg(
        prompt_ratio=("prompt_ratio", "mean"),
        completion_ratio=("completion_ratio", "mean"),
    )
    if not daily_ratios.empty:
        line("date", "prompt_ratio", daily_ratios, "Avg prompt ratio per day", "date", "ratio", "daily_avg_prompt_ratio.png", out)
        line("date", "completion_ratio", daily_ratios, "Avg completion ratio per day", "date", "ratio", "daily_avg_completion_ratio.png", out)

    # Cache impact on latency
    cache_latency = df.groupby("cache_hit", as_index=False)["latency_s"].mean()
    if not cache_latency.empty:
        bar(cache_latency.set_index("cache_hit")["latency_s"], "Avg latency by cache_hit", "cache_hit", "avg latency (s)", "cache_vs_latency.png", out, top=len(cache_latency))

    # Failure rate by model_group
    fail_rate_mg = df.groupby("model_group", as_index=False).agg(
        requests=("request_id", "count"),
        failures=("status", lambda s: (s != "success").sum()),
        spend=("spend", "sum"),
    )
    fail_rate_mg["fail_rate"] = np.where(
        fail_rate_mg["requests"] > 0,
        fail_rate_mg["failures"] / fail_rate_mg["requests"],
        np.nan,
    )
    s_fail = fail_rate_mg.sort_values("spend", ascending=False).head(top).set_index("model_group")["fail_rate"]
    if not s_fail.empty:
        bar(s_fail, f"Failure rate by model_group (Top {len(s_fail)} by spend)", "model_group", "failure rate", "fail_rate_by_modelgroup.png", out, top=len(s_fail))

    # Distinct users per model
    pop_model = df.groupby("model")["end_user"].nunique().sort_values(ascending=False).head(top)
    if not pop_model.empty:
        bar(pop_model, f"Distinct users per model (Top {len(pop_model)})", "model", "distinct users", "model_distinct_users.png", out, top=len(pop_model))

    # Per-user and per-model anomalies (z-score on daily spend)
    def _z_anoms(df_in: pd.DataFrame, key: str, out_name: str, top_k: int = 10) -> None:
        top_keys = df_in.groupby(key)["spend"].sum().sort_values(ascending=False).head(top_k).index
        rows = []
        for k in top_keys:
            d = df_in[df_in[key] == k].groupby("date", as_index=False).agg(spend=("spend", "sum")).sort_values("date")
            if len(d) < 7:
                continue
            d["mean"] = d["spend"].rolling(14, min_periods=7).mean()
            d["std"] = d["spend"].rolling(14, min_periods=7).std()
            d["z"] = (d["spend"] - d["mean"]) / d["std"]
            d["key"] = k
            rows.append(d)
        if rows:
            out_df = pd.concat(rows, ignore_index=True)
            save_csv(out_df.sort_values("z", ascending=False), out_name, out)

    _z_anoms(df, "end_user", "anomalies_top_users_daily_spend.csv", top_k=top)
    _z_anoms(df, "model_group", "anomalies_top_modelgroups_daily_spend.csv", top_k=top)
