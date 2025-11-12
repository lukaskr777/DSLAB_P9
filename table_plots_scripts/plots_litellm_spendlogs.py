"""
Exploratory plots for LiteLLM_SpendLogs.
Writes figures to figs/litellm_spendlogs.
"""

import numpy as np
import pandas as pd

from utility_scripts.file_utils import PathLike, ensure_empty_dir, read_table, sanitize_fname, save_csv
from utility_scripts.plot_utils import (
    line, lines_quantiles, bar, hist, bin_and_quantiles, heatmap, scatter, scatter_with_fit, pareto_frontier_plot
)
from utility_scripts.df_reading_utils import (
    require,
    to_utc,
    clean_table,
    ensure_date_column,
    aggregate_by_time,
    add_rates,
    top_by_value,
    cumulative_share,
    safe_div,
    safe_quantile_cut,
)


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
        "status",
    }
    miss = require(df, needed, strict=False)
    if miss:
        raise SystemExit(f"Missing required columns: {sorted(miss)}")

    # Canonical cleaning and enrichment
    df = clean_table(df)  # clamps spend, coerces numerics, adds latency_s/ttfb_s/gen_s, completion_ratio, date
    # Convert status to numeric helper used by aggregations
    df["__success__"] = (
        df["status"].astype("string").str.lower().eq("success").fillna(False)
        if "status" in df.columns
        else pd.Series(False, index=df.index, dtype="boolean")
    )
    df["__count__"] = 1  # for counting via sum    
    # add hour/dow/datehour/week/month
    ts = to_utc(df["startTime"])
    df["date"] = ts.dt.floor("D")
    df = df.dropna(subset=["date"])
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek  # 0=Mon
    df["datehour"] = ts.dt.floor("h")
    df["week"] = ts.dt.isocalendar().week.astype(int)
    df["month"] = ts.dt.tz_localize(None).dt.to_period("M").astype(str)

    df["ttfb_share_row"] = safe_div(df["ttfb_s"], df["latency_s"]).clip(lower=0, upper=1)
    df["gen_share_row"]  = safe_div(df["gen_s"],  df["latency_s"]).clip(lower=0, upper=1)

    if "prompt_ratio" not in df.columns and {"prompt_tokens", "total_tokens"} <= set(df.columns):
        df["prompt_ratio"] = safe_div(df["prompt_tokens"], df["total_tokens"]).clip(lower=0, upper=1)

    if "completion_ratio" not in df.columns and {"completion_tokens", "total_tokens"} <= set(df.columns):
        df["completion_ratio"] = safe_div(df["completion_tokens"], df["total_tokens"]).clip(lower=0, upper=1)

    # Daily aggregates
    # ensure_date_column already handled in clean_table via date; kept here for idempotence if needed elsewhere
    df = ensure_date_column(df, time_col="startTime", out_col="date", floor="D", dropna=True, sort=True)
    daily = aggregate_by_time(
        df,
        time_col="date",
        freq="D",
        sums=("spend", "total_tokens", "prompt_tokens", "completion_tokens", "__success__", "__count__"),
        means=("latency_s", "ttfb_s", "gen_s"),
        include_count=False,
        custom={
            # Provide explicit names for clarity
            "spend": ("spend", "sum"),
            "requests": ("__count__", "sum"),
            "success": ("__success__", "sum"),
            "total_tokens": ("total_tokens", "sum"),
            "prompt_tokens": ("prompt_tokens", "sum"),
            "completion_tokens": ("completion_tokens", "sum"),
            "avg_latency_s": ("latency_s", "mean"),
            "avg_ttfb_s": ("ttfb_s", "mean"),
            "avg_gen_s": ("gen_s", "mean"),
        },
    )
    daily = add_rates(
        daily,
        ratios=(
            ("success", "requests", "success_rate"),
            ("total_tokens", "requests", "avg_tokens_per_req"),
            ("avg_ttfb_s", "avg_latency_s", "ttfb_share"),
            ("avg_gen_s", "avg_latency_s", "gen_share"),
        ),
    )
    daily = daily.sort_values("date", kind="stable")

    # Daily lines
    line(
        "date", "spend", daily, "Total spend per day", 
        "date", "spend", "daily_total_spend.png", out
    )
    line(
        "date", "requests", daily, "Requests per day", 
        "date", "requests", "daily_requests.png", out
    )
    line(
        "date", "success_rate", daily, "Success rate per day", 
        "date", "success rate", "daily_success_rate.png", out
    )
    line(
        "date", "avg_tokens_per_req", daily, "Avg tokens per request per day", 
        "date", "avg tokens/req", "daily_avg_tokens_per_req.png", out
    )
    line(
        "date", "avg_latency_s", daily, "Avg latency per day", 
        "date", "latency (s)", "daily_avg_latency.png", out
    )
    line(
        "date", "avg_ttfb_s", daily, "Avg time-to-first-token per day", 
        "date", "ttfb (s)", "daily_avg_ttfb.png", out
    )
    line(
        "date", "avg_gen_s", daily, "Avg generation time per day", 
        "date", "gen time (s)", "daily_avg_gen.png", out
    )
    line(
        "date", "ttfb_share", daily, "TTFB share of total latency (daily)", 
        "date", "share", "daily_ttfb_share.png", out
    )
    line(
        "date", "gen_share", daily, "Generation share of total latency (daily)", 
        "date", "share", "daily_gen_share.png", out
    )

    lines_quantiles(
        "date", "total_tokens", df, "Tokens/request per day: Q1/Median/Q3",
        "date", "tokens/request", "daily_tokens_quantiles.png", out
    )
    lines_quantiles(
        "date", "latency_s", df, "Latency per day: Q1/Median/Q3",
        "date", "latency (s)", "daily_latency_quantiles.png", out
    )
    lines_quantiles(
        "date", "ttfb_s", df, "TTFB per day: Q1/Median/Q3",
        "date", "ttfb (s)", "daily_ttfb_quantiles.png", out
    )
    lines_quantiles(
        "date", "gen_s", df, "Generation time per day: Q1/Median/Q3",
        "date", "gen time (s)", "daily_gen_quantiles.png", out
    )
    lines_quantiles(
        "date", "ttfb_share_row", df, "TTFB share per day: Q1/Median/Q3",
        "date", "share", "daily_ttfb_share_quantiles.png", out
    )
    lines_quantiles(
        "date", "gen_share_row", df, "Generation share per day: Q1/Median/Q3",
        "date", "share", "daily_gen_share_quantiles.png", out
    )

    # Bars: Top by spend
    for key, fname, title in [
        ("end_user", "top_users_spend.png", "Top end_users by total spend"),
        ("model", "top_models_spend.png", "Top models by total spend"),
        ("model_group", "top_model_groups_spend.png", "Top model_groups by total spend"),
        ("custom_llm_provider", "top_providers_spend.png", "Top providers by total spend"),
        ("api_key", "top_apikeys_spend.png", "Top API keys by total spend"),
        ("call_type", "top_call_types_spend.png", "Top call types by total spend"),
    ]:
        s = top_by_value(df, keys=key, value_col="spend", top=top, ascending=False)
        if not s.empty:
            bar(s, f"{title} (Top {len(s)})", key, "total spend", fname, out, top=len(s))

    # Histograms
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
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            hist(series, title, col, "count", fname, out, bins=50)

    for col, title, fname in [
        ("latency_s", "Latency per request (s)", "hist_latency_s.png"),
        ("ttfb_s", "Time to first token (s)", "hist_ttfb_s.png"),
        ("gen_s", "Generation time (s)", "hist_gen_s.png"),
    ]:
        series = safe_quantile_cut(df[col], 0.999)
        if not series.empty:
            hist(series, title, col, "count", fname, out, bins=50)

    # Whale curve
    spend_by_user = df.groupby("end_user", dropna=False)["spend"].sum()
    whale = cumulative_share(
        df=pd.DataFrame({"end_user": spend_by_user.index, "spend": spend_by_user.values}),
        key="end_user",
        value_col="spend",
        positive_only=True,
        normalize_rank=True,
        dropna_key=True,
    ).rename(columns={"rank": "user_rank_frac", "cum_share": "cum_share"})
    if not whale.empty:
        save_csv(whale, "whale_curve.csv", out)
        line(
            "user_rank_frac", "cum_share", whale, "Cumulative spend share vs user rank", 
            "user rank fraction", "cumulative share", "cumulative_spend_share.png", out
        )

    # Per-top-user daily lines
    top_users = (
        spend_by_user.sort_values(ascending=False)
        .loc[lambda s: s > 0]
        .head(top)
        .index.tolist()
    )
    if top_users:
        per_user_daily = (
            df[df["end_user"].isin(top_users)]
            .groupby(["date", "end_user"], as_index=False)
            .agg(spend=("spend", "sum"))
            .sort_values(["end_user", "date"], kind="stable")
        )
        for u in top_users:
            sub = per_user_daily[per_user_daily["end_user"] == u]
            safe = sanitize_fname(f"daily_spend_{u}.png")
            line("date", "spend", sub, f"Daily spend for {u}", "date", "spend", safe, out)

    # Hourly patterns
    hourly = df.groupby("hour", as_index=False).agg(
        spend=("spend", "sum"),
        requests=("request_id", "count"),
        avg_latency_s=("latency_s", "mean"),
    ).sort_values("hour", kind="stable")
    order = list(range(24))
    bar(
        hourly.set_index("hour")["spend"], "Spend by hour of day", 
        "hour", "spend", "byhour_spend.png", out, 
        top=24, order=order, sort_values=False
    )
    bar(
        hourly.set_index("hour")["requests"], "Requests by hour of day", 
        "hour", "requests", "byhour_requests.png", out, 
        top=24, order=order, sort_values=False
    )
    line(
        "hour", "avg_latency_s", hourly, "Avg latency by hour of day", 
        "hour", "latency (s)", "byhour_latency.png", out
    )
    lines_quantiles(
        "hour", "latency_s", df, "Latency by hour: Q1/Median/Q3",
        "hour", "latency (s)", "byhour_latency_quantiles.png", out
    )


    # Weekday patterns
    weekday = df.groupby("dow", as_index=False).agg(
        spend=("spend", "sum"),
        requests=("request_id", "count"),
        avg_latency_s=("latency_s", "mean"),
    ).sort_values("dow", kind="stable")
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday["dow_label"] = weekday["dow"].map({i: l for i, l in enumerate(dow_labels)})
    bar(
        weekday.set_index("dow_label")["spend"], "Spend by weekday", 
        "weekday", "spend", "byweekday_spend.png", out, 
        top=7, order=dow_labels, sort_values=False
    )
    bar(
        weekday.set_index("dow_label")["requests"], "Requests by weekday", 
        "weekday", "requests", "byweekday_requests.png", out, 
        top=7, order=dow_labels, sort_values=False
    )
    line(
        "dow", "avg_latency_s", weekday, "Avg latency by weekday", 
        "weekday (0=Mon)", "latency (s)", "byweekday_latency.png", out
    )
    lines_quantiles(
        "dow", "latency_s", df, "Latency by weekday: Q1/Median/Q3",
        "weekday (0=Mon)", "latency (s)", "byweekday_latency_quantiles.png", out
    )


    # Weekday × hour heatmap of spend
    pivot_hw = (
        df.pivot_table(index="dow", columns="hour", values="spend", aggfunc="sum")
        .reindex(index=range(7), columns=range(24), fill_value=0.0)
    )
    pivot_hw.index = pd.Index(dow_labels, name=pivot_hw.index.name)
    heatmap(
        pivot_hw, "Spend heatmap by weekday x hour", 
        "hour", "weekday", "heatmap_weekday_hour_spend.png", out
    )

    # Daily spend share for top-K groups
    def _daily_share(df_in: pd.DataFrame, key: str, fname_prefix: str, title_prefix: str) -> None:
        top_keys = (
            df_in.groupby(key)["spend"]
            .sum()
            .sort_values(ascending=False)
            .head(top_share_k)
            .index
        )
        if not len(top_keys):
            return
        daily_k = df_in[df_in[key].isin(top_keys)].groupby(["date", key], as_index=False).agg(spend=("spend", "sum"))
        totals = df_in.groupby("date", as_index=False).agg(total=("spend", "sum"))
        merged = daily_k.merge(totals, on="date", how="left")
        merged["share"] = safe_div(merged["spend"], merged["total"])
        for k_ in top_keys:
            kk = merged[merged[key] == k_]
            safe = sanitize_fname(f"{fname_prefix}_{k_}.png")
            line(
                "date", "share", kk, f"{title_prefix}: {k_}", 
                "date", "spend share", safe, out
            )

    _daily_share(df, "model_group", "evol_share_modelgroup", "Daily spend share for model_group")
    _daily_share(df, "custom_llm_provider", "evol_share_provider", "Daily spend share for provider")

    # Spend anomalies: z-score
    daily_anom = daily.copy()
    daily_anom["spend_rolling_mean"] = daily_anom["spend"].rolling(14, min_periods=7).mean()
    daily_anom["spend_rolling_std"] = daily_anom["spend"].rolling(14, min_periods=7).std()
    daily_anom["spend_z"] = (daily_anom["spend"] - daily_anom["spend_rolling_mean"]) / daily_anom["spend_rolling_std"]
    line(
        "date", "spend_z", daily_anom, "Daily spend z-score (window=14d)", 
        "date", "z-score", "daily_spend_zscore.png", out
    )

    # Throughput
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
            "startTime", "rpm", per_min, "Requests per minute", 
            "time", "req/min", "throughput_rpm.png", out
        )
    if not per_hr.empty:
        line(
            "startTime", "rph", per_hr, "Requests per hour", 
            "time", "req/hour", "throughput_rph.png", out
        )

    # Cost per 1k tokens
    def _cost_per_1k(group_key: str, fname: str, title: str) -> None:
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
        bar(
            s, f"{title} (Top {len(s)})", 
            group_key, "cost per 1k tokens", fname, out, 
            top=len(s)
        )

    _cost_per_1k("model_group", "cost_per_1k_by_modelgroup.png", "Cost per 1k tokens by model_group")
    _cost_per_1k("custom_llm_provider", "cost_per_1k_by_provider.png", "Cost per 1k tokens by provider")

    # Spend vs tokens (trimmed)
    tokens_clean = safe_quantile_cut(df["total_tokens"], 0.999)
    if not tokens_clean.empty:
        idx = tokens_clean.index
        scatter(
            df.loc[idx, "total_tokens"], df.loc[idx, "spend"], "Spend vs total tokens (trimmed 99.9%)",
            "total tokens", "spend", "scatter_spend_vs_tokens.png", out
        )

    # Correlations
    corr_cols = [
        "spend",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "latency_s",
        "ttfb_s",
        "gen_s",
        "__success__",
    ]
    corr_df = df[corr_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if not corr_df.empty:
        corr = corr_df.corr(method="pearson")
        save_csv(corr, "correlations.csv", out)
        for target, fname in [("spend", "corr_abs_with_spend.png"), ("latency_s", "corr_abs_with_latency.png")]:
            v = corr[target].drop(target, errors="ignore").abs().sort_values(ascending=False)
            if not v.empty:
                bar(
                    v, f"|corr| with {target}", 
                    "feature", "|corr|", fname, out, 
                    top=len(v)
                )

    # User analytics
    user_stats = df.groupby("end_user", as_index=False).agg(
        spend=("spend", "sum"),
        requests=("request_id", "count"),
        tokens=("total_tokens", "sum"),
        first=("date", "min"),
        last=("date", "max"),
        active_days=("date", pd.Series.nunique),
    )
    hist(
        user_stats["active_days"], "Active days per user", 
        "active days", "count", "hist_user_active_days.png", out, 
        bins=30
    )
    s_active = user_stats.sort_values("active_days", ascending=False).head(top).set_index("end_user")["active_days"]
    if not s_active.empty:
        bar(
            s_active, f"Top {len(s_active)} users by active days", 
            "end_user", "active days", "top_users_active_days.png", out, 
            top=len(s_active)
        )

    pop = df.groupby("model_group")["end_user"].nunique().sort_values(ascending=False).head(top)
    if not pop.empty:
        bar(
            pop, f"Distinct users per model_group (Top {len(pop)})", 
            "model_group", "distinct users", "modelgroup_distinct_users.png", out, 
            top=len(pop)
        )

    # Provider mix over time
    _daily_share(df, "custom_llm_provider", "provider_mix_share", "Provider spend share")

    # Outliers
    def _write_top_csv(col: str, frac: float, fname: str) -> None:
        n = max(1, int(len(df) * frac))
        top_rows = df.nlargest(n, col)
        save_csv(top_rows, fname, out)

    _write_top_csv("spend", 0.001, "top_0p1pct_spend_rows.csv")
    _write_top_csv("total_tokens", 0.001, "top_0p1pct_tokens_rows.csv")
    _write_top_csv("latency_s", 0.001, "top_0p1pct_latency_rows.csv")

    for col, fname in [
        ("spend", "hist_spend_trimmed.png"),
        ("total_tokens", "hist_tokens_trimmed.png"),
        ("latency_s", "hist_latency_trimmed.png"),
    ]:
        series = safe_quantile_cut(df[col], 0.999)
        if not series.empty:
            hist(
                series, f"{col} distribution (trimmed 99.9%)", 
                col, "count", fname, out, 
                bins=50
            )

    # Request type mix
    call_mix = df["call_type"].astype("string").value_counts()
    if not call_mix.empty:
        bar(
            call_mix, "Request type mix", 
            "call_type", "count", "call_type_counts.png", out, 
            top=len(call_mix)
        )
    call_daily = df.groupby(["date", "call_type"], as_index=False).agg(req=("request_id", "count"))
    total_daily = df.groupby("date", as_index=False)["request_id"].count().rename({"request_id": "total"}, axis=1)
    call_daily = call_daily.merge(total_daily, on="date", how="left")
    call_daily["share"] = safe_div(call_daily["req"], call_daily["total"])
    for ct in call_daily["call_type"].unique():
        sub = call_daily[call_daily["call_type"] == ct]
        safe = sanitize_fname(f"daily_share_calltype_{ct}.png")
        line(
            "date", "share", sub, f"Daily share for call_type={ct}", 
            "date", "share", safe, out
        )

    # Failures
    failures = df[~df["__success__"]]
    if not failures.empty:
        fail_daily = (
            failures.groupby("date", as_index=False)["request_id"]
            .count()
            .rename({"request_id": "failures"}, axis=1)
        )
        line(
            "date", "failures", pd.DataFrame(fail_daily), "Failures per day", 
            "date", "count", "failures_per_day.png", out
        )
        series = safe_quantile_cut(failures["latency_s"], 0.999)
        if not series.empty:
            hist(
                series, "Failure latency (trimmed 99.9%)", 
                "latency_s", "count", "fail_latency_hist.png", out, 
                bins=50
            )

    # Team splits
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
                bar(
                    s, f"Team {t}: spend by model_group (Top {len(s)})", 
                    "model_group", "spend", safe_team, out, 
                    top=len(s)
                )
            du = sub.groupby("end_user")["request_id"].count()
            if not du.empty:
                k = min(top, len(du))
                safe_team2 = sanitize_fname(f"team_{t}_enduser_requests.png")
                bar(
                    du.sort_values(ascending=False).head(k), f"Team {t}: requests by end_user (Top {k})", 
                    "end_user", "requests", safe_team2, out, 
                    top=k
                )

    # Elasticity: spend vs tokens slope
    agg_daily = df.groupby("date", as_index=False).agg(spend=("spend", "sum"), tokens=("total_tokens", "sum"))
    agg_daily = agg_daily.replace([np.inf, -np.inf], np.nan).dropna()
    if not agg_daily.empty and agg_daily["tokens"].gt(0).any():
        scatter_with_fit(
            x=agg_daily["tokens"].to_numpy(dtype=float),
            y=agg_daily["spend"].to_numpy(dtype=float),
            title="Spend vs tokens (daily) — least-squares slope",
            xlabel="tokens",
            ylabel="spend",
            fname="elasticity_spend_vs_tokens.png",
            outdir=out,
            write_params_path="elasticity_spend_vs_tokens.txt",
        )

    # Pareto front: model_group
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

    # Avg latency by (model_group × provider) heatmap
    lat_pivot = df.pivot_table(index="model_group", columns="custom_llm_provider", values="latency_s", aggfunc="mean")
    top_mg = df.groupby("model_group")["spend"].sum().sort_values(ascending=False).head(top_share_k).index
    top_prov = df.groupby("custom_llm_provider")["spend"].sum().sort_values(ascending=False).head(top_share_k).index
    lat_pivot = (
        lat_pivot.loc[lat_pivot.index.intersection(top_mg), 
        lat_pivot.columns.intersection(top_prov)].fillna(0.0)
    )
    if lat_pivot.shape[0] and lat_pivot.shape[1]:
        heatmap(
            lat_pivot, "Avg latency heatmap: model_group x provider", 
            "provider", "model_group", "heatmap_latency_modelgroup_provider.png", out
        )

    # Save selected tables
    save_csv(user_stats, "user_stats.csv", out)
    save_csv(mg_agg, "modelgroup_agg.csv", out)
    save_csv(hourly, "byhour_stats.csv", out)
    save_csv(weekday, "byweekday_stats.csv", out)

    # Binned latency curves
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
            .sort_values("avg_x", kind="stable")
        )
        line(
            "avg_x", "avg_y", g, f"Avg latency vs {label} (binned)", 
            label, "avg latency (s)", fname, out
        )

    _bin_and_avg(
        df["total_tokens"], df["latency_s"], bins=10, label="total tokens", fname="latency_vs_tokens_binned.png"
    )
    _bin_and_avg(
        df["spend"], df["latency_s"], bins=10, label="row spend", fname="latency_vs_spend_binned.png"
    )

    bin_and_quantiles(
        df["total_tokens"], df["latency_s"], bins=10,
        label="Latency vs total tokens (binned)", fname="latency_vs_tokens_binned_quantiles.png", out=out
    )
    bin_and_quantiles(
        df["spend"], df["latency_s"], bins=10,
        label="Latency vs row spend (binned)", fname="latency_vs_spend_binned_quantiles.png", out=out
    )

    # Throughput vs token volume
    per_min_tokens = (
        df.set_index("startTime")
        .resample("min")["total_tokens"]
        .sum()
        .reset_index()
        .rename(columns={"total_tokens": "tpm"})
    )
    if not per_min_tokens.empty:
        line(
            "startTime", "tpm", per_min_tokens, "Tokens per minute",
            "time", "tokens/min", "throughput_tokens_per_min.png", out
        )

    # Token composition histograms and daily ratios
    for c, title, fname in [
        ("prompt_ratio", "Prompt ratio = prompt/total", "hist_prompt_ratio.png"),
        ("completion_ratio", "Completion ratio = completion/total", "hist_completion_ratio.png"),
    ]:
        if c in df.columns:
            s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
            if not s.empty:
                hist(s.clip(lower=0, upper=1), title, c, "count", fname, out, bins=50)

    # Daily ratios
    have_daily = []
    if "prompt_ratio" in df.columns:
        have_daily.append(("prompt_ratio", "daily_avg_prompt_ratio.png", "Avg prompt ratio per day"))
    if "completion_ratio" in df.columns:
        have_daily.append(("completion_ratio", "daily_avg_completion_ratio.png", "Avg completion ratio per day"))

    if have_daily:
        daily_ratios = df.groupby("date", as_index=False).agg(**{col: (col, "mean") for col, _, _ in have_daily})
        for col, fname, title in have_daily:
            if col in daily_ratios.columns:
                line("date", col, daily_ratios, title, "date", "ratio", fname, out)
        
        if "prompt_ratio" in df.columns:
            lines_quantiles(
                "date", "prompt_ratio", df, "Prompt ratio per day: Q1/Median/Q3",
                "date", "ratio", "daily_prompt_ratio_quantiles.png", out
            )
        if "completion_ratio" in df.columns:
            lines_quantiles(
                "date", "completion_ratio", df, "Completion ratio per day: Q1/Median/Q3",
                "date", "ratio", "daily_completion_ratio_quantiles.png", out
            )

    # Failure rate by model_group
    fail_rate_mg = df.groupby("model_group", as_index=False).agg(
        requests=("request_id", "count"),
        failures=("__success__", lambda s: (~s).sum()),
        spend=("spend", "sum"),
    )
    fail_rate_mg["fail_rate"] = safe_div(fail_rate_mg["failures"], fail_rate_mg["requests"])
    s_fail = fail_rate_mg.sort_values("spend", ascending=False).head(top).set_index("model_group")["fail_rate"]
    if not s_fail.empty:
        bar(
            s_fail, f"Failure rate by model_group (Top {len(s_fail)} by spend)", 
            "model_group", "failure rate", "fail_rate_by_modelgroup.png", out, 
            top=len(s_fail)
        )

    # Distinct users per model
    pop_model = df.groupby("model")["end_user"].nunique().sort_values(ascending=False).head(top)
    if not pop_model.empty:
        bar(
            pop_model, f"Distinct users per model (Top {len(pop_model)})", 
            "model", "distinct users", "model_distinct_users.png", out, 
            top=len(pop_model)
        )

    # Per-user/model anomalies
    def _z_anoms(df_in: pd.DataFrame, key: str, out_name: str, top_k: int = 10) -> None:
        top_keys = df_in.groupby(key)["spend"].sum().sort_values(ascending=False).head(top_k).index
        rows: list[pd.DataFrame] = []
        for k_ in top_keys:
            d = (
                df_in[df_in[key] == k_]
                .groupby("date", as_index=False)
                .agg(spend=("spend", "sum"))
                .sort_values("date", kind="stable")
            )
            if len(d) < 7:
                continue
            d["mean"] = d["spend"].rolling(14, min_periods=7).mean()
            d["std"] = d["spend"].rolling(14, min_periods=7).std()
            d["z"] = (d["spend"] - d["mean"]) / d["std"]
            d["key"] = k_
            rows.append(d)
        if rows:
            out_df = pd.concat(rows, ignore_index=True)
            save_csv(out_df.sort_values("z", ascending=False), out_name, out)

    _z_anoms(df, "end_user", "anomalies_top_users_daily_spend.csv", top_k=top)
    _z_anoms(df, "model_group", "anomalies_top_modelgroups_daily_spend.csv", top_k=top)


if __name__ == "__main__":
    plot_all_litellm_spendlogs()
