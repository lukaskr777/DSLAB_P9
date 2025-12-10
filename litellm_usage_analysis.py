"""
Usage & workload profiling for swiss-ai/apertus-70b-instruct based on LiteLLM_SpendLogs.

Produces plots in figs/litellm_apertus70b_usage:

- Volume over time (requests / tokens / spend for this model group)
- Distinct users over time
- Hour-of-day and weekday usage patterns (+ weekday x hour heatmap)
- Top users by request volume
- Session-level usage (requests per session)
- Concentration of usage across users ("whale" curve)
- User activity span (active days per user)

Helper utilities required:
  - utility_scripts.file_utils
  - utility_scripts.plot_utils
  - utility_scripts.df_reading_utils
"""

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
    heatmap,
)
from utility_scripts.df_reading_utils import (
    require,
    clean_table,
    to_utc,
    ensure_date_column,
    aggregate_by_time,
    cumulative_share,
)


TARGET_MODEL_GROUP = "swiss-ai/apertus-70b-instruct"

# Leave as None to include everyone.
DEFAULT_USER_TO_EXCLUDE_FROM_WHALE: str | None = "enduser_d41d8cd9"


def plot_apertus70b_usage(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/litellm_apertus70b_usage",
    top: int = 20,
    exclude_users_from_whale: list[str] | None = None,
) -> None:
    if exclude_users_from_whale is None and DEFAULT_USER_TO_EXCLUDE_FROM_WHALE is not None:
        exclude_users_from_whale = [DEFAULT_USER_TO_EXCLUDE_FROM_WHALE]

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
        "total_tokens",
        "spend",
        "model_group",
        "model",
        "end_user",
        "session_id",
    }
    miss = require(df, needed, strict=False)
    if "model_group" in miss:
        raise SystemExit("Missing required column 'model_group'.")

    # Canonical cleaning (adds latency columns, date, etc.)
    df = clean_table(df)

    # Filter to the single model_group we care about
    df = df[df["model_group"] == TARGET_MODEL_GROUP].copy()
    if df.empty:
        raise SystemExit(f"No rows for model_group={TARGET_MODEL_GROUP!r}.")

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------
    ts = to_utc(df["startTime"])
    df["date"] = ts.dt.floor("D")  # type: ignore[attr-defined]
    df = df.dropna(subset=["date"])
    df["hour"] = ts.dt.hour  # type: ignore[attr-defined]
    df["dow"] = ts.dt.dayofweek  # 0 = Monday  # type: ignore[attr-defined]
    df["datehour"] = ts.dt.floor("h")  # type: ignore[attr-defined]

    df = ensure_date_column(df, time_col="startTime", out_col="date", floor="D", dropna=True, sort=True)
    df["__count__"] = 1

    # ------------------------------------------------------------------
    # Daily volume and adoption
    # ------------------------------------------------------------------
    daily = aggregate_by_time(
        df,
        time_col="date",
        freq="D",
        sums=("__count__", "total_tokens", "spend"),
        include_count=False,
        custom={
            "requests": ("__count__", "sum"),
            "total_tokens": ("total_tokens", "sum"),
            "spend": ("spend", "sum"),
        },
    ).sort_values("date", kind="stable")

    # Distinct users  per day
    users_per_day = df.groupby("date")["end_user"].nunique().reset_index(name="distinct_users")
    daily = daily.merge(users_per_day, on="date", how="left")

    # Lines: requests, tokens, spend
    line(
        "date",
        "requests",
        daily,
        "Requests per day (Apertus 70B)",
        "date",
        "requests",
        "daily_requests.png",
        out,
    )
    line(
        "date",
        "total_tokens",
        daily,
        "Total tokens per day (Apertus 70B)",
        "date",
        "tokens",
        "daily_tokens.png",
        out,
    )
    line(
        "date",
        "spend",
        daily,
        "Spend per day (Apertus 70B)",
        "date",
        "spend",
        "daily_spend.png",
        out,
    )

    # Lines: distinct users 
    line(
        "date",
        "distinct_users",
        daily,
        "Distinct end_users per day (Apertus 70B)",
        "date",
        "distinct users",
        "daily_distinct_users.png",
        out,
    )

    save_csv(daily, "daily_usage_stats.csv", out)

    # ------------------------------------------------------------------
    # Hour-of-day and weekday patterns
    # ------------------------------------------------------------------
    hourly = (
        df.groupby("hour", as_index=False)
        .agg(
            requests=("request_id", "count"),
            tokens=("total_tokens", "sum"),
            spend=("spend", "sum"),
        )
        .sort_values("hour", kind="stable")
    )
    weekday = (
        df.groupby("dow", as_index=False)
        .agg(
            requests=("request_id", "count"),
            tokens=("total_tokens", "sum"),
            spend=("spend", "sum"),
        )
        .sort_values("dow", kind="stable")
    )

    # Bars by hour
    order_hours = list(range(24))
    bar(
        hourly.set_index("hour")["requests"],
        "Requests by hour of day (Apertus 70B)",
        "hour",
        "requests",
        "byhour_requests.png",
        out,
        top=24,
        order=order_hours,
        sort_values=False,
    )
    bar(
        hourly.set_index("hour")["tokens"],
        "Tokens by hour of day (Apertus 70B)",
        "hour",
        "tokens",
        "byhour_tokens.png",
        out,
        top=24,
        order=order_hours,
        sort_values=False,
    )
    bar(
        hourly.set_index("hour")["spend"],
        "Spend by hour of day (Apertus 70B)",
        "hour",
        "spend",
        "byhour_spend.png",
        out,
        top=24,
        order=order_hours,
        sort_values=False,
    )

    # Bars by weekday
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday["dow_label"] = weekday["dow"].map({i: l for i, l in enumerate(dow_labels)})

    bar(
        weekday.set_index("dow_label")["requests"],
        "Requests by weekday (Apertus 70B)",
        "weekday",
        "requests",
        "byweekday_requests.png",
        out,
        top=7,
        order=dow_labels,
        sort_values=False,
    )
    bar(
        weekday.set_index("dow_label")["tokens"],
        "Tokens by weekday (Apertus 70B)",
        "weekday",
        "tokens",
        "byweekday_tokens.png",
        out,
        top=7,
        order=dow_labels,
        sort_values=False,
    )
    bar(
        weekday.set_index("dow_label")["spend"],
        "Spend by weekday (Apertus 70B)",
        "weekday",
        "spend",
        "byweekday_spend.png",
        out,
        top=7,
        order=dow_labels,
        sort_values=False,
    )

    save_csv(hourly, "byhour_usage.csv", out)
    save_csv(weekday, "byweekday_usage.csv", out)

    # Weekday x hour heatmap (requests)
    pivot_hw = (
        df.pivot_table(index="dow", columns="hour", values="request_id", aggfunc="count")
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    pivot_hw.index = pd.Index(dow_labels, name=pivot_hw.index.name)
    heatmap(
        pivot_hw,
        "Requests heatmap by weekday × hour (Apertus 70B)",
        "hour",
        "weekday",
        "heatmap_weekday_hour_requests.png",
        out,
    )

    # ------------------------------------------------------------------
    # Top entities by request volume
    # ------------------------------------------------------------------
    def _top_bar(series: pd.Series, title: str, xlabel: str, fname: str, k: int) -> None:
        s = series.sort_values(ascending=False).head(k)
        if not s.empty:
            bar(
                s,
                f"{title} (Top {len(s)})",
                xlabel,
                "requests",
                fname,
                out,
                top=len(s),
            )

    # Top end_users by requests
    requests_by_user = df.groupby("end_user")["request_id"].count()
    _top_bar(
        requests_by_user,
        "Top end_users by requests (Apertus 70B)",
        "end_user",
        "top_users_requests.png",
        top,
    )

    # ------------------------------------------------------------------
    # Session-level usage
    # ------------------------------------------------------------------
    if "session_id" in df.columns:
        session_stats = df.groupby("session_id", dropna=True)["request_id"].count().rename("requests")
        if not session_stats.empty:
            hist(
                session_stats,
                "Requests per session (Apertus 70B)",
                "requests per session",
                "count",
                "hist_requests_per_session.png",
                out,
                bins=50,
            )
            save_csv(session_stats.reset_index(), "session_stats.csv", out)

    # ------------------------------------------------------------------
    # User concentration / activity
    # ------------------------------------------------------------------
    # Whale curve: share of requests vs user rank (optionally excluding some users)
    req_by_user = df.groupby("end_user", dropna=True)["request_id"].count()
    if exclude_users_from_whale:
        to_drop = [u for u in exclude_users_from_whale if u in req_by_user.index]
        if to_drop:
            req_by_user = req_by_user.drop(labels=to_drop)

    if not req_by_user.empty:
        whale = cumulative_share(
            df=pd.DataFrame({"end_user": req_by_user.index, "requests": req_by_user.values}),
            key="end_user",
            value_col="requests",
            positive_only=True,
            normalize_rank=True,
            dropna_key=True,
        ).rename(columns={"rank": "user_rank_frac", "cum_share": "cum_share"})
        save_csv(whale, "whale_requests.csv", out)
        line(
            "user_rank_frac",
            "cum_share",
            whale,
            "Cumulative request share vs user rank (Apertus 70B)",
            "user rank fraction",
            "cumulative share",
            "whale_requests.png",
            out,
        )

    # User activity span: active days per user
    user_stats = df.groupby("end_user", as_index=False).agg(
        requests=("request_id", "count"),
        first=("date", "min"),
        last=("date", "max"),
        active_days=("date", pd.Series.nunique),
    )
    hist(
        user_stats["active_days"],
        "Active days per user (Apertus 70B)",
        "active days",
        "count",
        "hist_user_active_days.png",
        out,
        bins=30,
    )
    top_active = (
        user_stats.sort_values("active_days", ascending=False)
        .head(top)
        .set_index("end_user")["active_days"]
    )
    if not top_active.empty:
        bar(
            top_active,
            f"Top {len(top_active)} users by active days (Apertus 70B)",
            "end_user",
            "active days",
            "top_users_active_days.png",
            out,
            top=len(top_active),
        )

    save_csv(user_stats, "user_stats.csv", out)


if __name__ == "__main__":
    plot_apertus70b_usage()
