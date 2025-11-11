"""
Exploratory plots for LiteLLM_EndUserTable.
Writes figures to figs/litellm_end_user_table.

Outputs:
- Histogram: user spend distribution
- Histogram: log10(user spend) for spend > 0
- Bar: Top-N users by spend (spend > 0)
- Bar: Spend by budget_id
- Bar: User count by budget_id
- Bar: Spend by blocked flag
- Bar: User count by blocked flag
- Line: Cumulative share of spend vs user rank (whale curve, spend > 0)
"""

import numpy as np
import pandas as pd
from utility_scripts.file_utils import PathLike, read_table, ensure_empty_dir
from utility_scripts.plot_utils import line, bar, hist


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["budget_id"] = out["budget_id"].fillna("").replace("", "unknown_budget")
    out["user_id"] = out["user_id"].fillna("").replace("", "unknown_user")
    out["blocked"] = out["blocked"].fillna(False).astype(bool)
    out["spend"] = pd.to_numeric(out["spend"], errors="coerce").fillna(0.0)
    out.loc[out["spend"] < 0, "spend"] = 0.0
    return out


def _top_by_spend(df: pd.DataFrame, key: str, top: int) -> pd.Series:
    s = df.groupby(key)["spend"].sum().sort_values(ascending=False)
    s = s[s > 0]
    return s.head(top)


def _cumulative_share(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby("user_id")["spend"].sum()
    s = s[s > 0].sort_values(ascending=False)
    if s.empty:
        return pd.DataFrame({"rank": [], "cum_share": []})
    cum = s.cumsum() / s.sum()
    rank = np.arange(1, len(s) + 1)
    n = len(s)
    return pd.DataFrame({"rank": rank / n, "cum_share": cum.values})


def plot_all_litellm_end_user_table(
    dir_name: PathLike = "data",
    dataset: str = "litellm",
    outdir: PathLike = "figs/litellm_end_user_table",
    top: int = 10,
) -> None:
    out = ensure_empty_dir(outdir)

    df = read_table("LiteLLM_EndUserTable", dataset=dataset, dir_name=dir_name)
    if df.empty:
        raise SystemExit("No data loaded from LiteLLM_EndUserTable.")
    needed = {"user_id", "spend", "budget_id", "blocked"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = _clean(df)

    hist(
        df["spend"],
        title="User spend distribution",
        xlabel="spend per end-user",
        ylabel="count of users",
        fname="hist_user_spend.png",
        outdir=out,
        bins=50
    )

    nonzero = df.loc[df["spend"] > 0, "spend"]
    if not nonzero.empty:
        hist(
            pd.Series(np.log10(nonzero)),
            title="Log10 user spend distribution (spend > 0)",
            xlabel="log10(spend)",
            ylabel="count of users",
            fname="hist_user_spend_log10.png",
            outdir=out,
            bins=min(50, max(10, nonzero.nunique()))
        )

    top_users = _top_by_spend(df, "user_id", top)
    if not top_users.empty:
        bar(
            top_users,
            title=f"Top {len(top_users)} users by total spend",
            xlabel="user_id",
            ylabel="total spend",
            fname=f"top{len(top_users)}_users_spend.png",
            outdir=out,
            top=len(top_users)
        )

    spend_by_budget = df.groupby("budget_id")["spend"].sum().sort_values(ascending=False)
    bar(
        spend_by_budget,
        title="Total spend by budget_id",
        xlabel="budget_id",
        ylabel="total spend",
        fname="spend_by_budget.png",
        outdir=out,
        top=None,
    )

    count_by_budget = df["budget_id"].value_counts()
    bar(
        count_by_budget,
        title="User count by budget_id",
        xlabel="budget_id",
        ylabel="count of users",
        fname="count_by_budget.png",
        outdir=out,
        top=None,
    )

    spend_by_blocked = df.groupby("blocked")["spend"].sum().sort_values(ascending=False)
    bar(
        spend_by_blocked,
        title="Total spend by blocked flag",
        xlabel="blocked",
        ylabel="total spend",
        fname="spend_by_blocked.png",
        outdir=out,
        top=None,
    )

    count_by_blocked = df["blocked"].value_counts().sort_values(ascending=False)
    bar(
        count_by_blocked,
        title="User count by blocked flag",
        xlabel="blocked",
        ylabel="count of users",
        fname="count_by_blocked.png",
        outdir=out,
        top=None,
    )

    cum = _cumulative_share(df)
    if not cum.empty:
        line(
            x="rank",
            y="cum_share",
            data=cum,
            title="Cumulative share of spend vs user rank (spend > 0)",
            xlabel="user rank fraction",
            ylabel="cumulative spend share",
            fname="cumulative_spend_share.png",
            outdir=out,
        )
