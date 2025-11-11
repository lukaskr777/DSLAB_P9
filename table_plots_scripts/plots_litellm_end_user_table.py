"""
Exploratory plots for LiteLLM_EndUserTable.
Writes figures to figs/litellm_end_user_table.
"""

import numpy as np
import pandas as pd

from utility_scripts.file_utils import PathLike, read_table, ensure_empty_dir, save_csv
from utility_scripts.plot_utils import line, bar, hist
from utility_scripts.df_reading_utils import (
    require,
    clean_table,
    top_by_value,
    cumulative_share,
    fill_missing,
    coerce_numeric,
)


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

    need = {"user_id", "spend", "budget_id", "blocked"}
    miss = require(df, need, strict=False)
    if miss:
        raise SystemExit(f"Missing required columns: {sorted(miss)}")

    # Standardize types and values:
    # - clamp negative spend; coerce to numeric
    # - normalize strings for ids
    # - coerce blocked to bool with NaN->False
    df = clean_table(df, strict=False)  # only touches 'spend' present here
    df = coerce_numeric(df, cols=["spend"])
    df = fill_missing(
        df,
        fill_str={"user_id": "unknown_user", "budget_id": "unknown_budget"},
        fill_bool={"blocked": False},
    )
    df["blocked"] = df["blocked"].astype(bool)

    # Keep a snapshot CSV for debugging or external analysis
    save_csv(df, "end_user_table_clean.csv", out)

    # Histogram: user spend distribution
    s_spend = pd.to_numeric(df["spend"], errors="coerce").dropna()
    if not s_spend.empty:
        hist(
            s_spend,
            title="User spend distribution",
            xlabel="spend per end-user",
            ylabel="count of users",
            fname="hist_user_spend.png",
            outdir=out,
            bins=50,
        )

        # Histogram: log10(user spend) for spend > 0
        nonzero = s_spend[s_spend > 0]
        if not nonzero.empty:
            hist(
                pd.Series(np.log10(nonzero), index=nonzero.index),
                title="Log10 user spend distribution (spend > 0)",
                xlabel="log10(spend)",
                ylabel="count of users",
                fname="hist_user_spend_log10.png",
                outdir=out,
                bins=min(50, max(10, int(nonzero.nunique()))),
            )

    # Bar: Top-N users by spend (spend > 0)
    top_users = top_by_value(df[df["spend"] > 0], "user_id", "spend", top=top)
    if not top_users.empty:
        bar(
            top_users,
            title=f"Top {len(top_users)} users by total spend",
            xlabel="user_id",
            ylabel="total spend",
            fname=f"top{len(top_users)}_users_spend.png",
            outdir=out,
            top=len(top_users),
        )

    # Bar: Spend by budget_id
    spend_by_budget = df.groupby("budget_id", dropna=False)["spend"].sum().sort_values(ascending=False)
    if not spend_by_budget.empty:
        bar(
            spend_by_budget,
            title="Total spend by budget_id",
            xlabel="budget_id",
            ylabel="total spend",
            fname="spend_by_budget.png",
            outdir=out,
            top=None,
        )

    # Bar: User count by budget_id
    count_by_budget = df["budget_id"].astype("string").value_counts()
    if not count_by_budget.empty:
        bar(
            count_by_budget,
            title="User count by budget_id",
            xlabel="budget_id",
            ylabel="count of users",
            fname="count_by_budget.png",
            outdir=out,
            top=None,
        )

    # Bar: Spend by blocked flag
    spend_by_blocked = df.groupby("blocked", dropna=False)["spend"].sum().sort_values(ascending=False)
    if not spend_by_blocked.empty:
        bar(
            spend_by_blocked,
            title="Total spend by blocked flag",
            xlabel="blocked",
            ylabel="total spend",
            fname="spend_by_blocked.png",
            outdir=out,
            top=None,
        )

    # Bar: User count by blocked flag
    count_by_blocked = df["blocked"].astype(bool).value_counts().sort_values(ascending=False)
    if not count_by_blocked.empty:
        bar(
            count_by_blocked,
            title="User count by blocked flag",
            xlabel="blocked",
            ylabel="count of users",
            fname="count_by_blocked.png",
            outdir=out,
            top=None,
        )

    # Line: Cumulative share of spend vs user rank (spend > 0)
    whale = cumulative_share(df, key="user_id", value_col="spend", positive_only=True, normalize_rank=True)
    if not whale.empty:
        line(
            x="rank",
            y="cum_share",
            data=whale,
            title="Cumulative share of spend vs user rank (spend > 0)",
            xlabel="user rank fraction",
            ylabel="cumulative spend share",
            fname="cumulative_spend_share.png",
            outdir=out,
        )
