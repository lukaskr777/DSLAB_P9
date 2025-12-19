"""
Generate an overview of the apertus-sft-mixture dataset via exploratory data analysis and plotting.

This script:
- Loads the small_train.parquet subset.
- Performs light EDA (column types, memory usage, basic statistics).
- Computes simple derived features such as string lengths.
- Produces structural, compositional, temporal, correlation, and aggregated metric plots.
- Saves all figures into figs/apertus_overview.
"""

from pathlib import Path
import json

import pandas as pd

from utility_scripts.file_utils import ensure_empty_dir
from utility_scripts.plot_utils import hist, bar, line, scatter
from utility_scripts.df_reading_utils import ensure_date_column


DATA_PATH = Path("data/swiss-ai_apertus-sft-mixture/small_train.parquet")
OUTDIR = Path("figs/apertus_overview")


def load_df(path: Path) -> pd.DataFrame:
    """Load a DataFrame from a Parquet file and print its shape."""
    print("\n=== Loading DataFrame ===")
    df = pd.read_parquet(path)
    print(f"Loaded shape: {df.shape}")
    return df


def basic_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Print basic EDA information and add simple *_len features for object/string columns."""
    print("\n=== Column names ===")
    print(list(df.columns))

    print("\n=== Dtypes ===")
    print(df.dtypes)

    print("\n=== Memory usage (MB) ===")
    print(df.memory_usage(deep=True).sum() / 1024**2)

    # Group columns by dtype (useful for deciding next steps)
    dtype_groups = df.columns.to_series().groupby(df.dtypes).apply(list)
    print("\n=== Columns grouped by dtype ===")
    for dtype, cols in dtype_groups.items():
        print(f"{dtype}: {cols}")

    # ---- Light text-derived features ----
    print("\n=== Computing text length features ===")
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        # String length per row; nulls → NaN
        df[f"{col}_len"] = df[col].astype("string").str.len()

    # Numeric columns for describe
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols):
        print("\n=== describe() on numeric columns ===")
        print(df[num_cols].describe())
    else:
        print("\n=== No numeric columns to describe() ===")

    return df


# -------------------------------------------------------------------
# 1. Conversation-level structural plots
# -------------------------------------------------------------------

def plot_messages_per_conversation(df: pd.DataFrame, outdir: Path) -> None:
    """Histogram of messages_len per conversation."""
    hist(
        series=df["messages_len"],
        title="Messages per Conversation",
        xlabel="messages_len",
        ylabel="count",
        fname="hist_messages_len",
        outdir=outdir,
        bins=50,
        log_scale=True,
    )


def plot_id_length_distribution(df: pd.DataFrame, outdir: Path) -> None:
    """Histogram of conversation_id string lengths."""
    hist(
        series=df["conversation_id_len"],
        title="Length of conversation_id",
        xlabel="conversation_id_len",
        ylabel="count",
        fname="hist_conversation_id_len",
        outdir=outdir,
        bins=40,
    )


def plot_dataset_source_length(df: pd.DataFrame, outdir: Path) -> None:
    """Histogram of dataset_source string lengths."""
    hist(
        series=df["dataset_source_len"],
        title="Length of dataset_source strings",
        xlabel="dataset_source_len",
        ylabel="count",
        fname="hist_dataset_source_len",
        outdir=outdir,
        bins=30,
    )


# -------------------------------------------------------------------
# 2. Dataset composition plots
# -------------------------------------------------------------------

def plot_dataset_source_counts(df: pd.DataFrame, outdir: Path) -> None:
    """Bar chart of dataset_source frequencies."""
    counts = df["dataset_source"].value_counts(dropna=False)
    bar(
        series=counts,
        title="Dataset Source Frequency",
        xlabel="dataset_source",
        ylabel="count",
        fname="bar_dataset_source_counts",
        outdir=outdir,
        top=None,
    )


def plot_original_metadata_top(df: pd.DataFrame, outdir: Path) -> None:
    """
    Bar chart of the most common original_metadata values.

    original_metadata can be dict-like, so we convert it to a stable JSON string representation before counting.
    """
    meta_str = df["original_metadata"].apply(
        lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else str(x)
    )
    counts = meta_str.value_counts(dropna=False)

    bar(
        series=counts,
        title="Top original_metadata Values",
        xlabel="original_metadata (stringified)",
        ylabel="count",
        fname="bar_original_metadata_top20",
        outdir=outdir,
        top=20,
    )


# -------------------------------------------------------------------
# 3. Temporal structure
# -------------------------------------------------------------------

def _with_created_dt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a UTC-normalized created_dt column using ensure_date_column.
    The output is always sorted and NaN timestamps removed.
    """
    return ensure_date_column(
        df,
        time_col="created_timestamp",
        out_col="created_dt",
        floor=None,       # => normalized UTC date
        dropna=True,
        sort=True,
    )


def plot_conversations_over_time(df: pd.DataFrame, outdir: Path) -> None:
    """Line plot of unique conversations per calendar day."""
    tmp = _with_created_dt(df)

    counts = (
        tmp.groupby("created_dt", observed=True)["conversation_id"]
        .nunique()
        .reset_index(name="n_conversations")
        .sort_values("created_dt")
    )

    line(
        x="created_dt",
        y="n_conversations",
        data=counts,
        title="Conversations per Day",
        xlabel="Date",
        ylabel="Number of unique conversation_id",
        fname="line_conversations_per_day",
        outdir=outdir,
    )


def plot_day_of_week_hist(df: pd.DataFrame, outdir: Path) -> None:
    """Bar chart of conversations by weekday."""
    tmp = _with_created_dt(df)
    tmp["weekday"] = tmp["created_dt"].dt.day_name()

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tmp["weekday"] = pd.Categorical(tmp["weekday"], categories=weekdays, ordered=True)

    counts = tmp["weekday"].value_counts().sort_index()

    bar(
        series=counts,
        title="Conversations by Day of Week",
        xlabel="weekday",
        ylabel="count",
        fname="bar_conversations_by_weekday",
        outdir=outdir,
    )


# -------------------------------------------------------------------
# 4. Correlation plots
# -------------------------------------------------------------------

def plot_messages_vs_source_length(df: pd.DataFrame, outdir: Path) -> None:
    """Scatter of messages_len vs dataset_source_len."""
    scatter(
        x=df["dataset_source_len"],
        y=df["messages_len"],
        title="messages_len vs dataset_source_len",
        xlabel="dataset_source_len",
        ylabel="messages_len",
        fname="scatter_messages_vs_dataset_source_len",
        outdir=outdir,
    )


def plot_timestamp_len_vs_messages(df: pd.DataFrame, outdir: Path) -> None:
    """Scatter of messages_len vs created_timestamp_len."""
    scatter(
        x=df["created_timestamp_len"],
        y=df["messages_len"],
        title="messages_len vs created_timestamp_len",
        xlabel="created_timestamp_len",
        ylabel="messages_len",
        fname="scatter_messages_vs_created_timestamp_len",
        outdir=outdir,
    )


# -------------------------------------------------------------------
# 5. Source-level aggregated metrics
# -------------------------------------------------------------------

def plot_mean_messages_per_source(df: pd.DataFrame, outdir: Path) -> None:
    """Bar chart of mean messages_len per dataset_source."""
    means = (
        df.groupby("dataset_source", observed=True)["messages_len"]
        .mean()
        .sort_values(ascending=False)
    )
    bar(
        series=means,
        title="Average Conversation Length per Dataset Source",
        xlabel="dataset_source",
        ylabel="mean(messages_len)",
        fname="bar_mean_messages_per_dataset_source",
        outdir=outdir,
    )


def main() -> None:
    ensure_empty_dir(OUTDIR)
    df = load_df(DATA_PATH)
    df = basic_eda(df)

    # Structural
    plot_messages_per_conversation(df, OUTDIR)
    plot_id_length_distribution(df, OUTDIR)
    plot_dataset_source_length(df, OUTDIR)

    # Composition
    plot_dataset_source_counts(df, OUTDIR)
    plot_original_metadata_top(df, OUTDIR)

    # Temporal
    plot_conversations_over_time(df, OUTDIR)
    plot_day_of_week_hist(df, OUTDIR)

    # Correlations
    plot_messages_vs_source_length(df, OUTDIR)
    plot_timestamp_len_vs_messages(df, OUTDIR)

    # Aggregated metrics
    plot_mean_messages_per_source(df, OUTDIR)

    print(f"Saved plots to: {OUTDIR}")


if __name__ == "__main__":
    main()
