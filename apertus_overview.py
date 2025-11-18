from pathlib import Path
import pandas as pd
import json

from utility_scripts.file_utils import ensure_empty_dir
from utility_scripts.plot_utils import hist, bar, line, scatter


DATA_PATH = Path("data/swiss-ai_apertus-sft-mixture/train_sampled.parquet")
OUTDIR = Path("figs/apertus_overview")


def load_df(path: Path) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    print("\n=== Loading DataFrame ===")
    df = pd.read_parquet(path)
    print(f"Loaded shape: {df.shape}")
    return df


def basic_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic exploratory data analysis on the DataFrame."""
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
    text_cols = [c for c in df.columns if df[c].dtype == object]

    for col in text_cols:
        # len of each string; errors=ignore covers NaN
        df[f"{col}_len"] = df[col].str.len()

    # Numeric columns for safe describe
    num_cols = df.select_dtypes(include=["number"]).columns

    print("\n=== describe() on numeric columns ===")
    print(df[num_cols].describe())

    return df


# -------------------------------------------------------------------
# 1. Conversation-level structural plots
# -------------------------------------------------------------------

def plot_messages_per_conversation(df: pd.DataFrame, outdir: Path) -> None:
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

    original_metadata can be dict-like, so we convert it to a stable string
    representation (JSON) before counting.
    """
    # Convert dicts (and other types) to JSON / string so they become hashable
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

def plot_conversations_over_time(df: pd.DataFrame, outdir: Path) -> None:
    tmp = df.copy()
    tmp["created_dt"] = pd.to_datetime(tmp["created_timestamp"], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=["created_dt"])
    tmp["day"] = tmp["created_dt"].dt.floor("D")

    counts = (
        tmp.groupby("day", observed=True)["conversation_id"]
        .nunique()
        .reset_index(name="n_conversations")
        .sort_values("day")
    )

    line(
        x="day",
        y="n_conversations",
        data=counts,
        title="Conversations per Day",
        xlabel="Date",
        ylabel="Number of unique conversation_id",
        fname="line_conversations_per_day",
        outdir=outdir,
    )


def plot_day_of_week_hist(df: pd.DataFrame, outdir: Path) -> None:
    tmp = df.copy()
    tmp["created_dt"] = pd.to_datetime(tmp["created_timestamp"], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=["created_dt"])
    tmp["weekday"] = tmp["created_dt"].dt.day_name()

    counts = tmp["weekday"].value_counts()
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


# -------------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------------

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
