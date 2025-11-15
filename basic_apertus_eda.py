import pandas as pd
from pathlib import Path

PATH = Path("data/swiss-ai_apertus-sft-mixture/train_sampled.parquet")


def basic_eda(path: Path) -> pd.DataFrame:
    print("\n=== Loading DataFrame ===")
    df = pd.read_parquet(path)
    print(f"Loaded shape: {df.shape}")

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

    # Save enriched version
    out_path = path.with_name(path.stem + "_enriched.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved enriched dataset to: {out_path}")

    return df


if __name__ == "__main__":
    basic_eda(PATH)
