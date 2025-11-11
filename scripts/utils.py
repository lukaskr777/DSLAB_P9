import re
from pathlib import Path
from typing import Optional

import pandas as pd


PathLike = str | Path


def read_table(name: str, dataset: str = "litellm", dir_name: PathLike = "data") -> pd.DataFrame:
    """
    Load and concatenate all Parquet parts for a logical table.

    Directory layout expected:
        {dir_name}/{dataset}/public.{name}/1/*.parquet

    Args:
        name: Table name without the "public." prefix.
        dataset: Dataset directory under dir_name. Defaults to "litellm".
        dir_name: Root data directory. Defaults to "data".

    Returns:
        A DataFrame containing all rows from the found Parquet files.
        Returns an empty DataFrame if the directory exists but has no Parquet files.
    """
    inner_dir = Path(dir_name) / dataset / f"public.{name}" / "1"
    if not inner_dir.exists():
        raise FileNotFoundError(f"Missing directory: {inner_dir}")

    parts: list[pd.DataFrame] = []
    for f in inner_dir.iterdir():
        if f.suffix == ".parquet":
            parts.append(pd.read_parquet(f))

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def ensure_outdir(outdir: PathLike) -> Path:
    """Ensure that outdir exists. Return Path object."""
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_empty_dir(outdir: PathLike) -> Path:
    """Ensure that outdir exists. Delete all files in outdir and return Path object."""
    out = ensure_outdir(outdir)
    for p in out.iterdir():
        if p.is_file():
            p.unlink()
    return out


def save_csv(obj: pd.DataFrame | pd.Series, fname: PathLike, outdir: PathLike, index: Optional[bool] = None) -> Path:
    """
    Save a DataFrame or Series to CSV in {outdir}/{fname}.

    Args:
        obj: DataFrame or Series to write.
        fname: Output filename.
        outdir: Output directory.
        index: Whether to write the index. If None, use False for DataFrame without meaningful index and True otherwise.

    Returns:
        Path to the saved CSV.
    """
    out_path = ensure_outdir(outdir) / str(fname)
    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=obj.name or "value")
        write_index = True if index is None else index
        df.to_csv(out_path, index=write_index)
    else:
        # Heuristic: write index only if it has a name or is non-default
        default_range = isinstance(obj.index, pd.RangeIndex)
        write_index = (not default_range) if index is None else index
        obj.to_csv(out_path, index=write_index)
    return out_path


def save_text(text: str, fname: PathLike, outdir: PathLike) -> Path:
    """Save text to {outdir}/{fname}. Return Path to the saved file."""
    out_path = ensure_outdir(outdir) / str(fname)
    Path(out_path).write_text(text)
    return out_path


def sanitize_fname(name: str) -> str:
    """Replace path separators and unsafe characters. Keep length reasonable."""
    s = str(name).replace("/", "_").replace("\\", "_")
    s = re.sub(r'[^A-Za-z0-9._\-+() ]+', "_", s)
    return s[:200]


def to_dt(s: pd.Series) -> pd.Series:
    """Convert a Series to datetime in UTC, coercing errors to NaT."""
    return pd.to_datetime(s, errors="coerce", utc=True)
