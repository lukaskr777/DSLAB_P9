import re
from pathlib import Path
import shutil

import pandas as pd

PathLike = str | Path


def ensure_outdir(outdir: PathLike) -> Path:
    """Ensure directory exists. Return Path."""
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_empty_dir(outdir: PathLike) -> Path:
    """Ensure directory exists, then remove its contents. Return Path. Dangerous!"""
    p = Path(outdir)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=False)
    return p


def sanitize_fname(name: PathLike) -> str:
    """Replace path separators and unsafe characters. Keep length reasonable."""
    s = str(name).replace("/", "_").replace("\\", "_")
    s = re.sub(r'[^A-Za-z0-9._\-+() ]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"_+", "_", s)
    return s[:200]


def make_outpath(fname: PathLike, outdir: PathLike, ext: str | None = None) -> Path:
    """Build a sanitized output path with the given extension. """
    name = sanitize_fname(fname)
    if ext is not None:
        if not ext.startswith("."):
            raise ValueError(f"ext must start with '.': {ext}")
        stem, dot, suffix = name.rpartition(".")
        if not dot or suffix.lower() != ext.lower().lstrip("."):
            name = (stem or name) + ext
    return ensure_outdir(outdir) / name


def read_table(name: str, dataset: str = "litellm", dir_name: PathLike = "data") -> pd.DataFrame:
    """
    Load and concatenate all Parquet parts for a logical table.

    Expected layout:
        {dir_name}/{dataset}/public.{name}/1/*.parquet

    Returns:
        A DataFrame containing all rows from the found Parquet files.
        Returns an empty DataFrame if the directory exists but has no Parquet files.
    """
    base = Path(dir_name) / dataset / f"public.{name}" / "1"
    if not base.exists():
        raise FileNotFoundError(f"Missing directory: {base}")

    files = sorted(base.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()

    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def save_csv(obj: pd.DataFrame | pd.Series, fname: PathLike, outdir: PathLike) -> Path:
    """Save a DataFrame or Series to CSV at {outdir}/{fname}. Only write index if non-default. Return saved path."""
    out_path = make_outpath(fname, outdir, ext=".csv")

    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=obj.name or "value")
        write_index = True
    else:
        df = obj
        write_index = not (
            isinstance(df.index, pd.RangeIndex) 
            and df.index.name is None 
            and df.index.start == 0 
            and df.index.step == 1
        )

    df.to_csv(out_path, index=write_index, encoding="utf-8")
    return out_path


def save_text(text: str, fname: PathLike, outdir: PathLike) -> Path:
    """Save text to {outdir}/{fname}. Return saved path."""
    out_path = make_outpath(fname, outdir, ext=".txt")
    out_path.write_text(text, encoding="utf-8")
    return out_path
