import warnings
from pathlib import Path
from typing import List, Optional, Union, Sequence

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


PathLike = Union[str, Path]


def read_table(name: str, dataset: str = "litellm", dir_name: PathLike = "data") -> pd.DataFrame:
    """
    Load and concatenate all Parquet parts for a logical table.

    Directory layout expected:
        {dir_name}/{dataset}/public.{name}/1/*.parquet

    Args:
        name: Table name without the "public." prefix.
        dataset: Dataset directory under dir_name.
        dir_name: Root data directory.

    Returns:
        A DataFrame containing all rows from the found Parquet files.
        Returns an empty DataFrame if the directory exists but has no Parquet files.

    Raises:
        FileNotFoundError: If the expected directory does not exist.
    """
    inner_dir = Path(dir_name) / dataset / f"public.{name}" / "1"
    if not inner_dir.exists():
        raise FileNotFoundError(f"Missing directory: {inner_dir}")

    parts: List[pd.DataFrame] = []
    for f in inner_dir.iterdir():
        if f.suffix == ".parquet":
            parts.append(pd.read_parquet(f))

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def ensure_outdir(outdir: PathLike) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_empty_dir(outdir: PathLike) -> Path:
    out = ensure_outdir(outdir)
    for p in out.iterdir():
        if p.is_file():
            p.unlink()
    return out


def _tilt_and_crop_ticklabels(
    ax: Axes,
    axes: Sequence[str] = ("x",),
    *,
    tilt_thresh: int = 10,
    crop_thresh: int = 25,
    max_len: int = 25,
    rotation: int = 35,
) -> None:
    """
    Tilt mildly long tick labels and crop very long ones with an ellipsis.

    Args:
        ax: Target axes.
        axes: Iterable over {"x","y"} to apply formatting.
        tilt_thresh: Rotate labels if any length >= this.
        crop_thresh: Crop labels whose length >= this.
        max_len: Max length after cropping (ellipsis counts as 1).
        rotation: Rotation angle in degrees.
    """
    ax.figure.canvas.draw()

    for axis in axes:
        if axis == "x":
            texts = ax.get_xticklabels()
        elif axis == "y":
            texts = ax.get_yticklabels()
        else:
            continue

        any_long = False
        for t in texts:
            s = t.get_text()
            if not s:
                continue
            if len(s) >= crop_thresh:
                s = (s[: max(1, max_len - 1)] + "…") if len(s) > max_len else s
                t.set_text(s)
            if len(s) >= tilt_thresh:
                any_long = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if axis == "x":
                ax.set_xticklabels([t.get_text() for t in texts])
            else:
                ax.set_yticklabels([t.get_text() for t in texts])

        if any_long:
            if axis == "x":
                plt.setp(texts, rotation=rotation, ha="right", rotation_mode="anchor")
            else:
                plt.setp(texts, rotation=rotation, ha="right", rotation_mode="anchor")



def line(x: str, y: str, data: pd.DataFrame, title: str, xlabel: str, ylabel: str, fname: PathLike, outdir: PathLike) -> Path:
    """
    Save a line plot from a DataFrame.

    Args:
        x: Column for x-axis.
        y: Column for y-axis.
        data: Source DataFrame.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        fname: Output filename.
        outdir: Output directory.

    Returns:
        Path to the saved image.

    Raises:
        KeyError: If `x` or `y` is missing in `data`.
    """
    if x not in data.columns or y not in data.columns:
        missing = {c for c in (x, y) if c not in data.columns}
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    out_path = ensure_outdir(outdir) / str(fname)
    fig, ax = plt.subplots()
    data.plot(x=x, y=y, kind="line", legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def bar(series: pd.Series, title: str, xlabel: str, ylabel: str, fname: PathLike, outdir: PathLike, top: Optional[int] = None) -> Path:
    """
    Save a bar chart from a Series aggregated by index.

    Args:
        series: Input Series (index will be grouped).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        fname: Output filename.
        outdir: Output directory.
        top: Keep only top-N largest after aggregation. If None, keep all.

    Returns:
        Path to the saved image.
    """
    s = series.groupby(series.index).sum().sort_values(ascending=False)
    if top is not None:
        s = s.head(top)

    out_path = ensure_outdir(outdir) / str(fname)
    fig, ax = plt.subplots()
    s.plot(kind="bar", rot=45, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def hist(series: pd.Series, title: str, xlabel: str, ylabel: str, fname: PathLike, outdir: PathLike, bins: int = 50) -> Path:
    """
    Save a histogram from a Series.

    Args:
        series: Input Series.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        fname: Output filename.
        outdir: Output directory.
        bins: Number of histogram bins. Defaults to 50.

    Returns:
        Path to the saved image.
    """
    out_path = ensure_outdir(outdir) / str(fname)
    fig, ax = plt.subplots()
    series.plot(kind="hist", bins=bins, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
