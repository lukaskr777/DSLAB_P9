import warnings
from pathlib import Path
from typing import Optional, Union, Sequence, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scripts.utils import PathLike, ensure_outdir, save_text


def _tilt_and_crop_ticklabels(
    ax: Axes,
    axes: Sequence[str] = ("x",),
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
        tilt_thresh: Rotate labels if any length >= this. Defaults to 10.
        crop_thresh: Crop labels whose length >= this. Defaults to 25.
        max_len: Max length after cropping (ellipsis counts as 1). Defaults to 25.
        rotation: Rotation angle in degrees. Defaults to 35.
    """
    ax.figure.canvas.draw()

    for axis in axes:
        if axis == "x":
            texts = ax.get_xticklabels()
            setter = ax.set_xticklabels
        elif axis == "y":
            texts = ax.get_yticklabels()
            setter = ax.set_yticklabels
        else:
            continue

        any_long = False
        new_texts = []
        changed = False

        for t in texts:
            s = t.get_text()
            if not s:
                new_texts.append(s)
                continue

            orig = s
            if len(s) >= crop_thresh:
                s = s[: max(1, max_len - 1)] + "…" if len(s) > max_len else s
            if len(s) >= tilt_thresh:
                any_long = True

            new_texts.append(s)
            changed |= (s != orig)

        if changed:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                setter(new_texts)

        if any_long:
            plt.setp(texts, rotation=rotation, ha="right", rotation_mode="anchor")
        else:
            plt.setp(texts, rotation=0, ha="center", rotation_mode=None)
            ax.tick_params(axis=axis, labelrotation=0)


def line(
    x: str, y: str, data: pd.DataFrame, title: str, xlabel: str, ylabel: str, fname: PathLike, outdir: PathLike
) -> Path:
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


def bar(
    series: pd.Series, 
    title: str, 
    xlabel: str,
    ylabel: str, 
    fname: PathLike, 
    outdir: PathLike, 
    top: Optional[int] = None,
    order: Optional[Sequence] = None,
    sort_values: bool = True,
    ascending: bool = False,
) -> Path:
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
        order: Optional sequence to reindex the Series before plotting.
        sort_values: Whether to sort by values before taking top-N. Defaults to True.
        ascending: If sorting, whether to sort in ascending order. Defaults to False.

    Returns:
        Path to the saved image.
    """
    s = series.groupby(series.index).sum() if series.index.has_duplicates else series.copy()

    if order is not None:
        idx = pd.CategoricalIndex(s.index, categories=list(order), ordered=True)
        s = pd.Series(s.values, index=idx).sort_index()
    elif sort_values:
        s = s.sort_values(ascending=ascending)

    if top is not None:
        s = s.head(top)

    out_path = ensure_outdir(outdir) / str(fname)
    fig, ax = plt.subplots()
    s.plot(kind="bar", rot=0, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def hist(
    series: pd.Series, 
    title: str, 
    xlabel: str, 
    ylabel: str, 
    fname: PathLike, 
    outdir: PathLike, 
    bins: int = 50, 
    log_scale: bool = False
) -> Path:
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
        log_scale: Whether to use logarithmic scale for y-axis. Defaults to False.

    Returns:
        Path to the saved image.
    """
    out_path = ensure_outdir(outdir) / str(fname)
    fig, ax = plt.subplots()
    series.plot(kind="hist", bins=bins, ax=ax)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def scatter(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    annotate: Optional[pd.Series] = None,
) -> Path:
    """
    Save a scatter plot from two Series.

    Args:
        x: X-axis values.
        y: Y-axis values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        fname: Output filename.
        outdir: Output directory.
        annotate: Optional labels for each point. Must match len(x).

    Returns:
        Path to the saved image.

    Raises:
        ValueError: If x and y lengths differ, or annotate length mismatches.
    """
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: len(x)={len(x)} != len(y)={len(y)}")
    if annotate is not None and len(annotate) != len(x):
        raise ValueError(f"Length mismatch: len(annotate)={len(annotate)} != len(x)={len(x)}")

    out_path = ensure_outdir(outdir) / str(fname)
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=12)
    if annotate is not None:
        for xi, yi, lab in zip(x, y, annotate):
            ax.annotate(str(lab), (xi, yi), fontsize=6, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def heatmap(
    pivot: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
) -> Path:
    """
    Save a heatmap from a pivoted DataFrame (index → rows, columns → cols).

    Args:
        pivot: 2D DataFrame with numeric values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        fname: Output filename.
        outdir: Output directory.

    Returns:
        Path to the saved image.
    """
    out_path = ensure_outdir(outdir) / str(fname)
    fig, ax = plt.subplots()
    im = ax.imshow(pivot.values, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax, axes=("x", "y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def scatter_with_fit(
    x: Union[Sequence[float], np.ndarray, pd.Series],
    y: Union[Sequence[float], np.ndarray, pd.Series],
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    write_params_path: Optional[str] = None,
) -> Path:
    """
    Scatter with least-squares fit (y = a*x + b). Slope added to title.

    Args:
        x, y: Data vectors. NaNs/inf are dropped pairwise.
        title: Plot title (slope appended).
        xlabel: X-axis label.
        ylabel: Y-axis label.
        fname: Output filename.
        outdir: Output directory.
        write_params_path: If set, also save a text file with a and b.

    Returns:
        Path to the saved image.

    Raises:
        ValueError: If input lengths differ or fewer than 2 valid points.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Length mismatch: len(x)={len(x_arr)} != len(y)={len(y_arr)}")

    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    xv = x_arr[m]
    yv = y_arr[m]
    if xv.size < 2:
        raise ValueError("Need at least 2 finite points for regression.")

    a, b = np.polyfit(xv, yv, 1)

    out_path = ensure_outdir(outdir) / str(fname)
    fig, ax = plt.subplots()
    ax.scatter(xv, yv, s=12)

    xs = np.linspace(xv.min(), xv.max(), 100)
    ys = a * xs + b
    ax.plot(xs, ys)

    ax.set_title(f"{title} — slope={a:.6g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if write_params_path is not None:
        save_text(f"slope={a}\nintercept={b}\n", write_params_path, outdir)

    return out_path


def pareto_frontier_plot(
    x: Union[Sequence[float], np.ndarray, pd.Series],
    y: Union[Sequence[float], np.ndarray, pd.Series],
    labels: Optional[Union[Sequence[str], np.ndarray, pd.Series]],
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
) -> Path:
    """
    Scatter of points and lower-left Pareto frontier for minimizing both axes.

    Args:
        x, y: Coordinates. NaNs/inf are dropped pairwise.
        labels: Optional point labels for annotations.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        fname: Output filename.
        outdir: Output directory.

    Returns:
        Path to the saved image.

    Raises:
        ValueError: If input lengths differ or 0 valid points.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Length mismatch: len(x)={len(x_arr)} != len(y)={len(y_arr)}")

    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    xv = x_arr[m]
    yv = y_arr[m]
    if labels is not None:
        labs = np.asarray(labels, dtype=object)[m]
    else:
        labs = None

    if xv.size == 0:
        raise ValueError("No finite points to plot.")

    # Compute frontier: increasing x, strictly improving (lower) y.
    order = np.argsort(xv)
    best_y = np.inf
    frontier_idx: List[int] = []
    for i in order:
        yi = yv[i]
        if yi < best_y:
            best_y = yi
            frontier_idx.append(i)

    out_path = ensure_outdir(outdir) / str(fname)
    fig, ax = plt.subplots()
    ax.scatter(xv, yv, s=16)

    if frontier_idx:
        fxs = xv[frontier_idx]
        fys = yv[frontier_idx]
        # Ensure frontier drawn in increasing x
        order_f = np.argsort(fxs)
        ax.plot(fxs[order_f], fys[order_f])

    if labs is not None:
        for xi, yi, lab in zip(xv, yv, labs):
            ax.annotate(str(lab), (xi, yi), fontsize=6, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path