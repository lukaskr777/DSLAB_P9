from pathlib import Path
from typing import Optional, Sequence 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FixedFormatter

from utility_scripts.file_utils import PathLike, make_outpath, save_text


def _tilt_and_crop_ticklabels(
    ax: Axes,
    x_axis: bool = True,
    y_axis: bool = False,
    tilt_thresh: int = 10,
    crop_thresh: int = 25,
    rotation: int = 35,
) -> None:
    """
    Tilt mildly long tick labels and crop very long ones with an ellipsis.

    Args:
        ax: Target axes.
        x_axis: Whether to process the X axis.
        y_axis: Whether to process the Y axis.
        tilt_thresh: Rotate labels if any length >= this.
        crop_thresh: Crop labels whose length >= this.
        rotation: Rotation angle in degrees.
    """
    ax.figure.canvas.draw()

    if x_axis:
        x_ticks = [float(t) for t in ax.get_xticks()]
        x_texts = [t.get_text() for t in ax.get_xticklabels()]
        any_long_x = any(len(s or "") >= tilt_thresh for s in x_texts)
        x_labels: list[str] = []
        for s in x_texts:
            if len("" if s is None else str(s)) < crop_thresh:
                x_labels.append("" if s is None else str(s))
            else:
                x_labels.append(("" if s is None else str(s))[: max(1, crop_thresh - 1)] + "…")

        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(x_labels))

        ax.tick_params(axis="x", labelrotation=rotation if any_long_x else 0)
        for lab in ax.get_xticklabels():
            lab.set_horizontalalignment("right" if any_long_x else "center")
            lab.set_rotation_mode("anchor" if any_long_x else None)

    if y_axis:
        y_ticks = [float(t) for t in ax.get_yticks()]
        y_texts = [t.get_text() for t in ax.get_yticklabels()]
        any_long_y = any(len(s or "") >= tilt_thresh for s in y_texts)
        y_labels: list[str] = []
        for s in y_texts:
            if len("" if s is None else str(s)) < crop_thresh:
                y_labels.append("" if s is None else str(s))
            else:
                y_labels.append(("" if s is None else str(s))[: max(1, crop_thresh - 1)] + "…")

        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(y_labels))

        ax.tick_params(axis="y", labelrotation=rotation if any_long_y else 0)
        for lab in ax.get_yticklabels():
            lab.set_horizontalalignment("right")
            lab.set_verticalalignment("center")
            lab.set_rotation_mode("anchor" if any_long_y else None)

    ax.figure.canvas.draw()


def line(
    x: str, y: str, data: pd.DataFrame, title: str, xlabel: str, ylabel: str, fname: PathLike, outdir: PathLike
) -> Path:
    """Save a line plot from DataFrame columns `x` and `y`."""
    if x not in data.columns or y not in data.columns:
        missing = {c for c in (x, y) if c not in data.columns}
        raise KeyError(f"Missing columns: {missing}")

    df = data[[x, y]].copy().dropna()
    if df.empty:
        raise ValueError("DataFrame is empty after dropping NaNs for plotting.")
    
    out_path = make_outpath(fname, outdir, ext=".png")

    fig, ax = plt.subplots()
    df.plot(x=x, y=y, kind="line", legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax, x_axis=True, y_axis=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
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
    order: Optional[Sequence[str | float | int]] = None,
    sort_values: bool = True,
    ascending: bool = False,
) -> Path:
    """
    Save a bar chart from a Series aggregated by index.

    If `order` is given, the Series is reindexed accordingly. Otherwise, it is sorted by values (descending by default).
    Only the top-N entries are kept if `top` is specified.
    """
    s = series.groupby(series.index).sum() if series.index.has_duplicates else series.copy()
    s = s.dropna()

    if order is not None:
        s = s.reindex(list(order)).dropna()
    elif sort_values:
        s = s.sort_values(ascending=ascending)

    if top is not None:
        s = s.head(top)

    if s.empty:
        raise ValueError("Series is empty after aggregation, sorting, and/or head filtering.")

    out_path = make_outpath(fname, outdir, ext=".png")

    fig, ax = plt.subplots()
    s.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax, x_axis=True, y_axis=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
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
    log_scale: bool = False,
) -> Path:
    """
    Save a histogram from a numeric Series.

    Non-numeric entries are ignored, and the Y-axis can be logarithmic if `log_scale` is True.
    """
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f"`bins` must be a positive int, got {bins}.")
    
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raise ValueError("Series is empty after converting to numeric and dropping NaNs.")

    out_path = make_outpath(fname, outdir, ext=".png")

    fig, ax = plt.subplots()
    s.plot(kind="hist", bins=bins, ax=ax)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax, x_axis=True, y_axis=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
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
    annotate: Optional[pd.Series | Sequence[str]] = None,
) -> Path:
    """
    Save a scatter plot from two Series.

    If `annotate` is provided, labels each point with its corresponding value.
    Raises ValueError if lengths of `x`, `y`, or `annotate` mismatch.
    """
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: len(x)={len(x)} != len(y)={len(y)}")
    if annotate is not None and len(annotate) != len(x):
        raise ValueError(f"Length mismatch: len(annotate)={len(annotate)} != len(x)={len(x)}")

    out_path = make_outpath(fname, outdir)
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=12)

    if annotate is not None:
        for xi, yi, lab in zip(x, y, annotate):
            if pd.notna(xi) and pd.notna(yi):
                ax.annotate(str(lab), (xi, yi), fontsize=6, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax, x_axis=True, y_axis=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


def scatter_with_fit(
    x: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    write_params_path: Optional[PathLike] = None,
) -> Path:
    """
    Scatter with least-squares line y = a*x + b.

    NaNs/±inf are dropped pairwise before fitting. 
    If `write_params_path` is set, writes `slope` and `intercept` to that file.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Length mismatch: len(x)={len(x_arr)} != len(y)={len(y_arr)}")

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    xv, yv = x_arr[mask], y_arr[mask]
    if xv.size < 2:
        raise ValueError("Need at least 2 finite points for regression.")

    a, b = np.polyfit(xv, yv, 1)

    out_path = make_outpath(fname, outdir)
    fig, ax = plt.subplots()
    ax.scatter(xv, yv, s=12)

    xs = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xs, a * xs + b, color='red', linestyle='--', label=f"Fit: y={a:.3g}x + {b:.3g}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax, x_axis=True, y_axis=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    if write_params_path is not None:
        save_text(f"slope={a}\nintercept={b}\n", write_params_path, outdir)

    return out_path


def heatmap(pivot: pd.DataFrame, title: str, xlabel: str, ylabel: str, fname: PathLike, outdir: PathLike) -> Path:
    """
    Save a heatmap from a pivoted DataFrame.

    Uses imshow for dense data and adds a colorbar.
    Row and column labels are drawn from the DataFrame index and columns respectively.
    """
    if pivot.empty:
        raise ValueError("Input DataFrame for heatmap is empty.")
    
    out_path = make_outpath(fname, outdir)
    fig, ax = plt.subplots()
    im = ax.imshow(pivot.values, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax, x_axis=True, y_axis=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


def pareto_frontier_plot(
    x: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    labels: Sequence[str] | np.ndarray | pd.Series | None,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
) -> Path:
    """
    Scatter with lower-left Pareto frontier (minimize both axes).

    Drops NaNs/±inf pairwise. Frontier keeps points with strictly decreasing y when scanning in increasing x. 
    If `labels` is given, points are annotated.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Length mismatch: len(x)={len(x_arr)} != len(y)={len(y_arr)}")

    if labels is not None and len(labels) != len(x_arr):
        raise ValueError(f"Length mismatch: len(labels)={len(labels)} != len(x)={len(x_arr)}")

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    xv, yv = x_arr[mask], y_arr[mask]
    labs = (np.asarray(labels, dtype=object)[mask] if labels is not None else None)

    if xv.size == 0:
        raise ValueError("No finite points to plot.")

    # Frontier: scan by increasing x, keep strict improvements in y.
    order = np.argsort(xv)
    best_y = np.inf
    frontier_idx: list[int] = []
    for i in order:
        yi = yv[i]
        if yi < best_y:
            best_y = yi
            frontier_idx.append(i)

    out_path = make_outpath(fname, outdir)
    fig, ax = plt.subplots()
    ax.scatter(xv, yv, s=16)

    if frontier_idx:
        fxs = xv[frontier_idx]
        fys = yv[frontier_idx]
        ord_f = np.argsort(fxs)
        ax.plot(fxs[ord_f], fys[ord_f], color='red', marker='o', linestyle='-', linewidth=2, label="Pareto Frontier")
        ax.legend()

    if labs is not None:
        for xi, yi, lab in zip(xv, yv, labs):
            ax.annotate(str(lab), (xi, yi), fontsize=6, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _tilt_and_crop_ticklabels(ax, x_axis=True, y_axis=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path
