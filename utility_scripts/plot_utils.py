from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FixedFormatter

from utility_scripts.file_utils import PathLike, make_outpath, save_text


# ---------- Global plotting defaults ----------

DEFAULT_FIGSIZE: tuple[float, float] = (7.0, 4.2)
DEFAULT_DPI: int = 300
DEFAULT_TITLE_FONTSIZE: int = 12
DEFAULT_LABEL_FONTSIZE: int = 10
DEFAULT_TICK_FONTSIZE: int = 8

LINE_WIDTH: float = 1.8
LINE_WIDTH_THIN: float = 1.2
MARKER_SIZE: float = 4.0

COLOR_PRIMARY = "#1f77b4"    # blue
COLOR_SECONDARY = "#ff7f0e"  # orange
COLOR_TERTIARY = "#2ca02c"   # green
COLOR_MUTED = "#7f7f7f"      # grey


def _new_fig_ax(
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Create a new figure/axes with library-wide defaults."""
    fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGSIZE)
    # Light background for axes; keep figure white
    ax.set_facecolor("#f8f9fb")
    fig.patch.set_facecolor("white")
    return fig, ax


# ---------- Tick label utilities ----------

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
    # Ensure tick labels exist before reading them
    ax.figure.canvas.draw()

    if x_axis:
        x_ticks = [float(t) for t in ax.get_xticks()]
        x_texts = [t.get_text() for t in ax.get_xticklabels()]
        any_long_x = any(len(s or "") >= tilt_thresh for s in x_texts)

        x_labels: list[str] = []
        for s in x_texts:
            s_str = "" if s is None else str(s)
            if len(s_str) < crop_thresh:
                x_labels.append(s_str)
            else:
                x_labels.append(s_str[: max(1, crop_thresh - 1)] + "…")

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
            s_str = "" if s is None else str(s)
            if len(s_str) < crop_thresh:
                y_labels.append(s_str)
            else:
                y_labels.append(s_str[: max(1, crop_thresh - 1)] + "…")

        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(y_labels))

        ax.tick_params(axis="y", labelrotation=rotation if any_long_y else 0)
        for lab in ax.get_yticklabels():
            lab.set_horizontalalignment("right")
            lab.set_verticalalignment("center")
            lab.set_rotation_mode("anchor" if any_long_y else None)

    ax.figure.canvas.draw()


# ---------- Common finalization helper ----------

def _final_plotting(
    ax: Axes,
    fig: Figure,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: PathLike,
    *,
    x_axis: bool = True,
    y_axis: bool = False,
    xgrid: bool = True,
    ygrid: bool = True,
    tick_tilt: bool = True,
    title_fontsize: int = DEFAULT_TITLE_FONTSIZE,
    label_fontsize: int = DEFAULT_LABEL_FONTSIZE,
    tick_fontsize: int = DEFAULT_TICK_FONTSIZE,
    dpi: int = DEFAULT_DPI,
) -> None:
    ax.set_title(title, fontsize=title_fontsize, loc="left")
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    # Cleaner spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.6)
    ax.spines["bottom"].set_alpha(0.6)

    if tick_tilt:
        _tilt_and_crop_ticklabels(ax, x_axis=x_axis, y_axis=y_axis)

    if xgrid or ygrid:
        if xgrid and ygrid:
            axis = "both"
        elif xgrid:
            axis = "x"
        else:
            axis = "y"
        ax.set_axisbelow(True)
        ax.grid(which="major", axis=axis, linestyle="--", linewidth=0.6, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ---------- Plotting functions ----------

def line(
    x: str,
    y: str,
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    *,
    xgrid: bool = True,
    ygrid: bool = True,
    **style_kwargs: Any,
) -> Path:
    """Save a line plot from DataFrame columns `x` and `y`."""
    if x not in data.columns or y not in data.columns:
        missing = {c for c in (x, y) if c not in data.columns}
        raise KeyError(f"Missing columns: {missing}")

    df = data[[x, y]].copy().dropna()
    if df.empty:
        raise ValueError("DataFrame is empty after dropping NaNs for plotting.")

    out_path = make_outpath(fname, outdir, ext=".png")

    fig, ax = _new_fig_ax()
    df.plot(
        x=x,
        y=y,
        kind="line",
        legend=False,
        ax=ax,
        color=COLOR_PRIMARY,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
    )

    _final_plotting(
        ax,
        fig,
        title,
        xlabel,
        ylabel,
        out_path,
        x_axis=True,
        y_axis=False,
        xgrid=xgrid,
        ygrid=ygrid,
        **style_kwargs,
    )
    return out_path


def lines_quantiles(
    x: str,
    y: str,
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    *,
    q: tuple[float, float, float] = (0.25, 0.5, 0.75),
    plot_extrema: bool = False,
    xgrid: bool = True,
    ygrid: bool = True,
    **style_kwargs: Any,
) -> Path:
    """
    Save a multi-line plot with min, Q1, median, Q3, mean, and max of `y` grouped by `x`.
    Keeps only rows with finite y and non-null x.
    """
    if x not in data.columns or y not in data.columns:
        missing = {c for c in (x, y) if c not in data.columns}
        raise KeyError(f"Missing columns: {missing}")

    df = data[[x, y]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning for quantile plotting.")

    grouped = df.groupby(x, observed=True, sort=False)[y]
    qvals = grouped.quantile(np.array(q)).unstack().rename(
        columns={q[0]: "Q1", q[1]: "Median", q[2]: "Q3"}
    )
    qvals["Mean"] = grouped.mean()
    if plot_extrema:
        qvals["Min"] = grouped.min()
        qvals["Max"] = grouped.max()
    qvals = qvals.sort_index(kind="stable")

    out_path = make_outpath(fname, outdir, ext=".png")
    fig, ax = _new_fig_ax()

    ax.plot(
        qvals.index,
        qvals["Q1"],
        label="Q1",
        color=COLOR_MUTED,
        linestyle="--",
        linewidth=LINE_WIDTH_THIN,
        marker="",
    )
    ax.plot(
        qvals.index,
        qvals["Median"],
        label="Median",
        color=COLOR_PRIMARY,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
    )
    ax.plot(
        qvals.index,
        qvals["Q3"],
        label="Q3",
        color=COLOR_MUTED,
        linestyle="--",
        linewidth=LINE_WIDTH_THIN,
        marker="",
    )
    ax.plot(
        qvals.index,
        qvals["Mean"],
        label="Mean",
        color=COLOR_SECONDARY,
        linewidth=LINE_WIDTH,
        linestyle=":",
    )

    if plot_extrema:
        ax.plot(
            qvals.index,
            qvals["Min"],
            "--",
            alpha=0.5,
            label="Min",
            color="#bbbbbb",
            linewidth=LINE_WIDTH_THIN,
        )
        ax.plot(
            qvals.index,
            qvals["Max"],
            "--",
            alpha=0.5,
            label="Max",
            color="#bbbbbb",
            linewidth=LINE_WIDTH_THIN,
        )

    ax.legend(frameon=False)

    _final_plotting(
        ax,
        fig,
        title,
        xlabel,
        ylabel,
        out_path,
        x_axis=True,
        y_axis=False,
        xgrid=xgrid,
        ygrid=ygrid,
        **style_kwargs,
    )
    return out_path


def bar(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    *,
    top: int | None = None,
    order: Sequence[str | float | int] | None = None,
    sort_values: bool = True,
    ascending: bool = False,
    xgrid: bool = False,
    ygrid: bool = True,
    **style_kwargs: Any,
) -> Path:
    """
    Save a bar chart from a Series aggregated by index.

    If `order` is given, the Series is reindexed accordingly. Otherwise, it is sorted by values
    (descending by default). Only the top-N entries are kept if `top` is specified.
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

    fig, ax = _new_fig_ax()
    s.plot(
        kind="bar",
        ax=ax,
        color=COLOR_PRIMARY,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    _final_plotting(
        ax,
        fig,
        title,
        xlabel,
        ylabel,
        out_path,
        x_axis=True,
        y_axis=False,
        xgrid=xgrid,
        ygrid=ygrid,
        **style_kwargs,
    )
    return out_path


def hist(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    *,
    bins: int = 50,
    log_scale: bool = False,
    xgrid: bool = True,
    ygrid: bool = True,
    **style_kwargs: Any,
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

    fig, ax = _new_fig_ax()
    s.plot(kind="hist", bins=bins, ax=ax, color=COLOR_PRIMARY, alpha=0.75, edgecolor="white", linewidth=0.4)
    if log_scale:
        ax.set_yscale("log")

    _final_plotting(
        ax,
        fig,
        title,
        xlabel,
        ylabel,
        out_path,
        x_axis=True,
        y_axis=False,
        xgrid=xgrid,
        ygrid=ygrid,
        **style_kwargs,
    )
    return out_path


def grouped_hist(
    values: pd.Series,
    groups: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    *,
    bins: int | Sequence[float] = 30,
    stacked: bool = True,
    log_scale: bool = False,
    xgrid: bool = True,
    ygrid: bool = True,
    **style_kwargs: Any,
) -> Path:
    """
    Stacked histogram by group with common bin edges.

    Args:
        values: Numeric sample values.
        groups: Group labels aligned to `values`.
        bins: Int for automatic binning or explicit bin edges.
        stacked: Stack groups if True, otherwise overlay.
        log_scale: Logarithmic Y axis if True.
    """
    out_path = make_outpath(fname, outdir, ext=".png")

    v = pd.to_numeric(values, errors="coerce")
    g = groups.astype("string")
    mask = v.notna() & g.notna()
    v, g = v[mask], g[mask]
    if v.empty:
        raise ValueError("No finite values to plot.")

    if isinstance(bins, int):
        edges = np.histogram_bin_edges(v.to_numpy(), bins=bins).tolist()
    else:
        edges = list(bins)

    cats = list(pd.unique(g))
    data = [v[g == c].to_numpy() for c in cats]

    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(cats))]

    fig, ax = _new_fig_ax()
    ax.hist(
        data,
        bins=edges,
        stacked=stacked,
        color=colors,
        label=[str(c) for c in cats],
        edgecolor="white",
        linewidth=0.4,
        log=log_scale,
        alpha=0.85,
    )
    ax.legend(frameon=False)

    _final_plotting(
        ax,
        fig,
        title,
        xlabel,
        ylabel,
        out_path,
        x_axis=True,
        y_axis=False,
        xgrid=xgrid,
        ygrid=ygrid,
        **style_kwargs,
    )
    return out_path


def bin_and_quantiles(
    x: pd.Series,
    y: pd.Series,
    bins: int,
    xlabel: str,
    ylabel: str,
    fname: str,
    out: PathLike,
    *,
    plot_extrema: bool = False,
    xgrid: bool = True,
    ygrid: bool = True,
    **style_kwargs: Any,
) -> None:
    """
    Plot min, Q1, median, Q3, mean, and max of `y` across quantile bins of `x`.
    Dashed, semi-transparent lines for min/max.
    """
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return
    d = d[d["x"] >= 0]
    qn = min(bins, max(2, d["x"].nunique()))
    d["bin"] = pd.qcut(d["x"], q=qn, duplicates="drop")

    g = (
        d.groupby("bin", observed=True)
        .agg(
            avg_x=("x", "mean"),
            q1=("y", lambda s: s.quantile(0.25)),
            med=("y", "median"),
            q3=("y", lambda s: s.quantile(0.75)),
            mean_y=("y", "mean"),
            min_y=("y", "min"),
            max_y=("y", "max"),
        )
        .sort_values("avg_x", kind="stable")
    )

    out_path = make_outpath(fname, out, ext=".png")
    fig, ax = _new_fig_ax()

    ax.plot(
        g["avg_x"],
        g["q1"],
        label="Q1",
        color=COLOR_MUTED,
        linestyle="--",
        linewidth=LINE_WIDTH_THIN,
    )
    ax.plot(
        g["avg_x"],
        g["med"],
        label="Median",
        color=COLOR_PRIMARY,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
    )
    ax.plot(
        g["avg_x"],
        g["q3"],
        label="Q3",
        color=COLOR_MUTED,
        linestyle="--",
        linewidth=LINE_WIDTH_THIN,
    )
    ax.plot(
        g["avg_x"],
        g["mean_y"],
        label="Mean",
        color=COLOR_SECONDARY,
        linewidth=LINE_WIDTH,
        linestyle=":",
    )

    if plot_extrema:
        ax.plot(
            g["avg_x"],
            g["min_y"],
            "--",
            alpha=0.5,
            label="Min",
            color="#bbbbbb",
            linewidth=LINE_WIDTH_THIN,
        )
        ax.plot(
            g["avg_x"],
            g["max_y"],
            "--",
            alpha=0.5,
            label="Max",
            color="#bbbbbb",
            linewidth=LINE_WIDTH_THIN,
        )

    ax.legend(frameon=False)

    title_suffix = "Q1 / Median / Q3 / Mean" + (" / Min / Max" if plot_extrema else "")
    title = f"{xlabel}: {title_suffix}"
    _final_plotting(
        ax,
        fig,
        title,
        xlabel,
        ylabel,
        out_path,
        x_axis=True,
        y_axis=False,
        xgrid=xgrid,
        ygrid=ygrid,
        **style_kwargs,
    )


def scatter(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    *,
    annotate: pd.Series | Sequence[str] | None = None,
    xgrid: bool = True,
    ygrid: bool = True,
    point_size: float = 16.0,
    alpha: float = 0.6,
    annotation_fontsize: int = 6,
    **style_kwargs: Any,
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

    out_path = make_outpath(fname, outdir, ext=".png")
    fig, ax = _new_fig_ax()

    ax.scatter(
        x,
        y,
        s=point_size,
        alpha=alpha,
        linewidths=0.3,
        edgecolors="white",
        c=COLOR_PRIMARY,
    )

    if annotate is not None:
        for xi, yi, lab in zip(x, y, annotate):
            if pd.notna(xi) and pd.notna(yi):
                ax.annotate(str(lab), (xi, yi), fontsize=annotation_fontsize, alpha=0.8)

    _final_plotting(
        ax,
        fig,
        title,
        xlabel,
        ylabel,
        out_path,
        x_axis=True,
        y_axis=False,
        xgrid=xgrid,
        ygrid=ygrid,
        **style_kwargs,
    )
    return out_path


def scatter_with_fit(
    x: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    *,
    write_params_path: PathLike | None = None,
    x_scale: float = 1.0,  # divide x by this before plotting/fitting (e.g., 1e6 for “tokens (M)”)
    y_scale: float = 1.0,  # divide y by this before plotting/fitting (e.g., 1e3 for “spend (k$)”)
    xgrid: bool = True,
    ygrid: bool = True,
    **style_kwargs: Any,
) -> Path:
    """Scatter with least-squares line y = a*x + b in *displayed* units."""
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.size != y_arr.size:
        raise ValueError(f"Length mismatch: len(x)={x_arr.size} != len(y)={y_arr.size}")

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    xv_raw, yv_raw = x_arr[mask], y_arr[mask]
    if xv_raw.size < 2:
        raise ValueError("Need at least 2 finite points for regression.")

    # Convert to displayed units
    xv = xv_raw / x_scale
    yv = yv_raw / y_scale

    # OLS in displayed units: [x 1][a b]^T ≈ y
    X = np.c_[xv, np.ones_like(xv)]
    a_disp, b_disp = np.linalg.lstsq(X, yv, rcond=None)[0]

    yhat = a_disp * xv + b_disp
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out_path = make_outpath(fname, outdir, ext=".png")
    fig, ax = _new_fig_ax()

    ax.scatter(
        xv,
        yv,
        s=16,
        alpha=0.7,
        linewidths=0.3,
        edgecolors="white",
        c=COLOR_PRIMARY,
    )

    xs = np.linspace(xv.min(), xv.max(), 200)
    ax.plot(
        xs,
        a_disp * xs + b_disp,
        linestyle="--",
        color=COLOR_SECONDARY,
        linewidth=LINE_WIDTH,
        label=f"Fit: y = {a_disp:.6g} x + {b_disp:.6g}  (R²={r2:.3f}, n={xv.size})",
    )
    ax.legend(loc="best", frameon=False)

    ax.ticklabel_format(style="plain", useOffset=False, axis="both")

    _final_plotting(
        ax,
        fig,
        title,
        xlabel,
        ylabel,
        out_path,
        x_axis=True,
        y_axis=False,
        xgrid=xgrid,
        ygrid=ygrid,
        **style_kwargs,
    )

    if write_params_path is not None:
        # Also record raw-units params for programmatic use
        slope_raw = a_disp * (y_scale / x_scale)
        intercept_raw = b_disp * y_scale
        save_text(
            f"# displayed_units\nslope={a_disp:.17g}\nintercept={b_disp:.17g}\n"
            f"# raw_units\nslope={slope_raw:.17g}\nintercept={intercept_raw:.17g}\n"
            f"R2={r2:.17g}\nn={int(xv.size)}\n",
            write_params_path,
            outdir,
        )

    return out_path


def heatmap(
    pivot: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: PathLike,
    outdir: PathLike,
    *,
    annotate: bool = False,
    fmt: str = ".2g",
    xgrid: bool = False,
    ygrid: bool = False,
    **style_kwargs: Any,
) -> Path:
    """
    Save a heatmap from a pivoted DataFrame.

    Uses imshow for dense data and adds a colorbar.
    Row and column labels are drawn from the DataFrame index and columns respectively.
    """
    if pivot.empty:
        raise ValueError("Input DataFrame for heatmap is empty.")

    out_path = make_outpath(fname, outdir, ext=".png")
    fig, ax = _new_fig_ax()

    M = pivot.apply(pd.to_numeric, errors="coerce")
    if M.isna().all().all():
        raise ValueError("Heatmap has only NaNs after numeric coercion.")
    A = M.to_numpy(dtype=float)  # real float array, no object dtype

    im = ax.imshow(A, aspect="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=DEFAULT_TICK_FONTSIZE)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    if annotate:
        nrows, ncols = A.shape
        for i in range(nrows):
            for j in range(ncols):
                val = A[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        format(val, fmt),
                        ha="center",
                        va="center",
                        fontsize=DEFAULT_TICK_FONTSIZE,
                    )

    _final_plotting(
        ax,
        fig,
        title,
        xlabel,
        ylabel,
        out_path,
        x_axis=True,
        y_axis=True,
        xgrid=xgrid,
        ygrid=ygrid,
        **style_kwargs,
    )
    return out_path
