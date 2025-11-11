from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd


def require(df: pd.DataFrame, cols: Iterable[str], strict: bool = False) -> list[str]:
    """Return the list of missing columns in `df`. If `strict=True`, raise on any miss."""
    missing = [c for c in cols if c not in df.columns]
    if missing and strict:
        raise KeyError(f"Missing required columns: {missing}")
    return missing


def safe_div(
    num: pd.Series,
    den: pd.Series,
    *,
    allow_zero: bool = False,
    min_den: float | None = None,
) -> pd.Series:
    """
    Safe division with null propagation.

    If `allow_zero=False`, treat den <= 0 as NA. If `min_den` is set, treat den < min_den as NA.
    """
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    mask = d.notna()
    if not allow_zero:
        mask &= d > 0
    if min_den is not None:
        mask &= d >= min_den
    out = pd.Series(pd.NA, index=n.index, dtype="Float64")
    if mask.any():
        out[mask] = (n[mask] / d[mask]).astype("Float64")
    return out


def to_utc(s: pd.Series) -> pd.Series:
    """Convert to timezone-aware UTC datetimes. Invalids -> NaT."""
    return pd.to_datetime(s, errors="coerce", utc=True)


def ensure_date_column(
    df: pd.DataFrame,
    time_col: str,
    out_col: str = "date",
    floor: str | None = None,
    dropna: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Create a calendar key from a timestamp column.

    If `floor` is None, writes UTC-normalized midnights (datetime64[ns, UTC]).
    Else writes UTC timestamps floored to `floor` ('D','h','min', ...).
    """
    out = df.copy()
    ts = to_utc(out[time_col])
    if floor is None:
        out[out_col] = ts.dt.normalize()
    else:
        out[out_col] = ts.dt.floor(floor)
    if dropna:
        out = out.dropna(subset=[out_col])
    if sort:
        out = out.sort_values(out_col, kind="stable")
    return out


def fill_missing(
    df: pd.DataFrame,
    fill_str: Mapping[str, str] | None = None,
    fill_bool: Mapping[str, bool] | None = None,
    fill_num: Mapping[str, float | int] | None = None,
) -> pd.DataFrame:
    """Fill missing values by dtype intent. Only columns present are touched."""
    out = df.copy()
    if fill_str:
        for c, v in fill_str.items():
            if c in out.columns:
                s = out[c].astype("string")
                s = s.str.strip()
                s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NaT": pd.NA})
                out[c] = s.fillna(v)
    if fill_bool:
        for c, v in fill_bool.items():
            if c in out.columns:
                out[c] = out[c].fillna(v).astype(bool)
    if fill_num:
        for c, v in fill_num.items():
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(v)
    return out


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Coerce listed columns to numeric with invalids -> NaN. Columns not present are ignored."""
    out = df.copy()
    present_cols = [c for c in cols if c in out.columns]
    for c in present_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Float64")
    return out


def clamp_nonnegative(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Clamp negative values to 0.0 on listed columns. Ignores missing columns."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Float64")
            out[c] = out[c].where(out[c] >= 0, 0.0)
    return out


@dataclass(frozen=True)
class TableSchema:
    """Column mapping for usage/spend tables. Map only what exists."""
    spend: str | None = "spend"
    total_tokens: str | None = "total_tokens"
    prompt_tokens: str | None = "prompt_tokens"
    completion_tokens: str | None = "completion_tokens"
    t_start: str | None = "startTime"
    t_comp_start: str | None = "completionStartTime"
    t_end: str | None = "endTime"
    user: str | None = "end_user"
    status: str | None = "status"


def clean_table(
    df: pd.DataFrame,
    schema: TableSchema | None = None,
    *,
    clamp_negative_spend: bool = True,
    drop_missing_user: bool = True,
    success_predicate: Callable[[pd.Series], pd.Series] | None = None,
    add_date_key: bool = True,
    date_from_col: str | None = None,
    date_key: str = "date",
    date_freq: str = "D",
    compute_durations: bool = True,
    compute_ratios: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Generic cleaner for usage/spend logs. Schema-driven and context-agnostic.

    - `schema`: column names. Only mapped columns are touched.
    - `success_predicate`: function taking the status Series -> boolean mask. If None, no status filter.
    - `add_date_key`: add `date_key` from `date_from_col or schema.t_start` using `date_freq` (e.g. 'D','h','min').
    - `compute_durations`: create latency_s, ttfb_s, gen_s if the three timestamps exist.
    - `compute_ratios`: add completion_ratio = completion_tokens / total_tokens.
    - `strict=True`: raise if a requested transform lacks its input columns.
    """
    schema = schema or TableSchema()
    out = df.copy()

    # Spend
    if schema.spend and schema.spend in out.columns:
        out[schema.spend] = pd.to_numeric(out[schema.spend], errors="coerce")
        if clamp_negative_spend:
            out.loc[out[schema.spend] < 0, schema.spend] = 0.0
    elif strict and schema.spend:
        raise KeyError(f"Missing spend column '{schema.spend}'")

    # Tokens
    token_cols = [schema.total_tokens, schema.prompt_tokens, schema.completion_tokens]
    for c in token_cols:
        if c and c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        elif strict and c:
            raise KeyError(f"Missing token column '{c}'")

    # Timestamps
    time_cols = [schema.t_start, schema.t_comp_start, schema.t_end]
    for c in time_cols:
        if c and c in out.columns:
            out[c] = to_utc(out[c])
        elif strict and c:
            raise KeyError(f"Missing timestamp column '{c}'")

    # Durations
    if compute_durations and all(c and c in out.columns for c in time_cols):
        t0, tc, t1 = schema.t_start, schema.t_comp_start, schema.t_end
        out["latency_s"] = (out[t1] - out[t0]).dt.total_seconds()
        out["ttfb_s"] = (out[tc] - out[t0]).dt.total_seconds()
        out["gen_s"] = (out[t1] - out[tc]).dt.total_seconds()
        for c in ("latency_s", "ttfb_s", "gen_s"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Float64")
            out[c] = out[c].where(out[c] >= 0, pd.NA)
    elif compute_durations and strict:
        raise KeyError("Cannot compute durations: need t_start, t_comp_start, t_end")

    # Success filter
    if success_predicate is not None:
        if schema.status and schema.status in out.columns:
            mask = success_predicate(out[schema.status])
            if not isinstance(mask, pd.Series) or mask.dtype != bool or len(mask) != len(out):
                raise ValueError("success_predicate must return a boolean Series aligned to df rows.")
            out = out[mask]
        elif strict:
            raise KeyError("Cannot filter by status: status column missing")

    # User validity
    if drop_missing_user and schema.user and schema.user in out.columns:
        u = out[schema.user].astype("string").str.strip()
        bad = {"", "nan", "none", "null", "na", "n/a"}
        out = out[u.notna() & (~u.str.lower().isin(bad))]

    # Ratios
    if compute_ratios and schema.total_tokens and schema.completion_tokens:
        if all(c in out.columns for c in (schema.total_tokens, schema.completion_tokens)):
            out["completion_ratio"] = safe_div(out[schema.completion_tokens], out[schema.total_tokens])
        elif strict:
            raise KeyError("Cannot compute completion_ratio: missing token columns")

    # Date key
    if add_date_key:
        base = date_from_col or schema.t_start
        if base and base in out.columns:
            out[date_key] = to_utc(out[base]).dt.floor(date_freq)
            out = out.dropna(subset=[date_key])
        elif strict:
            raise KeyError(f"Cannot create '{date_key}': base time column missing")

    return out


def aggregate_by_time(
    df: pd.DataFrame,
    time_col: str,
    *,
    freq: str = "D",
    sums: Sequence[str] = (),
    means: Sequence[str] = (),
    maxes: Sequence[str] = (),
    mins: Sequence[str] = (),
    custom: Mapping[str, tuple[str, str | Callable[[pd.Series], object]]] | None = None,
    include_count: bool = False,
    sort: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Resample or bucket by time and compute aggregates.

    - `freq`: any pandas offset alias ('D','h','min', etc.). Uses floor on `time_col`.
    - `sums`/`means`/`maxes`/`mins`: columns to aggregate with the named reducer.
    - `custom`: mapping {out_name: (source_col, reducer)}. Reducer is a string or callable.
    """
    require(df, [time_col], strict=True)
    out = df.copy()
    out[time_col] = to_utc(out[time_col]).dt.floor(freq)

    agg_spec: dict[str, tuple[str, str | Callable[[pd.Series], object]]] = {}
    for c in sums:
        agg_spec[f"{c}__sum"] = (c, "sum")
    for c in means:
        agg_spec[f"{c}__mean"] = (c, "mean")
    for c in maxes:
        agg_spec[f"{c}__max"] = (c, "max")
    for c in mins:
        agg_spec[f"{c}__min"] = (c, "min")
    if custom:
        for out_name, (src, reducer) in custom.items():
            if strict:
                require(out, [src], strict=True)
            agg_spec[out_name] = (src, reducer)
    if include_count:
        agg_spec["__count__"] = (time_col, "size")

    if not agg_spec:
        g = out[[time_col]].drop_duplicates().sort_values(time_col, kind="stable")
        g = g.reset_index(drop=True)
        return g

    g = out.groupby(time_col, as_index=False).agg(
        **{k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in agg_spec.items()}
    )
    if sort:
        g = g.sort_values(time_col, kind="stable")
    return g


def add_rates(
    df: pd.DataFrame,
    *,
    ratios: Sequence[tuple[str, str, str]] = (),
    sums_to_avg_per_count: Sequence[tuple[Sequence[str], str, str]] = (),
    strict: bool = False,
) -> pd.DataFrame:
    """
    Add rate columns on an aggregated DataFrame.

    - `ratios`: list of (numerator_col, denominator_col, out_name)
    - `sums_to_avg_per_count`: list of ((sum_cols...), count_col, out_name). Sums are added then divided by count.
    """
    out = df.copy()
    for num, den, out_name in ratios:
        if strict and (num not in out or den not in out):
            raise KeyError(f"Missing columns for ratio {out_name}: {num}, {den}")
        if num in out and den in out:
            out[out_name] = safe_div(out[num], out[den])
    for sum_cols, count_col, out_name in sums_to_avg_per_count:
        if strict and any(c not in out for c in (*sum_cols, count_col)):
            missing = [c for c in (*sum_cols, count_col) if c not in out]
            raise KeyError(f"Missing columns for {out_name}: {missing}")
        if count_col in out and all(c in out for c in sum_cols):
            zero = pd.Series(0, index=out.index, dtype="Float64")
            total = reduce(lambda a, b: a.add(b.astype("Float64"), fill_value=0), (out[c] for c in sum_cols), zero)
            out[out_name] = safe_div(total, out[count_col])
    return out


def top_by_value(
    df: pd.DataFrame,
    keys: str | Sequence[str],
    value_col: str,
    *,
    top: int = 10,
    ascending: bool = False,
    normalize: bool = False,
    normalize_over: str = "slice",
    dropna_keys: bool = True,
    filter_values: Sequence[object] | None = None,
) -> pd.Series:
    """
    Group by `keys`, sum `value_col`, return top `top` groups as a Series.
    - `keys` can be a single column or a list (will form a MultiIndex).
    - `normalize=True` returns percentage share within the returned slice or all groups.
    - `normalize_over` in {"slice","all"} controls normalization base.
    """
    out = df.copy()
    keys_list = [keys] if isinstance(keys, str) else list(keys)
    need = keys_list + [value_col]
    missing = require(out, need, strict=False)
    if missing:
        return pd.Series(dtype="float64")

    if dropna_keys:
        for k in keys_list:
            out = out[out[k].notna()]
    if filter_values is not None and len(keys_list) == 1:
        out = out[out[keys_list[0]].isin(filter_values)]

    grouped = out.groupby(keys_list, dropna=False)[value_col].sum()
    ordered = grouped.sort_values(ascending=ascending)
    result = ordered.head(top)

    if normalize and not result.empty:
        if normalize_over not in {"slice", "all"}:
            raise ValueError("normalize_over must be 'slice' or 'all'")
        denom = result.sum() if normalize_over == "slice" else ordered.sum()
        if denom == 0:
            return (result * 0).rename(f"{value_col}_pct")
        result = (result / denom * 100).rename(f"{value_col}_pct")
    return result


def cumulative_share(
    df: pd.DataFrame,
    key: str,
    value_col: str,
    *,
    positive_only: bool = True,
    normalize_rank: bool = True,
    dropna_key: bool = True,
) -> pd.DataFrame:
    """
    Lorenz-style cumulative share curve for totals by `key`.

    Returns columns:
      - rank: in [0,1] if normalize_rank else 1..N
      - cum_share: in [0,1]
    """
    out = df.copy()
    miss = require(out, [key, value_col], strict=False)
    if miss:
        return pd.DataFrame({"rank": [], "cum_share": []})

    if dropna_key:
        out = out[out[key].notna()]

    totals = out.groupby(key, dropna=False)[value_col].sum()
    if positive_only:
        totals = totals[totals > 0]
    totals = totals.sort_values(ascending=False)

    if totals.empty:
        return pd.DataFrame({"rank": [], "cum_share": []})

    cum = totals.cumsum() / totals.sum()
    rank = np.arange(1, len(totals) + 1, dtype=float)
    if normalize_rank:
        rank /= len(totals)

    return pd.DataFrame({"rank": rank, "cum_share": cum.to_numpy()})


def add_full_time_index(df: pd.DataFrame, time_col: str, freq: str = "D") -> pd.DataFrame:
    """Ensure a continuous time index between min and max at `freq`. Fills missing rows with NaNs."""
    t = pd.to_datetime(df[time_col], utc=True)
    if t.empty:
        return df
    idx = pd.date_range(t.min(), t.max(), freq=freq, tz="UTC")
    return df.assign(**{time_col: t}).set_index(time_col).reindex(idx, copy=False).rename_axis(time_col).reset_index()


def safe_quantile_cut(s: pd.Series, q: float) -> pd.Series:
    """Trim at the q-quantile after dropping ±inf and NaNs."""
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s
    return s[s <= s.quantile(q, interpolation="linear")]
