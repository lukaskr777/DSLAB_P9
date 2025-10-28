from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from textwrap import shorten

from utils import read_table

from plots_tables.plots_litellm_daily_tag_spend import plot_all_itellm_daily_tag_spend
from plots_tables.plots_litellm_daily_team_spend import plot_all_litellm_daily_team_spend
from plots_tables.plots_litellm_dayly_user_spend import plot_all_litellm_daily_user_spend
from plots_tables.plots_litellm_end_user_table import plot_all_litellm_end_user_table
from plots_tables.plots_litellm_spendlogs import plot_all_litellm_spendlogs


def md_escape(s: Any) -> str:
    """
    Escape characters that break Markdown tables.

    Args:
        s: Any value to render inside a Markdown table cell.

    Returns:
        A single-line, escaped string.
    """
    txt = str(s)
    return txt.replace("|", r"\|").replace("\n", r"\n")


def example_val(series: pd.Series) -> Any | None:
    """
    Pick a representative non-null example from a Series.

    Strategy:
        Return the first non-null among the first 10 non-null values.

    Args:
        series: Input Series.

    Returns:
        The example value or None if none found.
    """
    for v in series.dropna().head(10):
        return v
    return None


def schema_markdown(df: pd.DataFrame, max_examples_len: int = 60) -> str:
    """
    Build a compact Markdown table describing schema and basic stats.

    Columns:
      - column
      - dtype
      - non-null %
      - unique
      - example

    Args:
        df: Input DataFrame.
        max_examples_len: Max characters to show for the example cell.

    Returns:
        Markdown-formatted string.
    """
    rows: List[str] = [
        "| column | dtype | non-null % | unique | example |",
        "|:--|:--|--:|--:|:--|",
    ]
    if df.empty:
        return "\n".join(rows + ["| — | — | 0 | 0 | — |"])

    nunique = df.nunique(dropna=True)
    notnull_pct = (df.notna().mean() * 100).round(1)

    for c in df.columns:
        ex = example_val(df[c])
        ex_str = "—" if pd.isna(ex) else shorten(md_escape(ex), max_examples_len, placeholder="…")
        rows.append(f"| {md_escape(c)} | {df[c].dtype} | {notnull_pct[c]:.1f} | {int(nunique[c])} | {ex_str} |")
    return "\n".join(rows)


def preview_markdown(df: pd.DataFrame, max_cols: int = 30, max_rows: int = 5) -> str:
    """
    Produce a small Markdown preview table.

    Args:
        df: Input DataFrame.
        max_cols: Max number of leftmost columns to include.
        max_rows: Max number of top rows to include.

    Returns:
        Markdown-formatted table or "_No rows._" if empty.
    """
    if df.empty:
        return "_No rows._"

    view = df.iloc[:max_rows, :max_cols].copy()
    for c in view.columns:
        view[c] = view[c].map(lambda x: shorten(md_escape(x), 80, placeholder="…") if isinstance(x, str) else x)
    return view.to_markdown(index=False)


def anchor(name: str) -> str:
    """Create a simple Markdown anchor slug."""
    return name.strip().lower().replace(" ", "-")


def summarize_tables_markdown(
    dataset: str = "litellm",
    dir_name: str = "data",
    md_file_name: str = "tables_summary.md",
    max_preview_cols: int = 30,
    max_preview_rows: int = 5,
) -> str:
    """
    Scan a dataset directory, load each table once, and write a Markdown summary.

    The summary includes:
      - A table of contents with row & column counts per table.
      - A section per table with schema and a small data preview.

    Args:
        dataset: Dataset folder under dir_name.
        dir_name: Root data directory.
        md_file_name: Output Markdown filename (written under dir_name).
        max_preview_cols: Columns shown in the preview.
        max_preview_rows: Rows shown in the preview.

    Returns:
        Path to the written Markdown file.
    """
    base = Path(dir_name) / dataset
    out_path = Path(dir_name) / md_file_name

    if not base.exists():
        raise FileNotFoundError(f"Missing dataset directory: {base}")

    table_folders = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("public.")])
    names = [p.name.split("public.", 1)[1] for p in table_folders]

    cache: Dict[str, Tuple[Optional[pd.DataFrame], Optional[str]]] = {}
    for t in names:
        try:
            df = read_table(t, dataset=dataset, dir_name=dir_name)
            cache[t] = (df, None)
        except Exception as e:
            cache[t] = (None, str(e))

    toc: List[str] = ["# Tables summary", "", "## Contents"]
    sections: List[str] = []

    # TOC
    for t in names:
        df, err = cache[t]
        if err is not None or df is None:
            toc.append(f"- [{t}](#{anchor(t)})  (error)")
        else:
            toc.append(f"- [{t}](#{anchor(t)})  ({len(df)}x{len(df.columns)})")

    # Sections
    for t in names:
        df, err = cache[t]
        if err is not None or df is None:
            sections.append(f"## {t}\n\n_Error while loading_: `{md_escape(err)}`\n")
            continue

        n_rows, n_cols = len(df), len(df.columns)
        sec = [
            f"## {t}",
            f"*Rows*: **{n_rows}**  •  *Columns*: **{n_cols}**",
            "",
            "### Schema",
            schema_markdown(df),
            "",
            "### Preview",
            preview_markdown(df, max_preview_cols, max_preview_rows),
            "",
        ]
        sections.append("\n".join(sec))

    content = "\n".join(toc + [""] + sections) + "\n"
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


if __name__ == "__main__":
    p1 = summarize_tables_markdown(dataset="litellm", md_file_name="litellm_summary.md")
    print(f"Wrote {p1}")
    p2 = summarize_tables_markdown(dataset="openwebui", md_file_name="openwebui_summary.md")
    print(f"Wrote {p2}")

    plot_all_itellm_daily_tag_spend()
    plot_all_litellm_daily_team_spend()
    plot_all_litellm_daily_user_spend()
    plot_all_litellm_end_user_table()
    plot_all_litellm_spendlogs()
