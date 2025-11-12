import re
from pathlib import Path
import math
import numbers

import pandas as pd
from textwrap import shorten

from utility_scripts.file_utils import PathLike, make_outpath, read_table


def md_escape(s: object) -> str:
    """Escape characters that break Markdown tables."""
    txt = str(s)
    return txt.replace("|", r"\|").replace("\n", r"\n")


def example_val(series: pd.Series) -> object | None:
    """Return a representative non-null example or None."""
    for v in series.dropna().head(10):
        return v
    return None


def schema_markdown(df: pd.DataFrame, max_examples_len: int = 60) -> str:
    """Build a summary Markdown table of a DataFrame with columns: column, dtype, non-null %, unique, example."""
    rows = ["| column | dtype | non-null % | unique | example |", "|:--|:--|--:|--:|:--|"]
    if df.empty:
        return "\n".join(rows + ["| — | — | 0 | 0 | — |"])

    nunique = df.nunique(dropna=True)
    notnull_pct = (df.notna().mean() * 100).round(1)

    for c in df.columns:
        ex = example_val(df[c])
        is_null = ex is None or ex is pd.NA or (isinstance(ex, numbers.Real) and math.isnan(float(ex)))
        ex_str = "—" if is_null else shorten(md_escape(ex), max_examples_len, placeholder="…")
        rows.append(f"| {md_escape(c)} | {df[c].dtype} | {notnull_pct[c]:.1f} | {int(nunique[c])} | {ex_str} |")
    return "\n".join(rows)


def preview_markdown(df: pd.DataFrame, max_cols: int = 30, max_rows: int = 5, max_cell_length: int = 80) -> str:
    """Small Markdown preview of top-left corner of a DataFrame."""
    if df.empty:
        return "_No rows._"

    view = df.iloc[:max_rows, :max_cols].copy()
    for c in view.columns:
        view[c] = view[c].map(
            lambda x: shorten(md_escape(x), max_cell_length, placeholder="…") if isinstance(x, str) else x
        )
    return view.to_markdown(index=False)


def anchor(name: str) -> str:
    """Create a simple Markdown anchor slug."""
    slug = name.strip().lower().replace(" ", "-")
    return slug


def summarize_tables_markdown(
    dataset: str = "litellm",
    dir_name: PathLike = "data",
    md_file_name: str = "tables_summary.md",
    max_preview_cols: int = 30,
    max_preview_rows: int = 5,
) -> str:
    """
    Load each table once and write a Markdown summary with:
    - A TOC showing row/column counts per table
    - Per-table sections with schema and a small preview
    """
    base = Path(dir_name) / dataset
    if not base.exists():
        raise FileNotFoundError(f"Missing dataset directory: {base}")
    
    out_path = make_outpath(md_file_name, dir_name, ext=".md")

    table_folders = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("public.")])
    names = [p.name.split("public.", 1)[1] for p in table_folders]

    cache: dict[str, tuple[pd.DataFrame | None, str | None]] = {}
    for t in names:
        try:
            df = read_table(t, dataset=dataset, dir_name=dir_name)
            cache[t] = (df, None)
        except Exception as e:
            cache[t] = (None, str(e))

    toc = ["# Tables summary", "", "## Contents"]
    sections: list[str] = []

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
    DATA_DIR = "data"

    p1 = summarize_tables_markdown(dataset="litellm", dir_name=DATA_DIR, md_file_name="litellm_summary.md")
    print(f"Wrote {p1}")
    p2 = summarize_tables_markdown(dataset="openwebui", dir_name=DATA_DIR, md_file_name="openwebui_summary.md")
    print(f"Wrote {p2}")
