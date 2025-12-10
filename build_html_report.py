# build_html_report.py

from pathlib import Path
from typing import Iterable
import html


OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")
EMBEDDINGS_PATH = OUT_DIR / (
    "small_train_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.npy"
)
TAG = EMBEDDINGS_PATH.stem
RUN_SUFFIX = "_langwise"
RUN_TAG = f"{TAG}{RUN_SUFFIX}"

FIGS_ROOT = Path("figs/apertus_clustering")
FIGS_DIR = FIGS_ROOT / RUN_TAG
HTML_PATH = FIGS_DIR / "index.html"


def _img_tag(src: Path, alt: str = "", css_class: str = "plot") -> str:
    rel = src.relative_to(FIGS_DIR)
    # Normalize Windows-style backslashes to forward slashes for HTML
    rel_str = str(rel).replace("\\", "/")
    return f'<img src="{html.escape(rel_str)}" alt="{html.escape(alt)}" class="{css_class}">'


def _section(title: str, content: str, section_id: str | None = None) -> str:
    if section_id is None:
        section_id = title.lower().replace(" ", "-")
    return f"""
<section id="{html.escape(section_id)}">
  <h2>{html.escape(title)}</h2>
  {content}
</section>
"""


def _gallery(img_paths: Iterable[Path], title_prefix: str = "") -> str:
    items = []
    for p in sorted(img_paths):
        caption = title_prefix + p.stem
        items.append(
            f"""
      <figure class="gallery-item">
        {_img_tag(p, alt=caption)}
        <figcaption>{html.escape(caption)}</figcaption>
      </figure>
"""
        )
    if not items:
        return "<p>No figures found.</p>"
    return "<div class=\"gallery\">\n" + "\n".join(items) + "\n</div>"


def _parse_top_words(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 1:
                continue
            row: dict[str, str] = {"cluster": parts[0]}
            for i in range(1, min(len(parts), 11), 2):
                word = parts[i]
                count = parts[i + 1] if i + 1 < len(parts) else ""
                rank = (i + 1) // 2
                row[f"word{rank}"] = word
                row[f"count{rank}"] = count
            rows.append(row)
    return rows


def _top_words_table(path: Path, title: str) -> str:
    rows = _parse_top_words(path)
    if not rows:
        return f"<p>No top-words file found at {html.escape(str(path))}.</p>"

    header_cells = ["cluster"]
    for k in range(1, 6):
        header_cells.append(f"word{k}")
        header_cells.append(f"count{k}")

    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in header_cells)

    body_rows = []
    for r in rows:
        cells = []
        for h in header_cells:
            cells.append(f"<td>{html.escape(r.get(h, ''))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
<h3>{html.escape(title)}</h3>
<div class="table-wrapper">
  <table>
    <thead>
      <tr>{header_html}</tr>
    </thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
</div>
"""
    return table_html


def _parse_representatives(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", maxsplit=2)
            if len(parts) != 3:
                continue
            cluster, title, full = parts
            rows.append(
                {
                    "cluster": cluster,
                    "title": title,
                    "full_prompt": full,
                }
            )
    return rows


def _representatives_table(path: Path, title: str) -> str:
    rows = _parse_representatives(path)
    if not rows:
        return f"<p>No representatives file found at {html.escape(str(path))}.</p>"

    body_rows = []
    for r in rows:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(r['cluster'])}</td>"
            f"<td>{html.escape(r['title'])}</td>"
            f"<td>{html.escape(r['full_prompt'])}</td>"
            "</tr>"
        )

    table_html = f"""
<h3>{html.escape(title)}</h3>
<div class="table-wrapper">
  <table>
    <thead>
      <tr>
        <th>cluster</th>
        <th>title (truncated)</th>
        <th>representative prompt</th>
      </tr>
    </thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
</div>
"""
    return table_html


def build_html() -> None:
    if not FIGS_DIR.exists():
        raise FileNotFoundError(f"FIGS_DIR does not exist: {FIGS_DIR}")

    # ------- Sections for all languages -------

    # Scatter plots / cluster sizes / feature summaries
    all_lang_plots = _gallery(FIGS_DIR.glob("scatter_dim1_dim2_*.png"), title_prefix="")
    all_lang_sizes = _gallery(FIGS_DIR.glob("cluster_sizes_*.png"), title_prefix="")
    all_lang_feats = _gallery(FIGS_DIR.glob("cluster_feature_means_heatmap_*.png"), title_prefix="")

    section_overview = _section(
        "Overview",
        """
<p>This page summarizes the language-wise clustering results for the swiss-ai / apertus-sft-mixture dataset.
Each figure and table is generated from the same RUN_TAG as the poster.</p>
<p>Use the navigation at the top to jump to specific sections.</p>
""",
        section_id="overview",
    )

    section_all = _section(
        "All languages – cluster plots",
        all_lang_plots + "<h3>Cluster sizes</h3>" + all_lang_sizes + "<h3>Feature heatmaps</h3>" + all_lang_feats,
        section_id="all-languages",
    )

    # Top words and representatives (all languages)
    top_words_all_path = FIGS_DIR / "top_words_cluster_langwise_final.txt"
    reps_all_path = FIGS_DIR / "cluster_representatives_cluster_langwise_final.txt"

    section_all_words = _section(
        "All languages – top words and representatives",
        _top_words_table(top_words_all_path, "Top 5 words per final cluster (all languages)")
        + _representatives_table(reps_all_path, "Representative prompts per final cluster (all languages)"),
        section_id="all-languages-words",
    )

    # ------- English-only sections -------

    figs_dir_en = FIGS_DIR / "english"
    if figs_dir_en.exists():
        en_plots = _gallery(figs_dir_en.glob("scatter_dim1_dim2_*.png"))
        en_sizes = _gallery(figs_dir_en.glob("cluster_sizes_*.png"))
        en_feats = _gallery(figs_dir_en.glob("cluster_feature_means_heatmap_*.png"))

        # English-only top words / reps
        top_words_en_path = figs_dir_en / "top_words_cluster_langwise_final.txt"
        reps_en_path = figs_dir_en / "cluster_representatives_cluster_langwise_final_en.txt"

        # English word clouds
        wc_dir_en = figs_dir_en / "wordclouds" / "cluster_langwise_final"
        en_wcs = _gallery(wc_dir_en.glob("*.png"), title_prefix="")

        section_en_plots = _section(
            "English-only – cluster plots",
            en_plots + "<h3>Cluster sizes (English only)</h3>" + en_sizes
            + "<h3>Feature heatmaps (English only)</h3>" + en_feats,
            section_id="english-plots",
        )

        section_en_wcs = _section(
            "English-only – word clouds",
            en_wcs,
            section_id="english-wordclouds",
        )

        section_en_words = _section(
            "English-only – top words and representatives",
            _top_words_table(top_words_en_path, "Top 5 words per final cluster (English only)")
            + _representatives_table(reps_en_path, "Representative prompts per final cluster (English only)"),
            section_id="english-words",
        )
    else:
        section_en_plots = _section(
            "English-only results",
            "<p>No English-only directory found. Run the plotting script with English outputs enabled.</p>",
            section_id="english-plots",
        )
        section_en_wcs = ""
        section_en_words = ""

    # ------- Navigation -------

    nav_html = """
<nav>
  <ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#all-languages">All languages – cluster plots</a></li>
    <li><a href="#all-languages-words">All languages – words & reps</a></li>
    <li><a href="#english-plots">English-only – cluster plots</a></li>
    <li><a href="#english-wordclouds">English-only – word clouds</a></li>
    <li><a href="#english-words">English-only – words & reps</a></li>
  </ul>
</nav>
"""

    # ------- Final HTML -------

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Apertus clustering report – {html.escape(RUN_TAG)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      line-height: 1.5;
      background-color: #f7f7f7;
    }}
    header {{
      background-color: #222;
      color: #fff;
      padding: 1rem 2rem;
    }}
    header h1 {{
      margin: 0;
      font-size: 1.8rem;
    }}
    header p {{
      margin: 0.3rem 0 0;
      font-size: 0.9rem;
      opacity: 0.8;
    }}
    nav {{
      background-color: #333;
      padding: 0.5rem 1.5rem;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    nav ul {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
    }}
    nav a {{
      color: #eee;
      text-decoration: none;
      font-size: 0.9rem;
      padding: 0.2rem 0.4rem;
    }}
    nav a:hover {{
      background-color: #555;
      border-radius: 4px;
    }}
    main {{
      padding: 1.5rem 2rem 3rem;
      max-width: 1200px;
      margin: 0 auto;
      background-color: #fff;
    }}
    section {{
      margin-bottom: 2.5rem;
    }}
    section h2 {{
      border-bottom: 1px solid #ddd;
      padding-bottom: 0.25rem;
      margin-bottom: 0.75rem;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1rem;
    }}
    .gallery-item {{
      background-color: #fafafa;
      border: 1px solid #eee;
      border-radius: 6px;
      padding: 0.5rem;
      box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }}
    .gallery-item img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .gallery-item figcaption {{
      font-size: 0.75rem;
      color: #555;
      margin-top: 0.3rem;
      word-break: break-word;
    }}
    .table-wrapper {{
      overflow-x: auto;
      margin-top: 0.75rem;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.85rem;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 0.3rem 0.5rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background-color: #f0f0f0;
    }}
    tr:nth-child(even) td {{
      background-color: #fafafa;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Apertus clustering report – {html.escape(RUN_TAG)}</h1>
    <p>Interactive companion page for the poster (all figures & cluster summaries).</p>
  </header>
  {nav_html}
  <main>
    {section_overview}
    {section_all}
    {section_all_words}
    {section_en_plots}
    {section_en_wcs}
    {section_en_words}
  </main>
</body>
</html>
"""

    HTML_PATH.write_text(html_doc, encoding="utf-8")
    print(f"HTML report written to: {HTML_PATH}")


if __name__ == "__main__":
    build_html()
