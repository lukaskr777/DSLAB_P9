from pathlib import Path
from typing import Iterable
import html
import json


OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")
EMBEDDINGS_PATH = OUT_DIR / (
    "small_train_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.npy"
)
TAG = EMBEDDINGS_PATH.stem
RUN_SUFFIX = "_langwise"
RUN_TAG = f"{TAG}{RUN_SUFFIX}"

# apertus_sft_mixture clustering figs
FIGS_ROOT = Path("figs/apertus_clustering")
FIGS_DIR = FIGS_ROOT / RUN_TAG

# Public AI (LiteLLM_SpendLogs) figure roots
USAGE_DIR = Path("figs") / "litellm_apertus70b_usage"
PERF_DIR = Path("figs") / "litellm_apertus70b_performance"
BEHAV_DIR = Path("figs") / "litellm_apertus70b_behavior"
USERCLUST_DIR = Path("figs") / "litellm_apertus70b_user_clusters"

# Create index.html at the top level of the repo
HTML_PATH = Path("index.html")


def _rel_from_html(path: Path) -> str:
    """Compute a relative path from the HTML file location to `path`, normalized with forward slashes."""
    try:
        rel = path.relative_to(HTML_PATH.parent)
    except ValueError:
        # Fallback: best-effort
        rel = path
    return str(rel).replace("\\", "/")


def _img_tag(src: Path, alt: str = "", css_class: str = "plot") -> str:
    rel_str = _rel_from_html(src)
    escaped_src = html.escape(rel_str)
    escaped_alt = html.escape(alt)
    return (
        f'<a href="{escaped_src}" target="_blank" class="plot-link">'
        f'<img src="{escaped_src}" alt="{escaped_alt}" class="{css_class}">'
        "</a>"
    )


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
    return '<div class="gallery">\n' + "\n".join(items) + "\n</div>"


def _parse_top_words(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        _ = f.readline()  # header
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
        _ = f.readline()  # header
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


def _cluster_summary_block() -> str:
    """Small block for the Public-AI user clustering TXT + CSV links and optional preview."""
    cluster_txt_path = USERCLUST_DIR / "apertus70b_cluster_summary.txt"
    feature_csv_path = USERCLUST_DIR / "apertus70b_feature_summary.csv"

    lines: list[str] = []
    lines.append("<p>Additional user-level clustering outputs:</p>")
    lines.append("<ul>")
    if cluster_txt_path.exists():
        lines.append(
            "<li>Cluster summary TXT: "
            f'<a href="{html.escape(_rel_from_html(cluster_txt_path))}">apertus70b_cluster_summary.txt</a></li>'
        )
    else:
        lines.append("<li>Cluster summary TXT: <em>not found</em></li>")

    if feature_csv_path.exists():
        lines.append(
            "<li>Feature summary CSV: "
            f'<a href="{html.escape(_rel_from_html(feature_csv_path))}">apertus70b_feature_summary.csv</a></li>'
        )
    else:
        lines.append("<li>Feature summary CSV: <em>not found</em></li>")
    lines.append("</ul>")

    # Optional TXT preview
    if cluster_txt_path.exists():
        try:
            data = json.loads(cluster_txt_path.read_text(encoding="utf-8"))
            pretty = json.dumps(data, indent=2)
            lines.append("<details><summary>Preview cluster summary TXT</summary>")
            lines.append("<pre>")
            lines.append(html.escape(pretty))
            lines.append("</pre></details>")
        except Exception:
            pass

    return "\n".join(lines)


def build_html() -> None:
    if not FIGS_DIR.exists():
        raise FileNotFoundError(f"FIGS_DIR does not exist: {FIGS_DIR}")

    # ============================================================
    # 1. Public AI logs (LiteLLM_SpendLogs) – Apertus 70B analysis
    # ============================================================

    usage_gallery = _gallery(USAGE_DIR.glob("*.png"))
    perf_gallery = _gallery(PERF_DIR.glob("*.png"))
    behav_gallery = _gallery(BEHAV_DIR.glob("*.png"))
    userclust_gallery = _gallery(USERCLUST_DIR.glob("*.png"))

    public_content = f"""
<p>
This section summarizes the analysis of the <code>LiteLLM_SpendLogs</code> Public AI logs for the
<code>swiss-ai/apertus-70b-instruct</code> model group. It focuses on four main aspects:
usage & workload, performance & efficiency, behavioral stability, and user-level clustering.
</p>

<h3>Usage & workload profiling</h3>
<p>
Time-series usage patterns, hourly/weekday behavior, request volume per session, and concentration of usage
across end users ("whale" curves).
</p>
{usage_gallery}

<h3>Performance & efficiency</h3>
<p>
Daily averages and quantiles for latency, TTFT, generation time, tokens per request, success rate,
and latency shares, plus trimmed latency/TTFT/gen distributions and latency-versus-load plots.
Log-scale variants are included where appropriate.
</p>
{perf_gallery}

<h3>Behavior & stability</h3>
<p>
Prompt/completion length distributions (with log-scale variants), daily quantiles of completion length,
prompt-vs-completion scatter, token-composition ratios (prompt/completion), inter-arrival behavior,
and correlation plots between tokens and latency components.
</p>
{behav_gallery}

<h3>User-level clustering</h3>
<p>
Per-end-user feature vectors (volume, temporal behavior, token structure, latency, and success rate)
are clustered with KMeans (model selection via silhouette + degeneracy checks). PCA is used for 2D
visualization, and per-feature cluster profiles are shown.
</p>
{userclust_gallery}
{_cluster_summary_block()}
"""

    section_public = _section(
        "Public AI logs – Apertus 70B analysis",
        public_content,
        section_id="public-ai-analysis",
    )

    # ============================================================
    # 2. apertus_sft_mixture language-wise clustering
    # ============================================================

    # All languages – scatter / sizes / feature heatmaps
    all_lang_plots = _gallery(FIGS_DIR.glob("scatter_dim1_dim2_*.png"))
    all_lang_sizes = _gallery(FIGS_DIR.glob("cluster_sizes_*.png"))
    all_lang_feats = _gallery(FIGS_DIR.glob("cluster_feature_means_heatmap_*.png"))

    top_words_all_path = FIGS_DIR / "top_words_cluster_langwise_final.txt"
    reps_all_path = FIGS_DIR / "cluster_representatives_cluster_langwise_final.txt"

    all_lang_words_html = (
        _top_words_table(top_words_all_path, "Top 5 words per final cluster (all languages)")
        + _representatives_table(reps_all_path, "Representative prompts per final cluster (all languages)")
    )

    # English-only sections
    figs_dir_en = FIGS_DIR / "english"
    if figs_dir_en.exists():
        en_plots = _gallery(figs_dir_en.glob("scatter_dim1_dim2_*.png"))
        en_sizes = _gallery(figs_dir_en.glob("cluster_sizes_*.png"))
        en_feats = _gallery(figs_dir_en.glob("cluster_feature_means_heatmap_*.png"))

        top_words_en_path = figs_dir_en / "top_words_cluster_langwise_final.txt"
        reps_en_path = figs_dir_en / "cluster_representatives_cluster_langwise_final_en.txt"
        wc_dir_en = figs_dir_en / "wordclouds" / "cluster_langwise_final"
        en_wcs = _gallery(wc_dir_en.glob("*.png"), title_prefix="")

        english_html = f"""
<h3>English-only – cluster plots</h3>
{en_plots}
<h4>Cluster sizes (English only)</h4>
{en_sizes}
<h4>Feature heatmaps (English only)</h4>
{en_feats}

<h3>English-only – word clouds</h3>
{en_wcs}

<h3>English-only – top words and representatives</h3>
{_top_words_table(top_words_en_path, "Top 5 words per final cluster (English only)")
 + _representatives_table(reps_en_path, "Representative prompts per final cluster (English only)")}
"""
    else:
        english_html = """
<h3>English-only results</h3>
<p>No English-only directory found. Run the plotting script with English outputs enabled.</p>
"""

    apertus_content = f"""
<p>
This section summarizes the language-wise clustering results for the
<code>swiss-ai/apertus-sft-mixture</code> dataset. All figures and tables share the same
<code>RUN_TAG = {html.escape(RUN_TAG)}</code> as the poster.
</p>

<h3>All languages – cluster plots</h3>
{all_lang_plots}
<h4>Cluster sizes</h4>
{all_lang_sizes}
<h4>Feature heatmaps</h4>
{all_lang_feats}

<h3>All languages – top words and representatives</h3>
{all_lang_words_html}

{english_html}
"""

    section_apertus = _section(
        "apertus_sft_mixture – language-wise clustering",
        apertus_content,
        section_id="apertus-clustering",
    )

    # ---------------- Navigation ----------------

    nav_html = """
<nav>
  <ul>
    <li><a href="#public-ai-analysis">Public AI logs – Apertus 70B</a></li>
    <li><a href="#apertus-clustering">apertus_sft_mixture clustering</a></li>
  </ul>
</nav>
"""

    # ---------------- Final HTML ----------------

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Apertus analysis & clustering report – {html.escape(RUN_TAG)}</title>
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
    h3 {{
      margin-top: 1.5rem;
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
    details summary {{
      cursor: pointer;
      font-size: 0.9rem;
      margin-top: 0.5rem;
    }}
    pre {{
      background: #111;
      color: #eee;
      padding: 0.75rem;
      font-size: 0.8rem;
      border-radius: 4px;
      overflow-x: auto;
    }}
    a.plot-link {{
      text-decoration: none;
    }}
    a.plot-link img {{
      display: block;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Apertus analysis & clustering report – {html.escape(RUN_TAG)}</h1>
    <p>Public AI logs analysis for Apertus 70B and language-wise clustering of the apertus_sft_mixture dataset.</p>
  </header>
  {nav_html}
  <main>
    {section_public}
    {section_apertus}
  </main>
</body>
</html>
"""

    HTML_PATH.write_text(html_doc, encoding="utf-8")
    print(f"HTML report written to: {HTML_PATH}")


if __name__ == "__main__":
    build_html()
