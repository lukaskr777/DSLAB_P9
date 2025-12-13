"""
DON'T RUN THIS SCRIPT, IT WILL MESS UP THE CURRENT REPORT, WHERE IMAGES HAVE BEEN MANUALLY CURATED.

Build a complete HTML report combining:
1) Public-AI (LiteLLM_SpendLogs) analysis for the Apertus-70B model group, and
2) Clustering results for the swiss-ai/apertus-sft-mixture dataset.

The script:
- Locates all generated figures (usage, performance, behavior, user clustering, mixed-languages clustering, 
  per-language clustering, English-only subsets, word clouds, top-word tables, representative prompts).
- Builds galleries, tables, navigation anchors, and an overview section.
- Uses collapsible sections for large galleries and tables.
- Reads human-friendly plot titles from a JSON file; behavior for missing titles is configurable.
- Produces a styled, self-contained `index.html` with clickable, lazy-loaded plots 
  that open full-resolution versions in a new tab.

All paths are resolved relative to the repo layout; the script assumes that the plotting scripts have 
already produced their outputs in the expected `figs/` subdirectories.
"""

import argparse
import html
import json
from pathlib import Path
from typing import Sequence


# ---------- Defaults (can be overridden via CLI) ----------

DEFAULT_OUT_DIR = Path("data/swiss-ai_apertus-sft-mixture")
DEFAULT_EMBEDDINGS_PATH = DEFAULT_OUT_DIR / (
    "small_train_conversation_embeddings__sentence-transformers__paraphrase-multilingual-mpnet-base-v2.npy"
)

DEFAULT_FIGS_ROOT_APERTUS = Path("figs/apertus_clustering")

DEFAULT_USAGE_DIR = Path("figs/litellm_apertus70b_usage")
DEFAULT_PERF_DIR = Path("figs/litellm_apertus70b_performance")
DEFAULT_BEHAV_DIR = Path("figs/litellm_apertus70b_behavior")
DEFAULT_USERCLUST_DIR = Path("figs/litellm_apertus70b_users_clustering")

DEFAULT_HTML_PATH = Path("index.html")

# ---------- Plot title configuration ----------

# JSON file mapping from file names to human-readable titles to display in the gallery.
PLOT_TITLES_PATH = Path("figs/plot_titles.json")

# Behavior for plots whose file name does not appear in PLOT_TITLES_PATH:
#   "use_filename": use a prettified version of the file stem as the caption
#   "skip":         discard the plot (do not display it at all)
PLOT_TITLE_FALLBACK = "use_filename"  # or "skip"

_PLOT_TITLES_CACHE: dict[str, str] = {}


def _load_plot_titles() -> dict[str, str]:
    """Load plot titles from JSON once and cache them."""
    global _PLOT_TITLES_CACHE
    if _PLOT_TITLES_CACHE:
        return _PLOT_TITLES_CACHE

    if not PLOT_TITLES_PATH.exists():
        _PLOT_TITLES_CACHE = {}
        return _PLOT_TITLES_CACHE

    try:
        data = json.loads(PLOT_TITLES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _PLOT_TITLES_CACHE = {str(k): str(v) for k, v in data.items()}
        else:
            _PLOT_TITLES_CACHE = {}
    except Exception:
        _PLOT_TITLES_CACHE = {}

    return _PLOT_TITLES_CACHE


# ---------- HTML helpers ----------


def _rel_from_html(path: Path, html_path: Path) -> str:
    """Compute a relative path from the HTML file location to `path`, normalized with forward slashes."""
    try:
        rel = path.relative_to(html_path.parent)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def _prettify_stem(stem: str) -> str:
    """Prettify a filename stem into a readable caption."""
    # Strip some common prefixes
    for prefix in (
        "scatter_dim1_dim2_",
        "cluster_sizes_",
        "cluster_feature_means_heatmap_",
        "top_words_",
        "cluster_representatives_",
    ):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]

    return stem.replace("_", " ")


def _get_plot_caption(p: Path) -> str | None:
    """
    Resolve a human-friendly caption for a plot path.

    Priority:
    1) Exact file name key in the JSON mapping (e.g., "scatter_dim1_dim2_all.png")
    2) Stem key in the JSON mapping (e.g., "scatter_dim1_dim2_all")
    3) Fallback policy (use prettified filename or skip)
    """
    titles = _load_plot_titles()
    name = p.name
    stem = p.stem

    if name in titles:
        return titles[name]
    if stem in titles:
        return titles[stem]

    if PLOT_TITLE_FALLBACK == "skip":
        return None

    # Default: use prettified filename
    return _prettify_stem(stem)


def _img_tag(src: Path, html_path: Path, alt: str = "", css_class: str = "plot") -> str:
    """Build an HTML <img> tag wrapped in a clickable <a> to the full-resolution image."""
    rel_str = _rel_from_html(src, html_path)
    escaped_src = html.escape(rel_str)
    escaped_alt = html.escape(alt)
    return (
        f'<a href="{escaped_src}" target="_blank" class="plot-link">'
        f'<img src="{escaped_src}" alt="{escaped_alt}" class="{css_class}" loading="lazy">'
        "</a>"
    )


def _gallery(img_paths: Sequence[Path], html_path: Path) -> str:
    """Build a responsive gallery from a list of image paths, using title mapping and fallback policy."""
    items: list[str] = []
    for p in img_paths:
        caption = _get_plot_caption(p)
        if caption is None:
            # Skip plots not in the mapping if fallback policy says so.
            continue
        figure_html = [
            '<figure class="gallery-item">',
            _img_tag(p, html_path, alt=caption),
            f"<figcaption>{html.escape(caption)}</figcaption>",
            "</figure>",
        ]
        items.append("\n".join(figure_html))

    if not items:
        return "<p>No figures found.</p>"

    return '<div class="gallery">\n' + "\n".join(items) + "\n</div>"


def _details(
    summary: str, inner_html: str, css_class: str = "collapsible", open_: bool = False, details_id: str | None = None
) -> str:
    """Wrap content in a <details> element."""
    open_attr = " open" if open_ else ""
    id_attr = f' id="{html.escape(details_id)}"' if details_id else ""
    return (
        f'<details class="{html.escape(css_class)}"{open_attr}{id_attr}>\n'
        f"  <summary>{html.escape(summary)}</summary>\n"
        f"  {inner_html}\n"
        "</details>"
    )


def _html_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    title: str | None = None,
    numeric_cols: Sequence[int] | None = None,
    table_id: str | None = None,
) -> str:
    """Generic HTML table builder with optional numeric alignment and title."""
    numeric_cols = set(numeric_cols or [])

    header_cells: list[str] = []
    for idx, h in enumerate(headers):
        classes: list[str] = []
        if idx in numeric_cols:
            classes.append("numeric")
        class_attr = f' class="{" ".join(classes)}"' if classes else ""
        header_cells.append(f"<th{class_attr}>{html.escape(h)}</th>")

    body_rows_html: list[str] = []
    for row in rows:
        cells: list[str] = []
        for idx, val in enumerate(row):
            classes = []
            if idx in numeric_cols:
                classes.append("numeric")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f"<td{class_attr}>{html.escape(val)}</td>")
        body_rows_html.append("<tr>" + "".join(cells) + "</tr>")

    title_html = f"<h3>{html.escape(title)}</h3>\n" if title else ""
    id_attr = f' id="{html.escape(table_id)}"' if table_id else ""

    return (
        title_html
        + '<div class="table-wrapper">\n'
        + f'  <table{id_attr}>\n'
        + "    <thead>\n"
        + "      <tr>"
        + "".join(header_cells)
        + "</tr>\n"
        + "    </thead>\n"
        + "    <tbody>\n"
        + "      "
        + "\n      ".join(body_rows_html)
        + "\n"
        + "    </tbody>\n"
        + "  </table>\n"
        + "</div>\n"
    )


def _section(title: str, content: str, section_id: str | None = None) -> str:
    """Wrap content in a <section> with an <h2> heading."""
    if section_id is None:
        section_id = title.lower().replace(" ", "-")
    return f"""
<section id="{html.escape(section_id)}">
  <h2>{html.escape(title)}</h2>
  {content}
</section>
"""


# ---------- Parsing helpers ----------


def _parse_top_words(path: Path) -> list[dict[str, str]]:
    """Parse top-words TSV: cluster, word1, count1, word2, count2, ..."""
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        _ = f.readline()  # skip header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            # Require at least cluster + one (word, count) pair
            if len(parts) < 3:
                continue

            row: dict[str, str] = {"cluster": parts[0]}
            # Parts: [cluster, word1, count1, word2, count2, ...]
            # Use up to 5 (word, count) pairs
            max_pairs = min((len(parts) - 1) // 2, 5)
            for rank in range(1, max_pairs + 1):
                word_idx = 2 * rank - 1
                count_idx = 2 * rank
                word = parts[word_idx]
                count = parts[count_idx] if count_idx < len(parts) else ""
                row[f"word{rank}"] = word
                row[f"count{rank}"] = count

            rows.append(row)

    # Sort numerically by cluster if possible
    def _cluster_sort_key(r: dict[str, str]):
        c = r.get("cluster", "")
        cs = c.lstrip("-")
        if cs.isdigit():
            return (0, int(c))
        return (1, c)

    rows.sort(key=_cluster_sort_key)
    return rows


def _top_words_table(path: Path, title: str, table_id: str | None = None) -> str:
    """Generate an HTML table of top words per cluster from the given TSV file."""
    rows = _parse_top_words(path)
    if not rows:
        return f"<p>No top-words file found at {html.escape(str(path))}.</p>"

    headers: list[str] = ["cluster"]
    for k in range(1, 6):
        headers.append(f"word{k}")
        headers.append(f"count{k}")

    table_rows: list[list[str]] = []
    for r in rows:
        row_vals: list[str] = []
        for h in headers:
            row_vals.append(r.get(h, ""))
        table_rows.append(row_vals)

    # numeric columns: counts only
    numeric_cols = [idx for idx, h in enumerate(headers) if h.startswith("count")]

    return _html_table(
        headers=headers,
        rows=table_rows,
        title=title,
        numeric_cols=numeric_cols,
        table_id=table_id,
    )


def _parse_representatives(path: Path) -> list[dict[str, str]]:
    """Parse representative prompts TSV: cluster, full_prompt, ..."""
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        _ = f.readline()  # skip header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            cluster = parts[0]
            full = parts[1]
            rows.append({"cluster": cluster, "full_prompt": full})

    # Sort by cluster (numeric if possible)
    def _cluster_sort_key(r: dict[str, str]):
        c = r.get("cluster", "")
        cs = c.lstrip("-")
        if cs.isdigit():
            return (0, int(c))
        return (1, c)

    rows.sort(key=_cluster_sort_key)
    return rows


def _representatives_table(path: Path, title: str) -> str:
    """Generate an HTML table of representative prompts per cluster from the given TSV file."""
    rows = _parse_representatives(path)
    if not rows:
        return f"<p>No representatives file found at {html.escape(str(path))}.</p>"

    headers = ["cluster", "representative prompt"]
    table_rows: list[list[str]] = []
    for r in rows:
        table_rows.append([r["cluster"], r["full_prompt"]])

    table_html = _html_table(headers=headers, rows=table_rows, title=None, numeric_cols=[])

    return _details(
        summary=title,
        inner_html=table_html,
        css_class="collapsible",
        open_=False,
    )


def _cluster_summary_block(path: Path) -> str:
    """
    Return HTML describing cluster summary (stats + feature means) for the Public AI user clusters.

    Feature means are rendered as one feature per row, with columns:
    [cluster, feature, mean_value].
    """
    if not path.exists():
        return f"<p>No cluster summary file found at {html.escape(str(path))}.</p>"

    try:
        raw_text = path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except Exception:
        # Fallback: show raw text if parsing fails
        return """
<p>Cluster summary could not be parsed as JSON. Raw content:</p>
<pre>{}</pre>
""".format(
            html.escape(path.read_text(encoding="utf-8"))
        )

    k_sel = data.get("k_selected", "N/A")
    sil = data.get("silhouette", "N/A")
    clusters = data.get("clusters", [])

    if not isinstance(clusters, list) or not clusters:
        return f"<p>Cluster summary has no cluster entries in {html.escape(str(path))}.</p>"

    base_keys = ("cluster", "n_users", "median_requests", "median_req_per_active_day")
    other_keys = {key for c in clusters for key in c.keys() if key not in base_keys}

    # --- Main per-cluster table (core stats) ---
    core_headers = list(base_keys)
    core_rows: list[list[str]] = []
    for c in clusters:
        row: list[str] = []
        for h in core_headers:
            val = c.get(h, "")
            row.append(str(val))
        core_rows.append(row)

    core_numeric_cols = [1, 2, 3]  # n_users, median_requests, median_req_per_active_day
    core_table_html = _html_table(
        headers=core_headers,
        rows=core_rows,
        title="Per-cluster statistics",
        numeric_cols=core_numeric_cols,
        table_id="publicai-cluster-core",
    )

    # --- Feature means as one feature per line: [cluster, feature, mean_value] ---

    feat_means_key = "feature_means"
    has_feat_means = feat_means_key in other_keys and any(
        isinstance(c.get(feat_means_key), dict) for c in clusters
    )

    if has_feat_means:
        feat_headers: list[str] = ["cluster", "feature", "mean_value"]
        feat_rows: list[list[str]] = []

        for c in clusters:
            cluster_id = str(c.get("cluster", ""))
            feat_means = c.get(feat_means_key) or {}
            if not isinstance(feat_means, dict):
                continue

            # Sort features by name for stable output
            for feat_name, val in sorted(feat_means.items()):
                if isinstance(val, (int, float)):
                    val_str = f"{val:.4g}"  # compact numeric formatting
                else:
                    val_str = str(val)
                feat_rows.append([cluster_id, feat_name, val_str])

        feat_numeric_cols = [2]  # mean_value
        feat_table_html = _html_table(
            headers=feat_headers,
            rows=feat_rows,
            title="Per-cluster feature means",
            numeric_cols=feat_numeric_cols,
            table_id="publicai-cluster-features",
        )
    else:
        feat_table_html = ""

    metrics_html = f"""
<p>
  <span class="metric-badge">k = {html.escape(str(k_sel))}</span>
  <span class="metric-badge">silhouette = {html.escape(str(sil))}</span>
</p>
"""

    inner = metrics_html + core_table_html + feat_table_html
    return f'<div class="cluster-summary-block">\n{inner}\n</div>'


# ---------- Section builders ----------


def _overview_section_html() -> str:
    """Short overview section with methodology."""
    content = """
<p>
This report combines two complementary analyses:
</p>
<ul>
  <li><strong>Public AI logs – Apertus 70B:</strong> Usage, performance, behavior, and user-level clustering
      derived from <code>LiteLLM_SpendLogs</code> for the <code>swiss-ai/apertus-70b-instruct</code> model group.</li>
  <li><strong>Clustering of apertus_sft_mixture conversations:</strong> UMAP-based visualization and clustering
      of conversations in the <code>swiss-ai/apertus-sft-mixture</code> dataset, including a mixed-languages run,
      per-language clustering, and an English-only subset.</li>
</ul>
"""
    return _section("Overview", content, section_id="overview")


def _public_ai_section_html(
    html_path: Path,
    usage_paths: Sequence[Path],
    perf_paths: Sequence[Path],
    behav_paths: Sequence[Path],
    userclust_paths: Sequence[Path],
    cluster_summary_path: Path,
) -> str:
    usage_gallery = _gallery(usage_paths, html_path)
    perf_gallery = _gallery(perf_paths, html_path)
    behav_gallery = _gallery(behav_paths, html_path)
    userclust_gallery = _gallery(userclust_paths, html_path)
    cluster_summary_html = _cluster_summary_block(cluster_summary_path)

    # Section mini-TOC
    section_toc = """
<ul class="section-toc">
  <li><a href="#public-ai-usage">Usage &amp; workload profiling</a></li>
  <li><a href="#public-ai-performance">Performance &amp; efficiency</a></li>
  <li><a href="#public-ai-behavior">Behavior &amp; stability</a></li>
  <li><a href="#public-ai-userclust">User-level clustering</a></li>
</ul>
"""

    public_content = f"""
<p>
This section summarizes the analysis of the <code>LiteLLM_SpendLogs</code> Public AI logs for the
<code>swiss-ai/apertus-70b-instruct</code> model group.
It focuses on usage &amp; workload, performance &amp; efficiency, behavioral stability, and user-level clustering.
</p>

{section_toc}

<h3 id="public-ai-usage">Usage &amp; workload profiling</h3>
{_details("Show usage plots", usage_gallery, open_=False)}

<h3 id="public-ai-performance">Performance &amp; efficiency</h3>
{_details("Show performance plots", perf_gallery, open_=False)}

<h3 id="public-ai-behavior">Behavior &amp; stability</h3>
{_details("Show behavior plots", behav_gallery, open_=False)}

<h3 id="public-ai-userclust">User-level clustering</h3>
{_details("Show user-clustering plots", userclust_gallery, open_=False)}
{_details("Show numeric cluster summary", cluster_summary_html, open_=False)}
"""

    return _section(
        "Public AI logs – Apertus 70B analysis",
        public_content,
        section_id="public-ai-analysis",
    )


def _apertus_section_html(
    html_path: Path,
    mixed_dir: Path,
    langwise_dir: Path,
) -> str:
    """
    Build the apertus_sft_mixture clustering section, split into:
    - Mixed languages
    - Per-language clustering
    - English only

    All feature heatmaps are intentionally discarded in this section.
    """

    # ---------- Mixed languages (single clustering over all languages) ----------

    if mixed_dir.exists():
        mixed_scatter_paths = sorted(mixed_dir.glob("scatter_dim1_dim2_*.png"))
        mixed_sizes_paths = sorted(mixed_dir.glob("cluster_sizes_*.png"))

        # Updated paths for mixed-languages wordclouds, top words, and representatives
        mixed_wc_dir = mixed_dir / "wordclouds" / "cluster_hdbscan"
        mixed_wc_paths = sorted(mixed_wc_dir.glob("*.png")) if mixed_wc_dir.exists() else []

        mixed_top_words_path = mixed_dir / "top_words_cluster_hdbscan.txt"
        mixed_reps_path = mixed_dir / "cluster_representatives_cluster_hdbscan.txt"

        print(f"[INFO] Found {len(mixed_scatter_paths)} mixed-languages scatter plots in {mixed_dir}")
        print(f"[INFO] Found {len(mixed_sizes_paths)} mixed-languages cluster-size plots in {mixed_dir}")
        print(f"[INFO] Found {len(mixed_wc_paths)} mixed-languages wordcloud plots in {mixed_wc_dir}")

        mixed_scatter_gallery = _gallery(mixed_scatter_paths, html_path)
        mixed_sizes_gallery = _gallery(mixed_sizes_paths, html_path)
        mixed_wc_gallery = _gallery(mixed_wc_paths, html_path)

        mixed_words_html = (
            _top_words_table(
                mixed_top_words_path,
                "Top 5 words per cluster (mixed languages)",
                table_id="mixed-top-words",
            )
            + _representatives_table(
                mixed_reps_path,
                "Show representative prompts per cluster (mixed languages)",
            )
        )

        mixed_html = f"""
<h3 id="apertus-mixed">Mixed languages</h3>

{_details("Show mixed-languages cluster scatter plots", mixed_scatter_gallery, open_=False)}
<h4 id="apertus-mixed-sizes">Cluster sizes (mixed languages)</h4>
{_details("Show mixed-languages cluster size plots", mixed_sizes_gallery, open_=False)}

<h4 id="apertus-mixed-wordclouds">Mixed-languages word clouds</h4>
{_details("Show mixed-languages word clouds", mixed_wc_gallery, open_=False)}

<h4 id="apertus-mixed-topwords">Mixed-languages top words &amp; representatives</h4>
{_details(
    "Show mixed-languages top words and representative prompts",
    mixed_words_html,
    open_=False,
)}
"""
    else:
        mixed_html = f"""
<h3 id="apertus-mixed">Mixed languages</h3>
<p>Mixed-languages figures directory not found: <code>{html.escape(str(mixed_dir))}</code>.</p>
"""

    # ---------- Per-language clustering (current "all_languages") ----------

    all_lang_scatter_paths = sorted(langwise_dir.glob("scatter_dim1_dim2_*.png"))
    all_lang_sizes_paths = sorted(langwise_dir.glob("cluster_sizes_*.png"))

    print(f"[INFO] Found {len(all_lang_scatter_paths)} per-language scatter plots in {langwise_dir}")
    print(f"[INFO] Found {len(all_lang_sizes_paths)} per-language cluster-size plots in {langwise_dir}")

    all_lang_scatter_gallery = _gallery(all_lang_scatter_paths, html_path)
    all_lang_sizes_gallery = _gallery(all_lang_sizes_paths, html_path)

    top_words_all_path = langwise_dir / "top_words_cluster_langwise_final.txt"
    reps_all_path = langwise_dir / "cluster_representatives_cluster_langwise_final.txt"

    all_lang_words_html = (
        _top_words_table(
            top_words_all_path,
            "Top 5 words per final cluster (per-language clustering)",
            table_id="per-lang-top-words",
        )
        + _representatives_table(
            reps_all_path,
            "Show representative prompts per final cluster (per-language clustering)",
        )
    )

    per_lang_html = f"""
<h3 id="apertus-per-lang">Per-language clustering</h3>

{_details("Show per-language cluster scatter plots", all_lang_scatter_gallery, open_=False)}
<h4 id="apertus-per-lang-sizes">Cluster sizes (per-language)</h4>
{_details("Show per-language cluster size plots", all_lang_sizes_gallery, open_=False)}

<h4 id="apertus-per-lang-topwords">Per-language top words &amp; representatives</h4>
{_details(
    "Show per-language top words and representative prompts",
    all_lang_words_html,
    open_=False,
)}
"""

    # ---------- English-only subset (inside langwise_dir/english) ----------

    figs_dir_en = langwise_dir / "english"
    if figs_dir_en.exists():
        en_scatter_paths = sorted(figs_dir_en.glob("scatter_dim1_dim2_*.png"))
        en_sizes_paths = sorted(figs_dir_en.glob("cluster_sizes_*.png"))
        wc_dir_en = figs_dir_en / "wordclouds" / "cluster_langwise_final"
        en_wc_paths = sorted(wc_dir_en.glob("*.png")) if wc_dir_en.exists() else []

        print(f"[INFO] Found {len(en_scatter_paths)} English-only scatter plots in {figs_dir_en}")
        print(f"[INFO] Found {len(en_sizes_paths)} English-only cluster-size plots in {figs_dir_en}")
        print(f"[INFO] Found {len(en_wc_paths)} English-only wordcloud plots in {wc_dir_en}")

        en_scatter_gallery = _gallery(en_scatter_paths, html_path)
        en_sizes_gallery = _gallery(en_sizes_paths, html_path)
        en_wc_gallery = _gallery(en_wc_paths, html_path)

        top_words_en_path = figs_dir_en / "top_words_cluster_langwise_final.txt"
        reps_en_path = figs_dir_en / "cluster_representatives_cluster_langwise_final_en.txt"

        english_html = f"""
<h3 id="apertus-english">English only</h3>

{_details("Show English-only cluster scatter plots", en_scatter_gallery, open_=False)}
<h4 id="apertus-english-sizes">Cluster sizes (English only)</h4>
{_details("Show English-only cluster size plots", en_sizes_gallery, open_=False)}

<h4 id="apertus-english-wordclouds">English-only word clouds</h4>
{_details("Show English-only word clouds", en_wc_gallery, open_=False)}

<h4 id="apertus-english-topwords">English-only top words &amp; representatives</h4>
{_details(
    "Show English-only top words and representative prompts",
    _top_words_table(
        top_words_en_path,
        "Top 5 words per final cluster (English only)",
        table_id="en-top-words",
    )
    + _representatives_table(
        reps_en_path,
        "Show representative prompts per final cluster (English only)",
    ),
    open_=False,
)}
"""
    else:
        english_html = f"""
<h3 id="apertus-english">English only</h3>
<p>No English-only directory found. Expected at <code>{html.escape(str(figs_dir_en))}</code>.</p>
"""

    # ---------- Section mini-TOC and wrapper ----------

    section_toc = """
<ul class="section-toc">
  <li><a href="#apertus-mixed">Mixed languages</a></li>
  <li><a href="#apertus-per-lang">Per-language clustering</a></li>
  <li><a href="#apertus-english">English only</a></li>
</ul>
"""

    apertus_content = f"""
<p>
This section summarizes the clustering results for the
<code>swiss-ai/apertus-sft-mixture</code> dataset.
We show a mixed-languages clustering run, per-language clustering, and an English-only subset.
</p>

{section_toc}

{mixed_html}
{per_lang_html}
{english_html}
"""
    return _section(
        "Clustering of apertus_sft_mixture conversations",
        apertus_content,
        section_id="apertus-clustering",
    )


# ---------- Main HTML builder ----------


def build_html(
    html_path: Path,
    figs_root_apertus: Path,
    usage_dir: Path,
    perf_dir: Path,
    behav_dir: Path,
    userclust_dir: Path,
    embeddings_path: Path,
) -> None:
    tag = embeddings_path.stem
    run_suffix = "_langwise"
    run_tag = f"{tag}{run_suffix}"

    # Mixed-languages clustering (all languages in one run)
    mixed_dir = figs_root_apertus / tag
    if not mixed_dir.exists():
        print(f"[WARN] Mixed-languages figs directory does not exist: {mixed_dir}")

    # Per-language clustering (the previous 'all_languages' langwise run)
    langwise_dir = figs_root_apertus / run_tag
    if not langwise_dir.exists():
        raise FileNotFoundError(f"Langwise FIGS_DIR does not exist: {langwise_dir}")

    # Public AI logs – gather figures and log counts
    usage_paths = sorted(p for p in usage_dir.glob("*.png") if p.is_file())
    perf_paths = sorted(p for p in perf_dir.glob("*.png") if p.is_file())
    behav_paths = sorted(p for p in behav_dir.glob("*.png") if p.is_file())
    userclust_paths = sorted(p for p in userclust_dir.glob("*.png") if p.is_file())

    print(f"[INFO] Found {len(usage_paths)} usage plots in {usage_dir}")
    print(f"[INFO] Found {len(perf_paths)} performance plots in {perf_dir}")
    print(f"[INFO] Found {len(behav_paths)} behavior plots in {behav_dir}")
    print(f"[INFO] Found {len(userclust_paths)} user-clustering plots in {userclust_dir}")

    cluster_summary_path = userclust_dir / "apertus70b_cluster_summary.txt"

    # Build sections
    section_overview = _overview_section_html()
    section_public = _public_ai_section_html(
        html_path=html_path,
        usage_paths=usage_paths,
        perf_paths=perf_paths,
        behav_paths=behav_paths,
        userclust_paths=userclust_paths,
        cluster_summary_path=cluster_summary_path,
    )
    section_apertus = _apertus_section_html(
        html_path=html_path,
        mixed_dir=mixed_dir,
        langwise_dir=langwise_dir,
    )

    # Navigation
    nav_html = """
<nav>
  <ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#public-ai-analysis">Public AI logs – Apertus 70B analysis</a></li>
    <li><a href="#apertus-clustering">Clustering of apertus_sft_mixture conversations</a></li>
  </ul>
</nav>
"""

    # Final HTML
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Public AI Logs Analysis &amp; Clustering of the apertus_sft_mixture dataset</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      line-height: 1.5;
      background-color: #f7f7f7;
      font-size: 16px;
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
    h4 {{
      margin-top: 1.2rem;
    }}
    .section-toc {{
      list-style: disc;
      padding-left: 1.5rem;
      margin-bottom: 1rem;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
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
      object-fit: contain;
      display: block;
    }}
    .gallery-item figcaption {{
      font-size: 0.85rem;
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
    thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    td.numeric, th.numeric {{
      text-align: right;
      font-variant-numeric: tabular-nums;
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
    details.collapsible summary {{
      cursor: pointer;
      font-weight: 500;
      margin: 0.25rem 0 0.5rem;
    }}
    .metric-badge {{
      display: inline-block;
      padding: 0.2rem 0.5rem;
      margin-right: 0.5rem;
      border-radius: 4px;
      background: #eef3ff;
      font-weight: 600;
      font-size: 0.85rem;
    }}
    .cluster-summary-block {{
      margin-top: 0.75rem;
    }}

    @media print {{
      body {{
        background: #fff;
      }}
      nav {{
        display: none;
      }}
      main {{
        max-width: none;
        margin: 0;
        padding: 1rem;
      }}
      header {{
        background: #fff;
        color: #000;
        border-bottom: 1px solid #000;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Public AI Logs Analysis &amp; Clustering of the apertus_sft_mixture dataset</h1>
    <p>Public AI logs analysis (for the Apertus 70B model) and clustering of the apertus_sft_mixture dataset.</p>
  </header>
  {nav_html}
  <main>
    {section_overview}
    {section_public}
    {section_apertus}
  </main>
</body>
</html>
"""

    html_path.write_text(html_doc, encoding="utf-8")
    print(f"[INFO] HTML report written to: {html_path}")


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HTML report for Public AI logs and apertus_sft_mixture clustering."
    )
    parser.add_argument(
        "--html-path",
        type=Path,
        default=DEFAULT_HTML_PATH,
        help="Output HTML path (default: index.html)",
    )
    parser.add_argument(
        "--figs-root-apertus",
        type=Path,
        default=DEFAULT_FIGS_ROOT_APERTUS,
        help="Root directory for apertus_sft_mixture clustering figures.",
    )
    parser.add_argument(
        "--usage-dir",
        type=Path,
        default=DEFAULT_USAGE_DIR,
        help="Directory for usage/workload figures.",
    )
    parser.add_argument(
        "--perf-dir",
        type=Path,
        default=DEFAULT_PERF_DIR,
        help="Directory for performance/efficiency figures.",
    )
    parser.add_argument(
        "--behav-dir",
        type=Path,
        default=DEFAULT_BEHAV_DIR,
        help="Directory for behavior/stability figures.",
    )
    parser.add_argument(
        "--userclust-dir",
        type=Path,
        default=DEFAULT_USERCLUST_DIR,
        help="Directory for user-clustering figures and summary JSON.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to the (small) embeddings file used to derive the run tag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_html(
        html_path=args.html_path,
        figs_root_apertus=args.figs_root_apertus,
        usage_dir=args.usage_dir,
        perf_dir=args.perf_dir,
        behav_dir=args.behav_dir,
        userclust_dir=args.userclust_dir,
        embeddings_path=args.embeddings_path,
    )


if __name__ == "__main__":
    main()
