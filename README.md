# Data Science Lab — Project 9  
**Requires Python 3.11**

This repository contains two components:

1. **Analysis of Public-AI (LiteLLM) inference logs**  
2. **Clustering of the Swiss-AI *apertus-sft-mixture* dataset**

Datasets are confidential and must be placed in `{DATA_DIR}`. Figures are written to `{FIGS_DIR}`.

---

# 1. Public-AI Inference Logs

- **`litellm_tables_summary.py`** — Summary tables for the LiteLLM/OpenWebUI datasets.  
- **`apertus70b_usage.py`** — Usage profiling: request volumes, tokens, users, temporal patterns.  
- **`apertus70b_performance.py`** — Latency/TTFT statistics, efficiency curves, latency distributions.  
- **`apertus70b_behavior.py`** — Verbosity, length distributions, retries, length–latency correlations.  
- **`apertus70b_users_clustering.py`** — User-level feature extraction and KMeans clustering.

---

# 2. Apertus-SFT-Mixture Clustering

- **`apertus_overview.py`** — Basic exploratory analysis and overview plots.  
- **`apertus_embeddings.py`** — Conversation-level embeddings using a multilingual MPNet encoder.  
- **`apertus_clustering.py`** — UMAP reduction + HDBSCAN, Leiden, and KMeans clustering.  
- **`apertus_clustering_plots.py`** — Post-clustering plots (scatter, sizes, heatmaps, word clouds).  
- **`apertus_langwise_clustering.py`** — Language detection + per-language clustering with fallbacks.  
- **`apertus_langwise_clustering_plots.py`** — Language-wise plots and English-only summaries.  
- **`apertus_full_clustering.py`** — Apply sampled KMeans model to the full embedding dataset in batches.

---

# HTML Report

- **`build_html_report.py`** — Generates a consolidated, styled `index.html` with all figures and sections.
