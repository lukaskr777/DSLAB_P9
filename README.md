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

---

# Extension of the clustering script to Public-AI logs

The clustering pipeline can be applied to LiteLLM conversations via an export step.

- **`export_litellm_to_parquet.py`** - Exports LiteLLM conversations from the PostgreSQL database into a single Parquet file compatible with the Apertus clustering pipeline.



# 3. Cost & Latency Estimation

This module provides a **high-level simulation framework for estimating LLM inference cost and latency** under varying **user demand patterns** and **compute cluster configurations**. It connects user-level prompt arrivals with cluster-level batching, scheduling, and throughput constraints to model realistic end-to-end performance.

All related code lives in **`cost_estimation_hl/`**.

---

## Codebase Structure

### `cost_estimation_hl/configs/`
Configuration layer defining all simulation parameters:

- **`cluster_configs.json`** — Compute cluster definitions (GPU type, counts, memory, parallelism, throughput)
- **`api_configs.json`** — API-level congigs
- **`usage_simulation_config.py`** — Classes for arrival process configuration
- **`computation_config.py`** — Classes for batching and compute-side simulation parameters

Two Python files define **dataclasses** that load and validate these configurations for use in the simulation.

---

### `cost_estimation_hl/notebooks/`
In-depth analytical notebooks:

- **`userbase_modelling.ipynb`** — Analysis and modelling of user behavior and prompt arrival processes
- **`cost_modelling.ipynb`** — Compute cluster modelling and cost breakdowns
- **`full_modelling.ipynb`** — End-to-end latency modelling combining userbase demand with cluster capacity to evaluate latency across different cluster setups

These notebooks are intended for **exploration, validation, and analysis**.

---

### Simulation Modules

Located at the root of `cost_estimation_hl/`:

- **`usage_simulation.py`** — Defines the userbase simulator and prompt arrival processes
- **`computation_simulation.py`** — Encapsulates prompt processing, batching, and compute-side latency simulation

Together, these modules form the core **simulation loop**.

---

### Local Simulation UI (FastAPI)

The **`html_page/`** folder provides a lightweight **FastAPI-based REST interface** for running simulations locally and inspecting results via a browser.

To run the local server:

```bash
cd cost_estimation_hl/html_page
uvicorn main:app --reload


















