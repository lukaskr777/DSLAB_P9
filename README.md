# Data Science Lab: Project 9

This project requires **Python 3.11**.

## Analysis of the data logs from Public AI Inference Utility

Data should be placed in the {DATA_DIR} folder. All figures will be created in the {FIGS_DIR} folder.
By default these are "data" and "figs", but these can be changed when running the scripts.

Note: The datasets used in this project are confidential and therefore not included in the repository.

- `litellm_tables_summary.py` generates Markdown summary tables for the two datasets: `litellm` and `openwebui`.
- `apertus70b_usage.py` profiles usage and workload, including time-series of requests, tokens, distinct users, and temporal patterns (hour-of-day / weekday), and produces aggregate “whale” curves and activity-span plots.
- `apertus70b_performance.py` analyzes performance and efficiency, producing daily latency/TTFT/generation statistics, token-per-request curves, latency distributions, and latency–throughput/token correlations.
- `apertus70b_behavior.py` focuses on model behavior and stability, studying verbosity, prompt/completion length distributions, token-composition ratios, inter-arrival times/retries, and correlations between length and latency.
- `apertus70b_user_clusters.py` builds per-user feature vectors (volume, efficiency, temporal behavior, latency, reliability), clusters users with KMeans (with automatic k-selection), and visualizes the resulting user segments in PCA space and via per-cluster feature profiles.

## Clustering of the Swiss AI Apertus SFT Mixture data

- `apertus_overview.py` does basic exploratory data analysis and generates basic plots regarding the data.
- `apertus_embeddings.py` generates embeddings based on the data and the chosen `sentence-transformers` model.
- `apertus_clustering.py` clusters the previously created embeddings according to different algorithms.
- `apertus_clustering_plots.py` generates different plots to compare and evaluate the different clusterings generated.
