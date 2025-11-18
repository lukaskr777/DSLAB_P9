# Data Science Lab: Project 9

This project requires **Python 3.11.14**.

## Analysis of the data logs from Public AI Inference Utility

Data should be placed in the {DATA_DIR} folder. All figures will be created in the {FIGS_DIR} folder.
By default these are "data" and "figs", but you can change them when running the scripts.

Note: The datasets used in this project are confidential and therefore not included in the repository.

- `table_summary.py` generates Markdown summary tables for the two datasets: `litellm` and `openwebui`.
- `main_table_plots.py` produces several plots analyzing the main tables of the `litellm` dataset.
- `users_usage.py` performs analyses of usage logs, including plotting tasks and clustering of end users of the Public AI LLM.

## Clustering of the Swiss AI Apertus SFT Mixture data

- `apertus_overview.py` does basic exploratory data analysis and generates basic plots regarding the data.
- `apertus_embeddings.py` generates embeddings based on the data and the chosen `sentence-transformers` model.
- `apertus_clustering.py` clusters the previously created embeddings according to different algorithms.
- `apertus_clustering_plots.py` generates different plots to compare and evaluate the different clusterings generated.
