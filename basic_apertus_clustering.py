"""
Conversation-level clustering for swiss-ai/apertus-sft-mixture.

Pipeline:
1. Load sampled Parquet.
2. Parse/flatten `messages` into a single `conversation_text` string.
3. Derive simple conversation-level features (lengths, number of turns).
4. Compute sentence embeddings for each conversation.
5. Reduce dimensionality with PCA.
6. Cluster conversations (HDBSCAN if available, else KMeans).
7. Save enriched DataFrame and arrays (embeddings, PCA coords).
8. Produce diagnostic plots for PCA and cluster-wise feature values.

Edit the constants in the __main__ block as needed.
"""

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud, STOPWORDS


# ---------- Helpers for messages parsing ----------

def _ensure_list_of_dicts(obj: Any) -> list[dict[str, Any]]:
    """Best-effort conversion of `obj` to a list of message dicts."""
    # numpy array of dicts
    if isinstance(obj, np.ndarray):
        return [m for m in obj.tolist() if isinstance(m, dict)]

    # plain list
    if isinstance(obj, list):
        return [m for m in obj if isinstance(m, dict)]

    # single dict
    if isinstance(obj, dict):
        return [obj]

    # JSON string (if ever needed)
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return _ensure_list_of_dicts(parsed)
        except Exception:
            return []

    # generic iterable fallback
    try:
        it = list(obj)
        return [m for m in it if isinstance(m, dict)]
    except Exception:
        return []


def _extract_text_from_content(content: Any) -> str:
    """
    Extract human-readable text from the nested `content` structure.

    Priority:
    - content["text"] if non-empty
    - join all parts[i]["text"] in content["parts"]
    - join all blocks[i]["text"] in content["blocks"]
    """
    if content is None:
        return ""

    if not isinstance(content, dict):
        return str(content)

    texts: list[str] = []

    # 1) direct text field
    txt = content.get("text")
    if isinstance(txt, str) and txt.strip():
        texts.append(txt.strip())

    # 2) parts -> [{'text': ..., 'type': ...}, ...]
    parts = content.get("parts")
    if parts is not None:
        try:
            parts_iter = list(parts)
        except TypeError:
            parts_iter = [parts]
        for p in parts_iter:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())

    # 3) blocks -> [{'text': ..., 'type': 'response', ...}, ...]
    blocks = content.get("blocks")
    if blocks is not None:
        try:
            blocks_iter = list(blocks)
        except TypeError:
            blocks_iter = [blocks]
        for b in blocks_iter:
            if isinstance(b, dict):
                t = b.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())

    return "\n".join(texts)


def _flatten_messages_to_text(messages: Iterable[dict[str, Any]]) -> str:
    """Join a list of message dicts into a single conversation string."""
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip()
        content_text = _extract_text_from_content(m.get("content"))

        if not content_text:
            continue

        if role:
            parts.append(f"{role}: {content_text}")
        else:
            parts.append(content_text)

    return "\n".join(parts)


def add_conversation_text_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse `messages` column into:
    - conversation_text
    - n_turns
    - user_msg_len
    - assistant_msg_len
    - text_length
    """
    parsed_messages: list[list[dict[str, Any]]] = []
    for msgs in df["messages"]:
        parsed_messages.append(_ensure_list_of_dicts(msgs))

    df = df.copy()
    df["messages_parsed"] = parsed_messages

    empty_ratio = (df["messages_parsed"].str.len() == 0).mean()
    print(f"Fraction of conversations with 0 parsed messages: {empty_ratio:.3f}")

    conversation_texts: list[str] = []
    n_turns: list[int] = []
    user_msg_len: list[int] = []
    assistant_msg_len: list[int] = []

    for msgs in parsed_messages:
        conversation_text = _flatten_messages_to_text(msgs)
        conversation_texts.append(conversation_text)
        n_turns.append(len(msgs))

        u_len = 0
        a_len = 0
        for m in msgs:
            content = str(m.get("content", ""))
            role = m.get("role", "")
            if role == "user":
                u_len += len(content)
            elif role == "assistant":
                a_len += len(content)
        user_msg_len.append(u_len)
        assistant_msg_len.append(a_len)

    df["conversation_text"] = conversation_texts
    df["n_turns"] = n_turns
    df["user_msg_len"] = user_msg_len
    df["assistant_msg_len"] = assistant_msg_len
    df["text_length"] = df["conversation_text"].str.len().fillna(0).astype(int)

    # Drop heavy intermediate column if not needed later
    df = df.drop(columns=["messages_parsed"])

    return df


# ---------- Embeddings + PCA + clustering ----------

def compute_embeddings(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute sentence embeddings for each conversation text.

    Requires `sentence-transformers` to be installed.
    """

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return emb


def reduce_dimensionality(
    emb: np.ndarray,
    n_components: int = 50,
    random_state: int = 0,
) -> np.ndarray:
    """
    PCA reduction on embedding matrix (n_samples, dim) -> (n_samples, n_components_eff).
    """
    n_samples, dim = emb.shape
    if n_samples <= 1:
        return emb.copy()

    n_components_eff = min(n_components, dim, n_samples - 1)
    pca = PCA(n_components=n_components_eff, random_state=random_state)
    return pca.fit_transform(emb)


def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 30) -> np.ndarray:
    """
    Cluster with HDBSCAN. Returns label array (noise labeled as -1).

    Requires `hdbscan` to be installed.
    """
    import hdbscan  # type: ignore[import]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    return labels


def cluster_kmeans(
    X: np.ndarray,
    n_clusters: int = 20,
    random_state: int = 0,
) -> np.ndarray:
    """
    Cluster with KMeans. Returns label array.
    """
    n_samples = X.shape[0]
    if n_samples < n_clusters:
        n_clusters = max(2, n_samples)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return labels


# ---------- Plotting ----------

FEATURE_COLUMNS: list[str] = [
    "n_turns",
    "user_msg_len",
    "assistant_msg_len",
    "text_length",
]


def make_cluster_plots(
    df: pd.DataFrame,
    X_pca: np.ndarray,
    label_col: str,
    figs_dir: Path,
) -> None:
    """
    Produce basic diagnostic plots:
    - PCA scatter (PC1 vs PC2) colored by cluster
    - Cluster size bar chart
    - Heatmap of per-cluster feature means
    - Per-feature bar charts of per-cluster means
    """
    figs_dir.mkdir(parents=True, exist_ok=True)
    labels = df[label_col].to_numpy()

    # --- PCA scatter: PC1 vs PC2 colored by cluster ---
    if X_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            s=5,
            alpha=0.7,
            cmap="tab20",
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA scatter (colored by {label_col})")
        cbar = plt.colorbar(scatter)
        cbar.set_label(label_col)
        plt.tight_layout()
        out_path = figs_dir / "pca_scatter_pc1_pc2.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved PCA scatter to: {out_path}")

    # --- Cluster size bar chart ---
    counts = df[label_col].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.xlabel("Cluster")
    plt.ylabel("Number of conversations")
    plt.title(f"Cluster sizes ({label_col})")
    plt.tight_layout()
    out_path = figs_dir / "cluster_sizes.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved cluster size bar chart to: {out_path}")

    # --- Per-cluster feature means ---
    numeric_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not numeric_features:
        print("No FEATURE_COLUMNS found in DataFrame; skipping feature plots.")
        return

    cluster_means = df.groupby(label_col, observed=True)[numeric_features].mean().sort_index()

    # Heatmap of feature means
    plt.figure(figsize=(1.5 * len(numeric_features) + 2, 0.4 * len(cluster_means) + 2))
    im = plt.imshow(cluster_means.values, aspect="auto")
    plt.colorbar(im, label="Mean value")

    plt.xticks(
        ticks=np.arange(len(numeric_features)),
        labels=numeric_features,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        ticks=np.arange(len(cluster_means)),
        labels=cluster_means.index.astype(str).tolist(),
    )
    plt.xlabel("Feature")
    plt.ylabel("Cluster")
    plt.title(f"Per-cluster feature means ({label_col})")
    plt.tight_layout()
    out_path = figs_dir / "cluster_feature_means_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved feature heatmap to: {out_path}")

    # Individual bar plots per feature
    for feat in numeric_features:
        plt.figure(figsize=(8, 4))
        cluster_means[feat].plot(kind="bar")
        plt.xlabel("Cluster")
        plt.ylabel(f"Mean {feat}")
        plt.title(f"Mean {feat} per cluster ({label_col})")
        plt.tight_layout()
        out_path = figs_dir / f"cluster_mean_{feat}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved per-cluster mean plot for {feat} to: {out_path}")


def make_cluster_wordclouds(
    df: pd.DataFrame,
    label_col: str,
    text_col: str,
    figs_dir: Path,
) -> None:
    """
    Compute and save a word cloud image for each cluster, based on `text_col`.

    Saves PNG files into `figs_dir / "wordclouds"`.
    """
    if text_col not in df.columns:
        print(f"Column {text_col!r} not found; skipping word clouds.")
        return

    wc_dir = figs_dir / "wordclouds"
    wc_dir.mkdir(parents=True, exist_ok=True)

    # Basic stopwords, plus the role prefixes we added in conversation_text
    base_stopwords = set(STOPWORDS)
    extra_stopwords = {"user", "assistant", "system"}
    stopwords = base_stopwords | extra_stopwords

    # Group by cluster and build one big text blob per cluster
    grouped = df.groupby(label_col, observed=True)[text_col]

    for cluster_label, texts in grouped:
        # Concatenate all conversation texts for this cluster
        combined = "\n".join(
            t for t in texts.astype(str)
            if isinstance(t, str) and t.strip()
        )
        if not combined.strip():
            print(f"Skipping cluster {cluster_label} (no non-empty text).")
            continue

        # Generate word cloud
        wc = WordCloud(
            width=1600,
            height=900,
            background_color="white",
            stopwords=stopwords,
            max_words=200,
        ).generate(combined)

        # Plot and save
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Cluster {cluster_label}")
        plt.tight_layout()

        out_path = wc_dir / f"cluster_{cluster_label}_wordcloud.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved word cloud for cluster {cluster_label} to: {out_path}")


# ---------- Main pipeline ----------

def main() -> None:
    # Paths and basic config — adjust as needed
    data_path = Path("data/swiss-ai_apertus-sft-mixture/train_sampled_enriched.parquet")
    out_dir = Path("data/swiss-ai_apertus-sft-mixture")
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = out_dir / "train_sampled_conversation_embeddings.npy"
    pca_path = out_dir / "train_sampled_conversation_embeddings_pca.npy"
    clustered_df_path = out_dir / "train_sampled_conversations_clustered.parquet"

    figs_dir = Path("figs/apertus_clustering")

    print(f"Loading sampled data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded shape: {df.shape}")

    # Step 1: add conversation_text and simple features
    print("Parsing `messages` and computing conversation-level features...")
    df = add_conversation_text_and_features(df)

    # Step 2: embeddings
    texts = df["conversation_text"].fillna("").astype(str).tolist()
    print("Computing embeddings for conversations...")
    emb = compute_embeddings(texts)
    print(f"Embeddings shape: {emb.shape}")
    np.save(embeddings_path, emb)
    print(f"Saved embeddings to: {embeddings_path}")

    # Step 3: PCA
    print("Running PCA on embeddings...")
    X_pca = reduce_dimensionality(emb, n_components=50)
    print(f"PCA shape: {X_pca.shape}")
    np.save(pca_path, X_pca)
    print(f"Saved PCA-reduced embeddings to: {pca_path}")

    # Step 4: clustering (try HDBSCAN first, fallback to KMeans)
    try:
        print("Clustering with HDBSCAN...")
        labels = cluster_hdbscan(X_pca, min_cluster_size=30)
        df["cluster_hdbscan"] = labels
        used_algo = "HDBSCAN"
    except ImportError:
        print("hdbscan not installed; falling back to KMeans.")
        labels = cluster_kmeans(X_pca, n_clusters=20)
        df["cluster_kmeans"] = labels
        used_algo = "KMeans"

    # Step 5: save enriched DataFrame with cluster labels
    df.to_parquet(clustered_df_path, index=False)
    print(f"Saved clustered conversations to: {clustered_df_path}")
    print(f"Clustering algorithm used: {used_algo}")

    # Quick cluster summary
    label_col = "cluster_hdbscan" if "cluster_hdbscan" in df.columns else "cluster_kmeans"
    print("\nCluster counts:")
    print(df[label_col].value_counts(dropna=False).sort_index())

     # Step 6: plots
    print("\nGenerating cluster plots...")
    make_cluster_plots(df, X_pca, label_col=label_col, figs_dir=figs_dir)

    print("\nGenerating word clouds per cluster...")
    make_cluster_wordclouds(
        df=df,
        label_col=label_col,
        text_col="conversation_text",
        figs_dir=figs_dir,
    )


if __name__ == "__main__":
    main()
