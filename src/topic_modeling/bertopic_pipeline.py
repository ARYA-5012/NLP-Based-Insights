"""
BERTopic Pipeline for financial earnings call topic modeling.
Discovers themes across the corpus, generates topic labels,
and provides temporal analysis capabilities.
"""
from typing import List, Dict, Tuple, Optional
from loguru import logger


def train_topic_model(
    documents: List[str],
    embeddings=None,
    nr_topics: str = "auto",
    min_topic_size: int = 15,
    n_gram_range: Tuple[int, int] = (1, 3),
    seed_topics: List[List[str]] = None,
):
    """
    Train a BERTopic model on the document corpus.

    Args:
        documents: List of text strings (chunks or full segments).
        embeddings: Pre-computed embeddings (optional, speeds up training).
        nr_topics: Number of topics ("auto" for automatic selection).
        min_topic_size: Minimum documents per topic.
        n_gram_range: N-gram range for topic representation.
        seed_topics: Optional seed topics to guide discovery.

    Returns:
        (topic_model, topics, probs) — the model, topic assignments, probabilities.
    """
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    logger.info(f"Training BERTopic on {len(documents)} documents...")

    # Custom UMAP for better financial text separation
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        random_state=42,
    )

    # Custom HDBSCAN for stable clusters
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        prediction_data=True,
    )

    # Seed topics for finance domain
    if seed_topics is None:
        seed_topics = [
            ["artificial intelligence", "AI", "machine learning", "automation", "generative AI"],
            ["supply chain", "logistics", "inventory", "sourcing", "manufacturing"],
            ["interest rate", "federal reserve", "monetary policy", "inflation", "borrowing"],
            ["regulation", "compliance", "SEC", "antitrust", "privacy"],
            ["revenue growth", "earnings", "margin", "profitability", "guidance"],
            ["cloud", "SaaS", "subscription", "digital transformation"],
            ["workforce", "hiring", "layoffs", "talent", "remote work"],
            ["cybersecurity", "data breach", "ransomware", "security"],
            ["ESG", "sustainability", "climate", "carbon", "renewable"],
            ["M&A", "acquisition", "merger", "partnership", "divestiture"],
        ]

    topic_model = BERTopic(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        top_n_words=10,
        min_topic_size=min_topic_size,
        n_gram_range=n_gram_range,
        seed_topic_list=seed_topics,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info.Topic != -1])
    logger.info(f"BERTopic discovered {n_topics} topics (+ outlier topic)")

    return topic_model, topics, probs


def get_topic_labels(topic_model) -> Dict[int, str]:
    """
    Extract human-readable topic labels from the model.

    Returns:
        {0: "AI & Machine Learning", 1: "Supply Chain", ...}
    """
    topic_info = topic_model.get_topic_info()
    labels = {}

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            labels[-1] = "Outlier / Noise"
            continue

        # Use top keywords as label
        keywords = topic_model.get_topic(topic_id)
        if keywords:
            top_words = [w for w, _ in keywords[:3]]
            labels[topic_id] = " / ".join(top_words).title()
        else:
            labels[topic_id] = f"Topic {topic_id}"

    return labels


def get_topic_details(topic_model) -> List[Dict]:
    """
    Get detailed information about all discovered topics.

    Returns:
        [{"id", "label", "keywords", "count", "coherence"}, ...]
    """
    topic_info = topic_model.get_topic_info()
    details = []

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue

        keywords = topic_model.get_topic(topic_id)
        details.append({
            "id": topic_id,
            "label": get_topic_labels(topic_model).get(topic_id, f"Topic {topic_id}"),
            "keywords": [{"word": w, "weight": round(s, 4)} for w, s in (keywords or [])[:10]],
            "count": int(row.get("Count", 0)),
        })

    return sorted(details, key=lambda x: x["count"], reverse=True)


def topics_over_time(
    topic_model,
    documents: List[str],
    timestamps: List[str],
) -> Dict:
    """
    Track topic evolution over time.

    Args:
        topic_model: Trained BERTopic model.
        documents: Same documents used for training.
        timestamps: List of period labels (e.g., "Q3_2024") parallel to documents.

    Returns:
        DataFrame-like dict with topic frequency per time period.
    """
    try:
        topics_over_time = topic_model.topics_over_time(documents, timestamps)
        return topics_over_time.to_dict("records")
    except Exception as e:
        logger.error(f"topics_over_time failed: {e}")
        return []


def find_representative_docs(
    topic_model,
    documents: List[str],
    topic_id: int,
    top_n: int = 5,
) -> List[str]:
    """Find the most representative documents for a given topic."""
    try:
        repr_docs = topic_model.get_representative_docs(topic_id)
        return repr_docs[:top_n] if repr_docs else []
    except Exception as e:
        logger.warning(f"Could not get representative docs: {e}")
        return []


if __name__ == "__main__":
    # Quick test with mock data
    test_docs = [
        "We are investing heavily in artificial intelligence and machine learning infrastructure.",
        "Our AI platform saw significant adoption across enterprise customers this quarter.",
        "Supply chain disruptions continue to impact our manufacturing operations.",
        "We have diversified our supplier base to mitigate supply chain risks.",
        "Interest rates remain a headwind for our lending business.",
        "The Federal Reserve's monetary policy decisions will impact our outlook.",
        "Revenue grew 15% year-over-year driven by our cloud platform.",
        "Our SaaS subscription model continues to drive recurring revenue growth.",
    ] * 5  # Repeat to meet minimum topic size

    model, topics, probs = train_topic_model(test_docs, min_topic_size=5)
    labels = get_topic_labels(model)
    print(f"Discovered topics: {labels}")
    details = get_topic_details(model)
    for d in details:
        print(f"  Topic {d['id']}: {d['label']} ({d['count']} docs)")
