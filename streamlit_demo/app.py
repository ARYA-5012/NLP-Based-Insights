import sys
import os
import json

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.retrieval.vector_store import get_collection

# ─── PATHS ───
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPICS_JSON = os.path.join(ROOT, "data", "models", "bertopic_model", "topics.json")
SIZING_CSV  = os.path.join(ROOT, "data", "raw", "insights", "sizing_analysis.csv")
TWEETS_CSV  = os.path.join(ROOT, "data", "raw", "insights", "tweet_issues.csv")

# ─── PAGE CONFIG ───
st.set_page_config(page_title="Customer Voice Intelligence", layout="wide", page_icon="🗣️")

# ─── CACHED LOADERS ───
@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-mpnet-base-v2")

@st.cache_resource
def load_collection():
    return get_collection()

@st.cache_data
def load_sizing_data():
    if os.path.exists(SIZING_CSV):
        return pd.read_csv(SIZING_CSV)
    return None

@st.cache_data
def load_tweet_issues():
    if os.path.exists(TWEETS_CSV):
        return pd.read_csv(TWEETS_CSV)
    return None

@st.cache_data
def load_topic_data():
    """Parse topics.json into a usable DataFrame."""
    if not os.path.exists(TOPICS_JSON):
        return None, None
    with open(TOPICS_JSON, "r") as f:
        data = json.load(f)

    # Build topic_sizes from topic_freq list
    # Format: [[topic_id, topic_id, count], ...]
    freq_list = data.get("topic_freq", [])
    sizes = {}
    for item in freq_list:
        if len(item) >= 3:
            tid = item[0]
            count = item[2]
            sizes[str(tid)] = count

    # Build topic representations
    reps = data.get("topic_representations", {})
    labels = data.get("topic_labels", {})

    rows = []
    for tid, words in reps.items():
        if tid == "-1":  # Skip outliers
            continue
        top_word = words[0][0] if words else "unknown"
        top_score = words[0][1] if words else 0.0
        label = labels.get(tid, f"Topic {tid}")
        # Human-readable label: take first meaningful word
        readable = top_word.title()
        rows.append({
            "topic_id": int(tid),
            "label": readable,
            "full_label": label,
            "top_word": top_word,
            "top_score": round(top_score, 4),
            "size": sizes.get(tid, 0),
            "top_words": ", ".join([w[0] for w in words[:5]])
        })

    df = pd.DataFrame(rows).sort_values("size", ascending=False)
    return df, reps

def search_chromadb(query_text, n_results=5, where_filter=None):
    model = load_embedding_model()
    collection = load_collection()
    query_embedding = model.encode(query_text).tolist()
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_params["where"] = where_filter
    return collection.query(**query_params)

# ─── HEADER ───
st.title("🗣️ Customer Voice Intelligence")
st.markdown("### Insights from E-Commerce Reviews & Support Tweets")

collection = load_collection()
st.sidebar.metric("📊 Total Indexed Documents", f"{collection.count():,}")

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Universal Search", "👗 Product Insights", "🐦 Support Ops", "📊 Evaluation"]
)

# ════════════════════════════════════════════════════
# PAGE 1: UNIVERSAL SEARCH
# ════════════════════════════════════════════════════
if page == "🔍 Universal Search":
    st.header("Semantic Search")
    query = st.text_input(
        "Ask anything",
        placeholder="e.g. 'shipping delays', 'dress sizing issues', 'refund problems'"
    )
    filter_source = st.selectbox("Filter by Source", ["All", "clothing_reviews", "twitter_support"])

    if st.button("🔍 Search") and query:
        with st.spinner("Searching 20,000+ documents..."):
            where = {"source": filter_source} if filter_source != "All" else None
            results = search_chromadb(query, n_results=8, where_filter=where)

            if results and results["documents"][0]:
                st.success(f"Found {len(results['documents'][0])} relevant results")
                for i in range(len(results["documents"][0])):
                    doc  = results["documents"][0][i]
                    meta = results["metadatas"][0][i]
                    dist = results["distances"][0][i]
                    score = max(0, 1 - dist)
                    icon = "👗" if meta.get("source") == "clothing_reviews" else "🐦"
                    with st.expander(f"{icon} Score: {score:.3f} — {doc[:80]}..."):
                        st.write(doc)
                        st.caption(
                            f"Source: **{meta.get('source','N/A')}** | "
                            f"Category: {meta.get('category','N/A')} | "
                            f"Rating: {meta.get('rating','N/A')}"
                        )
            else:
                st.warning("No results found. Try a different query.")

# ════════════════════════════════════════════════════
# PAGE 2: PRODUCT INSIGHTS
# ════════════════════════════════════════════════════
elif page == "👗 Product Insights":
    st.header("Women's E-Commerce Reviews Analysis")
    sizing_df = load_sizing_data()

    col1, col2 = st.columns(2)
    with col1:
        if sizing_df is not None:
            st.metric("Total Reviews Analyzed", f"{len(sizing_df):,}")
            counts = sizing_df["sizing_feedback"].value_counts()
            fig = px.pie(
                values=counts.values, names=counts.index,
                title="Sizing Feedback Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Place `sizing_analysis.csv` in `data/raw/insights/`.")

    with col2:
        if sizing_df is not None and "rating" in sizing_df.columns:
            rc = sizing_df["rating"].value_counts().sort_index()
            fig2 = px.bar(
                x=[f"{int(r)} ⭐" for r in rc.index], y=rc.values,
                title="Rating Distribution", color_discrete_sequence=["#636EFA"]
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Aspect-Based Search")
    category = st.selectbox("Product Category", ["Dresses", "Knits", "Blouses", "Jeans", "Pants", "Shorts"])
    if st.button("Analyze Category"):
        with st.spinner(f"Searching reviews about {category}..."):
            results = search_chromadb(
                f"issues problems complaints about {category}", n_results=5,
                where_filter={"source": "clothing_reviews"}
            )
            if results and results["documents"][0]:
                st.subheader(f"Common Feedback for {category}")
                for doc in results["documents"][0]:
                    st.warning(doc)

# ════════════════════════════════════════════════════
# PAGE 3: SUPPORT OPS
# ════════════════════════════════════════════════════
elif page == "🐦 Support Ops":
    st.header("Twitter Support Operations")
    tweet_df = load_tweet_issues()

    col1, col2 = st.columns(2)
    with col1:
        if tweet_df is not None:
            st.metric("Tweets Classified", f"{len(tweet_df):,}")
            ic = tweet_df["issue_type"].value_counts()
            fig = px.bar(
                x=ic.values, y=ic.index, orientation='h',
                title="Ticket Volume by Issue Type",
                color_discrete_sequence=["#EF553B"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Place `tweet_issues.csv` in `data/raw/insights/`.")

    with col2:
        if tweet_df is not None and "confidence" in tweet_df.columns:
            fig2 = px.histogram(
                tweet_df, x="confidence", nbins=20,
                title="Classification Confidence Distribution",
                color_discrete_sequence=["#00CC96"]
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Find Similar Tickets")
    ticket_text = st.text_area("Paste incoming tweet:", "My package was supposed to be here yesterday.")
    if st.button("Find Similar Cases"):
        with st.spinner("Searching support history..."):
            results = search_chromadb(ticket_text, n_results=5, where_filter={"source": "twitter_support"})
            if results and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    st.success(f"**Similar Case {i+1}:** {doc}")
                    st.caption(f"Author: {meta.get('author','N/A')}")

# ════════════════════════════════════════════════════
# PAGE 4: EVALUATION  ← NEW
# ════════════════════════════════════════════════════
elif page == "📊 Evaluation":
    st.header("📊 Model Evaluation & Quality Metrics")
    st.markdown(
        "This page shows how well the NLP models performed on real data — "
        "cluster coherence, topic distributions, and classification confidence."
    )

    topic_df, topic_reps = load_topic_data()
    tweet_df = load_tweet_issues()
    sizing_df = load_sizing_data()

    # ── Section 1: BERTopic Cluster Sizes ──
    st.markdown("---")
    st.subheader("🗂️ BERTopic: Cluster Sizes")
    st.caption(
        "Each bar is a discovered topic. Larger clusters = more documents share that theme. "
        "Well-separated, large clusters indicate good topic coherence."
    )

    if topic_df is not None:
        top20 = topic_df.head(20)
        fig_clusters = px.bar(
            top20, x="size", y="label", orientation="h",
            title="Top 20 Topics by Document Count",
            labels={"size": "Documents in Cluster", "label": "Topic"},
            color="size",
            color_continuous_scale="Blues",
            text="size"
        )
        fig_clusters.update_traces(textposition="outside")
        fig_clusters.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_clusters, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Topics Discovered", len(topic_df))
        col2.metric("Largest Cluster", f"{topic_df['size'].max():,} docs")
        col3.metric("Avg Cluster Size", f"{int(topic_df['size'].mean()):,} docs")
    else:
        st.warning("BERTopic model not found at `data/models/bertopic_model/topics.json`.")

    # ── Section 2: Topic Coherence (c-TF-IDF scores) ──
    st.markdown("---")
    st.subheader("🎯 Topic Coherence: c-TF-IDF Top-Word Scores")
    st.caption(
        "The c-TF-IDF score of the top word in each topic measures how uniquely that word "
        "defines the cluster. Higher scores = more coherent, distinct topics."
    )

    if topic_df is not None:
        fig_coherence = px.scatter(
            topic_df.head(25),
            x="top_score", y="size",
            text="label",
            title="Topic Coherence vs. Cluster Size",
            labels={"top_score": "c-TF-IDF Score (Top Word)", "size": "Cluster Size"},
            color="top_score",
            color_continuous_scale="Viridis",
            size="size",
            size_max=40
        )
        fig_coherence.update_traces(textposition="top center")
        fig_coherence.update_layout(height=500)
        st.plotly_chart(fig_coherence, use_container_width=True)

        st.markdown("**Top 10 Most Coherent Topics (by c-TF-IDF score):**")
        top_coherent = topic_df.nlargest(10, "top_score")[["label", "top_words", "top_score", "size"]]
        top_coherent.columns = ["Topic", "Top Keywords", "Coherence Score", "Cluster Size"]
        st.dataframe(top_coherent.reset_index(drop=True), use_container_width=True)

    # ── Section 3: Classification Confidence ──
    st.markdown("---")
    st.subheader("🤖 Zero-Shot Classification: Confidence Analysis")
    st.caption(
        "Confidence scores from BART-large-MNLI classifying support tweets. "
        "High average confidence (>0.7) indicates the model is making clear, reliable predictions."
    )

    if tweet_df is not None and "confidence" in tweet_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            avg_conf = tweet_df["confidence"].mean()
            high_conf = (tweet_df["confidence"] >= 0.7).sum()
            low_conf  = (tweet_df["confidence"] < 0.5).sum()

            st.metric("Average Confidence", f"{avg_conf:.1%}")
            st.metric("High Confidence Predictions (≥70%)", f"{high_conf:,} ({high_conf/len(tweet_df):.0%})")
            st.metric("Low Confidence Predictions (<50%)", f"{low_conf:,} ({low_conf/len(tweet_df):.0%})")

            # Confidence by issue type
            conf_by_type = tweet_df.groupby("issue_type")["confidence"].mean().sort_values(ascending=False)
            fig_conf_type = px.bar(
                x=conf_by_type.values, y=conf_by_type.index, orientation="h",
                title="Avg Confidence by Issue Type",
                color=conf_by_type.values,
                color_continuous_scale="RdYlGn",
                labels={"x": "Avg Confidence", "y": "Issue Type"}
            )
            st.plotly_chart(fig_conf_type, use_container_width=True)

        with col2:
            fig_hist = px.histogram(
                tweet_df, x="confidence", nbins=25,
                title="Confidence Score Distribution",
                color_discrete_sequence=["#636EFA"],
                labels={"confidence": "Confidence Score", "count": "# Tweets"}
            )
            fig_hist.add_vline(x=0.7, line_dash="dash", line_color="green",
                               annotation_text="High Confidence Threshold (0.7)")
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red",
                               annotation_text="Low Confidence Threshold (0.5)")
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Tweet issues data not found. Place `tweet_issues.csv` in `data/raw/insights/`.")

    # ── Section 4: Sizing Analysis Quality ──
    st.markdown("---")
    st.subheader("📏 Sizing Analysis: Coverage & Distribution")
    st.caption(
        "How many reviews contained actionable sizing signals vs. neutral text. "
        "High 'Neutral' % means the rule-based system could be improved with ML."
    )

    if sizing_df is not None and "sizing_feedback" in sizing_df.columns:
        col1, col2 = st.columns(2)
        with col1:
            counts = sizing_df["sizing_feedback"].value_counts()
            coverage = (1 - counts.get("Neutral", 0) / len(sizing_df)) * 100
            st.metric("Sizing Signal Coverage", f"{coverage:.1f}%",
                      help="% of reviews with actionable sizing feedback (non-Neutral)")
            st.metric("Total Reviews Analyzed", f"{len(sizing_df):,}")

            fig_sz = px.pie(
                values=counts.values, names=counts.index,
                title="Sizing Feedback Breakdown",
                color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]
            )
            st.plotly_chart(fig_sz, use_container_width=True)

        with col2:
            if "rating" in sizing_df.columns:
                # Avg rating per sizing category
                avg_rating = sizing_df.groupby("sizing_feedback")["rating"].mean().sort_values()
                fig_rating = px.bar(
                    x=avg_rating.values, y=avg_rating.index, orientation="h",
                    title="Avg Rating by Sizing Feedback",
                    color=avg_rating.values,
                    color_continuous_scale="RdYlGn",
                    labels={"x": "Avg Star Rating", "y": "Sizing Feedback"}
                )
                st.plotly_chart(fig_rating, use_container_width=True)
                st.caption(
                    "💡 Insight: Products that 'Run Small' tend to have lower ratings — "
                    "sizing issues directly impact customer satisfaction."
                )
    else:
        st.warning("Sizing data not found. Place `sizing_analysis.csv` in `data/raw/insights/`.")
