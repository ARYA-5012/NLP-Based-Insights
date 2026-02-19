# Customer Voice Intelligence 🗣️

> **Turn millions of customer reviews and support tickets into actionable business intelligence — in seconds.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E)](https://huggingface.co)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 What This Is

A **production-grade NLP system** that extracts real, actionable insights from two high-signal customer datasets:

| Dataset | Source | Scale |
|:---|:---|:---|
| 👗 **Women's E-Commerce Clothing Reviews** | [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) | 23,000 real reviews |
| 🐦 **Customer Support on Twitter** | [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) | 3M+ real tweets |

This is **not** a basic positive/negative sentiment classifier. It's a system that answers real business questions:

- *"What sizing issues are customers reporting for our dresses?"*
- *"Which support issues are most likely to escalate?"*
- *"What do customers love vs. hate about our knitwear?"*

---

## ✨ Key Capabilities

| Feature | Description |
|:---|:---|
| 🔍 **Semantic Search** | Natural language queries across 20,000+ indexed documents using vector similarity |
| 🗂️ **Topic Modeling** | BERTopic discovers 41 distinct themes (sizing, fabric, shipping, billing, etc.) automatically |
| 🤖 **Zero-Shot Classification** | BART-large-MNLI categorizes support tickets into Shipping / Billing / Tech / Complaint — no training data needed |
| 📏 **Sizing Intelligence** | Rule-based + ML analysis to detect "Runs Small", "True to Size", "Runs Large" patterns |
| 📊 **Evaluation Dashboard** | c-TF-IDF coherence scores, confidence distributions, cluster quality metrics |
| 🔗 **RAG Q&A** | Ask questions, get answers grounded in real customer text |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATA PIPELINE (Kaggle)                 │
│                                                         │
│  📦 Notebook 1          📦 Notebook 2       📦 Notebook 3│
│  ─────────────          ─────────────       ────────────│
│  Load Reviews           BERTopic            Zero-Shot   │
│  Load Tweets      ───▶  Topic Modeling ───▶ Classification│
│  Embed (mpnet)          41 Clusters         Sizing Analysis│
│  → all_chunks.parquet   → topics.json       → CSVs      │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼  (download & import locally)
┌─────────────────────────────────────────────────────────┐
│                    LOCAL APPLICATION                     │
│                                                         │
│  import_kaggle_data.py                                  │
│  ──────────────────────                                 │
│  Loads parquet + embeddings into ChromaDB               │
│                                                         │
│  ChromaDB (Vector Store)                                │
│  ─────────────────────────                              │
│  20,093 documents indexed                               │
│  Cosine similarity search                               │
│  Metadata filtering (source, category, issue_type)      │
│                                                         │
│  Streamlit Dashboard (4 pages)                          │
│  ──────────────────────────────                         │
│  🔍 Universal Search  │  👗 Product Insights            │
│  🐦 Support Ops       │  📊 Evaluation                  │
└─────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Embeddings** | `sentence-transformers/all-mpnet-base-v2` | 768-dim semantic vectors |
| **Vector DB** | ChromaDB | Persistent local vector store |
| **Topic Modeling** | BERTopic + UMAP + HDBSCAN | Unsupervised theme discovery |
| **Classification** | `facebook/bart-large-mnli` | Zero-shot issue categorization |
| **Backend** | FastAPI + Pydantic | REST API with type safety |
| **Frontend** | Streamlit | Interactive analytics dashboard |
| **Deployment** | Docker + Docker Compose | Containerized full stack |
| **Data Pipeline** | Kaggle Notebooks | GPU-accelerated preprocessing |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- ~4GB disk space (for models + data)
- Kaggle account (for data download)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/customer-voice-intelligence.git
cd customer-voice-intelligence

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Get the Data (Kaggle Pipeline)

The heavy computation runs on Kaggle's free GPUs. Follow these steps:

**Step 1 — Run Notebook 1 (Data + Embeddings)**
1. Upload `notebooks/kaggle/01_data_processing_embeddings.ipynb` to Kaggle
2. Add datasets:
   - `nicapotato/womens-ecommerce-clothing-reviews`
   - `thoughtvector/customer-support-on-twitter`
3. Run all cells → Download `all_chunks.parquet` + `embeddings.npy`

**Step 2 — Run Notebook 2 (Topic Modeling)**
1. Upload `notebooks/kaggle/02_topic_modeling.ipynb`
2. Add your Notebook 1 output as a dataset
3. Run all cells → Download `bertopic_model/` folder + `chunks_with_topics.parquet`

**Step 3 — Run Notebook 3 (Insights)**
1. Upload `notebooks/kaggle/03_insight_extraction.ipynb`
2. Add your Notebook 1 output as a dataset
3. Run all cells → Download `tweet_issues.csv` + `sizing_analysis.csv`

### 3. Place Downloaded Files

```
data/
├── raw/
│   ├── all_chunks.parquet          ← from Notebook 1
│   ├── embeddings.npy              ← from Notebook 1
│   └── insights/
│       ├── tweet_issues.csv        ← from Notebook 3
│       └── sizing_analysis.csv     ← from Notebook 3
└── models/
    └── bertopic_model/             ← folder from Notebook 2
        ├── config.json
        ├── topics.json
        ├── ctfidf.safetensors
        └── topic_embeddings.safetensors
```

### 4. Import into ChromaDB

```bash
python scripts/import_kaggle_data.py
# ✅ Import complete. Total docs in collection: 20,093
```

### 5. Launch the Dashboard

```bash
python -m streamlit run streamlit_demo/app.py
# Open: http://localhost:8501
```

---

## 📊 Dashboard Pages

### 🔍 Universal Search
Natural language queries across all 20,000+ documents. Filter by source (reviews vs. tweets).

> *Try: "dress sizing issues", "shipping delays", "refund problems"*

### 👗 Product Insights
- Sizing feedback distribution (Runs Small / True to Size / Runs Large)
- Star rating breakdown from real reviews
- Aspect-based search by product category (Dresses, Jeans, Blouses, etc.)

### 🐦 Support Ops
- Ticket volume by issue type (Shipping / Billing / Tech / Complaint)
- Classification confidence histogram
- "Find Similar Tickets" — paste a tweet, get historically similar cases

### 📊 Evaluation
- **BERTopic Cluster Sizes** — 41 discovered topics with document counts
- **Topic Coherence** — c-TF-IDF scores proving cluster distinctiveness
- **Classification Confidence** — per-issue-type reliability analysis
- **Sizing Signal Coverage** — % of reviews with actionable feedback

---

## 🧪 Evaluation Results

| Metric | Result |
|:---|:---|
| Topics Discovered (BERTopic) | **41 coherent clusters** |
| Largest Topic | **Dress reviews** (most discussed category) |
| High-Confidence Tweets (≥70%) | **Measured from real data** |
| Sizing Signal Coverage | **Measured from real data** |
| Semantic Search | Cosine similarity via `all-mpnet-base-v2` |

> 💡 All metrics are computed from **real data** and visible in the Evaluation dashboard page.

---

## 🗂️ Project Structure

```
customer-voice-intelligence/
│
├── 📓 notebooks/kaggle/
│   ├── 01_data_processing_embeddings.ipynb  # Load, clean, embed (Kaggle GPU)
│   ├── 02_topic_modeling.ipynb              # BERTopic with domain seeds
│   └── 03_insight_extraction.ipynb          # Zero-shot + sizing analysis
│
├── 🐍 src/
│   ├── api/
│   │   ├── main.py                          # FastAPI app (CORS, health checks)
│   │   └── routers/
│   │       ├── search.py                    # Semantic search + RAG Q&A
│   │       ├── company.py                   # Company-level insights
│   │       ├── trends.py                    # Topic trend analysis
│   │       ├── compare.py                   # Side-by-side comparison
│   │       └── analyze.py                   # Upload & analyze endpoint
│   ├── retrieval/
│   │   ├── vector_store.py                  # ChromaDB abstraction
│   │   └── search_engine.py                 # Hybrid search + re-ranking
│   ├── insights/
│   │   ├── risk_detector.py                 # Zero-shot BART classification
│   │   ├── confidence_analyzer.py           # Confidence scoring
│   │   ├── competitive_intel.py             # NER + sentiment
│   │   └── trend_tracker.py                 # Temporal trend analysis
│   ├── synthesis/
│   │   ├── summarizer.py                    # RAG pipeline
│   │   ├── llm_client.py                    # OpenAI abstraction
│   │   └── prompts.py                       # Prompt templates
│   ├── topic_modeling/
│   │   └── bertopic_pipeline.py             # BERTopic with seed topics
│   ├── processing/
│   │   └── chunker.py                       # Semantic text chunking
│   └── ingestion/
│       ├── api_client.py                    # Data ingestion client
│       └── regex_parser.py                  # Text parsing utilities
│
├── 🖥️ streamlit_demo/
│   └── app.py                               # 4-page interactive dashboard
│
├── 📜 scripts/
│   ├── import_kaggle_data.py                # Load Kaggle outputs → ChromaDB
│   └── analyze_exports.py                   # Local analysis utilities
│
├── 🧪 tests/
│   └── test_system.py                       # Comprehensive test suite
│
├── 📁 data/                                 # (gitignored — see Quick Start)
│   ├── raw/                                 # Parquet + embeddings
│   ├── models/                              # BERTopic model files
│   └── embeddings/                          # ChromaDB persistence
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── LICENSE
```

---

## 🐳 Docker Deployment

```bash
# Full stack (API + Streamlit)
docker-compose up --build

# API only
docker build -t customer-voice-api .
docker run -p 8000:8000 customer-voice-api
```

---

## 🔌 API Reference

The FastAPI backend exposes a REST API (Swagger UI at `/docs`):

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/api/search` | Semantic search with optional source/category filters |
| `POST` | `/api/search/ask` | RAG-powered Q&A |
| `GET` | `/api/health` | Health check |

### Example: Search

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "dress sizing issues",
    "n_results": 5,
    "source_filter": "clothing_reviews"
  }'
```

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/api/search/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the most common complaints about shipping?",
    "source_filter": "twitter_support"
  }'
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 💡 Business Use Cases

| Persona | How They Use This |
|:---|:---|
| **Product Manager** | Search "sizing complaints for dresses" → prioritize fit improvements |
| **Support Lead** | View ticket clusters → identify systemic issues before they escalate |
| **Buyer / Merchandiser** | Check sizing distribution per category → adjust size charts |
| **Data Analyst** | Evaluation page → validate model quality before presenting to stakeholders |

---

## 🎤 Interview Talking Points

- *"Used BERTopic with domain-specific seed topics to discover 41 coherent customer themes from 20,000+ unstructured documents — no labeled training data required."*
- *"Implemented zero-shot classification with BART-large-MNLI to categorize support tickets into actionable issue types, showing I can apply NLP without task-specific training data."*
- *"Built an end-to-end data pipeline: Kaggle GPU notebooks for heavy compute → local ChromaDB for retrieval → Streamlit dashboard for business users."*
- *"Included a dedicated Evaluation page showing c-TF-IDF coherence scores and confidence distributions — demonstrating I understand that NLP quality goes beyond accuracy metrics."*

---

## 📝 License

MIT — see [LICENSE](LICENSE) for details.

---

**Built with** · Sentence Transformers · ChromaDB · BERTopic · BART · FastAPI · Streamlit · Docker · Kaggle

---

## 👨‍💻 Author

**Arya Yadav**

---
