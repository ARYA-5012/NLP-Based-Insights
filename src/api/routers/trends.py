"""
Trends Router — topic trends, emerging themes, sector analysis.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from loguru import logger

router = APIRouter()


@router.get("/trends")
async def get_trends(
    sector: Optional[str] = Query(None, description="Filter by sector"),
    year: Optional[int] = Query(None, description="Filter by year"),
):
    """
    Get trending topics across the earnings corpus.
    Shows topic frequency, emerging themes, and sector-level patterns.
    """
    try:
        from src.retrieval.vector_store import get_collection_stats

        stats = get_collection_stats()

        # Return available data and trend placeholders
        return {
            "corpus_stats": stats,
            "note": "Topic trends require BERTopic model training. Run the topic modeling pipeline first.",
            "sample_trends": [
                {
                    "theme": "AI & Machine Learning Investment",
                    "status": "EMERGING",
                    "current_frequency": "67% of tech companies",
                    "year_ago_frequency": "12% of tech companies",
                    "change": "+458%",
                },
                {
                    "theme": "Supply Chain Diversification",
                    "status": "STABLE",
                    "current_frequency": "45% of industrial companies",
                    "year_ago_frequency": "42% of industrial companies",
                    "change": "+7%",
                },
                {
                    "theme": "Remote Work / Hybrid",
                    "status": "DECLINING",
                    "current_frequency": "15% of companies",
                    "year_ago_frequency": "38% of companies",
                    "change": "-61%",
                },
            ],
        }

    except Exception as e:
        logger.error(f"Trends error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topics")
async def get_topics():
    """Get all discovered topics from the BERTopic model."""
    return {
        "note": "Run BERTopic pipeline to populate topics.",
        "sample_topics": [
            {"id": 0, "label": "AI & Technology Investment", "keywords": ["ai", "machine learning", "automation"], "doc_count": 450},
            {"id": 1, "label": "Supply Chain & Logistics", "keywords": ["supply chain", "logistics", "inventory"], "doc_count": 312},
            {"id": 2, "label": "Interest Rate Environment", "keywords": ["interest rate", "fed", "monetary policy"], "doc_count": 278},
            {"id": 3, "label": "Revenue Growth & Margins", "keywords": ["revenue", "margin", "earnings"], "doc_count": 520},
            {"id": 4, "label": "Regulatory & Compliance", "keywords": ["regulation", "compliance", "SEC"], "doc_count": 156},
            {"id": 5, "label": "Cloud & Digital Transformation", "keywords": ["cloud", "SaaS", "digital"], "doc_count": 389},
            {"id": 6, "label": "Workforce & Talent", "keywords": ["hiring", "talent", "workforce"], "doc_count": 201},
            {"id": 7, "label": "ESG & Sustainability", "keywords": ["sustainability", "ESG", "climate"], "doc_count": 134},
        ],
    }
