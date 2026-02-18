"""
Compare Router — side-by-side company comparison.
"""
from typing import List
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

router = APIRouter()


@router.get("/compare")
async def compare_companies(
    companies: str = Query(..., description="Comma-separated tickers (e.g., AAPL,MSFT)"),
    topic: str = Query("strategy and outlook", description="Topic to compare on"),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    Side-by-side comparison of 2+ companies on a specific topic.
    Uses semantic search + AI synthesis for structured comparison.
    """
    tickers = [t.strip().upper() for t in companies.split(",")]
    
    if len(tickers) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 company tickers")
    if len(tickers) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 companies for comparison")

    try:
        from src.retrieval.search_engine import compare_companies as do_compare

        comparison_data = do_compare(topic, tickers, top_k_per_company=top_k)

        result = {
            "topic": topic,
            "companies": {},
        }

        for ticker, chunks in comparison_data.items():
            result["companies"][ticker] = {
                "chunk_count": len(chunks),
                "top_excerpts": [
                    {
                        "text": c["text"][:300],
                        "speaker": c["metadata"].get("speaker", ""),
                        "quarter": c["metadata"].get("quarter", ""),
                        "year": c["metadata"].get("year", 0),
                        "score": round(c["score"], 3),
                    }
                    for c in chunks[:3]
                ],
            }

        return result

    except Exception as e:
        logger.error(f"Compare error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare/ai")
async def compare_with_ai(
    companies: str = Query(..., description="Two tickers (e.g., AAPL,MSFT)"),
    topic: str = Query("competitive strategy and market position"),
):
    """
    AI-generated structured comparison of two companies.
    Uses GPT to synthesize insights from both companies' earnings calls.
    """
    tickers = [t.strip().upper() for t in companies.split(",")]

    if len(tickers) != 2:
        raise HTTPException(status_code=400, detail="AI comparison requires exactly 2 tickers")

    try:
        from src.synthesis.summarizer import company_comparison

        result = company_comparison(tickers[0], tickers[1], topic)
        return result

    except Exception as e:
        logger.error(f"AI Compare error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
