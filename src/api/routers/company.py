"""
Company Profile Router — per-company insights, risk, confidence analysis.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Path, Query
from loguru import logger

router = APIRouter()


@router.get("/company/{ticker}")
async def get_company_insights(
    ticker: str = Path(..., description="Company ticker symbol (e.g., AAPL)"),
    quarter: Optional[str] = Query(None, description="Quarter filter (e.g., Q3)"),
    year: Optional[int] = Query(None, description="Year filter (e.g., 2024)"),
):
    """
    Get comprehensive insights for a specific company.
    Includes executive summary, risks, confidence analysis, and competitor mentions.
    """
    ticker = ticker.upper()
    try:
        from src.retrieval.search_engine import search_by_company
        from src.insights.confidence_analyzer import analyze_confidence, compute_sentiment_deviation
        from src.insights.competitive_intel import build_competitive_landscape

        # Fetch all relevant chunks
        query = f"{ticker} revenue growth strategy risks outlook"
        chunks = search_by_company(query, ticker, top_k=20, quarter=quarter, year=year)

        if not chunks:
            return {
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "message": "No transcript data found for this company/period.",
            }

        # Split by section
        pres_chunks = [c for c in chunks if c["metadata"].get("section") == "presentation"]
        qa_chunks = [c for c in chunks if c["metadata"].get("section") == "qa"]

        # Confidence analysis
        confidence = {}
        if pres_chunks and qa_chunks:
            confidence = compute_sentiment_deviation(pres_chunks, qa_chunks)
        elif chunks:
            all_text = " ".join(c["text"] for c in chunks)
            confidence = analyze_confidence(all_text)

        # Competitive landscape
        competitive = build_competitive_landscape(chunks, ticker)

        # Collect unique speakers
        speakers = list(set(c["metadata"].get("speaker", "") for c in chunks if c["metadata"].get("speaker")))

        return {
            "ticker": ticker,
            "quarter": quarter,
            "year": year,
            "chunk_count": len(chunks),
            "confidence_analysis": confidence,
            "competitive_landscape": competitive,
            "speakers": speakers,
            "top_excerpts": [
                {
                    "text": c["text"][:300],
                    "speaker": c["metadata"].get("speaker", ""),
                    "section": c["metadata"].get("section", ""),
                    "score": round(c["score"], 3),
                }
                for c in chunks[:5]
            ],
        }

    except Exception as e:
        logger.error(f"Company insights error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/company/{ticker}/summary")
async def get_company_summary(
    ticker: str = Path(...),
    quarter: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
):
    """Generate an AI-powered executive summary for a company."""
    ticker = ticker.upper()
    try:
        from src.synthesis.summarizer import executive_summary
        result = executive_summary(ticker, quarter or "Q3", year or 2024)
        return result
    except Exception as e:
        logger.error(f"Summary error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/company/{ticker}/risks")
async def get_company_risks(
    ticker: str = Path(...),
    quarter: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
):
    """Get risk analysis for a company."""
    ticker = ticker.upper()
    try:
        from src.retrieval.search_engine import search_by_company
        from src.insights.risk_detector import analyze_document

        query = f"risk challenge concern headwind uncertainty {ticker}"
        chunks = search_by_company(query, ticker, top_k=15, quarter=quarter, year=year)
        
        if not chunks:
            return {"ticker": ticker, "risks": [], "message": "No data available."}

        result = analyze_document(chunks)
        result["ticker"] = ticker
        return result

    except Exception as e:
        logger.error(f"Risk analysis error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
