"""
Analyze Router — upload and analyze individual transcripts.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

router = APIRouter()


class TranscriptInput(BaseModel):
    """Input for transcript analysis."""
    text: str = Field(..., description="Raw earnings call transcript text", min_length=100)
    ticker: str = Field("UNKNOWN", description="Company ticker symbol")
    quarter: str = Field("Q0", description="Quarter (e.g., Q3)")
    year: int = Field(2024, description="Year")


class AnalyzeResponse(BaseModel):
    """Response from transcript analysis."""
    ticker: str
    quarter: str
    year: int
    sections: dict
    confidence: dict
    risks: list
    competitive: dict


@router.post("/analyze")
async def analyze_transcript(input_data: TranscriptInput):
    """
    Upload and analyze a raw earnings call transcript.
    Returns structured insights including risks, confidence, and competitor mentions.
    """
    try:
        from src.ingestion.regex_parser import TranscriptParser
        from src.processing.chunker import chunk_segments
        from src.insights.risk_detector import detect_risks
        from src.insights.confidence_analyzer import analyze_confidence, compute_sentiment_deviation
        from src.insights.competitive_intel import extract_competitor_mentions

        # 1. Parse transcript
        parser = TranscriptParser()
        parsed = parser.parse(
            input_data.text,
            metadata={"ticker": input_data.ticker, "quarter": input_data.quarter, "year": input_data.year}
        )

        # 2. Analyze confidence
        pres_text = " ".join(s["text"] for s in parsed["presentation"])
        qa_text = " ".join(s["text"] for s in parsed["qa"])

        pres_confidence = analyze_confidence(pres_text, section="presentation")
        qa_confidence = analyze_confidence(qa_text, section="qa") if qa_text else {}

        # Sentiment deviation
        deviation = {}
        if parsed["presentation"] and parsed["qa"]:
            pres_chunks = [{"text": s["text"]} for s in parsed["presentation"]]
            qa_chunks = [{"text": s["text"]} for s in parsed["qa"]]
            deviation = compute_sentiment_deviation(pres_chunks, qa_chunks)

        # 3. Detect risks
        full_text = pres_text + " " + qa_text
        risks = detect_risks(full_text)

        # 4. Extract competitor mentions
        competitors = extract_competitor_mentions(full_text, source_ticker=input_data.ticker)

        return {
            "ticker": input_data.ticker,
            "quarter": input_data.quarter,
            "year": input_data.year,
            "sections": {
                "presentation_segments": len(parsed["presentation"]),
                "qa_segments": len(parsed["qa"]),
                "has_qa": parsed["stats"]["has_qa"],
                "speakers": parsed["stats"]["speakers"],
            },
            "confidence": {
                "presentation": pres_confidence,
                "qa": qa_confidence,
                "sentiment_deviation": deviation,
            },
            "risks": risks[:10],  # Top 10 risks
            "competitive": {
                "mentions": competitors[:10],
                "total_count": len(competitors),
            },
        }

    except Exception as e:
        logger.error(f"Analyze error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
