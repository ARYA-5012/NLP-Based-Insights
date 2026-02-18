"""
Context-Aware Summarization Pipeline (RAG).
Combines retrieval results with LLM generation to produce
structured, grounded financial summaries.
"""
from typing import List, Dict, Optional
from loguru import logger

from src.retrieval.search_engine import search, search_by_company
from src.synthesis.llm_client import generate, generate_with_context
from src.synthesis import prompts


def executive_summary(
    ticker: str, quarter: str, year: int, top_k: int = 10
) -> Dict:
    """
    Generate an executive summary for a company's earnings call.

    Returns:
        {"company", "quarter", "year", "summary", "sources"}
    """
    query = f"key results revenue growth strategy {ticker}"
    chunks = search_by_company(query, ticker, top_k=top_k, quarter=quarter, year=year)

    if not chunks:
        return {"company": ticker, "quarter": quarter, "year": year,
                "summary": "No transcript data available.", "sources": []}

    # Format context
    context = _format_chunks(chunks)
    prompt = prompts.EXECUTIVE_SUMMARY.format(
        company=ticker, quarter=quarter, year=year, context=context
    )
    summary = generate(prompt, system_prompt="You are a senior equity research analyst.")

    return {
        "company": ticker,
        "quarter": quarter,
        "year": year,
        "summary": summary,
        "sources": [c["id"] for c in chunks],
    }


def risk_report(
    ticker: str, quarter: str, year: int, top_k: int = 10
) -> Dict:
    """Generate a risk analysis report for a company."""
    query = f"risk challenge headwind uncertainty concern threat {ticker}"
    chunks = search_by_company(query, ticker, top_k=top_k, quarter=quarter, year=year)

    if not chunks:
        return {"company": ticker, "risks": [], "summary": "No data available."}

    context = _format_chunks(chunks)
    prompt = prompts.RISK_REPORT.format(
        company=ticker, quarter=quarter, year=year, context=context
    )
    report = generate(prompt, system_prompt="You are a financial risk analyst.")

    return {
        "company": ticker,
        "quarter": quarter,
        "year": year,
        "report": report,
        "sources": [c["id"] for c in chunks],
    }


def company_comparison(
    ticker_a: str, ticker_b: str, topic: str, top_k: int = 5
) -> Dict:
    """Compare two companies on a specific topic."""
    chunks_a = search_by_company(topic, ticker_a, top_k=top_k)
    chunks_b = search_by_company(topic, ticker_b, top_k=top_k)

    context_a = _format_chunks(chunks_a) if chunks_a else "No relevant excerpts found."
    context_b = _format_chunks(chunks_b) if chunks_b else "No relevant excerpts found."

    prompt = prompts.COMPANY_COMPARISON.format(
        company_a=ticker_a, context_a=context_a,
        company_b=ticker_b, context_b=context_b,
        topic=topic,
    )
    comparison = generate(prompt, system_prompt="You are a senior equity analyst.")

    return {
        "company_a": ticker_a,
        "company_b": ticker_b,
        "topic": topic,
        "comparison": comparison,
        "sources_a": [c["id"] for c in chunks_a],
        "sources_b": [c["id"] for c in chunks_b],
    }


def qa_alpha_analysis(
    ticker: str, quarter: str, year: int, top_k: int = 10
) -> Dict:
    """Analyze Q&A section for management confidence signals."""
    query = f"analyst question answer {ticker}"
    # Only fetch Q&A chunks
    chunks = search_by_company(query, ticker, top_k=top_k, quarter=quarter, year=year)
    qa_chunks = [c for c in chunks if c["metadata"].get("section") == "qa"]

    if not qa_chunks:
        return {"company": ticker, "analysis": "No Q&A data available.", "confidence_score": None}

    context = _format_chunks(qa_chunks)
    prompt = prompts.QA_ALPHA.format(
        company=ticker, quarter=quarter, year=year, context=context
    )
    analysis = generate(prompt, system_prompt="You are a behavioral finance analyst.")

    return {
        "company": ticker,
        "quarter": quarter,
        "year": year,
        "analysis": analysis,
        "sources": [c["id"] for c in qa_chunks],
    }


def answer_query(query: str, top_k: int = 10, filters: Dict = None) -> Dict:
    """
    General-purpose RAG: answer any natural language question using the corpus.
    This is the main entry point for the search/chat interface.
    """
    chunks = search(query, top_k=top_k, filters=filters)
    if not chunks:
        return {"query": query, "answer": "No relevant information found.", "sources": []}

    answer = generate_with_context(query, chunks)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "id": c["id"],
                "company": c["metadata"].get("ticker", ""),
                "quarter": c["metadata"].get("quarter", ""),
                "year": c["metadata"].get("year", ""),
                "speaker": c["metadata"].get("speaker", ""),
                "score": round(c["score"], 3),
                "excerpt": c["text"][:200] + "...",
            }
            for c in chunks
        ],
    }


def _format_chunks(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        header = (
            f"[{i}] {meta.get('ticker', '?')} Q{meta.get('quarter', '?')} {meta.get('year', '?')} "
            f"| {meta.get('speaker', 'Unknown')} ({meta.get('role', '')}) "
            f"| {meta.get('section', '')}"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
