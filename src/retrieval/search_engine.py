"""
Semantic Search Engine with Q&A Alpha boosting.
Combines vector similarity search with metadata filtering and re-ranking.
"""
from typing import List, Dict, Optional
from loguru import logger

from src.embeddings.embedder import generate_embeddings
from src.retrieval.vector_store import query_vectors

# ─── Configuration ───────────────────────────────────────────────
QA_BOOST_FACTOR = 1.5  # Boost Q&A chunks (unscripted = higher alpha)
DIVERSITY_PENALTY = 0.9  # Penalize multiple chunks from same speaker


def search(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict] = None,
    boost_qa: bool = True,
    diversify: bool = True,
) -> List[Dict]:
    """
    Perform semantic search across the earnings transcript corpus.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.
        filters: Metadata filters (e.g., {"ticker": "AAPL", "year": 2024}).
        boost_qa: Whether to boost Q&A section results.
        diversify: Whether to diversify results across speakers/companies.

    Returns:
        Ranked list of results: [{"text", "metadata", "score", "id"}, ...]
    """
    # 1. Generate query embedding
    query_emb = generate_embeddings([query])[0]

    # 2. Fetch more than needed for re-ranking
    fetch_k = top_k * 3 if (boost_qa or diversify) else top_k
    raw_results = query_vectors(query_emb, top_k=fetch_k, filters=filters)

    if not raw_results:
        logger.warning(f"No results found for query: '{query[:50]}...'")
        return []

    # 3. Re-rank: Apply Q&A boost
    if boost_qa:
        for r in raw_results:
            if r["metadata"].get("section") == "qa":
                r["score"] *= QA_BOOST_FACTOR

    # 4. Diversify: Penalize repeated speakers/companies
    if diversify:
        seen_speakers = {}
        for r in raw_results:
            key = f"{r['metadata'].get('ticker', '')}_{r['metadata'].get('speaker', '')}"
            count = seen_speakers.get(key, 0)
            r["score"] *= DIVERSITY_PENALTY ** count
            seen_speakers[key] = count + 1

    # 5. Sort by final score
    raw_results.sort(key=lambda x: x["score"], reverse=True)

    # 6. Return top_k
    results = raw_results[:top_k]
    logger.info(f"Search '{query[:40]}...' → {len(results)} results (top score: {results[0]['score']:.3f})")
    return results


def search_by_company(
    query: str,
    ticker: str,
    top_k: int = 10,
    quarter: Optional[str] = None,
    year: Optional[int] = None,
) -> List[Dict]:
    """Search within a specific company's transcripts."""
    filters = {"ticker": ticker}
    if year:
        filters["year"] = year
    # ChromaDB doesn't support multiple where clauses natively without $and
    # For simplicity, filter on ticker + post-filter on quarter/year
    results = search(query, top_k=top_k * 2, filters=filters)
    
    if quarter:
        results = [r for r in results if r["metadata"].get("quarter") == quarter]
    
    return results[:top_k]


def compare_companies(
    query: str,
    tickers: List[str],
    top_k_per_company: int = 5,
) -> Dict[str, List[Dict]]:
    """
    Search for a topic across multiple companies for comparison.

    Returns:
        {"AAPL": [results], "MSFT": [results], ...}
    """
    comparison = {}
    for ticker in tickers:
        results = search_by_company(query, ticker, top_k=top_k_per_company)
        comparison[ticker] = results
    
    return comparison


def find_similar_chunks(chunk_id: str, top_k: int = 5) -> List[Dict]:
    """Find chunks similar to a given chunk (for 'related insights' feature)."""
    from src.retrieval.vector_store import get_collection
    
    collection = get_collection()
    result = collection.get(ids=[chunk_id], include=["embeddings"])
    
    if not result["embeddings"]:
        return []
    
    embedding = result["embeddings"][0]
    # Exclude the source chunk
    results = query_vectors(embedding, top_k=top_k + 1)
    return [r for r in results if r["id"] != chunk_id][:top_k]
