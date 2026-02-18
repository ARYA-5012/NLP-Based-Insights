"""
Trend Tracker — tracks topic/theme evolution over time.
Computes quarterly topic frequencies, identifies emerging themes,
and optionally correlates with stock price data.
"""
from typing import List, Dict, Optional
from collections import defaultdict
from loguru import logger


def compute_topic_frequencies(
    chunks: List[Dict],
    topics: Dict[int, str],
    chunk_topics: List[int],
) -> Dict:
    """
    Compute topic frequency per quarter.

    Args:
        chunks: All chunks with metadata.
        topics: Topic ID → topic label mapping.
        chunk_topics: Topic ID for each chunk (parallel to chunks list).

    Returns:
        {
            "Q3_2024": {"AI Investment": 12, "Supply Chain": 5, ...},
            "Q4_2024": {...},
        }
    """
    freq = defaultdict(lambda: defaultdict(int))

    for chunk, topic_id in zip(chunks, chunk_topics):
        meta = chunk.get("metadata", {})
        quarter = meta.get("quarter", "Q0")
        year = meta.get("year", 0)
        period = f"{quarter}_{year}"

        topic_label = topics.get(topic_id, f"Topic_{topic_id}")
        if topic_id == -1:
            continue  # Skip outlier topic

        freq[period][topic_label] += 1

    return dict(freq)


def detect_emerging_themes(
    freq_data: Dict,
    threshold_multiplier: float = 3.0,
) -> List[Dict]:
    """
    Identify themes whose frequency has increased significantly.

    An "emerging theme" is one that appears 3x+ more frequently
    than it did one year ago.

    Returns:
        List of {"theme", "current_freq", "prev_freq", "change_pct", "status"}
    """
    periods = sorted(freq_data.keys())
    if len(periods) < 2:
        return []

    current = periods[-1]
    # Find period from ~1 year ago (4 quarters back)
    prev_idx = max(0, len(periods) - 5)
    prev = periods[prev_idx]

    current_data = freq_data[current]
    prev_data = freq_data[prev]

    emerging = []
    all_topics = set(list(current_data.keys()) + list(prev_data.keys()))

    for topic in all_topics:
        curr_freq = current_data.get(topic, 0)
        prev_freq = prev_data.get(topic, 0)

        if prev_freq == 0 and curr_freq > 0:
            change_pct = float("inf")
            status = "NEW"
        elif curr_freq == 0 and prev_freq > 0:
            change_pct = -100.0
            status = "DECLINING"
        elif prev_freq > 0:
            change_pct = ((curr_freq - prev_freq) / prev_freq) * 100
            if change_pct >= (threshold_multiplier - 1) * 100:
                status = "EMERGING"
            elif change_pct <= -50:
                status = "DECLINING"
            else:
                status = "STABLE"
        else:
            continue

        emerging.append({
            "theme": topic,
            "current_period": current,
            "previous_period": prev,
            "current_freq": curr_freq,
            "previous_freq": prev_freq,
            "change_pct": round(change_pct, 1) if change_pct != float("inf") else "NEW",
            "status": status,
        })

    # Sort by change magnitude
    emerging.sort(key=lambda x: x["current_freq"], reverse=True)
    return emerging


def sector_topic_heatmap(
    chunks: List[Dict],
    topics: Dict[int, str],
    chunk_topics: List[int],
) -> Dict:
    """
    Build a sector × topic frequency matrix for heatmap visualization.

    Returns:
        {
            "sectors": ["Technology", "Healthcare", ...],
            "topics": ["AI Investment", "Supply Chain", ...],
            "matrix": [[12, 5, ...], [3, 8, ...], ...]  # sectors × topics
        }
    """
    sector_topic_freq = defaultdict(lambda: defaultdict(int))

    for chunk, topic_id in zip(chunks, chunk_topics):
        if topic_id == -1:
            continue
        meta = chunk.get("metadata", {})
        # Derive sector from ticker (simplified — in production, use a lookup)
        sector = _get_sector(meta.get("ticker", ""))
        topic_label = topics.get(topic_id, f"Topic_{topic_id}")
        sector_topic_freq[sector][topic_label] += 1

    sectors = sorted(sector_topic_freq.keys())
    all_topics = sorted(set(t for s in sector_topic_freq.values() for t in s))

    matrix = []
    for sector in sectors:
        row = [sector_topic_freq[sector].get(t, 0) for t in all_topics]
        matrix.append(row)

    return {"sectors": sectors, "topics": all_topics, "matrix": matrix}


def correlate_with_stock(
    topic_freq: Dict,
    ticker: str,
    topic_name: str,
) -> Optional[Dict]:
    """
    Correlate topic frequency with stock price performance.
    Uses yfinance to fetch stock data (optional dependency).

    Returns:
        {
            "ticker": str,
            "topic": str,
            "correlation": float,
            "data_points": [{"period", "topic_freq", "stock_return"}, ...]
        }
    """
    try:
        import yfinance as yf
        import numpy as np

        stock = yf.Ticker(ticker)
        hist = stock.history(period="3y", interval="3mo")

        if hist.empty:
            return None

        # Match quarterly returns to topic frequency
        data_points = []
        for period, topics in sorted(topic_freq.items()):
            freq = topics.get(topic_name, 0)
            # Parse period (e.g., "Q3_2024")
            parts = period.split("_")
            if len(parts) != 2:
                continue

            data_points.append({
                "period": period,
                "topic_freq": freq,
            })

        if len(data_points) < 3:
            return None

        # Simple correlation
        freqs = [d["topic_freq"] for d in data_points]
        # Placeholder — in production, join with actual stock returns
        return {
            "ticker": ticker,
            "topic": topic_name,
            "data_points": data_points,
            "note": "Stock correlation requires aligned quarterly return data",
        }

    except ImportError:
        logger.warning("yfinance not installed — skipping stock correlation")
        return None


# ─── Sector Lookup (simplified) ─────────────────────────────────

_SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "META": "Technology", "AMZN": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "ADBE": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "WFC": "Financials", "V": "Financials",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
    "LLY": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare",
    "WMT": "Consumer", "PG": "Consumer", "KO": "Consumer",
    "NKE": "Consumer", "MCD": "Consumer", "COST": "Consumer",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
}


def _get_sector(ticker: str) -> str:
    return _SECTOR_MAP.get(ticker, "Other")
