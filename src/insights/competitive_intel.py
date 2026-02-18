"""
Competitive Intelligence Engine.
Extracts competitor mentions from earnings calls using NER + keyword matching,
analyzes sentiment toward competitors, and builds a competitive landscape.
"""
import re
from typing import List, Dict, Set
from collections import defaultdict
from loguru import logger

# ─── Known competitor mapping (sector-based) ────────────────────
COMPETITOR_MAP = {
    # Tech
    "AAPL": ["GOOGL", "MSFT", "SAMSUNG", "META"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL", "CRM", "META"],
    "GOOGL": ["MSFT", "AAPL", "META", "AMZN", "OPENAI"],
    "AMZN": ["MSFT", "GOOGL", "WMT", "SHOP", "TGT"],
    "META": ["GOOGL", "SNAP", "TIKTOK", "AAPL", "MSFT"],
    "NVDA": ["AMD", "INTC", "GOOGL", "MSFT"],
    # Finance
    "JPM": ["BAC", "GS", "MS", "WFC", "C"],
    "GS": ["MS", "JPM", "BAC"],
    # Healthcare
    "JNJ": ["PFE", "ABBV", "MRK", "BMY"],
    "PFE": ["MRK", "JNJ", "ABBV", "LLY"],
}

# Company name → ticker mapping for NER resolution
COMPANY_NAMES = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "nvidia": "NVDA",
    "amd": "AMD", "intel": "INTC", "tesla": "TSLA", "netflix": "NFLX",
    "jpmorgan": "JPM", "goldman sachs": "GS", "morgan stanley": "MS",
    "walmart": "WMT", "salesforce": "CRM", "oracle": "ORCL",
    "johnson & johnson": "JNJ", "pfizer": "PFE", "merck": "MRK",
    "samsung": "SAMSUNG", "openai": "OPENAI", "tiktok": "TIKTOK",
    "snapchat": "SNAP", "spotify": "SPOT",
}

# Sentiment words
POSITIVE_WORDS = {"advantage", "leading", "outperform", "stronger", "ahead", "better", "superior", "winning"}
NEGATIVE_WORDS = {"behind", "losing", "pressure", "threat", "compete", "challenged", "aggressive", "disruption"}


def extract_competitor_mentions(
    text: str,
    source_ticker: str = "",
    use_ner: bool = False,
) -> List[Dict]:
    """
    Extract competitor mentions from text.

    Args:
        text: Transcript text.
        source_ticker: The company whose transcript this is (excluded from results).
        use_ner: Whether to use spaCy NER (slower but catches unknown companies).

    Returns:
        List of {competitor, ticker, sentence, sentiment, context}
    """
    mentions = []
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Keyword-based detection
    for sentence in sentences:
        s_lower = sentence.lower()
        for name, ticker in COMPANY_NAMES.items():
            if name in s_lower and ticker != source_ticker:
                sentiment = _score_sentiment(sentence)
                mentions.append({
                    "competitor": name.title(),
                    "ticker": ticker,
                    "sentence": sentence.strip(),
                    "sentiment": sentiment,
                    "method": "keyword",
                })

    # Optional: spaCy NER for catching unlisted companies
    if use_ner:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    name_lower = ent.text.lower()
                    if name_lower in COMPANY_NAMES and COMPANY_NAMES[name_lower] != source_ticker:
                        continue  # Already handled above
                    if ent.text != source_ticker and len(ent.text) > 2:
                        # Try to find the sentence containing this entity
                        sent = ent.sent.text if ent.sent else ent.text
                        mentions.append({
                            "competitor": ent.text,
                            "ticker": COMPANY_NAMES.get(name_lower, ""),
                            "sentence": sent.strip(),
                            "sentiment": _score_sentiment(sent),
                            "method": "ner",
                        })
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")

    # Deduplicate by sentence
    seen = set()
    unique = []
    for m in mentions:
        key = (m["ticker"] or m["competitor"], m["sentence"][:50])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique


def build_competitive_landscape(
    chunks: List[Dict],
    source_ticker: str,
) -> Dict:
    """
    Analyze all chunks from a company to build a competitive landscape view.

    Returns:
        {
            "source_company": str,
            "competitors": [
                {
                    "name": str, "ticker": str, "mentions": int,
                    "avg_sentiment": float, "contexts": [str]
                }
            ],
            "competitive_intensity": "High" / "Medium" / "Low"
        }
    """
    all_mentions = []
    for chunk in chunks:
        text = chunk.get("text", "")
        mentions = extract_competitor_mentions(text, source_ticker)
        for m in mentions:
            m["source_metadata"] = chunk.get("metadata", {})
        all_mentions.extend(mentions)

    # Aggregate by competitor
    competitor_data = defaultdict(lambda: {"mentions": 0, "sentiments": [], "contexts": []})
    for m in all_mentions:
        key = m["ticker"] or m["competitor"]
        competitor_data[key]["name"] = m["competitor"]
        competitor_data[key]["ticker"] = m["ticker"]
        competitor_data[key]["mentions"] += 1
        competitor_data[key]["sentiments"].append(m["sentiment"]["score"])
        if len(competitor_data[key]["contexts"]) < 3:  # Keep top 3 contexts
            competitor_data[key]["contexts"].append(m["sentence"][:200])

    # Format output
    competitors = []
    for key, data in sorted(competitor_data.items(), key=lambda x: x[1]["mentions"], reverse=True):
        avg_sent = sum(data["sentiments"]) / max(len(data["sentiments"]), 1)
        competitors.append({
            "name": data["name"],
            "ticker": data["ticker"],
            "mentions": data["mentions"],
            "avg_sentiment": round(avg_sent, 3),
            "sentiment_label": "Positive" if avg_sent > 0.1 else "Negative" if avg_sent < -0.1 else "Neutral",
            "contexts": data["contexts"],
        })

    # Competitive intensity
    total_mentions = sum(c["mentions"] for c in competitors)
    intensity = "Low"
    if total_mentions > 15:
        intensity = "High"
    elif total_mentions > 5:
        intensity = "Medium"

    return {
        "source_company": source_ticker,
        "competitors": competitors,
        "total_competitor_mentions": total_mentions,
        "competitive_intensity": intensity,
    }


def _score_sentiment(sentence: str) -> Dict:
    """Score sentiment of a sentence toward a competitor mention."""
    s_lower = sentence.lower()
    words = set(s_lower.split())

    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)

    if pos_count > neg_count:
        score = min(pos_count * 0.3, 1.0)
        label = "Positive"
    elif neg_count > pos_count:
        score = -min(neg_count * 0.3, 1.0)
        label = "Negative"
    else:
        score = 0.0
        label = "Neutral"

    return {"score": score, "label": label}
