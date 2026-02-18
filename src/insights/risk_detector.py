"""
Risk Detection Engine using zero-shot classification.
Identifies, categorizes, and scores risk statements from earnings call chunks.
"""
from typing import List, Dict
from loguru import logger

# Lazy-loaded classifier
_classifier = None

RISK_CATEGORIES = [
    "operational risk",
    "financial risk",
    "regulatory risk",
    "competitive risk",
    "market risk",
    "geopolitical risk",
    "technology risk",
    "supply chain risk",
]

RISK_KEYWORDS = [
    "risk", "challenge", "headwind", "uncertain", "concern", "difficult",
    "pressure", "volatility", "threat", "decline", "weakness", "slowdown",
    "disruption", "exposure", "vulnerability", "downturn", "adverse",
    "impediment", "liability", "loss", "impairment", "default",
]

OPPORTUNITY_KEYWORDS = [
    "opportunity", "growth", "expansion", "innovation", "tailwind",
    "momentum", "upside", "strength", "improvement", "acceleration",
    "optimize", "efficiency", "margin expansion",
]


def _get_classifier():
    """Lazy-load the zero-shot classifier."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        logger.info("Loading zero-shot classifier (bart-large-mnli)...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU (use 0 for GPU)
        )
        logger.info("Classifier loaded.")
    return _classifier


def detect_risks(text: str, threshold: float = 0.5) -> List[Dict]:
    """
    Detect and categorize risks in a text passage.

    Args:
        text: Earnings call text to analyze.
        threshold: Minimum confidence for risk classification.

    Returns:
        List of {"text", "category", "confidence", "severity"}
    """
    # 1. Extract risk-bearing sentences
    sentences = _extract_risk_sentences(text)
    if not sentences:
        return []

    # 2. Classify each risk sentence
    classifier = _get_classifier()
    risks = []

    for sentence in sentences:
        result = classifier(sentence, candidate_labels=RISK_CATEGORIES, multi_label=True)
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        if top_score >= threshold:
            risks.append({
                "text": sentence,
                "category": top_label.replace(" risk", "").title(),
                "confidence": round(top_score, 3),
                "severity": _score_severity(sentence, top_score),
                "all_labels": dict(zip(result["labels"][:3], [round(s, 3) for s in result["scores"][:3]])),
            })

    logger.info(f"Detected {len(risks)} risks from {len(sentences)} risk-bearing sentences")
    return risks


def detect_opportunities(text: str) -> List[Dict]:
    """Detect growth opportunities and positive signals."""
    sentences = _split_sentences(text)
    opportunities = []

    for s in sentences:
        s_lower = s.lower()
        if any(kw in s_lower for kw in OPPORTUNITY_KEYWORDS):
            opportunities.append({
                "text": s,
                "type": _classify_opportunity(s),
            })

    return opportunities


def analyze_document(chunks: List[Dict], threshold: float = 0.5) -> Dict:
    """
    Analyze multiple chunks for risk and opportunity signals.

    Returns:
        {
            "risks": [...],
            "opportunities": [...],
            "risk_summary": {"Operational": count, ...},
            "overall_risk_level": "Medium"
        }
    """
    all_risks = []
    all_opportunities = []

    for chunk in chunks:
        text = chunk.get("text", "")
        risks = detect_risks(text, threshold)
        opps = detect_opportunities(text)

        for r in risks:
            r["source"] = chunk.get("metadata", {})
        for o in opps:
            o["source"] = chunk.get("metadata", {})

        all_risks.extend(risks)
        all_opportunities.extend(opps)

    # Summarize
    risk_counts = {}
    for r in all_risks:
        cat = r["category"]
        risk_counts[cat] = risk_counts.get(cat, 0) + 1

    overall = "Low"
    if len(all_risks) > 10:
        overall = "High"
    elif len(all_risks) > 5:
        overall = "Medium"

    return {
        "risks": all_risks,
        "opportunities": all_opportunities,
        "risk_summary": risk_counts,
        "overall_risk_level": overall,
        "total_risks": len(all_risks),
        "total_opportunities": len(all_opportunities),
    }


# ─── Helpers ─────────────────────────────────────────────────────

def _extract_risk_sentences(text: str) -> List[str]:
    """Extract sentences that contain risk-related keywords."""
    sentences = _split_sentences(text)
    return [s for s in sentences if any(kw in s.lower() for kw in RISK_KEYWORDS)]


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _score_severity(sentence: str, confidence: float) -> str:
    """Heuristic severity scoring."""
    high_markers = ["significant", "material", "major", "severe", "critical", "substantial"]
    low_markers = ["minor", "slight", "modest", "limited", "manageable"]

    s_lower = sentence.lower()
    if any(m in s_lower for m in high_markers) or confidence > 0.85:
        return "High"
    elif any(m in s_lower for m in low_markers) or confidence < 0.6:
        return "Low"
    return "Medium"


def _classify_opportunity(sentence: str) -> str:
    """Simple opportunity type classification."""
    s_lower = sentence.lower()
    if any(k in s_lower for k in ["growth", "expansion", "market"]):
        return "Growth"
    elif any(k in s_lower for k in ["efficiency", "cost", "optimize"]):
        return "Cost Optimization"
    elif any(k in s_lower for k in ["innovation", "technology", "ai", "ml"]):
        return "Innovation"
    return "General"
