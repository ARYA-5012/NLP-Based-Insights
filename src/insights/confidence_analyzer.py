"""
Management Confidence Analyzer.
Detects hedge words, certainty markers, deflection patterns, and
computes the "Q&A Alpha" signal: Presentation sentiment vs Q&A confidence.
"""
import re
from typing import List, Dict
from loguru import logger

# ─── Linguistic Feature Dictionaries ────────────────────────────

HEDGE_WORDS = [
    "might", "could", "possibly", "perhaps", "may", "potentially",
    "we believe", "we think", "we expect", "we hope", "we anticipate",
    "somewhat", "relatively", "to some extent", "in some ways",
    "it depends", "it's hard to say", "it's difficult to predict",
    "uncertain", "unclear", "remains to be seen",
]

CERTAINTY_MARKERS = [
    "will", "definitely", "certainly", "committed", "confident",
    "we are confident", "we will deliver", "absolutely", "guaranteed",
    "strong conviction", "clearly", "without a doubt", "undoubtedly",
    "we have demonstrated", "proven track record", "solid",
]

DEFLECTION_PATTERNS = [
    r"that's a (?:great|good|excellent|fair) question",
    r"as (?:you|we) (?:know|mentioned|discussed)",
    r"i think (?:the|what's) (?:important|key) (?:here |is )",
    r"let me (?:take a step back|provide some context|give you)",
    r"we (?:typically|generally) don't (?:comment|disclose|provide)",
    r"i'd (?:refer you to|point you toward)",
]

QUANTITATIVE_PATTERNS = [
    r'\d+(?:\.\d+)?%',           # percentages
    r'\$\d+(?:\.\d+)?\s*[BMK]',  # dollar amounts
    r'\d+(?:\.\d+)?\s*(?:billion|million|thousand)',
    r'(?:grew|increased|decreased|declined)\s+\d+',
]


def analyze_confidence(text: str, section: str = "unknown") -> Dict:
    """
    Analyze a text passage for management confidence signals.

    Args:
        text: Transcript text to analyze.
        section: "presentation" or "qa" (for the Q&A Alpha metric).

    Returns:
        {
            "confidence_score": float (0-1),
            "hedge_count": int,
            "certainty_count": int,
            "deflection_count": int,
            "quantitative_specificity": float (0-1),
            "details": {...}
        }
    """
    text_lower = text.lower()
    words = text_lower.split()
    total_words = max(len(words), 1)

    # 1. Count hedge words
    hedge_matches = []
    for hw in HEDGE_WORDS:
        count = text_lower.count(hw)
        if count > 0:
            hedge_matches.append({"word": hw, "count": count})
    hedge_count = sum(m["count"] for m in hedge_matches)

    # 2. Count certainty markers
    certainty_matches = []
    for cm in CERTAINTY_MARKERS:
        count = text_lower.count(cm)
        if count > 0:
            certainty_matches.append({"word": cm, "count": count})
    certainty_count = sum(m["count"] for m in certainty_matches)

    # 3. Count deflection patterns
    deflection_count = 0
    deflections = []
    for pattern in DEFLECTION_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            deflection_count += len(matches)
            deflections.extend(matches)

    # 4. Quantitative specificity (more numbers = more confident/transparent)
    quant_matches = []
    for pattern in QUANTITATIVE_PATTERNS:
        quant_matches.extend(re.findall(pattern, text))
    quant_score = min(len(quant_matches) / max(total_words / 100, 1), 1.0)

    # 5. Compute composite confidence score
    # Higher certainty + quantitative = more confident
    # Higher hedge + deflection = less confident
    hedge_penalty = min(hedge_count / max(total_words / 50, 1), 1.0)
    certainty_bonus = min(certainty_count / max(total_words / 50, 1), 1.0)
    deflection_penalty = min(deflection_count * 0.15, 0.5)

    confidence_score = max(0, min(1,
        0.5  # baseline
        + certainty_bonus * 0.3
        + quant_score * 0.2
        - hedge_penalty * 0.3
        - deflection_penalty
    ))

    return {
        "confidence_score": round(confidence_score, 3),
        "section": section,
        "hedge_count": hedge_count,
        "certainty_count": certainty_count,
        "deflection_count": deflection_count,
        "quantitative_specificity": round(quant_score, 3),
        "details": {
            "hedge_words_found": hedge_matches[:5],
            "certainty_markers_found": certainty_matches[:5],
            "deflections_found": deflections[:3],
            "quantitative_mentions": len(quant_matches),
        },
    }


def compute_sentiment_deviation(
    presentation_chunks: List[Dict],
    qa_chunks: List[Dict],
) -> Dict:
    """
    Compute the "Q&A Alpha" metric: deviation between
    scripted preparation sentiment and unscripted Q&A confidence.

    A HIGH deviation = Management is less confident than they appear.
    This is a warning signal for investors.

    Returns:
        {
            "presentation_confidence": float,
            "qa_confidence": float,
            "deviation_score": float,
            "flag": "NORMAL" | "WARNING" | "RED_FLAG",
            "interpretation": str
        }
    """
    # Analyze presentation chunks
    pres_scores = []
    for chunk in presentation_chunks:
        result = analyze_confidence(chunk.get("text", ""), section="presentation")
        pres_scores.append(result["confidence_score"])

    # Analyze Q&A chunks
    qa_scores = []
    for chunk in qa_chunks:
        result = analyze_confidence(chunk.get("text", ""), section="qa")
        qa_scores.append(result["confidence_score"])

    pres_avg = sum(pres_scores) / max(len(pres_scores), 1)
    qa_avg = sum(qa_scores) / max(len(qa_scores), 1)
    deviation = abs(pres_avg - qa_avg)

    # Flag thresholds
    if deviation > 0.3:
        flag = "RED_FLAG"
        interpretation = (
            f"Significant gap ({deviation:.2f}) between prepared remarks confidence "
            f"({pres_avg:.2f}) and Q&A confidence ({qa_avg:.2f}). "
            f"Management may be more uncertain than their scripted remarks suggest."
        )
    elif deviation > 0.15:
        flag = "WARNING"
        interpretation = (
            f"Moderate gap ({deviation:.2f}) between presentation ({pres_avg:.2f}) "
            f"and Q&A ({qa_avg:.2f}). Worth monitoring."
        )
    else:
        flag = "NORMAL"
        interpretation = (
            f"Consistent confidence across presentation ({pres_avg:.2f}) "
            f"and Q&A ({qa_avg:.2f}). No red flags detected."
        )

    logger.info(f"Sentiment Deviation: {deviation:.3f} → {flag}")

    return {
        "presentation_confidence": round(pres_avg, 3),
        "qa_confidence": round(qa_avg, 3),
        "deviation_score": round(deviation, 3),
        "flag": flag,
        "interpretation": interpretation,
        "presentation_samples": len(pres_scores),
        "qa_samples": len(qa_scores),
    }


def analyze_speaker(segments: List[Dict], speaker_name: str) -> Dict:
    """Analyze confidence for a specific speaker across all their segments."""
    speaker_segments = [s for s in segments if s.get("speaker") == speaker_name]
    if not speaker_segments:
        return {"speaker": speaker_name, "error": "No segments found"}

    full_text = " ".join(s.get("text", "") for s in speaker_segments)
    result = analyze_confidence(full_text)
    result["speaker"] = speaker_name
    result["segment_count"] = len(speaker_segments)
    return result
