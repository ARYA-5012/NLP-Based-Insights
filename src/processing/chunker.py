"""
Semantic Chunker for earnings call transcripts.
Splits parsed segments into retrieval-ready chunks respecting:
  1. Speaker boundaries (never cross speakers)
  2. Sentence boundaries (never break mid-sentence)
  3. Token limits (300-500 tokens per chunk)
  4. Overlap for context continuity
"""
import re
from typing import List, Dict
from loguru import logger

try:
    import tiktoken
    _encoder = tiktoken.encoding_for_model("gpt-4")
except ImportError:
    _encoder = None
    logger.warning("tiktoken not installed — falling back to word-based token counting")


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken if available, else rough word count."""
    if _encoder:
        return len(_encoder.encode(text))
    return len(text.split())


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex (fast, no spaCy dependency)."""
    # Split on period/question/exclamation followed by space + capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_segments(
    segments: List[Dict],
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    metadata_base: Dict = None,
) -> List[Dict]:
    """
    Convert parsed transcript segments into retrieval-ready chunks.

    Args:
        segments: Output from TranscriptParser.parse() — list of {"speaker", "role", "text"}
        max_tokens: Maximum tokens per chunk (target: 300-500)
        overlap_tokens: Tokens of overlap between consecutive chunks
        metadata_base: Base metadata to attach (ticker, quarter, year, section)

    Returns:
        List of chunk dicts: {"text", "metadata", "token_count"}
    """
    metadata_base = metadata_base or {}
    chunks = []

    for segment in segments:
        if not segment.get("text", "").strip():
            continue

        sentences = _split_sentences(segment["text"])
        if not sentences:
            continue

        # Build per-segment metadata
        seg_metadata = {
            **metadata_base,
            "speaker": segment.get("speaker", "Unknown"),
            "role": segment.get("role", "Unknown"),
        }

        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = _count_tokens(sentence)

            # If single sentence exceeds max, treat it as its own chunk
            if s_tokens > max_tokens:
                # Save accumulated
                if current_sentences:
                    chunks.append(_make_chunk(current_sentences, seg_metadata))
                # Save oversized sentence as its own chunk
                chunks.append(_make_chunk([sentence], seg_metadata))
                current_sentences = []
                current_tokens = 0
                continue

            # If adding this sentence would exceed max, save current chunk
            if current_tokens + s_tokens > max_tokens and current_sentences:
                chunks.append(_make_chunk(current_sentences, seg_metadata))

                # Overlap: keep last sentences that fit within overlap budget
                overlap_sents = _get_overlap(current_sentences, overlap_tokens)
                current_sentences = overlap_sents
                current_tokens = sum(_count_tokens(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_tokens += s_tokens

        # Final chunk for this segment
        if current_sentences:
            chunks.append(_make_chunk(current_sentences, seg_metadata))

    logger.info(f"Created {len(chunks)} chunks from {len(segments)} segments "
                f"(avg {sum(c['token_count'] for c in chunks) // max(len(chunks), 1)} tokens/chunk)")
    return chunks


def _make_chunk(sentences: List[str], metadata: Dict) -> Dict:
    """Create a chunk dict from sentences."""
    text = " ".join(sentences)
    return {
        "text": text,
        "metadata": dict(metadata),  # copy
        "token_count": _count_tokens(text),
    }


def _get_overlap(sentences: List[str], max_overlap_tokens: int) -> List[str]:
    """Get trailing sentences that fit within the overlap token budget."""
    overlap = []
    tokens = 0
    for sent in reversed(sentences):
        t = _count_tokens(sent)
        if tokens + t > max_overlap_tokens:
            break
        overlap.insert(0, sent)
        tokens += t
    return overlap


if __name__ == "__main__":
    # Quick test
    test_segments = [
        {
            "speaker": "Alice Smith",
            "role": "CEO",
            "text": (
                "We are pleased to report strong results. Revenue grew 15% year-over-year. "
                "Our AI platform saw significant adoption across enterprise customers. "
                "We launched three new products this quarter. Innovation remains at the core."
            ),
        },
        {
            "speaker": "Bob Jones",
            "role": "CFO",
            "text": (
                "Total revenue was $500 million. Gross margin expanded to 65%. "
                "We ended the quarter with $1.2 billion in cash."
            ),
        },
    ]
    result = chunk_segments(
        test_segments,
        max_tokens=30,  # low limit for testing
        overlap_tokens=5,
        metadata_base={"ticker": "MOCK", "quarter": "Q3", "year": 2024, "section": "presentation"},
    )
    for i, c in enumerate(result):
        print(f"Chunk {i}: [{c['token_count']} tok] {c['text'][:80]}...")
        print(f"  Meta: {c['metadata']}")
