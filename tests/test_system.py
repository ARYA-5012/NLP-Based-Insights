"""
Tests for the EarningsInsight AI system.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Test Regex Parser ───────────────────────────────────────────

class TestTranscriptParser:
    """Tests for the transcript parser."""

    def setup_method(self):
        from src.ingestion.regex_parser import TranscriptParser
        self.parser = TranscriptParser()

    def test_parse_basic_transcript(self):
        text = (
            "Operator: Welcome to the earnings call.\\n"
            "Alice Smith: Thank you. Revenue grew 15% this quarter.\\n"
            "We are very pleased with these results.\\n"
            "Operator: We will now begin the question-and-answer session.\\n"
            "Bob Analyst: Can you discuss the margin outlook?\\n"
            "Alice Smith: We expect margins to remain stable."
        )
        result = self.parser.parse(text)
        assert result["stats"]["has_qa"] is True
        assert result["stats"]["presentation_segments"] > 0
        assert result["stats"]["qa_segments"] > 0

    def test_parse_no_qa_section(self):
        text = (
            "Operator: Welcome.\\n"
            "CEO Name: Revenue was strong this quarter."
        )
        result = self.parser.parse(text)
        assert result["stats"]["has_qa"] is False
        assert result["stats"]["presentation_segments"] > 0

    def test_speaker_identification(self):
        text = "Alice Smith: Good morning everyone."
        result = self.parser.parse(text)
        found_speakers = result["stats"]["speakers"]
        assert "Alice Smith" in found_speakers

    def test_noise_removal(self):
        text = "---\nOperator: Welcome.\n---"
        result = self.parser.parse(text)
        # Should still find the operator
        assert len(result["presentation"]) > 0


# ─── Test Chunker ────────────────────────────────────────────────

class TestChunker:
    """Tests for the semantic chunker."""

    def test_basic_chunking(self):
        from src.processing.chunker import chunk_segments

        segments = [
            {
                "speaker": "CEO",
                "role": "CEO",
                "text": "Revenue grew 15% year-over-year. We are investing in AI. "
                        "Our platform saw significant growth. Innovation is key.",
            }
        ]
        chunks = chunk_segments(segments, max_tokens=20, overlap_tokens=5)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["speaker"] == "CEO"

    def test_speaker_boundary_respected(self):
        from src.processing.chunker import chunk_segments

        segments = [
            {"speaker": "CEO", "role": "CEO", "text": "We had a great quarter."},
            {"speaker": "CFO", "role": "CFO", "text": "Revenue was $500 million."},
        ]
        chunks = chunk_segments(segments, max_tokens=1000)
        speakers = [c["metadata"]["speaker"] for c in chunks]
        # Each chunk should have exactly one speaker
        assert all(s in ["CEO", "CFO"] for s in speakers)

    def test_metadata_attached(self):
        from src.processing.chunker import chunk_segments

        segments = [{"speaker": "CEO", "role": "CEO", "text": "Revenue grew."}]
        chunks = chunk_segments(
            segments,
            metadata_base={"ticker": "AAPL", "quarter": "Q3", "year": 2024}
        )
        assert chunks[0]["metadata"]["ticker"] == "AAPL"


# ─── Test Confidence Analyzer ────────────────────────────────────

class TestConfidenceAnalyzer:
    """Tests for the management confidence analyzer."""

    def test_high_confidence_text(self):
        from src.insights.confidence_analyzer import analyze_confidence

        text = "We will definitely deliver strong results. We are committed to 15% growth."
        result = analyze_confidence(text)
        assert result["confidence_score"] > 0.5
        assert result["certainty_count"] > 0

    def test_low_confidence_text(self):
        from src.insights.confidence_analyzer import analyze_confidence

        text = "We might possibly see some growth. It's uncertain and difficult to predict."
        result = analyze_confidence(text)
        assert result["hedge_count"] > 0

    def test_sentiment_deviation(self):
        from src.insights.confidence_analyzer import compute_sentiment_deviation

        pres = [{"text": "We will definitely deliver outstanding results. Strong conviction."}]
        qa = [{"text": "It might be difficult. We're uncertain about the outlook."}]
        
        result = compute_sentiment_deviation(pres, qa)
        assert "flag" in result
        assert result["deviation_score"] >= 0

    def test_quantitative_specificity(self):
        from src.insights.confidence_analyzer import analyze_confidence

        specific = "Revenue grew 15.3% to $500M. Margin expanded 200bps."
        vague = "Revenue grew nicely. Margins improved somewhat."

        specific_result = analyze_confidence(specific)
        vague_result = analyze_confidence(vague)
        assert specific_result["quantitative_specificity"] > vague_result["quantitative_specificity"]


# ─── Test Competitive Intel ──────────────────────────────────────

class TestCompetitiveIntel:
    """Tests for competitor extraction."""

    def test_extract_competitors(self):
        from src.insights.competitive_intel import extract_competitor_mentions

        text = "We continue to see competitive pressure from Microsoft in the cloud space."
        mentions = extract_competitor_mentions(text, source_ticker="AMZN")
        assert len(mentions) > 0
        assert any(m["ticker"] == "MSFT" for m in mentions)

    def test_exclude_source_company(self):
        from src.insights.competitive_intel import extract_competitor_mentions

        text = "Apple continues to innovate. We are ahead of Google in this space."
        mentions = extract_competitor_mentions(text, source_ticker="AAPL")
        # Should not include Apple as a competitor mention
        assert not any(m["ticker"] == "AAPL" for m in mentions)


# ─── Test API ────────────────────────────────────────────────────

class TestAPI:
    """Tests for FastAPI endpoints."""

    def setup_method(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "running"

    def test_detailed_health(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "api" in data
        assert data["api"] == "healthy"

    def test_search_validation(self):
        """Search should reject queries that are too short."""
        response = self.client.post("/api/search", json={"query": "ab"})
        assert response.status_code == 422  # Validation error

    def test_compare_requires_two(self):
        """Compare should require at least 2 tickers."""
        response = self.client.get("/api/compare?companies=AAPL&topic=test")
        assert response.status_code == 400

    def test_topics_endpoint(self):
        response = self.client.get("/api/topics")
        assert response.status_code == 200

    def test_trends_endpoint(self):
        response = self.client.get("/api/trends")
        assert response.status_code == 200
