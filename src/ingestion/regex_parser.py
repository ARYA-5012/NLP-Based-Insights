"""
Regex-based transcript parser for earnings call text.
Handles speaker diarization, section segmentation (Presentation vs Q&A),
and noise removal (Safe Harbor statements, legal disclaimers).
"""
import re
from typing import List, Dict, Tuple
from loguru import logger


class TranscriptParser:
    """Parses raw earnings call transcript text into structured segments."""

    # ─── Q&A Section Detection ───────────────────────────────────────
    QA_START_PATTERNS = [
        r"will now begin the question-and-answer",
        r"we will now begin the q\s*&\s*a",
        r"open the (?:floor|line) for questions",
        r"ready to take (?:your |the first )?questions",
        r"we are now ready for questions",
        r"(?:let's|let us) open (?:it |the call )?up for questions",
        r"begin (?:the )?q\s*&\s*a",
        r"operator.*?first question",
        r"instructions for the q\s*&\s*a",
    ]

    # ─── Speaker Identification ──────────────────────────────────────
    # Matches "FirstName LastName:" at start of line (2-4 word names)
    SPEAKER_PATTERN = re.compile(
        r"^([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+){0,3}):\s*"
    )

    # ─── Noise Patterns to Remove ────────────────────────────────────
    NOISE_PATTERNS = [
        r"(?:This|The following)\s+(?:call|presentation|conference).*?(?:forward[- ]looking|safe harbor).*?(?:\.|$)",
        r"(?:safe harbor|forward[- ]looking statements?).*?(?:10-K|10-Q|SEC|annual report).*?\.",
        r"(?:Copyright|©|\(c\)).*?(?:All rights reserved|Inc\.|Corp\.).*",
        r"^[-=]{3,}$",  # separator lines
        r"^\s*\[.*?\]\s*$",  # bracketed annotations like [Operator Instructions]
    ]

    # ─── Role Classification ─────────────────────────────────────────
    EXECUTIVE_KEYWORDS = ["ceo", "chief executive", "president", "chairman"]
    CFO_KEYWORDS = ["cfo", "chief financial", "finance officer", "treasurer"]
    IR_KEYWORDS = ["investor relations", "ir ", "head of ir"]
    OPERATOR_KEYWORDS = ["operator", "moderator", "conference operator"]

    def __init__(self):
        self._compiled_noise = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                                for p in self.NOISE_PATTERNS]

    def parse(self, text: str, metadata: Dict = None) -> Dict:
        """
        Main entry point. Parses raw transcript text into structured output.

        Returns:
            {
                "metadata": {...},
                "presentation": [{"speaker": str, "role": str, "text": str}, ...],
                "qa": [{"speaker": str, "role": str, "text": str}, ...],
                "stats": {"total_speakers": int, "presentation_segments": int, ...}
            }
        """
        metadata = metadata or {}

        # 1. Normalize newlines
        text = text.replace('\\n', '\n').replace('\r\n', '\n')

        # 2. Remove noise
        text = self._remove_noise(text)

        # 3. Split into Presentation vs Q&A
        presentation_text, qa_text = self._split_sections(text)

        # 4. Parse segments per section
        presentation = self._parse_segments(presentation_text)
        qa = self._parse_segments(qa_text)

        # 5. Classify speaker roles
        all_speakers = set(s["speaker"] for s in presentation + qa)
        role_map = self._classify_roles(presentation + qa, metadata)

        for segment in presentation + qa:
            segment["role"] = role_map.get(segment["speaker"], "Unknown")

        # 6. Compute stats
        stats = {
            "total_speakers": len(all_speakers),
            "presentation_segments": len(presentation),
            "qa_segments": len(qa),
            "has_qa": len(qa) > 0,
            "speakers": list(all_speakers),
        }

        logger.info(
            f"Parsed transcript: {stats['presentation_segments']} presentation + "
            f"{stats['qa_segments']} Q&A segments, {stats['total_speakers']} speakers"
        )

        return {
            "metadata": metadata,
            "presentation": presentation,
            "qa": qa,
            "stats": stats,
        }

    def _remove_noise(self, text: str) -> str:
        """Remove Safe Harbor statements, disclaimers, and formatting artifacts."""
        for pattern in self._compiled_noise:
            text = pattern.sub("", text)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_sections(self, text: str) -> Tuple[str, str]:
        """Split transcript into Presentation and Q&A sections."""
        lower_text = text.lower()

        for pattern in self.QA_START_PATTERNS:
            match = re.search(pattern, lower_text)
            if match:
                return text[: match.start()].strip(), text[match.start() :].strip()

        # Fallback: no Q&A found
        logger.warning("No Q&A section detected — treating entire text as presentation")
        return text, ""

    def _parse_segments(self, text: str) -> List[Dict[str, str]]:
        """Segment text by speaker turns."""
        if not text.strip():
            return []

        segments = []
        lines = text.split("\n")
        current_speaker = "Unknown"
        current_lines: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = self.SPEAKER_PATTERN.match(line)
            if match:
                # Save previous segment
                if current_lines:
                    segments.append({
                        "speaker": current_speaker,
                        "role": "",  # filled later
                        "text": " ".join(current_lines).strip(),
                    })
                    current_lines = []

                current_speaker = match.group(1)
                remainder = line[match.end() :].strip()
                if remainder:
                    current_lines.append(remainder)
            else:
                current_lines.append(line)

        # Final segment
        if current_lines:
            segments.append({
                "speaker": current_speaker,
                "role": "",
                "text": " ".join(current_lines).strip(),
            })

        return segments

    def _classify_roles(self, segments: List[Dict], metadata: Dict) -> Dict[str, str]:
        """Infer speaker roles from context and metadata."""
        role_map = {}
        
        for segment in segments:
            speaker = segment["speaker"]
            if speaker in role_map:
                continue

            text_lower = segment["text"].lower()
            speaker_lower = speaker.lower()

            if any(k in speaker_lower for k in self.OPERATOR_KEYWORDS):
                role_map[speaker] = "Operator"
            elif any(k in text_lower for k in self.IR_KEYWORDS):
                role_map[speaker] = "Investor Relations"
            elif any(k in text_lower for k in self.EXECUTIVE_KEYWORDS):
                role_map[speaker] = "CEO"
            elif any(k in text_lower for k in self.CFO_KEYWORDS):
                role_map[speaker] = "CFO"
            else:
                # Heuristic: speakers in presentation are usually executives,
                # new speakers in Q&A are usually analysts
                role_map[speaker] = "Executive" if segment in segments[:5] else "Analyst"

        return role_map


if __name__ == "__main__":
    import json

    # Test with mock data
    mock_path = "data/raw/mock_transcript.json"
    try:
        with open(mock_path, "r") as f:
            data = json.load(f)
        parser = TranscriptParser()
        result = parser.parse(data[0]["content"], metadata={"ticker": "MOCK", "quarter": 3, "year": 2024})
        print(f"Presentation: {result['stats']['presentation_segments']} segments")
        print(f"Q&A: {result['stats']['qa_segments']} segments")
        print(f"Speakers: {result['stats']['speakers']}")
    except FileNotFoundError:
        print("Mock file not found. Run api_client.py first or create data/raw/mock_transcript.json")
