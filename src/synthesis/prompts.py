"""
Centralized prompt templates for all LLM-powered features.
Each template uses {placeholders} for dynamic content injection.
"""

# ─── Executive Summary ──────────────────────────────────────────
EXECUTIVE_SUMMARY = """You are a senior equity research analyst. Based ONLY on the following excerpts from {company}'s Q{quarter} {year} earnings call, provide:

1. **Key Takeaways** (3-5 bullet points — the most important facts for an investor)
2. **Top Risks** (up to 3, with severity: High/Medium/Low)
3. **Strategic Initiatives** (major announcements or strategy shifts)
4. **Analyst Concerns** (what topics dominated the Q&A section)

## Excerpts:
{context}

## Rules:
- Do NOT hallucinate facts not present in the excerpts.
- If information is not available, say "Not discussed in this call."
- Cite the speaker for each insight (e.g., "CEO stated...").
- Be concise. Each bullet should be one sentence.
"""

# ─── Risk Report ─────────────────────────────────────────────────
RISK_REPORT = """You are a risk analyst reviewing {company}'s Q{quarter} {year} earnings call. From the excerpts below, extract ALL risk-related statements.

For each risk, provide:
- **Category**: Operational / Financial / Regulatory / Competitive / Geopolitical / Market
- **Severity**: High / Medium / Low
- **Description**: One-sentence summary
- **Direct Quote**: A relevant quote from the transcript
- **Speaker**: Who mentioned it

## Excerpts:
{context}

Format your response as a numbered list. If no risks are identified, state "No significant risks discussed."
"""

# ─── Competitive Intelligence ────────────────────────────────────
COMPETITIVE_INTEL = """You are a competitive intelligence analyst. From the following earnings call excerpts, extract all mentions of competitors.

For each competitor mentioned, provide:
- **Competitor Name**
- **Context**: Why they were mentioned
- **Sentiment**: Positive / Neutral / Negative (how the company views this competitor)
- **Strategic Implication**: What this means for competitive positioning

## Excerpts:
{context}

## Company Under Analysis: {company}
"""

# ─── Company Comparison ─────────────────────────────────────────
COMPANY_COMPARISON = """You are a senior analyst comparing two companies' earnings call narratives.

## Company A: {company_a}
{context_a}

## Company B: {company_b}
{context_b}

## Topic of Comparison: {topic}

Provide a structured comparison covering:
1. **Key Differences** in approach/strategy
2. **Tone & Confidence** (which company sounds more confident?)
3. **Risk Profiles** (who faces bigger challenges?)
4. **Strategic Focus** (where are they investing?)
5. **Analyst Concerns** (what are analysts worried about for each?)

Be specific and cite quotes where possible.
"""

# ─── Trend Analysis ──────────────────────────────────────────────
TREND_ANALYSIS = """You are a market strategist analyzing industry trends. Based on the following excerpts from multiple companies' earnings calls, identify:

1. **Emerging Themes** (topics appearing more frequently than before)
2. **Declining Themes** (topics fading from discussion)
3. **Consensus Views** (what most companies agree on)
4. **Contrarian Signals** (where one company diverges from the pack)
5. **Actionable Insights** (what should investors pay attention to?)

## Topic: {topic}
## Time Period: {period}

## Excerpts from Multiple Companies:
{context}
"""

# ─── Q&A Alpha Analysis ─────────────────────────────────────────
QA_ALPHA = """You are analyzing the Q&A section of {company}'s Q{quarter} {year} earnings call for signs of management evasiveness or confidence.

Analyze each analyst question and management response below:

{context}

For each Q&A exchange, assess:
1. **Directness**: Did management answer directly (1-5 scale, 5=very direct)?
2. **Hedge Words**: Count of uncertain language ("might", "could", "possibly", "we believe")
3. **Deflection**: Did they redirect to a different topic?
4. **Confidence Level**: High / Medium / Low

Finally, provide an **Overall Q&A Confidence Score** (1-10) and explain your reasoning.
"""
