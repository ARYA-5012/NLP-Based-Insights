"""
LLM Client abstraction for OpenAI GPT models.
Handles API communication, retries, and token management.
"""
import os
from typing import List, Dict, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    """Lazy-load the OpenAI client."""
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate(
    prompt: str,
    system_prompt: str = "You are a senior financial analyst.",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> str:
    """
    Generate text using OpenAI API.

    Args:
        prompt: User prompt with context.
        system_prompt: System instructions.
        model: Model name (default from env).
        temperature: Creativity (0=deterministic, 1=creative).
        max_tokens: Max response length.

    Returns:
        Generated text string.
    """
    model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content.strip()
        usage = response.usage
        logger.debug(f"LLM [{model}]: {usage.prompt_tokens} in → {usage.completion_tokens} out")
        return text

    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise


def generate_with_context(
    query: str,
    context_chunks: List[Dict],
    system_prompt: str = "You are a senior financial analyst.",
    template: str = None,
    model: str = None,
) -> str:
    """
    RAG-style generation: answer a query using retrieved context chunks.

    Args:
        query: User's question.
        context_chunks: Retrieved chunks with "text" and "metadata".
        system_prompt: System instructions.
        template: Custom prompt template (use {query} and {context} placeholders).
        model: LLM model name.

    Returns:
        Generated answer grounded in the provided context.
    """
    # Format context
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        source = f"{meta.get('ticker', '?')} Q{meta.get('quarter', '?')} {meta.get('year', '?')}"
        speaker = f"{meta.get('speaker', 'Unknown')} ({meta.get('role', '')})"
        section = meta.get("section", "")
        context_parts.append(
            f"[Source {i}: {source} | {speaker} | {section}]\n{chunk['text']}"
        )
    context_str = "\n\n".join(context_parts)

    # Build prompt
    if template:
        prompt = template.format(query=query, context=context_str)
    else:
        prompt = (
            f"Based ONLY on the following excerpts from earnings call transcripts, "
            f"answer this question:\n\n"
            f"**Question**: {query}\n\n"
            f"## Excerpts:\n{context_str}\n\n"
            f"## Rules:\n"
            f"- Only use facts from the excerpts above.\n"
            f"- Cite the source (company, quarter) for each claim.\n"
            f"- If the information is not available, say 'Not discussed in the provided excerpts.'\n"
        )

    return generate(prompt, system_prompt=system_prompt, model=model)
