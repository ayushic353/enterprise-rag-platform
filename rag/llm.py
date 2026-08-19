"""
LLM generation layer. Tries Groq first (fast, hosted); if that fails
(missing key, network error, rate limit) it falls back to a local
Ollama server. This is the "dual LLM support" feature.
"""
import logging
from typing import List, Optional

import requests

from rag import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the user's question "
    "using ONLY the provided context. If the answer is not contained in the "
    "context, say you don't have enough information — do not make things up. "
    "Cite which source(s) you used when relevant."
)


def _build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def _call_groq(question: str, context_chunks: List[str]) -> Optional[str]:
    if not config.GROQ_API_KEY:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=config.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(question, context_chunks)},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groq call failed, will try fallback: %s", exc)
        return None


def _call_ollama(question: str, context_chunks: List[str]) -> Optional[str]:
    try:
        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\n{_build_prompt(question, context_chunks)}",
            "stream": False,
        }
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json().get("response")
    except Exception as exc:  # noqa: BLE001
        logger.error("Ollama fallback also failed: %s", exc)
        return None


def generate_answer(question: str, context_chunks: List[str]) -> dict:
    """
    Returns {"answer": str, "provider": "groq" | "ollama" | "none"}
    """
    answer = _call_groq(question, context_chunks)
    if answer:
        return {"answer": answer, "provider": "groq"}

    answer = _call_ollama(question, context_chunks)
    if answer:
        return {"answer": answer, "provider": "ollama"}

    return {
        "answer": (
            "I couldn't reach either LLM provider (Groq or Ollama). "
            "Check your GROQ_API_KEY or make sure `ollama serve` is running."
        ),
        "provider": "none",
    }
