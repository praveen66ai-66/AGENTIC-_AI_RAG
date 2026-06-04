"""Shared helpers for all agent nodes: LLM singleton, prompt loader, formatters."""

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from langchain_groq import ChatGroq

_llm: Any = None


def get_llm() -> Any:
    """
    Returns a Groq LLM with:
      - 3-attempt retry with exponential jitter on transient errors
      - Gemini 1.5 Flash fallback if GOOGLE_API_KEY is set and Groq keeps failing
    """
    global _llm
    if _llm is None:
        primary = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=4096,
        )

        # Retry up to 3 times on any transient Groq error before giving up / falling back
        primary_with_retry = primary.with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
        )

        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            fallback = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_key,
                max_output_tokens=4096,
            )
            _llm = primary_with_retry.with_fallbacks([fallback])
        else:
            _llm = primary_with_retry

    return _llm


def load_prompt(agent: str) -> dict:
    """Load app/prompts/agents/{agent}.yml"""
    path = Path(f"app/prompts/agents/{agent}.yml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_guardrail_prompt(layer: str) -> dict:
    """Load app/prompts/guardrails/{layer}.yml"""
    path = Path(f"app/prompts/guardrails/{layer}.yml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_tokens(response) -> dict:
    """
    Extract real token usage from a LangChain LLM response.
    Returns {"in": N, "out": M, "total": N+M} or empty dict on failure.
    Works with both invoke() AIMessage and stream() AIMessageChunk (last chunk).
    """
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        m = response.usage_metadata
        return {
            "in":    m.get("input_tokens",  0),
            "out":   m.get("output_tokens", 0),
            "total": m.get("total_tokens",  0),
        }
    if hasattr(response, "response_metadata"):
        tu = (response.response_metadata or {}).get("token_usage", {})
        if tu:
            return {
                "in":    tu.get("prompt_tokens",     0),
                "out":   tu.get("completion_tokens", 0),
                "total": tu.get("total_tokens",      0),
            }
    return {}


def parse_json(text: str) -> dict:
    """Extract and parse JSON from an LLM response, stripping markdown fences.
    Always returns a dict — never a list, string, or None."""
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    try:
        result = json.loads(text.strip())
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def format_chunks(chunks: list[dict], max_chunks: int = 8) -> str:
    """Render retrieved chunks as numbered blocks for prompt injection."""
    if not chunks:
        return "No chunks retrieved."
    lines = []
    for i, c in enumerate(chunks[:max_chunks], 1):
        page    = c.get("page", "?")
        section = c.get("section", "") or ""
        text    = (c.get("text") or "").strip()
        lines.append(f"[{i}] Page {page} | {section}\n{text}")
    return "\n\n".join(lines)


def format_history(history: list[dict], last_n: int = 3) -> str:
    """Render Redis session history as plain text for prompt injection."""
    if not history:
        return "No prior conversation."
    lines = []
    for msg in history[-last_n:]:
        role    = (msg.get("role") or "user").capitalize()
        content = msg.get("content") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
