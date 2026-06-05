"""
Output guardrail — validates the generated answer against retrieved chunks.

Uses the LLM judge defined in app/prompts/guardrails/output.yml to detect:
- Hallucinated facts not present in any source chunk
- Financial figures not traceable to a source
- Unsourced claims

Returns GuardResult(PASS) or GuardResult(BLOCK, reason).
The generator node appends the reason as a warning rather than suppressing
the answer entirely, preserving utility while flagging quality issues.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from app.agents._utils import format_chunks, load_guardrail_prompt, parse_json

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    PASS  = "pass"
    BLOCK = "block"


@dataclass
class GuardResult:
    decision: Decision
    reason:   str = ""

    @property
    def blocked(self) -> bool:
        return self.decision == Decision.BLOCK


_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=1024,
        )
    return _llm


def check(answer: str, chunks: list[dict]) -> GuardResult:
    """Validate the generated answer against the source chunks used to produce it."""
    if not answer or not chunks:
        return GuardResult(Decision.PASS)

    prompt = load_guardrail_prompt("output")
    system_text = prompt["system"].format(
        retrieved_chunks=format_chunks(chunks),
        generated_answer=answer,
    )

    try:
        response = _get_llm().invoke([SystemMessage(content=system_text)])
        parsed   = parse_json(response.content)
    except Exception as exc:
        # Never block on guardrail failure — fail open (intentional)
        logger.warning("output_guard LLM call failed, failing open: %s", exc)
        return GuardResult(Decision.PASS)

    if not parsed.get("passes", True):
        risk   = parsed.get("hallucination_risk", "unknown")
        unsrc  = parsed.get("unsourced_claims") or []
        reason = f"Hallucination risk: {risk}."
        if unsrc:
            reason += " Unsourced: " + "; ".join(str(c) for c in unsrc[:3])
        return GuardResult(Decision.BLOCK, reason)

    return GuardResult(Decision.PASS)
