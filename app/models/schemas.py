"""Pydantic request/response schemas — shared between FastAPI and Streamlit."""

import re
import time
from datetime import datetime, timezone
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


class QueryRequest(BaseModel):
    question:   str
    tenant_id:  Optional[str] = None   # authoritative source is X-Tenant-Id header
    user_id:    Optional[str] = None   # authoritative source is X-User-Id header
    session_id: Optional[str] = None   # authoritative source is X-Session-Id header
    top_k:      int = 5

    @field_validator("question")
    @classmethod
    def question_clean(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 500:
            raise ValueError("Question must be 500 characters or fewer.")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_range(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError("top_k must be between 1 and 10.")
        return v


class AgentRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None

    @field_validator("question")
    @classmethod
    def question_clean(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 500:
            raise ValueError("Question must be 500 characters or fewer.")
        return v


class AgentSource(BaseModel):
    page:    Optional[int]
    section: str
    source:  str
    score:   float


class AgentResponse(BaseModel):
    answer:             str
    sources:            list[AgentSource]
    confidence:         float
    plan:               list[str]
    iteration_count:    int
    session_id:         str
    trajectory_id:      str
    cache_hit:          bool
    cache_type:         Optional[str]
    similarity:         Optional[float]
    idempotency_replay: bool
    created_at:         str
    duration_ms:        float


class Citation(BaseModel):
    number:     int
    page:       Optional[int]
    section:    Optional[str]
    source:     Optional[str]   # "text" | "table"
    collection: str


class RetrievedChunk(BaseModel):
    citation_number: int
    text:       str
    score:      float
    page:       Optional[int]
    section:    Optional[str]
    source:     Optional[str]
    collection: str


class QueryResponse(BaseModel):
    question:    str
    tenant_id:   str
    user_id:     str
    session_id:  str
    chunks:      list[RetrievedChunk]
    citations:   list[Citation]
    answer:      Optional[str] = None
    llm_ready:   bool = False
    created_at:  str  = ""          # ISO-8601 UAE (UTC+4) timestamp of the response
    duration_ms: float = 0.0        # end-to-end latency in milliseconds
