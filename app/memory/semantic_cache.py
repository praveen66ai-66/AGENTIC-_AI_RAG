"""
Semantic answer cache — avoids re-running the full agent pipeline for questions
that are semantically equivalent to one already answered this session.

Storage: Redis HASH  key = {tenant}:{user}:scache
         Field = MD5(normalized question)
         Value = JSON { question, answer, embedding, confidence, ts }

Lookup strategy (two-pass):
  1. Exact match  — MD5 hash lookup, O(1)
  2. Semantic sim — cosine similarity of BGE embeddings, O(N) over cached entries
     → hit if similarity >= threshold (default 0.92)

Cache eviction:
  - Per-user cap of MAX_ENTRIES (100) — oldest entries removed when exceeded
  - TTL of 24 hours, reset on each store

Why per user (not per session):
  Users repeat questions across sessions. Caching at the user level means
  a question answered yesterday is still cached today.
"""

import hashlib
import json
import re
import time
from typing import Optional

import numpy as np


def _clean(answer: str) -> str:
    """Strip ## Sources / ## Confidence scaffolding the LLM used to emit."""
    answer = re.sub(r'\n##\s+(Sources|Confidence|Source).*', '', answer,
                    flags=re.DOTALL | re.IGNORECASE)
    answer = re.sub(r'^##\s+Answer\s*\n', '', answer, flags=re.IGNORECASE)
    return answer.strip()

_CACHE_TTL     = 86400   # 24 hours
_MAX_ENTRIES   = 100     # per user
_SIM_THRESHOLD = 0.92    # cosine similarity hit threshold


# ── Key ───────────────────────────────────────────────────────────────────────

def _key(tenant_id: str, user_id: str) -> str:
    return f"{tenant_id}:{user_id}:scache"


def _question_hash(question: str) -> str:
    return hashlib.md5(question.lower().strip().encode()).hexdigest()


# ── Math ──────────────────────────────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom  = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def check(
    tenant_id: str,
    user_id:   str,
    question:  str,
    threshold: float = _SIM_THRESHOLD,
) -> Optional[dict]:
    """
    Look up the cache for this question.

    Returns dict with keys:
        answer      — the cached answer string
        confidence  — original confidence score
        cache_type  — 'exact' | 'semantic'
        similarity  — cosine similarity (1.0 for exact)

    Returns None if no match above threshold.
    """
    from app.memory.session import _get_client
    from app.retrieval.search import _embed

    r   = _get_client()
    key = _key(tenant_id, user_id)

    # ── Pass 1: exact hash match O(1) ────────────────────────────────────────
    field     = _question_hash(question)
    exact_raw = r.hget(key, field)
    if exact_raw:
        entry = json.loads(exact_raw)
        return {
            "answer":     _clean(entry["answer"]),
            "confidence": entry["confidence"],
            "cache_type": "exact",
            "similarity": 1.0,
            "sources":    entry.get("sources", []),
        }

    # ── Pass 2: semantic similarity O(N) ─────────────────────────────────────
    all_raw = r.hgetall(key)
    if not all_raw:
        return None

    q_emb      = _embed.encode(question).tolist()
    best_sim   = 0.0
    best_entry = None

    for _, raw in all_raw.items():
        entry = json.loads(raw)
        sim   = _cosine_sim(q_emb, entry["embedding"])
        if sim > best_sim:
            best_sim   = sim
            best_entry = entry

    if best_sim >= threshold and best_entry:
        return {
            "answer":     _clean(best_entry["answer"]),
            "confidence": best_entry["confidence"],
            "cache_type": "semantic",
            "similarity": round(best_sim, 4),
            "sources":    best_entry.get("sources", []),
        }

    return None


def store(
    tenant_id:  str,
    user_id:    str,
    question:   str,
    answer:     str,
    confidence: float = 0.5,
    sources:    list  = None,
) -> None:
    """
    Cache the answer for this question.
    Evicts the oldest entries if over MAX_ENTRIES.
    """
    from app.memory.session import _get_client
    from app.retrieval.search import _embed

    r     = _get_client()
    key   = _key(tenant_id, user_id)
    field = _question_hash(question)

    embedding = _embed.encode(question).tolist()
    entry = {
        "question":   question,
        "answer":     _clean(answer),   # strip scaffolding before storing
        "embedding":  embedding,
        "confidence": confidence,
        "sources":    sources or [],
        "ts":         time.time(),
    }

    pipe = r.pipeline()
    pipe.hset(key, field, json.dumps(entry))
    pipe.expire(key, _CACHE_TTL)
    pipe.execute()

    # Evict oldest entries if over cap
    _evict_if_needed(r, key)


def invalidate(tenant_id: str, user_id: str, question: str) -> None:
    """Remove a specific question from the cache (e.g. after document update)."""
    r = _get_client()
    from app.memory.session import _get_client as gc
    gc().hdel(_key(tenant_id, user_id), _question_hash(question))


def clear_user_cache(tenant_id: str, user_id: str) -> None:
    """Wipe the entire semantic cache for a user."""
    from app.memory.session import _get_client
    _get_client().delete(_key(tenant_id, user_id))


# ── Eviction ──────────────────────────────────────────────────────────────────

def _evict_if_needed(r, key: str) -> None:
    """Delete the oldest entries so the hash stays under MAX_ENTRIES."""
    all_raw = r.hgetall(key)
    if len(all_raw) <= _MAX_ENTRIES:
        return

    by_ts = sorted(
        ((f, json.loads(v)["ts"]) for f, v in all_raw.items()),
        key=lambda x: x[1],
    )
    to_delete = [f for f, _ in by_ts[:len(by_ts) - _MAX_ENTRIES]]
    if to_delete:
        r.hdel(key, *to_delete)
