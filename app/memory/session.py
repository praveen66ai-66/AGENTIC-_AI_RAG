"""
Redis short-term memory — tenant/user/session isolated.

Key layout (all TTL-refreshed on activity):
  {tenant}:{user}:{session}:history          LIST   — conversation turns
  {tenant}:{user}:{session}:meta             HASH   — session stats
  {tenant}:{user}:{session}:chunks           SET    — chunk MD5s seen (dedup)
  {tenant}:{user}:{session}:msgdedup:{id}    STRING — message dedup guard (SETNX)
  {tenant}:{user}:{session}:qguard:{req_id}  STRING — query-count guard (SETNX)
  {tenant}:sessions                          ZSET   — active sessions by last_active epoch

Idempotency guarantees:
  append_message  — msg_id SETNX guard prevents duplicate turns on retry
  bump_query_count — request_id SETNX guard prevents double-increment on retry
  init_meta       — HSETNX never overwrites existing fields
  filter_new_chunks — Lua script makes SISMEMBER+SADD atomic (no TOCTOU)
  add_chunks_seen — SADD is always idempotent
  _register_session — ZADD is always idempotent
"""

import hashlib
import json
import os
import time
from typing import Optional

import redis

SESSION_TTL  = 3600   # 1 hour  — reset on every message
TENANT_TTL   = 86400  # 24 hours — tenant session index
MAX_HISTORY  = 20     # rolling window of last N turns

_client: Optional[redis.Redis] = None

# ── Lua: atomic SISMEMBER-then-SADD for chunk dedup (eliminates TOCTOU) ──────
# KEYS[1] = set key
# ARGV    = list of chunk-id strings
# Returns list of IDs that were NOT already in the set (and are now added).
_ATOMIC_FILTER_ADD = """
local key  = KEYS[1]
local new  = {}
for _, id in ipairs(ARGV) do
    if redis.call('SISMEMBER', key, id) == 0 then
        redis.call('SADD', key, id)
        table.insert(new, id)
    end
end
return new
"""


# ── Connection ────────────────────────────────────────────────────────────────

def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        url = os.getenv("REDIS_URL")
        if url:
            _client = redis.Redis.from_url(url, decode_responses=True, protocol=2)
        else:
            _client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
                protocol=2,
            )
    return _client


def ping() -> bool:
    try:
        return _get_client().ping()
    except Exception:
        return False


# ── Key builders ──────────────────────────────────────────────────────────────

def _hist_key(prefix: str)    -> str: return f"{prefix}:history"
def _meta_key(prefix: str)    -> str: return f"{prefix}:meta"
def _chunks_key(prefix: str)  -> str: return f"{prefix}:chunks"
def _sessions_key(tid: str)   -> str: return f"{tid}:sessions"

def make_prefix(tenant_id: str, user_id: str, session_id: str) -> str:
    """Canonical Redis key prefix — isolates per tenant / user / session."""
    return f"{tenant_id}:{user_id}:{session_id}"


# ── Conversation history ──────────────────────────────────────────────────────

def append_message(
    prefix:  str,
    role:    str,
    content: str,
    msg_id:  str | None        = None,
    sources: list[dict] | None = None,
    tokens:  int | None        = None,
) -> bool:
    """
    Append one turn to the session history list.

    msg_id — caller-supplied idempotency token (e.g. request UUID + role).
             If provided: SETNX guard prevents the same message being appended
             twice on a client retry. Pass None to skip the guard.

    Returns True if the message was written, False if it was a duplicate.
    """
    r = _get_client()

    # ── Idempotency guard ─────────────────────────────────────────────────────
    if msg_id:
        guard = f"{prefix}:msgdedup:{msg_id}"
        if not r.setnx(guard, 1):
            return False          # already appended — skip silently
        r.expire(guard, SESSION_TTL)

    entry = {
        "role":    role,
        "content": content,
        "ts":      time.time(),
        "tokens":  tokens or max(1, len(content) // 4),
        "sources": sources or [],
    }

    pipe = r.pipeline()
    pipe.rpush(_hist_key(prefix), json.dumps(entry))
    pipe.ltrim(_hist_key(prefix), -MAX_HISTORY, -1)
    pipe.expire(_hist_key(prefix), SESSION_TTL)
    pipe.execute()

    _increment_token_meta(prefix, tokens=entry["tokens"])
    return True


def get_history(
    prefix:     str,
    tenant_id:  str | None = None,
    user_id:    str | None = None,
    session_id: str | None = None,
) -> list[dict]:
    """
    Return session history. Checks Redis first (fast path).
    If Redis is empty AND tenant/user/session are supplied, falls back to
    PostgreSQL episodic_memory so history survives the 1-hour Redis TTL.
    """
    r   = _get_client()
    raw = r.lrange(_hist_key(prefix), 0, -1)
    if raw:
        return [json.loads(m) for m in raw]

    # Cold-start fallback — Redis TTL expired, reload from PG
    if tenant_id and user_id and session_id:
        from app.memory.store import get_episodic_history
        pg_rows = get_episodic_history(tenant_id, user_id, session_id, limit=MAX_HISTORY)
        if pg_rows:
            # Rehydrate Redis so subsequent calls are fast again
            pipe = r.pipeline()
            for row in pg_rows:
                pipe.rpush(_hist_key(prefix), json.dumps({
                    "role":    row.get("role", ""),
                    "content": row.get("content", ""),
                    "ts":      row.get("created_at", ""),
                    "tokens":  row.get("tokens", 0),
                    "sources": row.get("sources") or [],
                }))
            pipe.ltrim(_hist_key(prefix), -MAX_HISTORY, -1)
            pipe.expire(_hist_key(prefix), SESSION_TTL)
            pipe.execute()
        return pg_rows

    return []


def clear_session(prefix: str) -> None:
    r = _get_client()
    r.delete(_hist_key(prefix), _meta_key(prefix), _chunks_key(prefix))


# ── Session metadata ──────────────────────────────────────────────────────────

def init_meta(prefix: str, tenant_id: str, user_id: str, session_id: str) -> None:
    """
    Create session metadata on first use.
    HSETNX guarantees this is fully idempotent — repeated calls are no-ops.
    """
    r   = _get_client()
    key = _meta_key(prefix)
    now = time.time()

    pipe = r.pipeline()
    pipe.hsetnx(key, "tenant_id",    tenant_id)
    pipe.hsetnx(key, "user_id",      user_id)
    pipe.hsetnx(key, "session_id",   session_id)
    pipe.hsetnx(key, "created_at",   now)
    pipe.hsetnx(key, "last_active",  now)
    pipe.hsetnx(key, "query_count",  0)
    pipe.hsetnx(key, "total_tokens", 0)
    pipe.expire(key, SESSION_TTL)
    pipe.execute()

    _register_session(tenant_id, session_id)


def get_meta(prefix: str) -> dict:
    raw = _get_client().hgetall(_meta_key(prefix))
    return {
        **raw,
        "query_count":  int(raw.get("query_count",  0)),
        "total_tokens": int(raw.get("total_tokens", 0)),
        "created_at":   float(raw.get("created_at",  0)),
        "last_active":  float(raw.get("last_active", 0)),
    }


def _increment_token_meta(prefix: str, tokens: int = 0) -> None:
    """Internal — update token total and last_active timestamp."""
    r   = _get_client()
    key = _meta_key(prefix)
    pipe = r.pipeline()
    pipe.hincrbyfloat(key, "total_tokens", tokens)
    pipe.hset(key, "last_active", time.time())
    pipe.expire(key, SESSION_TTL)
    pipe.execute()


def bump_query_count(prefix: str, request_id: str | None = None) -> int:
    """
    Increment the session query counter.

    request_id — caller-supplied idempotency token (e.g. the request UUID).
                 SETNX guard ensures one request increments the counter exactly
                 once even if the caller retries.

    Returns the current count after the operation.
    """
    r   = _get_client()
    key = _meta_key(prefix)

    # ── Idempotency guard ─────────────────────────────────────────────────────
    if request_id:
        guard = f"{prefix}:qguard:{request_id}"
        if not r.setnx(guard, 1):
            # Already counted for this request — return current value
            return int(r.hget(key, "query_count") or 0)
        r.expire(guard, SESSION_TTL)

    r.expire(key, SESSION_TTL)
    return int(r.hincrby(key, "query_count", 1))


# ── Chunk deduplication ───────────────────────────────────────────────────────

def filter_new_chunks(prefix: str, chunks: list[dict]) -> list[dict]:
    """
    Return only chunks NOT yet seen this session and mark them as seen atomically.

    Uses a Lua script so the SISMEMBER check and SADD happen in one round-trip
    with no TOCTOU window — safe under concurrent workers.
    """
    if not chunks:
        return chunks

    r   = _get_client()
    key = _chunks_key(prefix)

    ids = [
        hashlib.md5(c.get("text", "")[:200].encode()).hexdigest()
        for c in chunks
    ]

    new_ids = r.eval(_ATOMIC_FILTER_ADD, 1, key, *ids)
    r.expire(key, SESSION_TTL)

    new_id_set  = set(new_ids)
    return [c for c, cid in zip(chunks, ids) if cid in new_id_set]


def add_chunks_seen(prefix: str, chunk_ids: list[str]) -> None:
    """Unconditionally mark chunk IDs as seen. SADD is idempotent."""
    if not chunk_ids:
        return
    r = _get_client()
    pipe = r.pipeline()
    pipe.sadd(_chunks_key(prefix), *chunk_ids)
    pipe.expire(_chunks_key(prefix), SESSION_TTL)
    pipe.execute()


# ── Prior retrieval gap ───────────────────────────────────────────────────────

def set_last_gap(prefix: str, gap: str) -> None:
    """Store the last retrieval gap so the planner can address it next turn."""
    r   = _get_client()
    key = _meta_key(prefix)
    pipe = r.pipeline()
    pipe.hset(key, "last_gap", gap)
    pipe.expire(key, SESSION_TTL)
    pipe.execute()


def get_last_gap(prefix: str) -> str:
    return _get_client().hget(_meta_key(prefix), "last_gap") or ""


# ── Tenant session index ──────────────────────────────────────────────────────

def _register_session(tenant_id: str, session_id: str) -> None:
    """ZADD is idempotent — updates score if member already exists."""
    r   = _get_client()
    key = _sessions_key(tenant_id)
    pipe = r.pipeline()
    pipe.zadd(key, {session_id: time.time()})
    pipe.expire(key, TENANT_TTL)
    pipe.execute()


def get_active_sessions(tenant_id: str, since_seconds: int = 3600) -> list[str]:
    """O(log N) range query on the sorted set score."""
    r     = _get_client()
    since = time.time() - since_seconds
    return r.zrangebyscore(_sessions_key(tenant_id), since, "+inf")
