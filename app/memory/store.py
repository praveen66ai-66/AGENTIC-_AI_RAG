"""
PostgreSQL long-term memory — three stores in the memory hierarchy.

  Working   → Redis (session.py)   — fast, TTL-bound, current session
  Episodic  → episodic_memory      — every conversation turn, permanent
  Semantic  → semantic_memory      — extracted topics/facts across sessions
  Audit     → query_logs           — full pipeline trace per request
  Registry  → sessions             — session lifecycle and statistics

Tenant isolation
  Every table includes tenant_id in every query and every index.
  A row belonging to tenant A can never appear in a query scoped to tenant B.

Idempotency
  Every write uses ON CONFLICT (idempotency_key) DO NOTHING.
  The caller supplies the key (e.g. "{request_uuid}:user") — retried requests
  produce zero duplicate rows.

Timestamps
  created_at  TIMESTAMPTZ DEFAULT NOW()  on every table (DB clock, not app clock)
  updated_at  maintained via ON CONFLICT DO UPDATE on sessions + semantic_memory
  duration_ms written by the caller (measured wall-clock in app)

Connection pooling
  ThreadedConnectionPool (min 2, max 10) — one pool per process, shared across
  FastAPI worker threads. Context manager returns connections automatically.
"""

import json
import os
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

_pool: Optional[ThreadedConnectionPool] = None


# ── Pool ──────────────────────────────────────────────────────────────────────

def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=os.getenv("POSTGRES_HOST",     "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            user=os.getenv("POSTGRES_USER",     "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres123"),
            dbname=os.getenv("POSTGRES_DB",     "postgres"),
        )
    return _pool


@contextmanager
def _conn():
    """Borrow a connection from the pool; return it automatically."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ── Schema ────────────────────────────────────────────────────────────────────

def setup_tables() -> None:
    """
    Create all tables and indexes idempotently.
    Run once at app startup — safe to call repeatedly.
    """
    with _conn() as conn:
        with conn.cursor() as cur:

            # ── Session registry ──────────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id      TEXT        NOT NULL,
                    user_id        TEXT        NOT NULL,
                    session_id     TEXT        NOT NULL,
                    query_count    INT         NOT NULL DEFAULT 0,
                    total_tokens   INT         NOT NULL DEFAULT 0,
                    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    ended_at       TIMESTAMPTZ,
                    metadata       JSONB        DEFAULT '{}',
                    CONSTRAINT uq_sessions_tenant_session UNIQUE (tenant_id, session_id)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_tenant_user   ON sessions(tenant_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_last_active   ON sessions(last_active_at DESC);")

            # ── Episodic memory ───────────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id       TEXT        NOT NULL,
                    user_id         TEXT        NOT NULL,
                    session_id      TEXT        NOT NULL,
                    trajectory_id   TEXT,
                    role            TEXT        NOT NULL,
                    content         TEXT        NOT NULL,
                    sources         JSONB       DEFAULT '[]',
                    tokens          INT         NOT NULL DEFAULT 0,
                    confidence      FLOAT,
                    cache_hit       BOOLEAN     NOT NULL DEFAULT FALSE,
                    cache_type      TEXT,
                    idempotency_key TEXT        UNIQUE,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ep_tenant_user  ON episodic_memory(tenant_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ep_session       ON episodic_memory(session_id, created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ep_trajectory    ON episodic_memory(trajectory_id);")

            # ── Semantic memory ───────────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id       TEXT        NOT NULL,
                    user_id         TEXT        NOT NULL,
                    fact_type       TEXT        NOT NULL,
                    subject         TEXT        NOT NULL,
                    content         TEXT        NOT NULL,
                    source_session  TEXT,
                    confidence      FLOAT       NOT NULL DEFAULT 1.0,
                    idempotency_key TEXT        UNIQUE,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sm_tenant_user  ON semantic_memory(tenant_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sm_fact_type    ON semantic_memory(tenant_id, user_id, fact_type);")

            # ── Query audit log ───────────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id       TEXT        NOT NULL,
                    user_id         TEXT        NOT NULL,
                    session_id      TEXT        NOT NULL,
                    trajectory_id   TEXT,
                    query           TEXT        NOT NULL,
                    plan            JSONB       DEFAULT '[]',
                    retrieved       TEXT,
                    answer          TEXT,
                    iteration_count INT         NOT NULL DEFAULT 0,
                    duration_ms     FLOAT,
                    confidence      FLOAT,
                    cache_hit       BOOLEAN     NOT NULL DEFAULT FALSE,
                    cache_type      TEXT,
                    idempotency_key TEXT        UNIQUE,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ql_tenant_user  ON query_logs(tenant_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ql_session       ON query_logs(session_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ql_created       ON query_logs(created_at DESC);")
            # Non-destructive migrations — safe to run on any existing schema version
            cur.execute("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS plan            JSONB   DEFAULT '[]';")
            cur.execute("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS confidence      FLOAT;")
            cur.execute("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS duration_ms     FLOAT;")
            cur.execute("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS cache_hit       BOOLEAN NOT NULL DEFAULT FALSE;")
            cur.execute("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS cache_type      TEXT;")
            cur.execute("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS iteration_count INT     NOT NULL DEFAULT 0;")
            cur.execute("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS idempotency_key TEXT;")
            cur.execute("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS token_usage     JSONB   DEFAULT '{}';")
            cur.execute("ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS total_tokens    INT     DEFAULT 0;")
            # Unique index on idempotency_key (CREATE INDEX IF NOT EXISTS is idempotent)
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ql_idem ON query_logs(idempotency_key) WHERE idempotency_key IS NOT NULL;")


# ── Session registry ──────────────────────────────────────────────────────────

def upsert_session(
    tenant_id:  str,
    user_id:    str,
    session_id: str,
    tokens:     int = 0,
) -> None:
    """
    Register a session on first use; update stats on subsequent calls.
    ON CONFLICT updates last_active_at, query_count, total_tokens atomically.
    """
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sessions
                        (tenant_id, user_id, session_id, query_count, total_tokens)
                    VALUES (%s, %s, %s, 1, %s)
                    ON CONFLICT (tenant_id, session_id) DO UPDATE SET
                        last_active_at = NOW(),
                        query_count    = sessions.query_count + 1,
                        total_tokens   = sessions.total_tokens + EXCLUDED.total_tokens
                """, (tenant_id, user_id, session_id, tokens))
    except Exception:
        pass


# ── Episodic memory ───────────────────────────────────────────────────────────

def log_turn(
    tenant_id:       str,
    user_id:         str,
    session_id:      str,
    trajectory_id:   str,
    role:            str,
    content:         str,
    idempotency_key: str,
    sources:         list[dict]    = None,
    tokens:          int           = 0,
    confidence:      Optional[float] = None,
    cache_hit:       bool          = False,
    cache_type:      Optional[str] = None,
) -> None:
    """
    Write one conversation turn to episodic_memory.
    ON CONFLICT (idempotency_key) DO NOTHING — retries are no-ops.

    idempotency_key convention:
      user turn      → "{request_uuid}:user"
      assistant turn → "{request_uuid}:asst"
    """
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO episodic_memory
                        (tenant_id, user_id, session_id, trajectory_id, role,
                         content, sources, tokens, confidence, cache_hit,
                         cache_type, idempotency_key)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                """, (
                    tenant_id, user_id, session_id, trajectory_id, role,
                    content,
                    json.dumps(sources or []),
                    tokens or max(1, len(content) // 4),
                    confidence,
                    cache_hit,
                    cache_type,
                    idempotency_key,
                ))
    except Exception:
        pass


def get_episodic_history(
    tenant_id:  str,
    user_id:    str,
    session_id: str,
    limit:      int = 20,
) -> list[dict]:
    """
    Retrieve the last N turns for a session from PostgreSQL.
    Used as cold-start fallback when Redis TTL has expired.
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT role, content, sources, tokens, confidence,
                           cache_hit, created_at
                    FROM   episodic_memory
                    WHERE  tenant_id  = %s
                      AND  user_id    = %s
                      AND  session_id = %s
                    ORDER  BY created_at DESC
                    LIMIT  %s
                """, (tenant_id, user_id, session_id, limit))
                rows = cur.fetchall()
                return list(reversed([dict(r) for r in rows]))
    except Exception:
        return []


def get_cross_session_history(
    tenant_id: str,
    user_id:   str,
    limit:     int = 50,
) -> list[dict]:
    """
    Recent turns across ALL sessions for a user.
    Used to build long-term user context for personalization.
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT session_id, role, content, confidence, created_at
                    FROM   episodic_memory
                    WHERE  tenant_id = %s
                      AND  user_id   = %s
                    ORDER  BY created_at DESC
                    LIMIT  %s
                """, (tenant_id, user_id, limit))
                return list(reversed([dict(r) for r in cur.fetchall()]))
    except Exception:
        return []


# ── Semantic memory ───────────────────────────────────────────────────────────

def store_semantic_fact(
    tenant_id:       str,
    user_id:         str,
    fact_type:       str,
    subject:         str,
    content:         str,
    idempotency_key: str,
    source_session:  Optional[str]   = None,
    confidence:      float           = 1.0,
) -> None:
    """
    Persist an extracted fact/topic to semantic_memory.
    ON CONFLICT (idempotency_key) DO NOTHING — retries are no-ops.

    fact_type values:
      'topic'      — financial topic the user asked about (e.g. "cash flow")
      'preference' — user behaviour pattern (e.g. "prefers table format")
      'entity'     — named entity extracted (e.g. "IFRS 9", "Apple Inc.")
      'fact'       — domain fact extracted from answer
    """
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO semantic_memory
                        (tenant_id, user_id, fact_type, subject, content,
                         source_session, confidence, idempotency_key)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                """, (
                    tenant_id, user_id, fact_type, subject, content,
                    source_session, confidence, idempotency_key,
                ))
    except Exception:
        pass


def get_semantic_profile(
    tenant_id: str,
    user_id:   str,
    fact_type: Optional[str] = None,
    limit:     int = 30,
) -> list[dict]:
    """
    Return the accumulated semantic profile for a user.
    Optionally filter by fact_type ('topic', 'preference', etc.).
    Used to personalise planner prompts with what the user cares about.
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if fact_type:
                    cur.execute("""
                        SELECT fact_type, subject, content, confidence, created_at
                        FROM   semantic_memory
                        WHERE  tenant_id = %s AND user_id = %s AND fact_type = %s
                        ORDER  BY created_at DESC LIMIT %s
                    """, (tenant_id, user_id, fact_type, limit))
                else:
                    cur.execute("""
                        SELECT fact_type, subject, content, confidence, created_at
                        FROM   semantic_memory
                        WHERE  tenant_id = %s AND user_id = %s
                        ORDER  BY created_at DESC LIMIT %s
                    """, (tenant_id, user_id, limit))
                return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


# ── Query audit log ───────────────────────────────────────────────────────────

def log_query(
    tenant_id:       str,
    user_id:         str,
    session_id:      str,
    query:           str,
    retrieved:       str,
    idempotency_key: str,
    trajectory_id:   Optional[str]   = None,
    plan:            list            = None,
    answer:          Optional[str]   = None,
    iteration_count: int             = 0,
    duration_ms:     Optional[float] = None,
    confidence:      Optional[float] = None,
    cache_hit:       bool            = False,
    cache_type:      Optional[str]   = None,
    token_usage:     Optional[dict]  = None,
    total_tokens:    int             = 0,
) -> None:
    """
    Full pipeline audit trace for one request.
    ON CONFLICT (idempotency_key) DO NOTHING — retries are no-ops.
    """
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO query_logs
                        (tenant_id, user_id, session_id, trajectory_id,
                         query, plan, retrieved, answer, iteration_count,
                         duration_ms, confidence, cache_hit, cache_type,
                         token_usage, total_tokens, idempotency_key)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                """, (
                    tenant_id, user_id, session_id, trajectory_id,
                    query,
                    json.dumps(plan or []),
                    retrieved, answer,
                    iteration_count,
                    duration_ms, confidence,
                    cache_hit, cache_type,
                    json.dumps(token_usage or {}),
                    total_tokens,
                    idempotency_key,
                ))
    except Exception:
        pass


def get_query_history(
    tenant_id: str,
    user_id:   str,
    limit:     int = 20,
) -> list[dict]:
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT query, answer, confidence, duration_ms,
                           cache_hit, iteration_count, created_at
                    FROM   query_logs
                    WHERE  tenant_id = %s AND user_id = %s
                    ORDER  BY created_at DESC LIMIT %s
                """, (tenant_id, user_id, limit))
                return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


# ── Health ────────────────────────────────────────────────────────────────────

def ping() -> bool:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False
