"""
Inspect every PostgreSQL table used by the RAG system.

Shows:
  - Row counts and date ranges for each table
  - Recent rows with formatted output
  - JSONB columns pretty-printed
  - Idempotency key patterns (verify no duplicates)

Usage:
    uv run python scripts/inspect_postgres.py                  # all tables
    uv run python scripts/inspect_postgres.py sessions         # one table
    uv run python scripts/inspect_postgres.py episodic 20      # N most-recent rows
"""

import json
import os
import sys
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


# ── Connection ────────────────────────────────────────────────────────────────

def connect():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres123"),
        dbname=os.getenv("POSTGRES_DB", "postgres"),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def scalar(conn, sql: str, params=None):
    """Run a query that returns a single scalar value."""
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        row = cur.fetchone()
        return row[0] if row else None


def scalars(conn, sql: str, params=None):
    """Run a query that returns a single row of scalars as a tuple."""
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchone()


def fetchall_plain(conn, sql: str, params=None):
    """Return rows as plain tuples."""
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


def fetchall_dict(conn, sql: str, params=None):
    """Return rows as dicts (RealDictCursor)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params or ())
        return [dict(r) for r in cur.fetchall()]


def fmt_val(v) -> str:
    if v is None:
        return "(null)"
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S UTC")
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, indent=2)
    s = str(v)
    return s[:200] + ("..." if len(s) > 200 else "")


def divider(label: str = "") -> None:
    width = 80
    if label:
        pad = (width - len(label) - 2) // 2
        right = width - pad - len(label) - 2
        print("\n" + "-" * pad + " " + label + " " + "-" * right)
    else:
        print("-" * 80)


def print_rows(rows: list[dict], json_cols: set = None) -> None:
    json_cols = json_cols or set()
    if not rows:
        print("  (no rows)")
        return
    for i, row in enumerate(rows, 1):
        print(f"\n  --- Row {i} ---")
        for k, v in row.items():
            if k in json_cols:
                parsed = v
                if isinstance(v, str):
                    try:
                        parsed = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        pass
                pretty = json.dumps(parsed, indent=4) if isinstance(parsed, (dict, list)) else str(parsed)
                lines = pretty.splitlines()
                if len(lines) <= 1:
                    print(f"    {k:25s}: {pretty}")
                else:
                    print(f"    {k:25s}:")
                    for line in lines[:12]:
                        print(f"      {line}")
                    if len(lines) > 12:
                        print(f"      ... ({len(lines) - 12} more lines)")
            else:
                print(f"    {k:25s}: {fmt_val(v)}")


# ── Table inspectors ──────────────────────────────────────────────────────────

def inspect_sessions(conn, limit: int) -> None:
    divider("sessions -- session registry")
    print("  Purpose : One row per session. Tracks query count + token usage.")
    print("  Written : on every /query or /agent call (upsert).")
    print()

    row = scalars(conn, "SELECT COUNT(*), MIN(started_at), MAX(last_active_at) FROM sessions")
    count, oldest, newest = row if row else (0, None, None)
    print(f"  Total rows     : {count}")
    print(f"  Oldest session : {fmt_val(oldest)}")
    print(f"  Newest activity: {fmt_val(newest)}")

    if count == 0:
        print("\n  [!] No sessions found -- has the API been called yet?")
        return

    rows = fetchall_dict(conn, """
        SELECT tenant_id, user_id, session_id, query_count, total_tokens,
               started_at, last_active_at
        FROM   sessions
        ORDER  BY last_active_at DESC
        LIMIT  %s
    """, (limit,))
    print(f"\n  Most recent {min(limit, len(rows))} row(s):")
    print_rows(rows)


def inspect_episodic(conn, limit: int) -> None:
    divider("episodic_memory -- conversation history")
    print("  Purpose : Every conversation turn (user + assistant), permanent.")
    print("  Written : After generator_node produces an answer.")
    print("  Key col : idempotency_key ensures no duplicate turns on retry.")
    print()

    count = scalar(conn, "SELECT COUNT(*) FROM episodic_memory")
    u     = scalar(conn, "SELECT COUNT(*) FROM episodic_memory WHERE role = 'user'")
    a     = scalar(conn, "SELECT COUNT(*) FROM episodic_memory WHERE role = 'assistant'")
    s     = scalar(conn, "SELECT COUNT(DISTINCT session_id) FROM episodic_memory")
    print(f"  Total rows        : {count}")
    print(f"  User turns        : {u}")
    print(f"  Assistant turns   : {a}")
    print(f"  Distinct sessions : {s}")

    dups = fetchall_plain(conn, """
        SELECT idempotency_key, COUNT(*) AS n
        FROM   episodic_memory
        WHERE  idempotency_key IS NOT NULL
        GROUP  BY idempotency_key
        HAVING COUNT(*) > 1
        LIMIT  5
    """)
    if dups:
        print(f"\n  [!] DUPLICATE idempotency keys found (idempotency broken!):")
        for d in dups:
            print(f"      {d[0]}  x{d[1]}")
    else:
        print("  [ok] No duplicate idempotency keys")

    if count == 0:
        print("\n  [!] No episodic turns found -- has the API been called yet?")
        return

    rows = fetchall_dict(conn, """
        SELECT role, content, sources, tokens, confidence,
               cache_hit, cache_type, session_id, trajectory_id, created_at
        FROM   episodic_memory
        ORDER  BY created_at DESC
        LIMIT  %s
    """, (limit,))
    print(f"\n  Most recent {min(limit, len(rows))} turn(s):")
    print_rows(rows, json_cols={"sources"})


def inspect_semantic(conn, limit: int) -> None:
    divider("semantic_memory -- extracted facts / topics")
    print("  Purpose : Accumulated knowledge about what users ask about.")
    print("  Written : By the planner/generator when facts/topics are extracted.")
    print("  fact_type values: 'topic' | 'preference' | 'entity' | 'fact'")
    print()

    count = scalar(conn, "SELECT COUNT(*) FROM semantic_memory")
    print(f"  Total rows : {count}")

    by_type = fetchall_plain(conn, """
        SELECT fact_type, COUNT(*) AS n
        FROM   semantic_memory
        GROUP  BY fact_type
        ORDER  BY n DESC
    """)
    if by_type:
        print("  Breakdown by fact_type:")
        for row in by_type:
            print(f"    {str(row[0]):15s}: {row[1]}")
    else:
        print("  [!] No semantic facts stored yet.")
        return

    rows = fetchall_dict(conn, """
        SELECT fact_type, subject, content, confidence, source_session, created_at
        FROM   semantic_memory
        ORDER  BY created_at DESC
        LIMIT  %s
    """, (limit,))
    print(f"\n  Most recent {min(limit, len(rows))} fact(s):")
    print_rows(rows)


def inspect_query_logs(conn, limit: int) -> None:
    divider("query_logs -- full pipeline audit trail")
    print("  Purpose : One row per query -- question, plan, answer, latency, tokens.")
    print("  Written : At the end of every /query or /agent request.")
    print("  Key col : idempotency_key prevents duplicate log entries on retry.")
    print()

    count = scalar(conn, "SELECT COUNT(*) FROM query_logs")
    print(f"  Total rows : {count}")

    stats = scalars(conn, """
        SELECT AVG(duration_ms), MIN(duration_ms), MAX(duration_ms),
               AVG(confidence),  SUM(total_tokens)
        FROM   query_logs
        WHERE  duration_ms IS NOT NULL
    """)
    if stats and stats[0] is not None:
        avg_ms, min_ms, max_ms, avg_conf, total_tok = stats
        print(f"  Latency avg/min/max : {avg_ms:.0f} / {min_ms:.0f} / {max_ms:.0f} ms")
        if avg_conf is not None:
            print(f"  Avg confidence      : {avg_conf:.3f}")
        print(f"  Total tokens used   : {total_tok or 0:,}")

    cache_hits = scalar(conn, "SELECT COUNT(*) FROM query_logs WHERE cache_hit = TRUE")
    print(f"  Cache hits          : {cache_hits} / {count}")

    dups = fetchall_plain(conn, """
        SELECT idempotency_key, COUNT(*) AS n
        FROM   query_logs
        WHERE  idempotency_key IS NOT NULL
        GROUP  BY idempotency_key
        HAVING COUNT(*) > 1
        LIMIT  5
    """)
    if dups:
        print(f"\n  [!] DUPLICATE idempotency keys found:")
        for d in dups:
            print(f"      {d[0]}  x{d[1]}")
    else:
        print("  [ok] No duplicate idempotency keys")

    if count == 0:
        print("\n  [!] No query logs found -- has the API been called yet?")
        return

    rows = fetchall_dict(conn, """
        SELECT query, answer, plan, confidence, duration_ms,
               cache_hit, cache_type, iteration_count,
               token_usage, total_tokens, session_id, trajectory_id, created_at
        FROM   query_logs
        ORDER  BY created_at DESC
        LIMIT  %s
    """, (limit,))
    print(f"\n  Most recent {min(limit, len(rows))} query log(s):")
    print_rows(rows, json_cols={"plan", "token_usage"})


# ── Main ──────────────────────────────────────────────────────────────────────

TABLE_MAP = {
    "sessions":  inspect_sessions,
    "episodic":  inspect_episodic,
    "semantic":  inspect_semantic,
    "querylogs": inspect_query_logs,
    "query":     inspect_query_logs,
    "logs":      inspect_query_logs,
}

VALID_TABLES = ["sessions", "episodic", "semantic", "querylogs"]


def main() -> None:
    args   = sys.argv[1:]
    target = args[0].lower() if args else "all"
    limit  = int(args[1]) if len(args) > 1 else 5

    print("=" * 80)
    print("  Agentic AI RAG -- PostgreSQL Inspection")
    print(f"  Host : {os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', 5432)}")
    print(f"  DB   : {os.getenv('POSTGRES_DB', 'postgres')}")
    print("=" * 80)

    try:
        conn = connect()
    except Exception as e:
        print(f"\n  [!] Could not connect to PostgreSQL: {e}")
        print("  Check POSTGRES_HOST/PORT/USER/PASSWORD/DB in .env")
        sys.exit(1)

    print("  [ok] Connected\n")

    try:
        if target == "all":
            inspect_sessions(conn,   limit)
            inspect_episodic(conn,   limit)
            inspect_semantic(conn,   limit)
            inspect_query_logs(conn, limit)
        elif target in TABLE_MAP:
            TABLE_MAP[target](conn, limit)
        else:
            print(f"  Unknown table '{target}'. Choose from: all, {', '.join(VALID_TABLES)}")
            sys.exit(1)
    finally:
        conn.close()

    divider()
    print("  Done.")


if __name__ == "__main__":
    main()
