"""
Token usage + stage duration inspector.

Reads the last N queries from PostgreSQL query_logs and prints:
  - Per-query: total tokens, duration, confidence, cache hit
  - Per-stage breakdown: planner / reasoner / generator tokens
  - Groq context window usage % per query
  - Session totals

Usage:
    uv run python scripts/check_tokens.py
    uv run python scripts/check_tokens.py --limit 20 --tenant default --user default
"""

import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Groq context window for the configured model (tokens in + out per single call)
# Override via GROQ_CONTEXT_WINDOW env var if your model differs.
CONTEXT_WINDOW = int(os.getenv("GROQ_CONTEXT_WINDOW", "32768"))

# Max output tokens set in get_llm()
MAX_OUTPUT = 4096


def connect():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST",     "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER",     "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        dbname=os.getenv("POSTGRES_DB",     "postgres"),
    )


def fetch_queries(tenant_id, user_id, limit):
    with connect() as conn:
        with conn.cursor() as cur:
            # Migrate schema — safe to run on any existing table version
            migrations = [
                "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS plan            JSONB   DEFAULT '[]'",
                "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS confidence      FLOAT",
                "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS duration_ms     FLOAT",
                "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS cache_hit       BOOLEAN NOT NULL DEFAULT FALSE",
                "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS cache_type      TEXT",
                "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS iteration_count INT     NOT NULL DEFAULT 0",
                "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS idempotency_key TEXT",
                "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS token_usage     JSONB   DEFAULT '{}'",
                "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS total_tokens    INT     DEFAULT 0",
                "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS trajectory_id   TEXT",
            ]
            for m in migrations:
                cur.execute(m)
            conn.commit()

            cur.execute("""
                SELECT query, confidence, duration_ms,
                       cache_hit, iteration_count, token_usage, total_tokens,
                       created_at
                FROM   query_logs
                WHERE  tenant_id = %s AND user_id = %s
                ORDER  BY created_at DESC
                LIMIT  %s
            """, (tenant_id, user_id, limit))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def bar(used, limit, width=20):
    filled = int(used / limit * width) if limit else 0
    pct    = round(used / limit * 100, 1) if limit else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct}%"


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=10)
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user",   default="default")
    args = parser.parse_args()

    try:
        rows = fetch_queries(args.tenant, args.user, args.limit)
    except Exception as e:
        print(f"PostgreSQL error: {e}")
        print("Make sure the API has been started at least once (schema auto-creates on startup).")
        return

    if not rows:
        print("No queries found. Ask a question first, then run this script.")
        return

    section(f"TOKEN USAGE — last {len(rows)} queries  "
            f"(tenant={args.tenant}, user={args.user})")
    print(f"  Model context window : {CONTEXT_WINDOW:,} tokens")
    print(f"  Max output tokens    : {MAX_OUTPUT:,} tokens")

    session_total_tokens = 0
    session_total_ms     = 0

    for i, row in enumerate(rows, 1):
        created  = row["created_at"]
        if hasattr(created, "strftime"):
            created = created.strftime("%Y-%m-%d %H:%M:%S")

        total_tok = row.get("total_tokens") or 0
        duration  = row.get("duration_ms") or 0
        conf      = row.get("confidence") or 0
        iters     = row.get("iteration_count") or 0
        cache     = "⚡ cache" if row.get("cache_hit") else ""
        query_preview = (row.get("query") or "")[:70]

        session_total_tokens += total_tok
        session_total_ms     += duration

        print(f"\n  [{i}] {created}  {cache}")
        print(f"       Q: {query_preview}")
        print(f"       confidence={conf:.2f}  iterations={iters}  duration={duration:.0f}ms")

        # Per-stage breakdown
        token_usage = row.get("token_usage") or {}
        if isinstance(token_usage, str):
            try:
                token_usage = json.loads(token_usage)
            except Exception:
                token_usage = {}

        if token_usage:
            print(f"       ── Stage tokens ──────────────────────────────────")
            stages = ["planner", "reasoner", "reasoner_2", "generator"]
            for stage in stages:
                if stage not in token_usage:
                    continue
                t     = token_usage[stage]
                t_in  = t.get("in", 0)
                t_out = t.get("out", 0)
                t_tot = t.get("total", 0)
                ctx_pct = f"{t_in/CONTEXT_WINDOW*100:.1f}% of ctx window" if t_in else ""
                print(f"       {stage:<12} in={t_in:>5}  out={t_out:>5}  total={t_tot:>5}  {ctx_pct}")

            print(f"       ── Total: {total_tok:,} tokens  "
                  f"{bar(total_tok, CONTEXT_WINDOW*3)}")
        else:
            print(f"       (no token data — run a query after this update)")

    section("SESSION TOTALS")
    print(f"  Queries run     : {len(rows)}")
    print(f"  Total tokens    : {session_total_tokens:,}")
    print(f"  Avg tokens/query: {session_total_tokens // len(rows):,}")
    print(f"  Total duration  : {session_total_ms/1000:.1f}s")
    print(f"  Avg duration    : {session_total_ms/len(rows):.0f}ms")

    section("SPLUNK QUERIES — per-stage detail")
    print("""
  Per-stage duration:
    index=agent_trajectory sourcetype="rag:trajectory" event.phase="exit"
    | table event.node, event.duration_ms, event.tokens_in, event.tokens_out, event.tokens_total
    | sort event.node

  Token usage over time:
    index=agent_trajectory sourcetype="rag:trajectory" event.node="generator" event.phase="exit"
    | timechart avg(event.tokens_total) by event.node

  Queries near context limit (tokens_in > 20k):
    index=agent_trajectory sourcetype="rag:trajectory"
    | where event.tokens_in > 20000
    | table _time, event.trajectory_id, event.node, event.tokens_in
""")


if __name__ == "__main__":
    main()
