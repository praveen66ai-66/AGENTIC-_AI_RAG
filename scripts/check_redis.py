"""
Redis data inspector — shows all live keys with clear labels.

Usage:
    uv run python scripts/check_redis.py
    uv run python scripts/check_redis.py --tenant default --user default
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta

UAE_TZ = timezone(timedelta(hours=4))

from dotenv import load_dotenv

load_dotenv()

import redis

r = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True,
    protocol=2,
)


def ts(epoch) -> str:
    """Convert epoch float to UAE time (UTC+4)."""
    try:
        return datetime.fromtimestamp(float(epoch), tz=UAE_TZ).strftime("%Y-%m-%d %H:%M:%S GST")
    except Exception:
        return str(epoch)


def ttl_str(key: str) -> str:
    t = r.ttl(key)
    if t < 0:
        return "no TTL"
    m, s = divmod(t, 60)
    return f"expires in {m}m {s}s"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_session_index(tenant_id: str) -> list[str]:
    """
    KEY: {tenant}:sessions  (ZSET)
    WHAT: All active sessions for this tenant, scored by last_active epoch.
    """
    key = f"{tenant_id}:sessions"
    section(f"ACTIVE SESSIONS  [{key}]  (ZSET)")

    members = r.zrangebyscore(key, "-inf", "+inf", withscores=True)
    if not members:
        print("  (empty)")
        return []

    session_ids = []
    for session_id, score in members:
        print(f"  session_id : {session_id}")
        print(f"  last_active: {ts(score)}")
        print(f"  {ttl_str(key)}")
        print()
        session_ids.append(session_id)
    return session_ids


def check_meta(prefix: str) -> None:
    """
    KEY: {tenant}:{user}:{session}:meta  (HASH)
    WHAT: Session stats — query count, token usage, timestamps, last retrieval gap.
    """
    key = f"{prefix}:meta"
    section(f"SESSION META  [{key}]  (HASH)")

    data = r.hgetall(key)
    if not data:
        print("  (empty or expired)")
        return

    print(f"  tenant_id   : {data.get('tenant_id', '-')}")
    print(f"  user_id     : {data.get('user_id', '-')}")
    print(f"  session_id  : {data.get('session_id', '-')}")
    print(f"  query_count : {data.get('query_count', 0)}  ← how many questions asked this session")
    print(f"  total_tokens: {data.get('total_tokens', 0)}  ← estimated tokens used")
    print(f"  created_at  : {ts(data.get('created_at', 0))}")
    print(f"  last_active : {ts(data.get('last_active', 0))}")
    gap = data.get("last_gap", "")
    print(f"  last_gap    : {gap or '(none)'}  ← what the last retrieval could not find")
    print(f"  {ttl_str(key)}")


def check_history(prefix: str) -> None:
    """
    KEY: {tenant}:{user}:{session}:history  (LIST)
    WHAT: Conversation turns — user questions and assistant answers (last 20).
    """
    key = f"{prefix}:history"
    section(f"CONVERSATION HISTORY  [{key}]  (LIST)")

    items = r.lrange(key, 0, -1)
    if not items:
        print("  (empty or expired)")
        return

    print(f"  {len(items)} turn(s) stored  {ttl_str(key)}\n")
    for i, raw in enumerate(items, 1):
        try:
            msg = json.loads(raw)
        except Exception:
            msg = {"role": "?", "content": raw}
        role    = msg.get("role", "?").upper()
        content = msg.get("content", "")
        preview = content[:120] + ("..." if len(content) > 120 else "")
        print(f"  [{i}] {role} ({ts(msg.get('ts', 0))})")
        print(f"       {preview}")
        print()


def check_chunks(prefix: str) -> None:
    """
    KEY: {tenant}:{user}:{session}:chunks  (SET)
    WHAT: MD5 hashes of chunks already shown this session — prevents repetition.
    """
    key = f"{prefix}:chunks"
    section(f"SEEN CHUNKS DEDUP  [{key}]  (SET)")

    count = r.scard(key)
    if count == 0:
        print("  (empty or expired)")
        return

    print(f"  {count} unique chunk(s) seen this session  ← retriever won't show these again")
    print(f"  {ttl_str(key)}")


def check_semantic_cache(tenant_id: str, user_id: str) -> None:
    """
    KEY: {tenant}:{user}:scache  (HASH)
    WHAT: Cached answers per user (24h TTL). Skips the full agent pipeline on repeat questions.
    """
    key = f"{tenant_id}:{user_id}:scache"
    section(f"SEMANTIC ANSWER CACHE  [{key}]  (HASH)")

    fields = r.hkeys(key)
    if not fields:
        print("  (empty — no questions cached yet)")
        return

    print(f"  {len(fields)} cached answer(s)  {ttl_str(key)}\n")
    for i, field in enumerate(fields, 1):
        raw = r.hget(key, field)
        try:
            entry = json.loads(raw)
        except Exception:
            entry = {}
        q       = entry.get("question", "?")[:80]
        conf    = entry.get("confidence", 0)
        cached  = ts(entry.get("ts", 0))
        print(f"  [{i}] Q: {q}")
        print(f"       confidence={conf}  cached_at={cached}")
        print()


def check_idempotency_keys(prefix: str) -> None:
    """
    KEY: {tenant}:{user}:{session}:idem:*  (STRING, TTL=300s)
    WHAT: Short-lived guards — if a client retries the same request within 5min,
          the stored result is returned without re-running the pipeline.
    """
    pattern = f"{prefix}:idem:*"
    keys    = list(r.scan_iter(pattern))
    section(f"IDEMPOTENCY KEYS  [{pattern}]  (STRING, 5min TTL)")

    if not keys:
        print("  (none active — all expired or no retries)")
        return

    print(f"  {len(keys)} active guard(s)\n")
    for k in keys:
        print(f"  {k}  ({ttl_str(k)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user",   default="default")
    args = parser.parse_args()

    tenant_id = args.tenant
    user_id   = args.user

    print(f"\nRedis inspection — tenant={tenant_id}  user={user_id}")
    print(f"Connected to: {os.getenv('REDIS_URL', 'redis://localhost:6379')}\n")

    # 1. Show all active sessions for this tenant
    session_ids = check_session_index(tenant_id)

    # 2. For each session, show meta + history + chunks + idempotency
    if session_ids:
        for session_id in session_ids:
            prefix = f"{tenant_id}:{user_id}:{session_id}"
            check_meta(prefix)
            check_history(prefix)
            check_chunks(prefix)
            check_idempotency_keys(prefix)
    else:
        print("\nNo active sessions found. Start the app and ask a question first.")

    # 3. Semantic cache (per user, not per session)
    check_semantic_cache(tenant_id, user_id)

    print("\n" + "="*60)
    print("  SUMMARY OF ALL RAG KEYS")
    print("="*60)
    all_keys = list(r.scan_iter(f"{tenant_id}:*"))
    if not all_keys:
        print("  (no keys found for this tenant)")
    else:
        for k in sorted(all_keys):
            ktype = r.type(k)
            kttl  = r.ttl(k)
            print(f"  {ktype:<6}  TTL={kttl:>6}s  {k}")

    print()


if __name__ == "__main__":
    main()
