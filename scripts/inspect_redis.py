"""
Inspect every Redis key used by the RAG system.

Key layout (from app/memory/session.py + semantic_cache.py):

  {tenant}:{user}:{session}:history          LIST   -- conversation turns (rolling 20)
  {tenant}:{user}:{session}:meta             HASH   -- session stats
  {tenant}:{user}:{session}:chunks           SET    -- chunk MD5s seen (dedup)
  {tenant}:{user}:{session}:msgdedup:{id}    STRING -- message dedup guard (SETNX)
  {tenant}:{user}:{session}:qguard:{req_id}  STRING -- query-count guard (SETNX)
  {tenant}:sessions                          ZSET   -- active sessions by last_active epoch
  {tenant}:{user}:scache                     HASH   -- semantic answer cache (per user)

Usage:
    uv run python scripts/inspect_redis.py                   # all sections
    uv run python scripts/inspect_redis.py sessions          # session index only
    uv run python scripts/inspect_redis.py history           # conversation history
    uv run python scripts/inspect_redis.py meta              # session metadata
    uv run python scripts/inspect_redis.py chunks            # chunk dedup sets
    uv run python scripts/inspect_redis.py cache             # semantic answer cache
    uv run python scripts/inspect_redis.py overview          # key counts only
    uv run python scripts/inspect_redis.py history 10        # show up to N sessions
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import redis
from dotenv import load_dotenv

load_dotenv()

SESSION_TTL = 3600   # from session.py
TENANT_TTL  = 86400


# ── Connection ────────────────────────────────────────────────────────────────

def connect() -> redis.Redis:
    url = os.getenv("REDIS_URL")
    if url:
        return redis.Redis.from_url(url, decode_responses=True, protocol=2)
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=True,
        protocol=2,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def scan_keys(r: redis.Redis, pattern: str) -> list[str]:
    """Use SCAN (non-blocking) to find all keys matching pattern."""
    keys = []
    cursor = 0
    while True:
        cursor, batch = r.scan(cursor, match=pattern, count=200)
        keys.extend(batch)
        if cursor == 0:
            break
    return keys


def fmt_ts(ts) -> str:
    if ts is None:
        return "(null)"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError, OSError):
        return str(ts)


def fmt_ttl(seconds: int) -> str:
    if seconds < 0:
        return "no TTL (persistent)"
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def divider(label: str = "") -> None:
    width = 80
    if label:
        pad = (width - len(label) - 2) // 2
        right = width - pad - len(label) - 2
        print("\n" + "-" * pad + " " + label + " " + "-" * right)
    else:
        print("-" * 80)


def safe(s: str) -> str:
    """Encode to the terminal's encoding, replacing unencodable chars with '?'."""
    enc = sys.stdout.encoding or "utf-8"
    return s.encode(enc, errors="replace").decode(enc)


def trunc(s: str, n: int = 120) -> str:
    if s is None:
        return "(null)"
    return safe(s[:n] + ("..." if len(s) > n else ""))


# ── Key-type scanner ──────────────────────────────────────────────────────────

def get_all_keys_by_type(r: redis.Redis) -> dict:
    """Scan for every key pattern and return grouped results."""
    return {
        "sessions_zset":  scan_keys(r, "*:sessions"),
        "history":        scan_keys(r, "*:history"),
        "meta":           scan_keys(r, "*:meta"),
        "chunks":         scan_keys(r, "*:chunks"),
        "msgdedup":       scan_keys(r, "*:msgdedup:*"),
        "qguard":         scan_keys(r, "*:qguard:*"),
        "scache":         scan_keys(r, "*:scache"),
    }


# ── Section printers ──────────────────────────────────────────────────────────

def print_overview(r: redis.Redis) -> None:
    divider("OVERVIEW -- all Redis key types")
    print("  Key layout used by this system:\n")
    print(f"  {'Pattern':<40} {'Count':>6}  {'TTL':>10}  Purpose")
    print(f"  {'-'*40} {'------':>6}  {'-'*10}  -------")

    all_keys = get_all_keys_by_type(r)
    patterns = [
        ("*:sessions",      "sessions_zset", "ZSET",   "Tenant session index (scored by last_active)"),
        ("*:history",       "history",       "LIST",   "Conversation turns (rolling 20)"),
        ("*:meta",          "meta",          "HASH",   "Session stats (query_count, tokens, etc.)"),
        ("*:chunks",        "chunks",        "SET",    "Seen chunk MD5s (dedup within session)"),
        ("*:msgdedup:*",    "msgdedup",      "STRING", "Message dedup guards (SETNX)"),
        ("*:qguard:*",      "qguard",        "STRING", "Query-count guards (SETNX)"),
        ("*:scache",        "scache",        "HASH",   "Semantic answer cache (per user, 24h TTL)"),
    ]

    total = 0
    for pattern, key_type, dtype, purpose in patterns:
        keys  = all_keys[key_type]
        count = len(keys)
        total += count
        # Sample TTL from first key if available
        ttl_str = ""
        if keys:
            ttl = r.ttl(keys[0])
            ttl_str = fmt_ttl(ttl)
        print(f"  {pattern:<40} {count:>6}  {ttl_str:>10}  [{dtype}] {purpose}")

    print(f"\n  Total tracked keys : {total}")
    print(f"  Total DB key count : {r.dbsize()}")


def print_sessions(r: redis.Redis, limit: int) -> None:
    divider("ZSET -- tenant session indexes  (*:sessions)")
    print("  One ZSET per tenant. Members = session IDs, Score = last_active Unix timestamp.")
    print()

    zset_keys = scan_keys(r, "*:sessions")
    if not zset_keys:
        print("  [!] No session index keys found.")
        return

    for zkey in sorted(zset_keys)[:limit]:
        ttl     = r.ttl(zkey)
        members = r.zrangebyscore(zkey, "-inf", "+inf", withscores=True)
        print(f"  Key   : {zkey}")
        print(f"  TTL   : {fmt_ttl(ttl)}")
        print(f"  Count : {len(members)} session(s)")

        # Sort by score descending (most recent first)
        for sid, score in sorted(members, key=lambda x: x[1], reverse=True)[:limit]:
            print(f"    session={sid}   last_active={fmt_ts(score)}")
        print()


def print_history(r: redis.Redis, limit: int) -> None:
    divider("LIST -- conversation history  (*:history)")
    print("  One LIST per session. Each element is a JSON turn {role, content, ts, tokens, sources}.")
    print(f"  Rolling window: last 20 turns. TTL: {SESSION_TTL // 3600}h (reset on each message).")
    print()

    hist_keys = sorted(scan_keys(r, "*:history"))
    if not hist_keys:
        print("  [!] No history keys found -- all sessions may have expired.")
        return

    print(f"  Found {len(hist_keys)} history key(s). Showing up to {limit}:")

    for key in hist_keys[:limit]:
        ttl    = r.ttl(key)
        length = r.llen(key)
        print(f"\n  Key  : {key}")
        print(f"  TTL  : {fmt_ttl(ttl)}")
        print(f"  Turns: {length}")

        turns = r.lrange(key, 0, -1)
        for i, raw in enumerate(turns, 1):
            try:
                t = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                print(f"    [{i}] (unparseable) {raw[:80]}")
                continue
            sources = t.get("sources") or []
            src_str = f"  {len(sources)} source(s)" if sources else ""
            print(f"    [{i}] {t.get('role','?'):9s} | {fmt_ts(t.get('ts'))} | {t.get('tokens',0)} tokens{src_str}")
            print(f"         {trunc(t.get('content',''), 100)}")


def print_meta(r: redis.Redis, limit: int) -> None:
    divider("HASH -- session metadata  (*:meta)")
    print("  One HASH per session. Tracks query_count, total_tokens, timestamps, last_gap.")
    print()

    meta_keys = sorted(scan_keys(r, "*:meta"))
    if not meta_keys:
        print("  [!] No meta keys found.")
        return

    print(f"  Found {len(meta_keys)} meta key(s). Showing up to {limit}:")

    for key in meta_keys[:limit]:
        ttl  = r.ttl(key)
        data = r.hgetall(key)
        print(f"\n  Key : {key}")
        print(f"  TTL : {fmt_ttl(ttl)}")
        if not data:
            print("  (empty hash)")
            continue
        for field, val in sorted(data.items()):
            if field in ("created_at", "last_active"):
                display = f"{val}  ({fmt_ts(val)})"
            else:
                display = val
            print(f"    {field:20s}: {display}")


def print_chunks(r: redis.Redis, limit: int) -> None:
    divider("SET -- chunk dedup  (*:chunks)")
    print("  One SET per session. Contains MD5 hashes of chunk texts seen this session.")
    print("  Prevents the same chunk being returned twice in one conversation.")
    print()

    chunk_keys = sorted(scan_keys(r, "*:chunks"))
    if not chunk_keys:
        print("  [!] No chunk dedup keys found.")
        return

    print(f"  Found {len(chunk_keys)} chunk set(s). Showing up to {limit}:")

    for key in chunk_keys[:limit]:
        ttl   = r.ttl(key)
        count = r.scard(key)
        print(f"\n  Key    : {key}")
        print(f"  TTL    : {fmt_ttl(ttl)}")
        print(f"  Chunks seen : {count} unique MD5s")
        # Show a few sample hashes so you know it's not empty garbage
        sample = r.srandmember(key, 3) or []
        for h in sample:
            print(f"    {h}")


def print_cache(r: redis.Redis, limit: int) -> None:
    divider("HASH -- semantic answer cache  (*:scache)")
    print("  One HASH per user (across all sessions). TTL 24h, max 100 entries.")
    print("  Field = MD5(normalized question). Value = JSON with question, answer, embedding, confidence.")
    print("  Embedding vector is omitted from display (768 floats).")
    print()

    cache_keys = sorted(scan_keys(r, "*:scache"))
    if not cache_keys:
        print("  [!] No semantic cache keys found.")
        return

    print(f"  Found {len(cache_keys)} cache hash(es).")

    for key in cache_keys[:limit]:
        ttl    = r.ttl(key)
        fields = r.hgetall(key)
        print(f"\n  Key     : {key}")
        print(f"  TTL     : {fmt_ttl(ttl)}")
        print(f"  Entries : {len(fields)}")

        # Sort by ts desc
        entries = []
        for field, raw in fields.items():
            try:
                e = json.loads(raw)
                entries.append((field, e))
            except (json.JSONDecodeError, ValueError):
                pass
        entries.sort(key=lambda x: x[1].get("ts", 0), reverse=True)

        print(f"\n  {'#':<3} {'Confidence':>10}  {'Cached at':<22}  Question")
        print(f"  {'-'*3} {'-'*10}  {'-'*22}  {'-'*40}")
        for i, (field, e) in enumerate(entries[:limit], 1):
            conf     = e.get("confidence", 0)
            ts_str   = fmt_ts(e.get("ts"))
            question = trunc(e.get("question", ""), 55)
            emb_len  = len(e.get("embedding") or [])
            print(f"  {i:<3} {conf:>10.3f}  {ts_str:<22}  {question}")
            print(f"      field={field}  embedding={emb_len}d")
            print(f"      answer: {trunc(e.get('answer',''), 90)}")
            print()

        if len(entries) > limit:
            print(f"  ... and {len(entries) - limit} more entries")


def print_dedup_guards(r: redis.Redis) -> None:
    divider("STRING -- idempotency guards  (*:msgdedup:*  /  *:qguard:*)")
    print("  Short-lived SETNX locks. Expire with the session (1h TTL).")
    print("  Non-zero count means retried requests are being deduplicated correctly.")
    print()

    msg_keys = scan_keys(r, "*:msgdedup:*")
    q_keys   = scan_keys(r, "*:qguard:*")
    print(f"  msgdedup guards active : {len(msg_keys)}")
    print(f"  qguard   guards active : {len(q_keys)}")

    if msg_keys:
        print(f"\n  Sample msgdedup keys (up to 5):")
        for k in msg_keys[:5]:
            print(f"    {k}  TTL={fmt_ttl(r.ttl(k))}")

    if q_keys:
        print(f"\n  Sample qguard keys (up to 5):")
        for k in q_keys[:5]:
            print(f"    {k}  TTL={fmt_ttl(r.ttl(k))}")


# ── Main ──────────────────────────────────────────────────────────────────────

SECTION_MAP = {
    "sessions": print_sessions,
    "history":  print_history,
    "meta":     print_meta,
    "chunks":   print_chunks,
    "cache":    print_cache,
    "overview": lambda r, _: print_overview(r),
}

VALID_SECTIONS = list(SECTION_MAP.keys())


def main() -> None:
    args    = sys.argv[1:]
    target  = args[0].lower() if args else "all"
    limit   = int(args[1]) if len(args) > 1 else 5

    print("=" * 80)
    print("  Agentic AI RAG -- Redis Inspection")
    print(f"  Host : {os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}")
    print("=" * 80)

    try:
        r = connect()
        r.ping()
    except Exception as e:
        print(f"\n  [!] Could not connect to Redis: {e}")
        print("  Check REDIS_HOST / REDIS_PORT / REDIS_PASSWORD in .env")
        sys.exit(1)

    print("  [ok] Connected\n")

    if target == "all":
        print_overview(r)
        print_sessions(r,  limit)
        print_history(r,   limit)
        print_meta(r,      limit)
        print_chunks(r,    limit)
        print_cache(r,     limit)
        print_dedup_guards(r)
    elif target in SECTION_MAP:
        SECTION_MAP[target](r, limit)
    else:
        print(f"  Unknown section '{target}'. Choose from: all, {', '.join(VALID_SECTIONS)}")
        sys.exit(1)

    divider()
    print("  Done.")


if __name__ == "__main__":
    main()
