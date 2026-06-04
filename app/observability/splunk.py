"""
Splunk HEC (HTTP Event Collector) client.

All sends are non-blocking — events are queued and drained by a daemon
thread. If Splunk is unreachable, events are silently dropped.
Never raises — observability must not break the app.
"""

import json
import logging
import os
import queue
import ssl
import threading
import time
import urllib.request
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_QUEUE: "queue.Queue[dict | None]" = queue.Queue(maxsize=2000)
_STARTED      = False
_STARTED_LOCK = threading.Lock()


def _hec_url() -> str:
    host = os.getenv("SPLUNK_HOST", "")
    port = os.getenv("SPLUNK_HEC_PORT", "8088")
    return f"https://{host}:{port}/services/collector/event"


def _worker() -> None:
    token = os.getenv("SPLUNK_HEC_TOKEN", "")
    if not token:
        logger.warning("SPLUNK_HEC_TOKEN not set — Splunk logging disabled")
        return

    url     = _hec_url()
    verify  = os.getenv("SPLUNK_VERIFY_SSL", "true").lower() != "false"
    ctx     = ssl.create_default_context() if verify else ssl._create_unverified_context()
    headers = {
        "Authorization":            f"Splunk {token}",
        "Content-Type":             "application/json",
        "X-Splunk-Request-Channel": str(uuid.uuid4()),
    }

    while True:
        try:
            payload = _QUEUE.get(timeout=5)
        except queue.Empty:
            continue
        if payload is None:
            break
        try:
            data = json.dumps(payload, default=str).encode()
            req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
                if resp.status not in (200, 201):
                    logger.debug("Splunk HEC %s → HTTP %d", url, resp.status)
        except Exception as exc:
            logger.debug("Splunk send error: %s", exc)


def _ensure_started() -> None:
    global _STARTED
    with _STARTED_LOCK:
        if not _STARTED:
            threading.Thread(target=_worker, daemon=True, name="splunk-hec").start()
            _STARTED = True


def _send(index: str, sourcetype: str, event: dict[str, Any]) -> None:
    _ensure_started()
    payload: dict[str, Any] = {
        "time":       time.time(),
        "index":      index,
        "sourcetype": sourcetype,
        "event":      event,
    }
    try:
        _QUEUE.put_nowait(payload)
    except Exception:
        pass


# ── rag_pipeline index ────────────────────────────────────────────────────────

def latency_summary(
    *,
    trajectory_id: str,
    session_id:    str,
    tenant_id:     str,
    latency_trace: dict,
) -> None:
    """
    One event per query with every sub-step duration flat at the top level.
    This makes SPL trivial: no joins, no sub-searches — just table/timechart.

    Sent to rag_pipeline index, sourcetype rag:latency.
    Field naming convention: {node}_{substep}_ms  (e.g. retriever_qdrant_ms)
    """
    event: dict[str, Any] = {
        "event_type":    "latency_summary",
        "trajectory_id": trajectory_id,
        "session_id":    session_id,
        "tenant_id":     tenant_id,
        "total_ms":      latency_trace.get("_total_ms", 0),
    }

    for node, val in latency_trace.items():
        if node.startswith("_"):
            # _pipeline dict → flatten as-is (keys already have _ms suffix)
            if isinstance(val, dict):
                for k, v in val.items():
                    event[k] = v
        elif isinstance(val, dict):
            for k, v in val.items():
                event[f"{node}_{k}"] = v   # e.g. retriever_qdrant_ms

    _send("rag_pipeline", "rag:latency", event)

def rag_query(
    *,
    tenant_id: str,
    user_id: str,
    session_id: str,
    trajectory_id: str,
    question_length: int,
    answer_length: int,
    confidence: float,
    cache_hit: bool,
    cache_type: str | None,
    iteration_count: int,
    duration_ms: float,
    source_count: int,
    idempotency_replay: bool = False,
) -> None:
    _send("rag_pipeline", "rag:query", {
        "event_type":         "rag_query",
        "tenant_id":          tenant_id,
        "user_id":            user_id,
        "session_id":         session_id,
        "trajectory_id":      trajectory_id,
        "question_length":    question_length,
        "answer_length":      answer_length,
        "confidence":         confidence,
        "cache_hit":          cache_hit,
        "cache_type":         cache_type,
        "iteration_count":    iteration_count,
        "duration_ms":        duration_ms,
        "source_count":       source_count,
        "idempotency_replay": idempotency_replay,
    })


def retrieval_query(
    *,
    tenant_id: str,
    session_id: str,
    question_length: int,
    chunk_count: int,
    duration_ms: float,
) -> None:
    _send("rag_pipeline", "rag:retrieval", {
        "event_type":      "retrieval_query",
        "tenant_id":       tenant_id,
        "session_id":      session_id,
        "question_length": question_length,
        "chunk_count":     chunk_count,
        "duration_ms":     duration_ms,
    })


# ── agent_trajectory index ────────────────────────────────────────────────────

def node_step(
    *,
    node: str,
    phase: str,
    trajectory_id: str,
    session_id: str,
    iteration_count: int = 0,
    duration_ms: float | None = None,
    **extra: Any,
) -> None:
    event: dict[str, Any] = {
        "event_type":      "node_step",
        "node":            node,
        "phase":           phase,
        "trajectory_id":   trajectory_id,
        "session_id":      session_id,
        "iteration_count": iteration_count,
    }
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    event.update(extra)
    _send("agent_trajectory", "rag:trajectory", event)


# ── security_events index ─────────────────────────────────────────────────────

def security_event(
    *,
    event_type: str,
    guard_type: str,
    reason: str,
    tenant_id: str = "",
    session_id: str = "",
    trajectory_id: str = "",
    question_length: int = 0,
) -> None:
    _send("security_events", "rag:security", {
        "event_type":      event_type,
        "guard_type":      guard_type,
        "reason":          reason,
        "tenant_id":       tenant_id,
        "session_id":      session_id,
        "trajectory_id":   trajectory_id,
        "question_length": question_length,
    })


# ── Python logging integration ────────────────────────────────────────────────

class SplunkHECHandler(logging.Handler):
    """Route Python WARNING+ log records to rag_pipeline index as rag:applog events."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _send("rag_pipeline", "rag:applog", {
                "event_type": "applog",
                "level":      record.levelname,
                "logger":     record.name,
                "message":    self.format(record),
                "module":     record.module,
                "funcName":   record.funcName,
            })
        except Exception:
            self.handleError(record)


# ── System metrics background thread ─────────────────────────────────────────

def _metrics_worker(interval: int) -> None:
    try:
        import psutil
    except ImportError:
        logger.debug("psutil not installed — system metrics disabled")
        return

    while True:
        time.sleep(interval)
        try:
            mem  = psutil.virtual_memory()
            cpu  = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage(".")
            _send("rag_pipeline", "rag:system", {
                "event_type":   "system_metrics",
                "cpu_percent":  cpu,
                "mem_percent":  mem.percent,
                "mem_used_mb":  round(mem.used / 1024**2, 1),
                "mem_total_mb": round(mem.total / 1024**2, 1),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / 1024**3, 2),
            })
        except Exception:
            pass


def start_metrics(interval: int = 60) -> None:
    """Start background system-metrics sender. Call once at app startup."""
    threading.Thread(
        target=_metrics_worker,
        args=(interval,),
        daemon=True,
        name="splunk-metrics",
    ).start()
