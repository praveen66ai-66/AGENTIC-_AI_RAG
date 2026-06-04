"""
Professional Streamlit frontend — calls FastAPI only.

Usage:
    uv run streamlit run app/streamlit_app.py
"""

import json
import os
import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"
HEALTH   = "http://localhost:8000/health"


def _stream_agent(question: str, headers: dict):
    """
    Generator consumed by st.write_stream().
    Connects to /agent/stream, yields token strings for live rendering,
    and stashes metadata + sources in session_state for display after streaming.
    """
    st.session_state._stream_meta = {}
    st.session_state._stream_done = {}

    with requests.post(
        f"{API_BASE}/agent/stream",
        json={"question": question},
        headers=headers,
        stream=True,
        timeout=180,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except Exception:
                continue

            t = event.get("type")
            if t == "metadata":
                st.session_state._stream_meta = event
            elif t == "token":
                yield event.get("content", "")
            elif t == "done":
                st.session_state._stream_done = event
                break
            elif t == "error":
                yield f"\n\n⚠️ {event.get('detail', 'Unknown error')}"
                break

st.set_page_config(
    page_title="FinanceRAG · AI Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Base */
.stApp { background: #f8f9fa; color: #1a1a2e; }
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e0e0e0;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #1565c0 0%, #1976d2 50%, #1565c0 100%);
    border: none;
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(21,101,192,0.25);
}
.app-header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header p {
    font-size: 13px;
    color: #bbdefb;
    margin: 4px 0 0 0;
}
.header-badge {
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.4);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 12px;
    color: #ffffff;
    font-weight: 600;
}

/* Status pills */
.status-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid;
}
.pill-green { background:#e8f5e9; border-color:#43a047; color:#2e7d32; }
.pill-blue  { background:#e3f2fd; border-color:#1976d2; color:#1565c0; }
.pill-amber { background:#fff8e1; border-color:#fb8c00; color:#e65100; }
.pill-red   { background:#ffebee; border-color:#e53935; color:#c62828; }

/* Chat messages */
.stChatMessage {
    background: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 10px !important;
}

/* Chunk card */
.chunk-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #1565c0;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.chunk-card:hover {
    border-left-color: #1976d2;
    box-shadow: 0 4px 16px rgba(21,101,192,0.15);
    transform: translateX(2px);
}
.chunk-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    flex-wrap: wrap;
    gap: 6px;
}
.chunk-meta { font-size: 12px; color: #546e7a; display: flex; align-items: center; gap: 8px; }
.chunk-text { font-size: 14px; color: #37474f; line-height: 1.75; }

/* Citation badge */
.cite-badge {
    background: #e3f2fd;
    border: 1px solid #1976d2;
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 11px;
    color: #1565c0;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* Score bar */
.score-wrap { display: flex; align-items: center; gap: 8px; }
.score-bar-bg {
    background: #eceff1;
    border-radius: 4px;
    height: 6px;
    width: 80px;
    overflow: hidden;
    border: 1px solid #cfd8dc;
}
.score-bar-fill { height: 100%; border-radius: 4px; }

/* Source tag */
.source-tag {
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
}
.tag-text  { background:#e3f2fd; border:1px solid #1976d2; color:#1565c0; }
.tag-table { background:#e0f2f1; border:1px solid #00897b; color:#00695c; }
.tag-struct{ background:#f3e5f5; border:1px solid #8e24aa; color:#6a1b9a; }

/* Citations summary */
.citations-box {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
}
.citations-box h5 { color: #546e7a; font-size: 12px; text-transform: uppercase;
                    letter-spacing: 1px; margin: 0 0 10px 0; }
.cite-item { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.cite-num  { background: #e3f2fd; border: 1px solid #1976d2; border-radius: 4px;
             padding: 1px 8px; font-size: 11px; color: #1565c0; font-weight: 700;
             min-width: 28px; text-align: center; }
.cite-info { font-size: 12px; color: #546e7a; }

/* Sidebar */
.sidebar-section { margin-bottom: 20px; }
.sidebar-section h4 { color: #1565c0; font-size: 13px; text-transform: uppercase;
                       letter-spacing: 1px; margin-bottom: 12px; }
.sidebar-stat {
    background: #f5f7fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stat-label { font-size: 12px; color: #546e7a; }
.stat-value { font-size: 14px; color: #1565c0; font-weight: 600; }

/* Inputs */
.stTextInput input, .stSelectbox select {
    background: #ffffff !important;
    border: 1px solid #cfd8dc !important;
    color: #1a1a2e !important;
    border-radius: 8px !important;
}
.stTextInput input:focus { border-color: #1565c0 !important; }

/* Chat input */
.stChatInputContainer {
    background: #ffffff !important;
    border-top: 1px solid #e0e0e0 !important;
    padding: 12px !important;
}
[data-testid="stChatInput"] {
    background: #f8f9fa !important;
    border: 1px solid #cfd8dc !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
}

/* Divider */
hr { border-color: #e0e0e0 !important; margin: 16px 0 !important; }

/* Button */
.stButton button {
    background: #1565c0 !important;
    border: 1px solid #1976d2 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100%;
}
.stButton button:hover {
    background: #1976d2 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id"       not in st.session_state: st.session_state.session_id       = str(uuid.uuid4())
if "chat_history"     not in st.session_state: st.session_state.chat_history     = []
if "query_count"      not in st.session_state: st.session_state.query_count      = 0
if "cache_hits"       not in st.session_state: st.session_state.cache_hits       = 0
if "last_confidence"  not in st.session_state: st.session_state.last_confidence  = None
if "last_duration_ms"   not in st.session_state: st.session_state.last_duration_ms   = None
if "last_token_usage"  not in st.session_state: st.session_state.last_token_usage  = {}
if "last_total_tokens" not in st.session_state: st.session_state.last_total_tokens  = 0

# ── Health check ──────────────────────────────────────────────────────────────
health = {}
try:
    health = requests.get(HEALTH, timeout=3).json()
except Exception:
    pass

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 FinanceRAG")
    st.caption("AI Research Assistant · v0.1")
    st.markdown("---")

    # Service status
    st.markdown('<div class="sidebar-section"><h4>Services</h4>', unsafe_allow_html=True)
    services = [
        ("PostgreSQL", health.get("postgres", False), "🗄️"),
        ("Redis",      health.get("redis",    False), "⚡"),
        ("Qdrant",     bool(health),                  "🔍"),
        ("LLM Agent",  bool(health),                  "🤖"),
    ]
    for name, ok, icon in services:
        label  = "Ready" if ok else "Offline"
        cls    = "pill-green" if ok else "pill-red"
        dot    = "●"
        st.markdown(
            f'<div class="status-pill {cls}">{icon} {name} <span>{dot} {label}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Identity
    st.markdown('<div class="sidebar-section"><h4>Identity</h4>', unsafe_allow_html=True)
    tenant_id = st.text_input("Tenant ID", value="default_tenant", label_visibility="visible")
    user_id   = st.text_input("User ID",   value="default_user",   label_visibility="visible")
    st.caption(f"Session `{st.session_state.session_id[:12]}…`")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄  New Session"):
        st.session_state.session_id       = str(uuid.uuid4())
        st.session_state.chat_history     = []
        st.session_state.query_count      = 0
        st.session_state.cache_hits       = 0
        st.session_state.last_confidence   = None
        st.session_state.last_duration_ms  = None
        st.session_state.last_token_usage  = {}
        st.session_state.last_total_tokens = 0
        st.rerun()

    st.markdown("---")

    # Session stats
    st.markdown('<div class="sidebar-section"><h4>Session Stats</h4>', unsafe_allow_html=True)
    conf_val  = f"{st.session_state.last_confidence:.2f}" if st.session_state.last_confidence is not None else "—"
    dur_val   = f"{st.session_state.last_duration_ms:.0f} ms" if st.session_state.last_duration_ms is not None else "—"
    tok_val   = f"{st.session_state.last_total_tokens:,}" if st.session_state.last_total_tokens else "—"
    ctx_limit = int(os.getenv("GROQ_CONTEXT_WINDOW", "32768"))
    ctx_pct   = f" ({st.session_state.last_total_tokens/ctx_limit*100:.1f}% of ctx)" if st.session_state.last_total_tokens else ""
    st.markdown(f"""
    <div class="sidebar-stat">
        <span class="stat-label">Queries</span>
        <span class="stat-value">{st.session_state.query_count}</span>
    </div>
    <div class="sidebar-stat">
        <span class="stat-label">Cache hits</span>
        <span class="stat-value">{st.session_state.cache_hits}</span>
    </div>
    <div class="sidebar-stat">
        <span class="stat-label">Last confidence</span>
        <span class="stat-value">{conf_val}</span>
    </div>
    <div class="sidebar-stat">
        <span class="stat-label">Last duration</span>
        <span class="stat-value">{dur_val}</span>
    </div>
    <div class="sidebar-stat">
        <span class="stat-label">Tokens used</span>
        <span class="stat-value">{tok_val}{ctx_pct}</span>
    </div>
    """, unsafe_allow_html=True)

    # Per-stage token + duration breakdown
    usage = st.session_state.last_token_usage
    if usage:
        st.markdown("**Last query — stage detail**")
        for stage, t in usage.items():
            if not isinstance(t, dict):
                continue
            st.caption(
                f"`{stage}` · in {t.get('in',0):,} · out {t.get('out',0):,} · "
                f"total {t.get('total',0):,}"
            )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.caption("📖 Source: Corporate Finance — Vernimmen, 4th Ed.")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <h1>📊 Corporate Finance Assistant</h1>
        <p>Powered by RAG · Semantic + Keyword Search · Multi-collection retrieval</p>
    </div>
    <div class="header-badge">● Agent Ready</div>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"], unsafe_allow_html=True)

# ── Query input ───────────────────────────────────────────────────────────────
question = st.chat_input("Ask anything about corporate finance…")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.query_count += 1

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            headers = {
                "X-Tenant-Id":  tenant_id,
                "X-User-Id":    user_id,
                "X-Session-Id": st.session_state.session_id,
            }

            # Stream tokens live — renders word by word
            answer = st.write_stream(_stream_agent(question, headers))

            # After streaming — read metadata stashed during stream
            meta = st.session_state.get("_stream_meta", {})
            done = st.session_state.get("_stream_done", {})

            confidence = done.get("confidence", meta.get("confidence", 0.0))
            sources    = done.get("sources", [])
            plan       = meta.get("plan", [])
            duration   = done.get("duration_ms", 0)
            cache_hit  = done.get("cache_hit", meta.get("cache_hit", False))

            # Update sidebar stats
            st.session_state.last_confidence   = confidence
            st.session_state.last_duration_ms  = duration
            st.session_state.last_token_usage  = done.get("token_usage",  meta.get("token_usage", {}))
            st.session_state.last_total_tokens = done.get("total_tokens", meta.get("total_tokens", 0))
            if cache_hit:
                st.session_state.cache_hits += 1

            # ── Confidence + duration badge ───────────────────────────────────
            conf_pct = int(confidence * 100)
            conf_col = "#2e7d32" if confidence >= 0.7 else "#e65100" if confidence >= 0.4 else "#c62828"
            cache_tag = ' <span style="background:#e3f2fd;border:1px solid #1976d2;border-radius:10px;padding:1px 8px;font-size:11px;color:#1565c0;font-weight:600;">⚡ cached</span>' if cache_hit else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0 2px 0;'
                f'padding:6px 12px;background:#f5f7fa;border-radius:8px;border:1px solid #e0e0e0;">'
                f'<span style="font-size:11px;color:#78909c;font-weight:500;">CONFIDENCE</span>'
                f'<span style="font-size:13px;font-weight:700;color:{conf_col};">{conf_pct}%</span>'
                f'<span style="color:#e0e0e0;">|</span>'
                f'<span style="font-size:11px;color:#78909c;">{duration:.0f} ms</span>'
                f'{cache_tag}</div>',
                unsafe_allow_html=True,
            )

            # ── Citations ─────────────────────────────────────────────────────
            if sources:
                cite_html = '<div class="citations-box"><h5>📎 Sources Used</h5>'
                for i, s in enumerate(sources, 1):
                    section = (s.get("section") or "—")[:60]
                    score   = float(s.get("score") or 0.0)
                    cite_html += (
                        f'<div class="cite-item">'
                        f'<span class="cite-num">[{i}]</span>'
                        f'<span class="cite-info">Page {s.get("page","?")} &nbsp;·&nbsp; '
                        f'{section} &nbsp;·&nbsp; score {score:.3f}</span></div>'
                    )
                cite_html += "</div>"
                st.markdown(cite_html, unsafe_allow_html=True)

            # ── Retrieval plan ────────────────────────────────────────────────
            if plan:
                with st.expander(f"🧠 Retrieval plan ({len(plan)} sub-tasks)", expanded=False):
                    for i, task in enumerate(plan, 1):
                        st.markdown(f"**{i}.** {task}")

            # ── Latency breakdown (compact HTML table, not st.table) ──────────
            trace = done.get("latency_trace", {})
            if trace:
                total_ms = float(trace.get("_total_ms") or duration or 1)
                order = ["_pipeline", "planner", "retriever", "retriever_1",
                         "reasoner", "reasoner_1", "generator"]
                rows_html = ""
                for key in order:
                    if key not in trace:
                        continue
                    val = trace[key]
                    items = val.items() if isinstance(val, dict) else []
                    for sub, ms in items:
                        try:
                            ms = float(ms)
                        except (TypeError, ValueError):
                            continue
                        pct  = ms / total_ms * 100
                        label = sub.replace("_ms","") if key == "_pipeline" else f"{key}.{sub.replace('_ms','')}"
                        bar_w = min(int(pct), 100)
                        bar_col = "#1565c0" if pct > 30 else "#42a5f5" if pct > 10 else "#90caf9"
                        rows_html += (
                            f'<tr><td style="font-size:11px;color:#546e7a;padding:2px 8px;">{label}</td>'
                            f'<td style="font-size:11px;font-weight:600;padding:2px 8px;">{ms:.0f} ms</td>'
                            f'<td style="padding:2px 8px;">'
                            f'<div style="background:#eceff1;border-radius:3px;height:8px;width:100px;">'
                            f'<div style="background:{bar_col};width:{bar_w}px;height:8px;border-radius:3px;"></div>'
                            f'</div></td>'
                            f'<td style="font-size:11px;color:#78909c;padding:2px 8px;">{pct:.1f}%</td></tr>'
                        )
                if rows_html:
                    with st.expander(f"⏱ Latency — {total_ms:.0f} ms total", expanded=False):
                        st.markdown(
                            f'<table style="width:100%;border-collapse:collapse;">'
                            f'<thead><tr>'
                            f'<th style="font-size:10px;color:#90a4ae;text-align:left;padding:2px 8px;">STEP</th>'
                            f'<th style="font-size:10px;color:#90a4ae;text-align:left;padding:2px 8px;">TIME</th>'
                            f'<th style="font-size:10px;color:#90a4ae;padding:2px 8px;"></th>'
                            f'<th style="font-size:10px;color:#90a4ae;text-align:left;padding:2px 8px;">%</th>'
                            f'</tr></thead><tbody>{rows_html}</tbody></table>',
                            unsafe_allow_html=True,
                        )

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        except requests.exceptions.ConnectionError:
            st.error("⚠️ API server not reachable. Run: `uv run uvicorn app.main_api:app`")
        except Exception as e:
            st.error(f"Error: {e}")
