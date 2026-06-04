# Agentic AI RAG — Corporate Finance Intelligence Platform

> **Pilot Release v0.1.0** — Production-grade, agentic Retrieval-Augmented Generation system for large-scale corporate finance document analysis (1,000+ page PDF support).

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-orange)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3-yellow)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-orange)](https://groq.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-blue)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

**Agentic AI RAG** is an enterprise-ready Retrieval-Augmented Generation system purpose-built for corporate finance education and practice. It enables practitioners, analysts, and students to query massive financial documents — balance sheets, income statements, accounting standards, regulatory filings, and course materials — using natural language, with answers grounded strictly in the source material.

The system goes beyond simple Q&A. Built on LangGraph's stateful multi-agent orchestration, it plans, retrieves, reasons, and self-corrects across complex multi-step financial queries spanning hundreds of pages — with full guardrails, trajectory audit trails, and a layered memory architecture.

### Core Use Case

- **Input:** `CoreCourseFinancialAccounting.pdf` (1,000+ pages, to be expanded)
- **Query:** *"What is the difference between FIFO and LIFO inventory valuation and how does each method affect net income during inflation?"*
- **Output:** A cited, context-grounded answer with source page references and confidence score

> **Note:** `CoreCourseFinancialAccounting.pdf` lives in `data/documents/`. Run `scripts/ingest.py` → `scripts/describe_images.py` → `scripts/build_index.py` in order to populate the Qdrant vector store before querying.

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                          CLIENT / USER                                │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │  HTTP / REST
                              ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      FastAPI Application Layer                        │
│            /query    /ingest    /health    /sessions                  │
│                          Middleware                                   │
│                   (CORS · Auth · Rate Limit)                          │
└──────────────────────────┬────────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐  ┌──────────┐  ┌───────────────┐
     │  Guardrails  │  │  Memory  │  │  Trajectory   │
     │   (Input)    │  │ Manager  │  │   Tracker     │
     └──────┬───────┘  └────┬─────┘  └───────┬───────┘
            │               │                │
            ▼               ▼                │
┌───────────────────────────────────────┐    │
│        LangGraph Agent Orchestrator   │◄───┘
│                                       │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐
│  │ Planner │─▶│Retriever │─▶│ Reasoner │─▶│ Generator │
│  │  Node   │  │  Node    │  │   Node   │  │   Node    │
│  └─────────┘  └──────────┘  └──────────┘  └─────┬─────┘
│       ▲                                          │      │
│       └──────────── Self-Correction Loop ─────────┘      │
└──────────────────────────────────────────────────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
   ┌────────────────┐   ┌──────────────────┐   ┌──────────────┐
   │  Vector Store  │   │   Groq LLM       │   │  Guardrails  │
   │  Qdrant Cloud  │   │  llama-3.3-70b   │   │  (Output)    │
   │  Dense + BM25  │   │  + Gemini fallbk │   │              │
   └───────┬────────┘   └──────────────────┘   └──────────────┘
           │
  ┌────────┴──────────────────────────────────┐
  │              Memory Backends               │
  │  ┌──────────┐  ┌─────────────────────────┐│
  │  │  Redis   │  │      PostgreSQL          ││
  │  │ Working  │  │  Episodic  │  Semantic   ││
  │  │ Memory   │  │  Memory    │  Memory     ││
  │  └──────────┘  └─────────────────────────┘│
  └───────────────────────────────────────────┘
           ▲
           │  Docling Ingestion Pipeline
           ▼
  ┌────────────────┐
  │  PDF → Parse   │
  │  → Chunk       │
  │  → Embed       │
  │  (1,000+ pgs)  │
  └────────────────┘
```

---

## Agent Flow

```
User Query
    │
    ├──▶ [Input Guardrails]     Validate intent, sanitize, finance domain check
    │
    ▼
[Planner Node]                  Decomposes complex financial question into sub-tasks
    │
    ▼
[Retriever Node]                Semantic search across vectorized document corpus
    │                           Pulls from working memory (Redis) for session context
    ▼
[Reasoner Node]                 Validates retrieved context against question
    │                           Re-triggers retrieval if context is insufficient
    ▼
[Generator Node]                Synthesizes grounded answer with source citations
    │
    ├──▶ [Output Guardrails]    Hallucination check, citation validation, domain rules
    │
    ├──▶ [Trajectory Recorder]  Persists full agent step trace (audit trail)
    │
    ├──▶ [Episodic Memory]      Saves conversation turn to PostgreSQL
    │
    ▼
Grounded Response + Source Page References + Confidence Score
```

---

## Memory Architecture

The system implements a three-tier memory model aligned to cognitive memory theory:

| Memory Type | Backend | Scope | Purpose |
|---|---|---|---|
| **Working Memory** | Redis | Per-session (TTL-based) | Active conversation buffer, agent state, recent turns |
| **Episodic Memory** | PostgreSQL | Persistent | Full conversation history, session records, user interactions |
| **Semantic Memory** | PostgreSQL + pgvector | Persistent | Extracted financial concepts, facts, entities from documents |

> **Why this matters for finance:** Episodic memory enables the system to reason across multiple sessions — e.g., "Last week you asked about LIFO; here's how it relates to today's question on inventory turnover."

---

## Guardrails

Two-layer safety system wrapping every agent invocation:

**Input Guardrails** (`app/guardrails/input/`)
- Prompt injection detection
- Off-topic / out-of-domain rejection
- Input sanitization (length, encoding, special chars)

**Output Guardrails** (`app/guardrails/output/`)
- Hallucination detection (answer vs. retrieved chunks)
- Citation completeness check (every claim must have a source)
- Financial domain rules (e.g., no unlicensed investment advice)

**Domain Guardrails** (`app/guardrails/domain/`)
- GAAP compliance terminology checks
- Finance-specific terminology validation

---

## Trajectory Tracking

Every agent step is recorded for audit, debugging, and future RLHF:

```
trajectory record = {
  session_id, run_id, step_number,
  node_name, input_state, output_state,
  tokens_used, latency_ms, timestamp
}
```

> **Why trajectory is non-negotiable for finance:** Any corporate finance application must maintain an audit trail of how an answer was derived — for compliance, explainability, and trust.

Stored in PostgreSQL (`db/schemas/`). Future: exportable to Splunk for enterprise log aggregation.

---

## Prompt Layer

All prompts are stored as **YAML files** — no inline strings in code. The content inside each YAML file uses **XML + Markdown**:

- **YAML** = file container, version metadata, easy `yaml.safe_load()` in Python
- **XML tags** = structural sections (`<role>`, `<context>`, `<constraints>`, `<output_format>`) for clear instruction boundaries
- **Markdown** = human-readable formatting inside sections (headers, bullet lists, code blocks, tables)

```yaml
# app/prompts/agents/planner.yml  — abbreviated example
name: planner_system
version: "1.0.0"

system: |
  <role>
  You are a financial query planning agent...
  </role>

  <constraints>
  - Produce between 1 and 4 sub-tasks only
  - Do not answer the question yourself; only plan retrieval
  </constraints>

  <output_format>
  ## Expected JSON
  {"sub_tasks": [...], "reasoning": "..."}
  </output_format>

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI 0.136+ | Async REST API, OpenAPI docs |
| ASGI Server | Uvicorn 0.48+ | Production-grade async server |
| Agent Orchestration | LangGraph 1.2+ | Stateful multi-agent workflows |
| LLM Framework | LangChain 1.3+ | Chains, tools, document utilities |
| LLM Provider | Groq (llama-3.3-70b-versatile) | Primary LLM via langchain-groq; Gemini 1.5 Flash fallback if `GOOGLE_API_KEY` set |
| PDF Processing | Docling (IBM) + PyMuPDF | Structured PDF parsing (tables, charts, formulas) |
| Dense Embeddings | sentence-transformers (BAAI/bge-base-en-v1.5) | 768-dim dense vectors for semantic search |
| Sparse Embeddings | fastembed (Qdrant/BM25) | Keyword-level sparse vectors |
| Vector Store | Qdrant Cloud | Hybrid dense+sparse search with RRF fusion; 3 collections |
| Reranker | cross-encoder/ms-marco-MiniLM-L-2-v2 | Precision re-scoring pass after retrieval |
| Data Validation | Pydantic v2 | Type-safe request/response schemas |
| Short-term Memory | Redis | Working memory buffer (TTL-based) |
| Long-term Memory | PostgreSQL | Episodic memory store |
| DB Migrations | Alembic | PostgreSQL schema versioning |
| Package Manager | uv | Fast, reproducible dependency management |
| Runtime | Python 3.12 | Latest stable CPython |
| Observability | LangSmith + Langfuse + Splunk HEC | Tracing, cost tracking, enterprise log pipeline |

---

## Project Structure

```
Agentic_AI_Rag/
│
├── main.py                          # FastAPI application entry point
├── pyproject.toml                   # Project metadata and dependencies
├── uv.lock                          # Pinned dependency versions (reproducible builds)
├── .python-version                  # Runtime pin: Python 3.12
├── .env.example                     # Environment variable template
├── .gitignore
│
├── app/                             # Application source
│   │
│   ├── api/                         # FastAPI layer
│   │   ├── routes/                  # Endpoint handlers: ingest, query, health, sessions
│   │   └── middleware/              # CORS, auth, rate limiting
│   │
│   ├── agents/                      # LangGraph multi-agent system
│   │   ├── graph/
│   │   │   ├── state.py             # AgentState TypedDict (single source of truth for graph state)
│   │   │   ├── builder.py           # StateGraph construction, node registration, compile()
│   │   │   └── edges.py             # Conditional edge functions (route_after_reasoner)
│   │   ├── nodes/                   # planner.py · retriever.py · reasoner.py · generator.py
│   │   └── tools/                   # Agent tools: vector search, financial calculator
│   │
│   ├── guardrails/                  # Two-layer safety system
│   │   ├── input/                   # Prompt injection, domain check, sanitizer
│   │   ├── output/                  # Hallucination detection, citation validator
│   │   └── domain/                  # Finance-specific rules (GAAP, disclaimers)
│   │
│   ├── trajectory/                  # Agent step audit trail
│   │                                # tracker.py · recorder.py · analyzer.py
│   │
│   ├── memory/                      # Three-tier memory model
│   │   ├── working/                 # Redis — short-term conversation buffer (TTL)
│   │   ├── episodic/                # PostgreSQL — session history, user interactions
│   │   └── semantic/                # PostgreSQL + pgvector — extracted knowledge
│   │
│   ├── pipeline/                    # Document processing pipeline
│   │   ├── ingestion/               # Docling PDF loader, chunker, embedder
│   │   └── retrieval/               # Vector retriever, reranker
│   │
│   ├── store/                       # Storage backend clients
│   │   ├── vector/                  # Qdrant client wrappers
│   │   ├── relational/              # PostgreSQL client + Alembic integration
│   │   └── cache/                   # Redis client
│   │
│   ├── prompts/                     # Versioned prompt layer (YAML files · XML+MD content)
│   │   ├── agents/                  # planner.yml · retriever.yml · reasoner.yml · generator.yml
│   │   ├── guardrails/              # input.yml · output.yml · domain.yml
│   │   ├── few_shots/               # finance_qa.yml — domain few-shot examples in YAML
│   │   ├── templates/               # reusable query/response YAML templates
│   │   └── registry/                # versions.yml — active prompt version pointer per agent
│   │
│   ├── observability/               # Monitoring and tracing
│   │   ├── logging/                 # Structured JSON logging (Splunk HEC-ready)
│   │   ├── tracing/                 # LangSmith integration
│   │   └── metrics/                 # Custom performance metrics
│   │
│   └── models/                      # Pydantic schemas: request, response, domain
│
├── data/                            # Data assets
│   ├── documents/                   # Source PDF documents
│   ├── processed/                   # Intermediate pipeline output (chunks JSON, page map, images)
│   └── exports/                     # Evaluation results, exports
│
├── db/                              # Database management
│   ├── migrations/
│   │   └── versions/                # Alembic migration history
│   ├── schemas/                     # Raw SQL: episodic.sql, semantic.sql, trajectory.sql
│   └── seeds/                       # Dev / test seed data
│
├── tests/
│   ├── unit/
│   │   ├── agents/
│   │   ├── guardrails/
│   │   ├── memory/
│   │   └── pipeline/
│   ├── integration/
│   │   ├── api/
│   │   └── memory/
│   └── e2e/
│
├── scripts/                         # Pipeline scripts (run in order)
│   ├── ingest.py                    # Step 1: PDF → text/table/image chunks
│   ├── describe_images.py           # Step 2: Image → LLM text descriptions (Llama 4 Scout)
│   └── build_index.py               # Step 3: Chunk → embed → upsert to Qdrant
│
└── docker/                          # Container configs: App + Redis + PostgreSQL
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | Pinned via `.python-version` |
| uv | Latest | `pip install uv` |
| Groq API Key | — | For llama-3.3-70b-versatile (primary LLM) |
| Qdrant Cloud account | — | `QDRANT_URL` + `QDRANT_API_KEY` (free tier available) |
| Redis | 7.0+ | Working memory (Docker recommended) |
| PostgreSQL | 15+ | Episodic memory, trajectory |
| Git | Any | |

---

## Quick Start

### 1. Clone

```bash
git clone <repository-url>
cd Agentic_AI_Rag
```

### 2. Install Dependencies

```bash
pip install uv
uv sync
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env — fill in GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY at minimum
```

### 4. Start Services (Docker)

```bash
# Redis + PostgreSQL  (Qdrant is cloud-hosted — no local container needed)
docker compose -f docker/docker-compose.yml up -d
```

### 5. Run API Server

```bash
.venv\Scripts\activate          # Windows
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **Health:** `http://localhost:8000/health`

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | — | Groq API key (primary LLM) |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq chat model |
| `QDRANT_URL` | **Yes** | — | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | **Yes** | — | Qdrant Cloud API key |
| `GOOGLE_API_KEY` | No | — | Google API key for Gemini 1.5 Flash fallback LLM |
| `GOOGLE_MODEL` | No | `gemini-2.0-flash` | Gemini model (fallback) |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection string |
| `POSTGRES_HOST` | No | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | No | `5432` | PostgreSQL port |
| `POSTGRES_USER` | No | `postgres` | PostgreSQL user |
| `POSTGRES_PASSWORD` | No | — | PostgreSQL password |
| `POSTGRES_DB` | No | `postgres` | PostgreSQL database |
| `PDF_SOURCE_DIR` | No | `./data/documents` | Source PDF directory for ingestion |
| `CHUNK_SIZE` | No | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | No | `200` | Overlap characters between chunks |
| `PDF_PAGE_OFFSET` | No | `16` | Offset between PDF page index and printed book page |
| `RERANKER_ENABLED` | No | `true` | Enable cross-encoder reranker (`true`/`false`) |
| `RERANKER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L-2-v2` | Reranker model |
| `HOST` | No | `0.0.0.0` | API server host |
| `PORT` | No | `8000` | API server port |
| `ENV` | No | `development` | `development` or `production` |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | — | LangSmith project name |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse public key |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse endpoint |
| `SPLUNK_HOST` | No | — | Splunk HEC hostname |
| `SPLUNK_HEC_PORT` | No | `8088` | Splunk HEC port |
| `SPLUNK_HEC_TOKEN` | No | — | Splunk HEC token |
| `SPLUNK_VERIFY_SSL` | No | `true` | Verify Splunk TLS certificate |

---

## API Reference

### Health Check

```http
GET /health
```
```json
{ "status": "healthy", "version": "0.1.0", "redis": "connected", "postgres": "connected", "vector_store": "connected" }
```

### Ingest Document

```http
POST /ingest
Content-Type: multipart/form-data
```
```json
{ "job_id": "ingest_abc123", "status": "processing", "pages_detected": 1024, "chunks_created": 4200 }
```

### Query (RAG)

```http
POST /query
Content-Type: application/json

{
  "question": "Explain the matching principle in accrual accounting.",
  "session_id": "session_xyz",
  "top_k": 5,
  "stream": false
}
```
```json
{
  "answer": "The matching principle requires that expenses be recognized...",
  "sources": [
    { "page": 42, "excerpt": "...expenses must be recorded in the same period...", "score": 0.94 },
    { "page": 78, "excerpt": "...accrual basis recognizes revenue when earned...", "score": 0.91 }
  ],
  "session_id": "session_xyz",
  "trajectory_id": "traj_def456",
  "tokens_used": 1240,
  "confidence": 0.92
}
```

---

## Pilot Performance Targets

| Metric | Target |
|---|---|
| PDF ingestion — 1,000 pages | < 10 minutes |
| Query latency P95 | < 5 seconds |
| Retrieval relevance Top-5 | > 85% |
| Concurrent sessions | 10+ |
| Answer citation rate | 100% (every claim sourced) |

---

## Features Status

### Implemented
- [x] FastAPI + LangGraph + LangChain + Groq scaffold with full folder architecture
- [x] Reproducible environment via `uv` + lock file
- [x] Docling PDF ingestion pipeline: parse → chunk → embed → upsert to Qdrant (3-script pipeline)
- [x] Qdrant Cloud hybrid vector store: dense (BAAI/bge-base-en-v1.5) + sparse (BM25) with RRF fusion
- [x] Cross-encoder reranker (ms-marco-MiniLM-L-2-v2) — precision pass after retrieval
- [x] LangGraph state machine: planner, retriever, reasoner, generator nodes
- [x] Input + output guardrails (prompt injection, domain check, hallucination detection)
- [x] Redis working memory (conversation buffer, TTL-based)
- [x] PostgreSQL episodic memory (session persistence)
- [x] REST endpoints: `/query`, `/agent`, `/health`
- [x] SSE streaming responses
- [x] Streamlit UI
- [x] Structured JSON logging (Splunk HEC-compatible)
- [x] LangSmith + Langfuse observability integration

### Roadmap

**Next**
- [ ] Docker Compose: app + Redis + PostgreSQL (local dev stack)
- [ ] `.env.example` template for onboarding
- [ ] RAGAS evaluation framework (faithfulness, relevancy, context recall)
- [ ] PostgreSQL + pgvector semantic memory (knowledge extraction layer)
- [ ] Prompt versioning via registry

**Future**
- [ ] Multi-document corpus support
- [ ] `/sessions` endpoint + session management UI
- [ ] Splunk enterprise log aggregation (HEC pipeline is wired; index config pending)

---

## Corporate Finance Domain Coverage

| Area | Topics |
|---|---|
| Financial Statements | Income Statement, Balance Sheet, Cash Flow Statement |
| Revenue Recognition | Accrual basis, matching principle, ASC 606 |
| Inventory Valuation | FIFO, LIFO, Weighted Average |
| Asset Management | Depreciation methods, impairment |
| Liquidity | Working capital, current/quick ratios |
| Capital Structure | Cost of capital, WACC, leverage |
| Performance Metrics | ROE, ROA, EBITDA, financial ratios |
| Tax Accounting | Deferred tax, effective tax rate |
| Consolidation | Intercompany transactions, elimination entries |
| Standards | GAAP principles, IFRS awareness |

---

## Development

```bash
# Tests
uv run pytest tests/unit -v
uv run pytest tests/integration -v
uv run pytest --cov=app --cov-report=html

# Add dependency
uv add <package>
uv add --dev <package>

# Code quality
uv run ruff format .
uv run ruff check .
uv run mypy app/
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Powered by LangGraph · LangChain · Docling · Groq · Qdrant · FastEmbed · FastAPI · Redis · PostgreSQL · Python 3.12*
