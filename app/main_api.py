"""FastAPI application entry point."""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

from app.api.routes import query as query_route
from app.api.routes import agent as agent_route
from app.memory.store import setup_tables, ping as pg_ping
from app.memory.session import ping as redis_ping
from app.observability import splunk
from app.retrieval.reranker import warmup as warmup_reranker
from app.retrieval.search import ping as qdrant_ping

# Rate limiter — 20 requests/minute per IP
limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])

app = FastAPI(
    title="Agentic AI RAG",
    description="Financial document RAG API",
    version="0.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allows Streamlit + future AWS API Gateway to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    setup_tables()
    warmup_reranker()
    splunk.start_metrics(interval=60)
    pg    = pg_ping()
    redis = redis_ping()
    qdrant = qdrant_ping()
    print(f"  PostgreSQL : {'OK' if pg     else 'FAIL'}")
    print(f"  Redis      : {'OK' if redis  else 'FAIL'}")
    print(f"  Qdrant     : {'OK' if qdrant else 'FAIL'}")


@app.get("/health")
def health():
    pg     = pg_ping()
    redis  = redis_ping()
    qdrant = qdrant_ping()
    llm    = bool(os.environ.get("GROQ_API_KEY"))

    all_ok = pg and redis and qdrant
    status = "ok" if all_ok else ("unhealthy" if not qdrant else "degraded")

    return {
        "status":    status,
        "postgres":  pg,
        "redis":     redis,
        "qdrant":    qdrant,
        "llm_ready": llm,
    }


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


app.include_router(query_route.router, prefix="/api/v1")
app.include_router(agent_route.router, prefix="/api/v1")
