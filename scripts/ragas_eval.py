"""
RAGAS evaluation — Faithfulness, Answer Relevancy, Context Precision, Context Recall.

Runs the agent on every question in data/eval/ragas_dataset.json, then scores
all four RAGAS metrics using Groq as the LLM judge and the project's BGE model
as the embedder (no extra API keys needed).

What each metric tells you
--------------------------
Faithfulness      : Is every claim in the answer grounded in the retrieved chunks?
                    Low = hallucination. Critical for a finance Q&A system.
Answer Relevancy  : Does the answer actually address the question asked?
                    Low = answer drifted off-topic.
Context Precision : Were the retrieved chunks useful? (signal-to-noise of retrieval)
                    Low = reranker is surfacing irrelevant chunks.
Context Recall    : Did retrieval find ALL the information needed?
                    Low = relevant pages exist in Qdrant but weren't retrieved.

Reads:
    data/eval/ragas_dataset.json      — questions + ground truth answers
Writes:
    data/eval/ragas_results_<ts>.json — per-question scores + aggregate

Usage:
    uv run python scripts/ragas_eval.py
    uv run python scripts/ragas_eval.py --sample 5   # run on first 5 questions only
"""

import argparse
import importlib
import json
import os
import sys
import types
import uuid
from datetime import datetime
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Compatibility shim ────────────────────────────────────────────────────────
# ragas imports ChatVertexAI from langchain_community.chat_models.vertexai, but
# langchain_community >= 0.3 removed that module. We use Groq, not VertexAI —
# create a minimal stub so ragas can import without crashing.
def _patch_vertexai() -> None:
    mod_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(mod_name)
    except ImportError:
        stub = types.ModuleType(mod_name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})  # empty placeholder class
        sys.modules[mod_name] = stub

_patch_vertexai()

from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = Path("data/eval/ragas_dataset.json")
RESULTS_DIR  = Path("data/eval")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── BGE embedding wrapper for RAGAS ──────────────────────────────────────────
# Reuses the already-loaded SentenceTransformer singleton from app.retrieval.search
# so the model is not loaded twice.

from ragas.embeddings import BaseRagasEmbeddings


class _BGEEmbeddings(BaseRagasEmbeddings):
    """Wraps BAAI/bge-base-en-v1.5 for RAGAS answer_relevancy scoring."""

    def embed_query(self, text: str) -> list[float]:
        from app.retrieval.search import _embed
        return _embed.encode(text, normalize_embeddings=True).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from app.retrieval.search import _embed
        vecs = _embed.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


# ── Agent runner ──────────────────────────────────────────────────────────────

def run_agent_eval(question: str) -> dict:
    """
    Invoke the RAG graph directly (bypasses stateful.run Redis/PG side effects).

    Returns:
        answer   : str
        contexts : list[str]  — raw text of every retrieved chunk
    """
    from app.agents.graph.builder import rag_graph

    state = {
        "messages":           [],
        "question":           question,
        "session_id":         "",        # empty → skips Redis chunk-dedup in retriever
        "plan":               [],
        "retrieved_chunks":   [],
        "context_sufficient": False,
        "retrieval_gap":      "",
        "answer":             "",
        "sources":            [],
        "confidence":         0.0,
        "iteration_count":    0,
        "trajectory_id":      str(uuid.uuid4()),
        "retrieval_empty":    False,
        "stage_tokens":       {},
        "node_timings":       {},
    }
    config      = {"configurable": {"thread_id": f"eval:{uuid.uuid4()}"}}
    final_state = rag_graph.invoke(state, config=config)

    chunks   = final_state.get("retrieved_chunks", [])
    contexts = [c["text"] for c in chunks if c.get("text", "").strip()]

    return {
        "answer":   final_state.get("answer", ""),
        "contexts": contexts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Evaluate only the first N questions (0 = all)")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}")
        sys.exit(1)

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.sample > 0:
        dataset = dataset[: args.sample]

    print(f"Loaded {len(dataset)} questions from {DATASET_PATH}")
    print(f"LLM judge : {GROQ_MODEL}")
    print(f"Embeddings: BAAI/bge-base-en-v1.5 (project model)\n")

    # ── Step 1: Run agent on every question ───────────────────────────────────
    print("=" * 55)
    print("Step 1 — Running agent on all questions")
    print("=" * 55)

    rows = []
    for i, item in enumerate(dataset, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i}/{len(dataset)}] {q[:70]}...")
        result = run_agent_eval(q)
        rows.append({
            "question":      q,
            "ground_truth":  gt,
            "answer":        result["answer"],
            "contexts":      result["contexts"],
        })
        n_ctx = len(result["contexts"])
        print(f"          → {n_ctx} context chunk(s) retrieved, "
              f"answer length {len(result['answer'])} chars")

    # ── Step 2: Build RAGAS dataset ───────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("Step 2 — Building RAGAS EvaluationDataset")
    print("=" * 55)

    from ragas import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input      = r["question"],
            response        = r["answer"],
            retrieved_contexts = r["contexts"],
            reference       = r["ground_truth"],
        )
        for r in rows
    ]
    eval_dataset = EvaluationDataset(samples=samples)
    print(f"  {len(samples)} samples ready.")

    # ── Step 3: Configure RAGAS LLM + embeddings ──────────────────────────────
    print(f"\n{'=' * 55}")
    print("Step 3 — Configuring RAGAS (Groq judge + BGE embeddings)")
    print("=" * 55)

    from ragas.llms import LangchainLLMWrapper
    from langchain_groq import ChatGroq
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    groq_llm  = ChatGroq(
        model      = GROQ_MODEL,
        api_key    = os.getenv("GROQ_API_KEY"),
        max_tokens = 4096,   # 2048 was too low — RAGAS prompts are verbose
    )
    ragas_llm  = LangchainLLMWrapper(groq_llm)
    ragas_emb  = _BGEEmbeddings()

    metrics = [
        Faithfulness(llm=ragas_llm),
        # strictness=1 → sends n=1 per request; Groq rejects n > 1
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb, strictness=1),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]
    print("  Metrics : Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("Step 4 — Running RAGAS evaluation (LLM calls per question × 4 metrics)")
    print("=" * 55)

    from ragas import evaluate, RunConfig

    run_config = RunConfig(
        max_retries = 5,
        max_wait    = 120,   # seconds between retries
        timeout     = 180,   # seconds per LLM call
        max_workers = 2,     # limit concurrency — Groq rate-limits aggressive parallel calls
    )

    result = evaluate(
        dataset         = eval_dataset,
        metrics         = metrics,
        run_config      = run_config,
        raise_exceptions = False,   # log failures as NaN instead of crashing
    )
    scores_df = result.to_pandas()

    # ── Step 5: Print results ─────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("Results — per-question scores")
    print("=" * 55)

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    header = f"{'#':>3}  {'Faithfulness':>13}  {'Ans.Relevancy':>14}  {'Ctx.Precision':>14}  {'Ctx.Recall':>11}"
    print(header)
    print("-" * len(header))

    for idx, row_data in scores_df.iterrows():
        vals = [row_data.get(c, float("nan")) for c in metric_cols]
        print(f"  {idx + 1:>2}  "
              f"{vals[0]:>13.3f}  "
              f"{vals[1]:>14.3f}  "
              f"{vals[2]:>14.3f}  "
              f"{vals[3]:>11.3f}")

    print(f"\n{'Aggregate (mean)':>20}")
    print("-" * 55)
    import math
    for col in metric_cols:
        mean_val = scores_df[col].mean() if col in scores_df else float("nan")
        label    = col.replace("_", " ").title()
        if math.isnan(mean_val):
            print(f"  {label:<22}   NaN  (all calls failed — check Groq rate limits)")
        else:
            bar = "█" * int(mean_val * 20)
            print(f"  {label:<22} {mean_val:.3f}  {bar}")

    # ── Step 6: Save results ──────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"ragas_results_{ts}.json"

    output = {
        "run_at":    datetime.now().isoformat(),
        "model":     GROQ_MODEL,
        "n_questions": len(samples),
        "aggregate": {
            col: (None if math.isnan(v := float(scores_df[col].mean())) else round(v, 4))
            for col in metric_cols
            if col in scores_df
        },
        "per_question": [
            {
                "question":           r["question"],
                "answer":             r["answer"],
                "n_contexts":         len(r["contexts"]),
                **{
                    col: round(float(scores_df.loc[i, col]), 4)
                    for col in metric_cols
                    if col in scores_df
                },
            }
            for i, r in enumerate(rows)
        ],
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved → {output_path}")

    # ── Step 7: Optional Langfuse upload ─────────────────────────────────────
    lf_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    lf_sec = os.getenv("LANGFUSE_SECRET_KEY")
    lf_host = os.getenv("LANGFUSE_HOST")

    if lf_key and lf_sec and lf_host:
        try:
            from langfuse import Langfuse
            lf = Langfuse(public_key=lf_key, secret_key=lf_sec, host=lf_host)
            run_name = f"ragas_eval_{ts}"

            for i, r in enumerate(rows):
                trace = lf.trace(name=run_name, input=r["question"], output=r["answer"])
                for col in metric_cols:
                    if col in scores_df:
                        lf.score(
                            trace_id   = trace.id,
                            name       = col,
                            value      = float(scores_df.loc[i, col]),
                            data_type  = "NUMERIC",
                        )

            lf.flush()
            print(f"Langfuse scores uploaded — experiment: {run_name}")
        except Exception as exc:
            print(f"Langfuse upload failed (non-fatal): {exc}")
    else:
        print("Langfuse env vars not set — skipping upload.")


if __name__ == "__main__":
    main()
