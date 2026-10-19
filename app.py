"""
Production FastAPI Microservice for the LLM Twin System (Stage 3 Inference & LLMOps Pipeline).
Exposes:
- GET  /health     : Container & evaluation gate readiness probe
- POST /twin-chat  : Multi-Agent Advanced RAG + A/B Testing Router + SafetyCritic + Comet Opik trace
- POST /feedback   : User star rating (1-5) + automated DPO/SFT retraining queue ingestion
- GET  /telemetry  : Aggregated LLMOps telemetry, A/B variant metrics, and DPO queue size
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.llm_twin.config.configuration import ConfigurationManager
from src.llm_twin.components.inference_pipeline import InferencePipeline


app = FastAPI(
    title="Om Chavan LLM Twin — Production LLMOps API",
    description=(
        "3-Pipeline FTI Architecture serving Multi-Agent Advanced Hybrid RAG, "
        "PEFT LoRA Style Alignment, A/B Testing, DPO Feedback Loop, and Comet Opik Telemetry."
    ),
    version="1.0.0",
)

_INFERENCE_ENGINE: Optional[InferencePipeline] = None


def get_inference_engine() -> InferencePipeline:
    global _INFERENCE_ENGINE
    if _INFERENCE_ENGINE is None:
        cfg_mgr = ConfigurationManager()
        inf_cfg = cfg_mgr.get_inference_pipeline_config()
        feat_cfg = cfg_mgr.get_feature_pipeline_config()
        _INFERENCE_ENGINE = InferencePipeline(
            config=inf_cfg,
            chunks_path=feat_cfg.cleaned_chunks_path,
            embedding_dim=feat_cfg.embedding_dim,
        )
    return _INFERENCE_ENGINE


class TwinChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        examples=["How does the 3-Pipeline FTI architecture decouple MongoDB, Qdrant, and LoRA fine-tuning?"],
    )
    top_k: int = Field(default=3, ge=1, le=6)
    ab_variant: Optional[str] = Field(
        default=None,
        description="Optional override: Variant_A_Base_RAG or Variant_B_LLMTwin_LoRA_AdvRAG",
    )


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., examples=["opik-trace-9a8b7c6d5e4f"])
    query: str = Field(..., min_length=3)
    response: str = Field(..., min_length=3)
    rating: int = Field(..., ge=1, le=5)
    user_correction: Optional[str] = Field(default=None)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head><title>LLM Twin — Production LLMOps API</title></head>
      <body style="font-family: sans-serif; max-width: 820px; margin: 40px auto; line-height: 1.6;">
        <h1>Om Chavan — Production LLM Twin Microservice (v1.0.0)</h1>
        <p><strong>Architecture:</strong> 3-Pipeline FTI (Feature &rarr; Training &rarr; Inference) with MongoDB, Qdrant, PEFT LoRA, Multi-Agent RAG, and Comet Opik Telemetry.</p>
        <ul>
          <li><a href="/docs">Interactive OpenAPI Documentation (/docs)</a></li>
          <li><a href="/health">Health &amp; Evaluation Gate Probe (/health)</a></li>
          <li><a href="/telemetry">LLMOps Telemetry &amp; A/B Experiment Summary (/telemetry)</a></li>
        </ul>
      </body>
    </html>
    """


@app.get("/health")
def health_check() -> Dict[str, Any]:
    eval_path = Path("artifacts/training/evaluation_report.json")
    gate_ready = False
    if eval_path.exists():
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
            gate_ready = bool(data.get("evaluation_gate", {}).get("promotion_approved", False))
        except Exception:
            gate_ready = False

    return {
        "status": "healthy",
        "service": "llm-twin-inference-pipeline",
        "evaluation_gate_ready": gate_ready,
        "version": "1.0.0",
    }


@app.post("/twin-chat")
def twin_chat(req: TwinChatRequest) -> Dict[str, Any]:
    try:
        engine = get_inference_engine()
        return engine.generate_twin_response(
            query=req.query,
            top_k=req.top_k,
            variant=req.ab_variant,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    try:
        engine = get_inference_engine()
        return engine.record_user_feedback(
            trace_id=req.trace_id,
            query=req.query,
            response=req.response,
            rating=req.rating,
            user_correction=req.user_correction,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/telemetry")
def get_telemetry_summary() -> Dict[str, Any]:
    engine = get_inference_engine()
    traces: List[Dict[str, Any]] = []
    if engine.config.telemetry_log_path.exists():
        with open(engine.config.telemetry_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))

    dpo_queue = []
    if engine.config.feedback_store_path.exists():
        try:
            dpo_queue = json.loads(engine.config.feedback_store_path.read_text(encoding="utf-8"))
        except Exception:
            dpo_queue = []

    avg_latency = (
        round(sum(t.get("TotalLatencyMs", 0.0) for t in traces) / len(traces), 2) if traces else 0.0
    )
    avg_faithfulness = (
        round(sum(t.get("faithfulness_score", 0.0) for t in traces) / len(traces), 4) if traces else 0.0
    )

    return {
        "total_traces_logged": len(traces),
        "avg_latency_ms": avg_latency,
        "avg_faithfulness_score": avg_faithfulness,
        "dpo_feedback_records": len(dpo_queue),
        "recent_traces": traces[-5:],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
