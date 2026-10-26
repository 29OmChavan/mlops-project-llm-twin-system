"""
Production FastAPI Microservice & Multi-Page Dynamic Web Portal for the LLM Twin System (v2.0).
Exposes:
- GET  /                  : Multi-Agent LLM Twin Studio & Persona Composer
- GET  /intro             : "What This Project Is & How I Developed It" Engineering Walkthrough
- GET  /warehouse         : MongoDB ODM Digital Data Warehouse & Live Qdrant Document Ingestion
- GET  /llama-lab         : Interactive PyTorch LLaMA & Mistral Architecture Lab (RMSNorm, RoPE, GQA, SwiGLU)
- GET  /training-hub      : Instruction Dataset Distillation, PEFT LoRA & Registry Explorer
- GET  /health            : Health & Evaluation Gate Probe (Rich HTML in browser, JSON for API/TestClient)
- GET  /telemetry         : Comet Opik Telemetry & DPO Queue Dashboard (Rich HTML in browser, JSON for API)
- POST /twin-chat         : Multi-Agent Advanced RAG + A/B Router + SafetyCritic + Comet Opik trace
- POST /ab-duel           : Side-by-side Variant A vs. Variant B duel comparison
- POST /ingest-document   : Live MongoDB ODM + Qdrant + BM25 custom document ingestion
- POST /llama-diagnostics : Live PyTorch LLaMA/Mistral tensor & KV-cache benchmark
- POST /feedback          : User star rating (1-5) + automated DPO/SFT retraining queue ingestion
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.llm_twin.config.configuration import ConfigurationManager
from src.llm_twin.components.inference_pipeline import InferencePipeline
from src.llm_twin.utils.common import load_json


app = FastAPI(
    title="llm-twin-fti-pipeline (MongoDB + Qdrant + LoRA)",
    description=(
        "3-Pipeline Feature/Training/Inference service for personal code and article search, "
        "LLaMA/Mistral layer checks, LoRA fine-tuning, and Opik request logging."
    ),
    version="2.0.0",
    docs_url="/swagger-raw",
)

templates = Jinja2Templates(directory="templates")
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
    output_format: str = Field(
        default="technical_answer",
        description="Persona format: technical_answer | linkedin_post | medium_article_section | github_adr_review",
    )


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., examples=["opik-trace-9a8b7c6d5e4f"])
    query: str = Field(..., min_length=3)
    response: str = Field(..., min_length=3)
    rating: int = Field(..., ge=1, le=5)
    user_correction: Optional[str] = Field(default=None)


class DocumentIngestRequest(BaseModel):
    platform: str = Field(default="github", description="github | medium | linkedin")
    category: str = Field(default="advanced_rag")
    title: str = Field(..., min_length=3)
    content: str = Field(..., min_length=10)
    link: str = Field(default="https://github.com/29OmChavan/mlops-project-4-llm-twin-system")


class LLaMADiagnosticsRequest(BaseModel):
    dim: int = Field(default=64, ge=16, le=256)
    n_heads: int = Field(default=8, ge=2, le=16)
    n_kv_heads: int = Field(default=2, ge=1, le=16)
    seq_len: int = Field(default=16, ge=4, le=64)
    sliding_window: int = Field(default=4, ge=2, le=32)


def _wants_html(request: Request, format_param: Optional[str] = None) -> bool:
    if format_param == "json":
        return False
    if format_param == "html":
        return True
    accept = request.headers.get("accept", "").lower()
    return "text/html" in accept


def _build_dashboard_context(active_page: str = "studio") -> Dict[str, Any]:
    engine = get_inference_engine()

    eval_path = Path("artifacts/training/evaluation_report.json")
    eval_report = load_json(eval_path) if eval_path.exists() else {}

    rag_bench_path = Path("artifacts/inference/advanced_rag_benchmark.json")
    rag_benchmark = load_json(rag_bench_path) if rag_bench_path.exists() else {}

    warehouse_path = Path("artifacts/feature_store/mongodb_warehouse.json")
    warehouse_docs = load_json(warehouse_path) if warehouse_path.exists() else []

    inst_path = Path("artifacts/training/val_instruction_dataset.json")
    instructions = load_json(inst_path) if inst_path.exists() else []

    traces: List[Dict[str, Any]] = []
    if engine.config.telemetry_log_path.exists():
        with open(engine.config.telemetry_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        traces.append(json.loads(line))
                    except Exception:
                        pass

    dpo_queue: List[Dict[str, Any]] = []
    if engine.config.feedback_store_path.exists():
        try:
            dpo_queue = load_json(engine.config.feedback_store_path)
        except Exception:
            dpo_queue = []

    avg_latency = (
        round(sum(t.get("TotalLatencyMs", 0.0) for t in traces) / len(traces), 2) if traces else 0.0
    )
    avg_faithfulness = (
        round(sum(t.get("faithfulness_score", 0.0) for t in traces) / len(traces), 4) if traces else 0.0
    )
    avg_style = (
        round(sum(t.get("style_alignment_score", 0.0) for t in traces) / len(traces), 4) if traces else 0.0
    )

    telemetry_summary = {
        "total_traces_logged": len(traces),
        "avg_latency_ms": avg_latency,
        "avg_faithfulness_score": avg_faithfulness,
        "avg_style_alignment_score": avg_style,
        "dpo_feedback_records": len(dpo_queue),
        "recent_traces": traces[-8:][::-1],
        "dpo_queue": dpo_queue[-8:][::-1],
    }

    health_summary = {
        "status": "healthy",
        "service": "llm-twin-inference-pipeline",
        "evaluation_gate_ready": bool(
            eval_report.get("evaluation_gate", {}).get("promotion_approved", False)
        ),
        "qdrant_chunks_indexed": len(engine.chunks),
        "mongodb_documents_count": len(warehouse_docs),
        "version": "2.0.0",
    }

    return {
        "active_page": active_page,
        "health": health_summary,
        "telemetry": telemetry_summary,
        "eval_report": eval_report,
        "rag_benchmark": rag_benchmark,
        "warehouse_docs": warehouse_docs,
        "instructions": instructions,
        "chat_result": None,
        "duel_result": None,
        "llama_diag": None,
        "feedback_notice": None,
        "ingest_notice": None,
        "ops_diag": engine.get_pre2023_ops_diagnostics(),
        "initial_step": 1,
    }


# =====================================================================
# Multi-Page HTML Routes + Content-Negotiated /health & /telemetry
# =====================================================================
@app.get("/", response_class=HTMLResponse, tags=["UI Pages"])
def home_studio(request: Request):
    ctx = _build_dashboard_context(active_page="studio")
    # Provide an initial sample trace so the studio is immediately rich on load
    engine = get_inference_engine()
    ctx["chat_result"] = engine.generate_twin_response(
        query="How does the 3-Pipeline FTI architecture decouple MongoDB, Qdrant, and LoRA fine-tuning?",
        top_k=3,
        variant="Variant_B_LLMTwin_LoRA_AdvRAG",
        output_format="technical_answer",
    )
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/intro", response_class=HTMLResponse, tags=["UI Pages"])
def intro_page(request: Request):
    ctx = _build_dashboard_context(active_page="intro")
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/studio", response_class=HTMLResponse, tags=["UI Pages"])
def studio_page(request: Request):
    return home_studio(request)


@app.get("/warehouse", response_class=HTMLResponse, tags=["UI Pages"])
def warehouse_page(request: Request):
    ctx = _build_dashboard_context(active_page="warehouse")
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/llama-lab", response_class=HTMLResponse, tags=["UI Pages"])
def llama_lab_page(request: Request):
    ctx = _build_dashboard_context(active_page="llama_lab")
    engine = get_inference_engine()
    ctx["llama_diag"] = engine.run_llama_mistral_diagnostics(
        dim=64, n_heads=8, n_kv_heads=2, seq_len=16, sliding_window=4
    )
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/training-hub", response_class=HTMLResponse, tags=["UI Pages"])
def training_hub_page(request: Request):
    ctx = _build_dashboard_context(active_page="training")
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/help", response_class=HTMLResponse, tags=["UI Pages"])
def help_page(request: Request):
    ctx = _build_dashboard_context(active_page="help")
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/docs", response_class=HTMLResponse, tags=["UI Pages"])
def docs_page(request: Request):
    ctx = _build_dashboard_context(active_page="docs")
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/health", tags=["Operations"])
def health_check(request: Request, format: Optional[str] = None):
    ctx = _build_dashboard_context(active_page="telemetry")
    if _wants_html(request, format):
        return templates.TemplateResponse(request=request, name="index.html", context=ctx)
    return JSONResponse(content=ctx["health"])


@app.get("/telemetry", tags=["Operations"])
def get_telemetry_summary(request: Request, format: Optional[str] = None):
    ctx = _build_dashboard_context(active_page="telemetry")
    if _wants_html(request, format):
        return templates.TemplateResponse(request=request, name="index.html", context=ctx)
    return JSONResponse(
        content={
            "total_traces_logged": ctx["telemetry"]["total_traces_logged"],
            "avg_latency_ms": ctx["telemetry"]["avg_latency_ms"],
            "avg_faithfulness_score": ctx["telemetry"]["avg_faithfulness_score"],
            "dpo_feedback_records": ctx["telemetry"]["dpo_feedback_records"],
            "recent_traces": ctx["telemetry"]["recent_traces"][:5],
        }
    )


# =====================================================================
# Interactive Web Form Handlers
# =====================================================================
@app.post("/studio-action", response_class=HTMLResponse, tags=["UI Pages"])
def studio_form_action(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(3),
    mode_select: str = Form("Variant_B_LLMTwin_LoRA_AdvRAG"),
    output_format: str = Form("technical_answer"),
):
    engine = get_inference_engine()
    ctx = _build_dashboard_context(active_page="studio")
    if mode_select == "AB_DUEL":
        ctx["duel_result"] = engine.compare_ab_variants(
            query=query, top_k=int(top_k), output_format=output_format
        )
    else:
        ctx["chat_result"] = engine.generate_twin_response(
            query=query,
            top_k=int(top_k),
            variant=mode_select,
            output_format=output_format,
        )
    ctx["initial_step"] = 2
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.post("/feedback-form", response_class=HTMLResponse, tags=["UI Pages"])
def feedback_form_action(
    request: Request,
    trace_id: str = Form(...),
    query: str = Form(...),
    response: str = Form(...),
    rating: int = Form(5),
    user_correction: Optional[str] = Form(default=None),
):
    engine = get_inference_engine()
    fb = engine.record_user_feedback(
        trace_id=trace_id,
        query=query,
        response=response,
        rating=int(rating),
        user_correction=user_correction if user_correction and user_correction.strip() else None,
    )
    ctx = _build_dashboard_context(active_page="studio")
    ctx["feedback_notice"] = (
        f"Feedback {fb['feedback_entry']['feedback_id']} recorded (Rating={rating}/5). "
        f"Promoted to DPO Retraining Queue: {fb['feedback_entry']['promoted_to_dpo_retraining_queue']} "
        f"(Total DPO Queue Size: {fb['dpo_queue_size']})."
    )
    ctx["chat_result"] = engine.generate_twin_response(query=query, top_k=3)
    ctx["initial_step"] = 3
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.post("/warehouse-ingest-form", response_class=HTMLResponse, tags=["UI Pages"])
def warehouse_ingest_form(
    request: Request,
    platform: str = Form("github"),
    category: str = Form("advanced_rag"),
    title: str = Form(...),
    content: str = Form(...),
    link: str = Form("https://github.com/29OmChavan/mlops-project-4-llm-twin-system"),
):
    engine = get_inference_engine()
    res = engine.ingest_custom_document(
        platform=platform, category=category, title=title, content=content, link=link
    )
    ctx = _build_dashboard_context(active_page="warehouse")
    ctx["ingest_notice"] = (
        f"Successfully ingested '{title}' as {res['doc_id']} ({res['chunk_id']}) into MongoDB Warehouse "
        f"and Qdrant Vector DB! Total indexed chunks: {res['total_qdrant_chunks']}."
    )
    ctx["initial_step"] = 2
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.post("/llama-lab-form", response_class=HTMLResponse, tags=["UI Pages"])
def llama_lab_form(
    request: Request,
    dim: int = Form(64),
    n_heads: int = Form(8),
    n_kv_heads: int = Form(2),
    seq_len: int = Form(16),
    sliding_window: int = Form(4),
):
    engine = get_inference_engine()
    diag = engine.run_llama_mistral_diagnostics(
        dim=int(dim),
        n_heads=int(n_heads),
        n_kv_heads=int(n_kv_heads),
        seq_len=int(seq_len),
        sliding_window=int(sliding_window),
    )
    ctx = _build_dashboard_context(active_page="llama_lab")
    ctx["llama_diag"] = diag
    ctx["initial_step"] = 2
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


# =====================================================================
# JSON REST API Endpoints
# =====================================================================
@app.post("/twin-chat", tags=["Inference API"])
def twin_chat(req: TwinChatRequest) -> Dict[str, Any]:
    try:
        engine = get_inference_engine()
        return engine.generate_twin_response(
            query=req.query,
            top_k=req.top_k,
            variant=req.ab_variant,
            output_format=req.output_format,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ab-duel", tags=["Inference API"])
def ab_duel(req: TwinChatRequest) -> Dict[str, Any]:
    try:
        engine = get_inference_engine()
        return engine.compare_ab_variants(
            query=req.query, top_k=req.top_k, output_format=req.output_format
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ingest-document", tags=["Feature Pipeline API"])
def ingest_document_api(req: DocumentIngestRequest) -> Dict[str, Any]:
    try:
        engine = get_inference_engine()
        return engine.ingest_custom_document(
            platform=req.platform,
            category=req.category,
            title=req.title,
            content=req.content,
            link=req.link,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/llama-diagnostics", tags=["LLaMA & Mistral Lab API"])
def llama_diagnostics_api(req: LLaMADiagnosticsRequest) -> Dict[str, Any]:
    try:
        engine = get_inference_engine()
        return engine.run_llama_mistral_diagnostics(
            dim=req.dim,
            n_heads=req.n_heads,
            n_kv_heads=req.n_kv_heads,
            seq_len=req.seq_len,
            sliding_window=req.sliding_window,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/feedback", tags=["LLMOps Feedback & DPO"])
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


@app.get("/ops-diagnostics", tags=["Bytewax, Feast, Qdrant & Evidently Ops"])
def get_ops_diagnostics() -> Dict[str, Any]:
    """Pre-2023 production systems diagnostics: Bytewax dedup, Qdrant INT8 quantization, Feast registry, Evidently drift."""
    engine = get_inference_engine()
    return engine.get_pre2023_ops_diagnostics()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
