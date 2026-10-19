# Changelog

All notable changes to **Project 04: Production LLM Twin System (`mlops-project-4-llm-twin-system`)** are documented in this file.

## [1.0.0] - 2026-10-19

### Added
- **First-Principles LLaMA & Mistral Architecture (`src/llm_twin/components/llama_mistral_from_scratch.py`)**:
  - Implemented `RMSNorm`, complex-plane Rotary Positional Embeddings (`RoPE`), `SwiGLUFeedForward`, `GroupedQueryAttention` (`GQA`), and Mistral Sliding Window Attention (`SWA`) in PyTorch.
- **Stage 1 — Feature Pipeline (`src/llm_twin/components/feature_pipeline.py`)**:
  - Implemented Pydantic ODM models (`RepositoryDocument`, `ArticleDocument`, `PostDocument`) and multi-source ETL crawlers (GitHub, Medium, LinkedIn) persisting to a MongoDB Document Warehouse (`mongomock` / `pymongo`) and indexing overlapping embedded chunks into **Qdrant**.
- **Stage 2 — Training Pipeline (`src/llm_twin/components/training_pipeline.py`)**:
  - Implemented Instruction Dataset Distillation, **PEFT LoRA** fine-tuning (`r=8, alpha=16`) on `google/flan-t5-small`, quantitative evaluation (**ROUGE-1/2/L** + **Twin Style Alignment Score**), CI/CD promotion gating, **CometML** & **DagsHub MLflow** tracking, and automated upload to **Hugging Face Hub** (`29Oms/mlops-project-4-llm-twin-adapter`).
- **Stage 3 — Inference & LLMOps Monitoring Pipeline (`src/llm_twin/components/inference_pipeline.py`)**:
  - Implemented Advanced Hybrid RAG ($k=3$ Query Expansion + Self-Query Metadata Extraction + Qdrant Dense + BM25 Lexical RRF + Cross-Feature Reranker), 4-stage Multi-Agent Orchestrator (`RouterAgent` $\rightarrow$ `ResearcherAgent` $\rightarrow$ `TwinWriterAgent` $\rightarrow$ `SafetyCriticAgent`), A/B Testing Router, User Feedback $\rightarrow$ DPO Retraining Queue, and Comet Opik trace logging.
- **Serving & CI/CD (`app.py`, `gradio_app.py`, `Dockerfile`, `.github/workflows/main.yaml`)**:
  - Deployed FastAPI microservice (`/health`, `/twin-chat`, `/feedback`, `/telemetry`), interactive Gradio UI, Docker container `29omchavan/mlops-project-4-llm-twin-system:latest`, and 5-test `pytest` suite.
