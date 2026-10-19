# Production Handoff Runbook (HO-004): Operating the 3-Pipeline LLM Twin System

- **System:** `mlops-project-4-llm-twin-system`
- **Container Image:** `29omchavan/mlops-project-4-llm-twin-system:latest`
- **Model Registry:** `https://huggingface.co/29Oms/mlops-project-4-llm-twin-adapter`
- **Experiment Tracking:** CometML (`mlops-project-4-llm-twin-system`) & DagsHub MLflow (`29omchavan/mlops-project-4-llm-twin-system`)

---

## 1. Quick-Start Operations

### Run Full 3-Pipeline FTI Orchestrator (Feature $\rightarrow$ Training $\rightarrow$ Inference)
```bash
python main.py
```

### Run Unit & Integration Tests + Linter
```bash
pytest tests/ -v
flake8 src/ app.py tests/ --max-line-length=120 --ignore=E501,W503,E402,E203
```

### Run Production Docker Container
```bash
docker pull 29omchavan/mlops-project-4-llm-twin-system:latest
docker run -d --name llm-twin-api -p 8080:8080 29omchavan/mlops-project-4-llm-twin-system:latest
```

---

## 2. API Endpoints & SLA Contract

| Endpoint | Method | Purpose | Expected SLA |
| :--- | :---: | :--- | :---: |
| `/health` | `GET` | Verifies service liveness and CI/CD evaluation gate status (`evaluation_gate_ready: true`) | `< 5 ms` |
| `/twin-chat` | `POST` | Runs `RouterAgent` $\rightarrow$ `ResearcherAgent` ($k=3$ Query Expansion + Self-Query + Qdrant/BM25 RRF) $\rightarrow$ `TwinWriterAgent` $\rightarrow$ `SafetyCriticAgent` | `< 25 ms` |
| `/feedback` | `POST` | Logs user star rating (`1..5`) and promotes `rating >= 4` into `artifacts/inference/user_feedback_dpo_queue.json` | `< 10 ms` |
| `/telemetry` | `GET` | Returns aggregated Comet Opik trace count, average latency, faithfulness score, and DPO queue depth | `< 10 ms` |

---

## 3. Retraining & DPO Promotion Procedure
1. Inspect accumulated high-rated user feedback in `artifacts/inference/user_feedback_dpo_queue.json` via `GET /telemetry`.
2. When `dpo_feedback_records >= 25`, trigger `python main.py` (or the GitHub Actions workflow) to merge approved feedback triplets into `artifacts/training/instruction_dataset.json`, fine-tune a new LoRA adapter, verify the evaluation gate (`ROUGE-L >= 0.30` and `Style Alignment >= 0.70`), and push the updated weights to Hugging Face Hub.
