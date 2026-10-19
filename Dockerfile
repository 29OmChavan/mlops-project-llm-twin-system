FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/var/tmp

WORKDIR /app

RUN mkdir -p /var/tmp && chmod 1777 /var/tmp

COPY setup.py README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn>=0.28.0 \
    pydantic>=2.6.0 \
    python-box>=7.1.1 \
    PyYAML>=6.0.1 \
    qdrant-client>=1.8.0 \
    rank-bm25>=0.2.2 \
    pymongo>=4.6.0 \
    mongomock>=4.1.2 \
    rouge-score>=0.1.2 \
    scikit-learn>=1.4.0 \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    httpx>=0.27.0 \
    -e .

COPY config/ ./config/
COPY params.yaml main.py app.py gradio_app.py ./
COPY artifacts/feature_store/ ./artifacts/feature_store/
COPY artifacts/training/instruction_dataset.json artifacts/training/val_instruction_dataset.json artifacts/training/evaluation_report.json ./artifacts/training/
COPY artifacts/inference/ ./artifacts/inference/

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
