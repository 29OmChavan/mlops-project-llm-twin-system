from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class FeaturePipelineConfig:
    root_dir: Path
    raw_warehouse_path: Path
    cleaned_chunks_path: Path
    qdrant_path: Path
    collection_name: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class TrainingPipelineConfig:
    root_dir: Path
    instruction_dataset_path: Path
    val_dataset_path: Path
    base_model_name: str
    adapter_dir: Path
    metrics_path: Path
    hf_repo_id: str
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    max_source_length: int
    max_target_length: int
    seed: int
    min_rouge_l_gate: float
    min_style_alignment_gate: float


@dataclass(frozen=True)
class InferencePipelineConfig:
    root_dir: Path
    telemetry_log_path: Path
    ab_test_log_path: Path
    feedback_store_path: Path
    rag_benchmark_path: Path
    top_k: int
    num_query_expansions: int
    rrf_k: int
    rerank_alpha_dense: float
    rerank_beta_lexical: float
    variant_b_traffic_split: float
    min_feedback_rating_for_dpo: int
