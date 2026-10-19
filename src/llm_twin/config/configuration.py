from pathlib import Path
from src.llm_twin.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.llm_twin.utils.common import read_yaml, create_directories
from src.llm_twin.entity.config_entity import (
    FeaturePipelineConfig,
    TrainingPipelineConfig,
    InferencePipelineConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_feature_pipeline_config(self) -> FeaturePipelineConfig:
        cfg = self.config.feature_pipeline
        create_directories([cfg.root_dir])
        return FeaturePipelineConfig(
            root_dir=Path(cfg.root_dir),
            raw_warehouse_path=Path(cfg.raw_warehouse_path),
            cleaned_chunks_path=Path(cfg.cleaned_chunks_path),
            qdrant_path=Path(cfg.qdrant_path),
            collection_name=str(cfg.collection_name),
            embedding_dim=int(cfg.embedding_dim),
            chunk_size=int(cfg.chunk_size),
            chunk_overlap=int(cfg.chunk_overlap),
        )

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        cfg = self.config.training_pipeline
        p = self.params.lora_finetuning
        create_directories([cfg.root_dir])
        return TrainingPipelineConfig(
            root_dir=Path(cfg.root_dir),
            instruction_dataset_path=Path(cfg.instruction_dataset_path),
            val_dataset_path=Path(cfg.val_dataset_path),
            base_model_name=str(cfg.base_model_name),
            adapter_dir=Path(cfg.adapter_dir),
            metrics_path=Path(cfg.metrics_path),
            hf_repo_id=str(cfg.hf_repo_id),
            r=int(p.r),
            lora_alpha=int(p.lora_alpha),
            lora_dropout=float(p.lora_dropout),
            target_modules=list(p.target_modules),
            learning_rate=float(p.learning_rate),
            num_train_epochs=int(p.num_train_epochs),
            per_device_train_batch_size=int(p.per_device_train_batch_size),
            max_source_length=int(p.max_source_length),
            max_target_length=int(p.max_target_length),
            seed=int(p.seed),
            min_rouge_l_gate=float(p.min_rouge_l_gate),
            min_style_alignment_gate=float(p.min_style_alignment_gate),
        )

    def get_inference_pipeline_config(self) -> InferencePipelineConfig:
        cfg = self.config.inference_pipeline
        rag_p = self.params.advanced_rag
        ab_p = self.params.ab_testing
        create_directories([cfg.root_dir])
        return InferencePipelineConfig(
            root_dir=Path(cfg.root_dir),
            telemetry_log_path=Path(cfg.telemetry_log_path),
            ab_test_log_path=Path(cfg.ab_test_log_path),
            feedback_store_path=Path(cfg.feedback_store_path),
            rag_benchmark_path=Path(cfg.rag_benchmark_path),
            top_k=int(rag_p.top_k),
            num_query_expansions=int(rag_p.num_query_expansions),
            rrf_k=int(rag_p.rrf_k),
            rerank_alpha_dense=float(rag_p.rerank_alpha_dense),
            rerank_beta_lexical=float(rag_p.rerank_beta_lexical),
            variant_b_traffic_split=float(ab_p.variant_b_traffic_split),
            min_feedback_rating_for_dpo=int(ab_p.min_feedback_rating_for_dpo),
        )
