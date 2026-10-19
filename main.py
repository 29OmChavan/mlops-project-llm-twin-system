"""
Master Orchestrator for the 3-Pipeline FTI (Feature -> Training -> Inference) LLM Twin System.
"""
from src.llm_twin import logger
from src.llm_twin.config.configuration import ConfigurationManager
from src.llm_twin.components.feature_pipeline import FeaturePipeline
from src.llm_twin.components.training_pipeline import TrainingPipeline
from src.llm_twin.components.inference_pipeline import InferencePipeline


def run_all_pipelines():
    config_manager = ConfigurationManager()

    logger.info(">>>>>> Stage 01: Feature Pipeline (MongoDB ETL -> Cleaning/Chunking -> Qdrant) started <<<<<<")
    feature_cfg = config_manager.get_feature_pipeline_config()
    feature_pipeline = FeaturePipeline(config=feature_cfg)
    feature_summary = feature_pipeline.run()
    logger.info(f"Feature Pipeline Summary: {feature_summary}")
    logger.info(">>>>>> Stage 01: Feature Pipeline completed <<<<<<\n==========================================")

    logger.info(">>>>>> Stage 02: Training Pipeline (Instruction Distillation -> PEFT LoRA -> CometML/MLflow/HF) started <<<<<<")
    training_cfg = config_manager.get_training_pipeline_config()
    training_pipeline = TrainingPipeline(config=training_cfg)
    training_report = training_pipeline.run(chunks_path=str(feature_cfg.cleaned_chunks_path))
    logger.info(
        f"Training Pipeline Summary: ROUGE-L={training_report['lora_twin_metrics']['rougeL']}, "
        f"StyleAlignment={training_report['lora_twin_metrics']['style_alignment_score']}, "
        f"Gate={training_report['evaluation_gate']['promotion_approved']}"
    )
    logger.info(">>>>>> Stage 02: Training Pipeline completed <<<<<<\n==========================================")

    logger.info(">>>>>> Stage 03: Inference & LLMOps Monitoring Pipeline (Multi-Agent RAG + A/B + DPO + Opik) started <<<<<<")
    inference_cfg = config_manager.get_inference_pipeline_config()
    inference_pipeline = InferencePipeline(
        config=inference_cfg,
        chunks_path=feature_cfg.cleaned_chunks_path,
        embedding_dim=feature_cfg.embedding_dim,
    )
    rag_benchmark = inference_pipeline.evaluate_rag_benchmark()
    sample_trace = inference_pipeline.generate_twin_response(
        query="How does our 3-Pipeline FTI architecture decouple MongoDB, Qdrant, and LoRA fine-tuning?",
        top_k=3,
        variant="Variant_B_LLMTwin_LoRA_AdvRAG",
    )
    fb_summary = inference_pipeline.record_user_feedback(
        trace_id=sample_trace["trace_id"],
        query=sample_trace["query"],
        response=sample_trace["answer"],
        rating=5,
    )
    logger.info(f"Advanced RAG Benchmark: {rag_benchmark}")
    logger.info(f"Sample Opik Trace ID: {sample_trace['trace_id']} | DPO Queue Summary: {fb_summary}")
    logger.info(">>>>>> Stage 03: Inference & LLMOps Monitoring Pipeline completed <<<<<<\n==========================================")


if __name__ == "__main__":
    run_all_pipelines()
