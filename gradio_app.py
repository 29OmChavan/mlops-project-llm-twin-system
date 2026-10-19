"""
Interactive Gradio Playground UI for Om Chavan's LLM Twin System.
Allows testing Multi-Agent Advanced RAG, A/B Variant comparison, and DPO Feedback submission.
"""
from src.llm_twin.config.configuration import ConfigurationManager
from src.llm_twin.components.inference_pipeline import InferencePipeline


def build_interface():
    cfg_mgr = ConfigurationManager()
    engine = InferencePipeline(
        config=cfg_mgr.get_inference_pipeline_config(),
        chunks_path=cfg_mgr.get_feature_pipeline_config().cleaned_chunks_path,
    )

    def chat_fn(query: str, variant: str, top_k: int):
        res = engine.generate_twin_response(query=query, top_k=int(top_k), variant=variant)
        citations_str = ", ".join(res["citations"])
        meta = (
            f"Trace ID: {res['trace_id']} | Intent: {res['intent']} | "
            f"Faithfulness: {res['faithfulness_score']} | Style Alignment: {res['style_alignment_score']} | "
            f"Latency: {res['TotalLatencyMs']} ms"
        )
        return res["answer"], citations_str, meta

    try:
        import gradio as gr

        with gr.Blocks(title="Om Chavan — Production LLM Twin") as demo:
            gr.Markdown("# Om Chavan — Production LLM Twin & Multi-Agent RAG Playground")
            with gr.Row():
                query_box = gr.Textbox(
                    label="Ask Om Chavan's LLM Twin",
                    value="How does the 3-Pipeline FTI architecture decouple MongoDB, Qdrant, and LoRA?",
                )
                variant_dd = gr.Dropdown(
                    choices=["Variant_B_LLMTwin_LoRA_AdvRAG", "Variant_A_Base_RAG"],
                    value="Variant_B_LLMTwin_LoRA_AdvRAG",
                    label="A/B Serving Variant",
                )
                topk_slider = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Top-K Chunks")
            submit_btn = gr.Button("Generate Grounded Twin Response")
            answer_out = gr.Textbox(label="LLM Twin Response", lines=5)
            citations_out = gr.Textbox(label="Qdrant Citations")
            meta_out = gr.Textbox(label="Comet Opik Telemetry")
            submit_btn.click(
                fn=chat_fn,
                inputs=[query_box, variant_dd, topk_slider],
                outputs=[answer_out, citations_out, meta_out],
            )
        return demo
    except ImportError:
        return None


if __name__ == "__main__":
    ui = build_interface()
    if ui is not None:
        ui.launch(server_name="0.0.0.0", server_port=7860)
