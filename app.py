"""Gradio UI for the AAF Doctrine Intelligence System."""
import json
import os
from datetime import datetime, timezone

import gradio as gr

from src import config
from src.rag_system import MilitaryIntelligenceSystem

bot = MilitaryIntelligenceSystem()
config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def app_fn(question: str) -> str:
    answer, citations = bot.ask(question)

    with config.LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "q": question,
            "a": answer,
            "sources": citations,
        }, ensure_ascii=False) + "\n")

    if citations:
        return f"{answer}\n\n---\n📚 BURIMET ZYRTARE: {', '.join(citations)}"
    return answer


CUSTOM_CSS = """
#output_box { background-color: #f0f4f0; border: 2px solid #2d5a27;
              border-radius: 8px; }
#output_box textarea { font-size: 16px !important; line-height: 1.6 !important;
                       color: #1a1a1a !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🇦🇱 Sistemi i Inteligjencës Doktrinore — AAF")
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(label="Pyetja juaj:",
                                    placeholder="Shkruani këtu...", lines=3)
            submit_btn = gr.Button("Analizo", variant="primary")
        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Përgjigja e Sistemit:",
                                     interactive=False, lines=12,
                                     elem_id="output_box")
    submit_btn.click(fn=app_fn, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
