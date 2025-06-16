import os
import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()

creds = Credentials(url=os.getenv("WATSONX_URL"), api_key=os.getenv("WATSONX_APIKEY"))
client = APIClient(credentials=creds, project_id=os.getenv("WATSONX_PROJECT_ID"))

models = [e.value for e in client.foundation_models.TextModels]

DEFAULT_MODEL = "mistralai/mistral-medium-2505"
DEFAULT_PROMPT_FILE =  "prompts/default.txt"


def highlight_placeholders(prompt_text):
    # Highlight {placeholders} in yellow
    return re.sub(r"(\{[^}]+\})", r'<span style="background: #fffa90; color: #6a4a00;">\1</span>', prompt_text)

def preview_prompt(prompt_file):
    if prompt_file is None:
        return ""
    prompt_text = Path(prompt_file).read_text()
    highlighted = highlight_placeholders(prompt_text)
    # Make line breaks visible in HTML
    highlighted = highlighted.replace('\n', '<br>')
    return f"<div style='font-family: monospace; font-size: 1em;'>{highlighted}</div>"


def generate_template(model_id: str, prompt_path: str, dataset_description: str, example_question: str) -> str:
    if not model_id:
        raise gr.Error("⚠️ Model ID is required!")
    if not prompt_path:
        raise gr.Error("⚠️ Prompt file is required!")
    if not dataset_description:
        raise gr.Error("⚠️ Dataset description is required!")
    if not example_question:
        raise gr.Error("⚠️ Example question is required!")
    
    model = ModelInference(
        model_id=model_id,
        credentials=creds,
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )

    prompt_template = Path(prompt_path).read_text()
    prompt = prompt_template.format(
        dataset_description=dataset_description,
        example_question=example_question
    )

    return model.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        params={"max_tokens": 4096}
    )['choices'][0]['message']['content']


with gr.Blocks() as demo:
    gr.Markdown("# WatsonX Prompt Generator with Template Preview")
    with gr.Row():
        model_id = gr.Dropdown(choices=models, label="Model ID", value=DEFAULT_MODEL)
        prompt_file = gr.FileExplorer(label="Prompt Template", root_dir="prompts", file_count="single", value=lambda: DEFAULT_PROMPT_FILE)
    prompt_preview = gr.HTML(label="Prompt Template Preview (placeholders highlighted)", value=preview_prompt(DEFAULT_PROMPT_FILE))
    with gr.Row():
        with gr.Column(scale=1):
            dataset_description = gr.Textbox(label="Task Description", placeholder="Short description of the task and the dataset...")
            example_question = gr.Textbox(label="Example Question", placeholder="An example question to guide the model...")
        with gr.Column(scale=1):
            output = gr.Textbox(label="Model Output", show_copy_button=True)
    
    # When the prompt_file changes, show the preview
    prompt_file.change(preview_prompt, inputs=prompt_file, outputs=prompt_preview)

    # When user clicks the button, generate output
    generate_btn = gr.Button("Generate")
    generate_btn.click(
        generate_template,
        inputs=[model_id, prompt_file, dataset_description, example_question],
        outputs=output
    )

demo.launch()
