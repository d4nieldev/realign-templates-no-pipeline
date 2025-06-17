import os
import re
import json
import random
from pathlib import Path
import subprocess
import shlex

import gradio as gr
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from openai import OpenAI

load_dotenv()

TASKS_PATH = Path("tasks")
DEFAULT_MODEL = "mistralai/mistral-medium-2505"
DEFAULT_PROMPT_FILE =  os.path.abspath(os.getcwd()) + "/prompts/default.txt"


creds = Credentials(url=os.getenv("WATSONX_URL"), api_key=os.getenv("WATSONX_API_KEY"))
client = APIClient(credentials=creds, project_id=os.getenv("WATSONX_PROJECT_ID"))

models = ["openai/gpt-4.1"] + [e.value for e in client.foundation_models.TextModels]

ds_to_task = {
    ds_path.name: [task_path.stem for task_path in ds_path.iterdir()]
    for ds_path in TASKS_PATH.iterdir()
}

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
    return f"<h2>Template Generation Prompt Preview</h2><div style='font-family: monospace; font-size: 1em; background-color: azure'>{highlighted}</div>"


def generate_template(model_id: str, prompt_path: str, dataset_description: str, example_question: str) -> str:
    if not model_id:
        raise gr.Error("⚠️ Model ID is required!")
    if not prompt_path:
        raise gr.Error("⚠️ Prompt file is required!")
    if not dataset_description:
        raise gr.Error("⚠️ Dataset description is required!")
    if not example_question:
        raise gr.Error("⚠️ Example question is required!")
    
    prompt_template = Path(prompt_path).read_text()
    prompt = prompt_template.format(
        dataset_description=dataset_description,
        example_question=example_question
    )
    
    if model_id.startswith("openai/"):
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        response = client.responses.create(
            model=model_id.replace("openai/", ""),
            instructions="You are a helpful assistant.",
            input=prompt,
            tools=[{"type": "web_search_preview"}]
        ).output_text
    else:
        model = ModelInference(
            model_id=model_id,
            credentials=creds,
            project_id=os.getenv("WATSONX_PROJECT_ID")
        )
        response = model.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            params={"max_tokens": 4096}
        )['choices'][0]['message']['content']

    return response


def update_tasks(dataset_choice):
        tasks = ds_to_task[dataset_choice]
        return gr.update(choices=tasks, value=tasks[0])
    

def get_example(dataset: str, task: str) -> tuple[str, str]:
    file_path = TASKS_PATH / dataset / f"{task}.json"

    try:
        items = json.loads(file_path.read_text())
    except json.JSONDecodeError:
        raise gr.Error("⚠️ JSON file could not be parsed.")

    if not items:
        raise gr.Error("⚠️ No entries in this file.")

    ex = random.choice(items)
    instr = ex["instruction"]
    resp  = ex["output"]

    return instr, resp


def run_dgt(
        dataset: str,
        task: str,
        task_description: str,
        template: str,
        instruction: str,
        response: str,
        is_search: bool,
        grounding_doc: str | None
    ) -> tuple[str, bool, int, list[str] | None, dict]:
    if not dataset:
        raise gr.Error("⚠️ Dataset is required!")
    if not task:
        raise gr.Error("⚠️ Task is required!")
    if not template:
        raise gr.Error("⚠️ Template is required!")
    if not instruction:
        raise gr.Error("⚠️ Instruction is required!")
    if not response:
        raise gr.Error("⚠️ Response is required!")

    # Generate configuration
    config = """
task_name: UI
created_by: IBM Research
data_builder: realign
task_description: Reformats the responses of a general instruction dataset (the one used in the original paper of ReAlign) into a format that better aligns with pre-established criteria and collected evidence.
retriever:
  type: duckduckgo_search
  limit: 5
  process_webpages: True
  deduplicate_sources: True
  reorder_organic: True
seed_datastore:
  type: default
  data_path: ${DGT_DATA_DIR}/research/realign/example_data_ui.json
"""
    
    config_file_path = Path("fms-dgt/tasks/research/realign/ui/task.yaml")
    config_file_path.parent.mkdir(parents=True, exist_ok=True)
    config_file_path.write_text(config)

    # Example data
    data_path = Path("fms-dgt/data/research/realign/example_data_ui.json")
    data_example = {
        "instruction": instruction,
        "answer": response,
        "category": "UI",
        "subcategory": f'{dataset}:{task}',
    }
    if grounding_doc:
        data_example["grounding_doc"] = grounding_doc

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps([data_example], indent=2))

    # Template
    template_path = Path("fms-dgt/data/research/realign/templates/ui.json")
    template_data = {
        'name': 'UI',
        'subcategories': [{
            'name': f'{dataset}:{task}',
            'description': task_description,
            'structure': template,
            'requires_search': is_search,
            'requires_grounding_doc': bool(grounding_doc),
            'requires_rewrite': True
        }]
    }
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(json.dumps([template_data], indent=2))

    # Run the DGT command
    fms_dgt_path = Path("fms-dgt")
    cmd = "python3 -m fms_dgt.research --task-paths ./tasks/research/realign/ui --restart-generation --num-outputs-to-generate 1"
    with subprocess.Popen(
        shlex.split(cmd),
        cwd=fms_dgt_path,                        # run *as if* we had cd-ed
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=dict(os.environ, PYTHONBUFFERED="1")
    ) as proc:
        if proc.stdout is None:
            raise gr.Error("⚠️ Failed to run the DGT command. Please check the logs.")
        for line in proc.stdout:                    # stream live output
            print("\033[34m"+line, end="\033[0m", flush=True)
    
    exit_code = proc.wait()
    if exit_code != 0:
        raise gr.Error(f"⚠️ DGT command failed with exit code {exit_code}. Please check the logs.")

    output_path = Path("fms-dgt/output/UI/data.jsonl")
    dgt_output = json.loads(output_path.read_text())

    return (
        dgt_output['rewritten_answer'],
        True if dgt_output['judge_scores']['readability'] == 'rewritten' else False,
        dgt_output['judge_scores']['factuality'],
        dgt_output['search_results'],
        gr.update(minimum=0, maximum=len(dgt_output['search_results']) - 1 if dgt_output['search_results'] else 0, value=0)
    )



with gr.Blocks() as demo:
    gr.Markdown("# WatsonX Prompt Generator with Template Preview")

    gr.Markdown("## View Data")
    with gr.Row():
        dataset = gr.Dropdown(choices=list(ds_to_task.keys()), label="Dataset", value=list(ds_to_task.keys())[0])
        task = gr.Dropdown(choices=ds_to_task[dataset.value], label="Task", value=ds_to_task[dataset.value][0], interactive=True)
    with gr.Row():
        instruction  = gr.Textbox(label="Instruction", lines=4, interactive=False, show_copy_button=True)
        response   = gr.Textbox(label="Response",    lines=4, interactive=False, show_copy_button=True)
    with gr.Row():
        get_example_btn = gr.Button("Show random example")
    
    gr.Markdown("## Create Template")
    with gr.Row():
        model_id = gr.Dropdown(choices=models, label="Model ID", value=DEFAULT_MODEL)
        prompt_file = gr.FileExplorer(label="Prompt Template", root_dir="prompts", file_count="single", value=lambda: DEFAULT_PROMPT_FILE)
    prompt_preview = gr.HTML(label="Prompt Template Preview (placeholders highlighted)", value=preview_prompt(DEFAULT_PROMPT_FILE))
    with gr.Row():
        task_description = gr.Textbox(label="Task Description", placeholder="Short description of the task and the dataset...")
        template = gr.Textbox(label="Generated Template", show_copy_button=True)
    with gr.Row():
        generate_btn = gr.Button("Generate Template")

    gr.Markdown("## Run ReAlign Pipeline")
    with gr.Row():
        with gr.Column(scale=1):
            is_search = gr.Checkbox(label="Use Search", value=False, info="Whether to use search to ground the response.")
            grounding_doc = gr.Textbox(label="Grounding Document (optional)", placeholder="Document to ground the response...", lines=2)
            run_dgt_btn = gr.Button("RuAlign", variant="primary")
        with gr.Column(scale=1):
            realigned_response = gr.Textbox(label="Realigned Response")
            preferred = gr.Checkbox(label="Realign Preferred")
            factuality_score = gr.Slider(1, 10, step=1, label="Factuality Score")
            search_results_slider = gr.Slider(0, 0, step=1, label="Search Query Index")
            search_result = gr.Markdown(label="Search Results", show_copy_button=True)
            search_results_state = gr.State([])
    
    # When the prompt_file changes, show the preview
    prompt_file.change(preview_prompt, inputs=prompt_file, outputs=prompt_preview)

    # When user clicks the button, generate output
    generate_btn.click(
        generate_template,
        inputs=[model_id, prompt_file, task_description, instruction],
        outputs=template
    )

    # When the dataset changes, update the task dropdown
    dataset.change(update_tasks, inputs=dataset, outputs=task)

    # When the dataset and task are selected, get a random example
    get_example_btn.click(
        get_example,
        inputs=[dataset, task],
        outputs=[instruction, response]
    )

    # When the run_dgt_btn is clicked, run the DGT command
    run_dgt_btn.click(
        run_dgt,
        inputs=[dataset, task, task_description, template, instruction, response, is_search, grounding_doc],
        outputs=[realigned_response, preferred, factuality_score, search_results_state, search_results_slider]
    )
    def my_task(idx, results):
        return results[idx]
    # When the search results slider changes, update the search result display
    search_results_slider.change(
        my_task,
        inputs=[search_results_slider, search_results_state],
        outputs=[search_result]
    )

demo.launch()
