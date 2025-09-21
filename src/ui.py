import os
import re
import json
import random
from pathlib import Path

import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from secknowledge2_partial import run

load_dotenv()

TASKS_PATH = Path("tasks")
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_PROMPT_FILE =  os.path.abspath(os.getcwd()) + "/format-gen-prompts/fewshot.txt"


assert "OPENAI_API_KEY" in os.environ, "Please set the `OPENAI_API_KEY` environment variable."
models = [m.id for m in OpenAI().models.list().data]

ds_to_task = {
    ds_path.name: sorted([task_path.stem for task_path in ds_path.iterdir() if task_path.is_file() and task_path.suffix == '.json'])
    for ds_path in TASKS_PATH.iterdir() if ds_path.is_dir()
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
    return f"<h2>Format Generation Prompt Preview</h2><div style='font-family: monospace; font-size: 1em; background-color: azure'>{highlighted}</div>"


def generate_template(model_id: str, prompt_path: str, dataset_description: str, examples_table: list[list]) -> str:
    if not model_id:
        raise gr.Error("⚠️ Model ID is required!")
    if not prompt_path:
        raise gr.Error("⚠️ Prompt file is required!")
    if not dataset_description:
        raise gr.Error("⚠️ Dataset description is required!")
    
    examples_questions = table_to_markdown(examples_table)
    if examples_questions == "No examples provided." and 'fewshot' in prompt_path:
        raise gr.Error("⚠️ Please tick at least one example for fewshot!")
    
    prompt_template = Path(prompt_path).read_text()
    prompt = prompt_template.format(
        dataset_description=dataset_description,
        example_questions=examples_questions
    )
    
    response = OpenAI().responses.create(
        model=model_id,
        instructions="You are a helpful assistant.",
        input=prompt,
        tools=[{"type": "web_search_preview"}]
    ).output_text

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


def run_secknowledge_2(
        dataset: str,
        task: str,
        task_description: str,
        template: str,
        instruction: str,
        response: str,
        is_search: bool,
        grounding_doc: str | None,
        max_queries: int,
        limit: int,
        summarize: bool,
        model_id: str
    ) -> tuple[str, str, float, list[str] | None, dict, dict]:
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

    output = run(
        question=instruction,
        answer=response,
        format=template,
        is_search=is_search,
        max_queries=max_queries,
        limit=limit,
        summarize=summarize,
        grounding_doc=grounding_doc or "",
        model_id=model_id
    )

    search_results = [
        f"## Search Query: {query}\n\n{results}"
        for query, results in output['search_results'].items()
    ] if output.get('search_results') else ["No search results."]

    return (
        output['rewritten_answer'],
        output['judge']['readability'],
        output['judge']['factuality'],
        search_results,
        gr.update(minimum=0, maximum=len(search_results)-1, value=0),
        gr.update(value=search_results[0]),
    )


def sample_examples(dataset: str, task: str, n: int) -> list[list]:
    """
    Return an n-row table ready for gr.Dataframe:
    [selected?, instruction, response]
    """
    file_path = TASKS_PATH / dataset / f"{task}.json"
    items = json.loads(file_path.read_text())
    sample = random.sample(items, min(n, len(items)))
    return [[True, ex["instruction"], ex["output"]] for ex in sample]   # checkbox off by default


def deselect_all(df):
    """Toggle all 'selected' boxes to True."""
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        if not df.empty:
            df.iloc[:, 0] = False          # first col is the checkbox
        return df

    # Fallback: plain list-of-lists
    return [[False, *row[1:]] for row in df]


def table_to_markdown(table) -> str:
    # Convert to a uniform list-of-lists first
    if isinstance(table, pd.DataFrame):
        rows = table.values.tolist()
    else:
        rows = table

    picked = [r[1] for r in rows if r[0]]
    if not picked:
        return "No examples provided."
    return "\n\n--\n\n".join(picked)


def first_checked(table):
    """
    Return (instruction, response) from the first row whose
    checkbox == True.  Works whether `table` is a list-of-lists
    or a pandas.DataFrame (Gradio sends a DataFrame once the
    user edits it).  Raises gr.Error if nothing is checked.
    """
    # normalise → list-of-lists
    rows = table.values.tolist() if isinstance(table, pd.DataFrame) else table

    for checked, instr, resp in rows:
        if checked:
            return instr, resp
    raise gr.Error("⚠️ Please tick at least one example before running SecKnowledge 2.0.")


def push_first_selected(table):
    try:
        return first_checked(table)
    except gr.Error:
        return "", "" 



with gr.Blocks() as demo:
    gr.Markdown("# Format Generation Framework")

    gr.Markdown("## Data Exploration & Example Selection")
    with gr.Row():
        dataset = gr.Dropdown(choices=sorted(ds_to_task.keys()), label="Category", value=sorted(ds_to_task.keys())[0])
        task = gr.Dropdown(choices=ds_to_task[dataset.value], label="Task", value=ds_to_task[dataset.value][0], interactive=True)
    with gr.Row():
        num_examples = gr.Slider(0, 10, step=1, value=1, label="Number of examples to sample randomally")
        sample_btn = gr.Button("Sample")
    
    examples_df = gr.Dataframe(
            headers=["Select", "Instruction", "Response"],
            datatype=["bool", "str", "str"],
            interactive=True,
            wrap=True,
            label="Example Pool")

    deselect_all_btn = gr.Button("Deselect all") 
    
    gr.Markdown("## Format Generation")
    with gr.Row():
        model_id = gr.Dropdown(choices=models, label="Model ID", value=DEFAULT_MODEL)
        prompt_file = gr.FileExplorer(label="Prompt Template", root_dir="format-gen-prompts", file_count="single", value=lambda: DEFAULT_PROMPT_FILE)
    prompt_preview = gr.HTML(label="Prompt Template Preview (placeholders highlighted)", value=preview_prompt(DEFAULT_PROMPT_FILE))
    with gr.Row():
        task_description = gr.Textbox(label="Task Description", placeholder="Short description of the task and the dataset...", show_copy_button=True)
        template = gr.Textbox(label="Generated Format", show_copy_button=True)
    with gr.Row():
        generate_btn = gr.Button("Generate Format")

    gr.Markdown("## Evaluation Through Pipeline Execution (Without Search)")
    with gr.Row():
        instruction = gr.Textbox(label="Instruction", placeholder="Instruction for SecKnowledge 2.0...", show_copy_button=True)
        response = gr.Textbox(label="Original Response", placeholder="Response for SecKnowledge 2.0...", show_copy_button=True)
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                is_search = gr.Checkbox(label="Use Search", value=False, info="Whether to use search to ground the response")
                with gr.Row():
                    with gr.Column(scale=2):
                        max_queries = gr.Slider(1, 10, value=2, step=1,
                                                label="Max queries / instruction",
                                                visible=False)
                    with gr.Column(scale=2):
                        limit_slider = gr.Slider(1, 10, value=2, step=1,
                                                label="Results per query (limit)",
                                                visible=False)
                    with gr.Column(scale=1, min_width=0):
                        summarize_checkbox = gr.Checkbox(label="Summarize Search Results with LLM",
                                                        value=False,
                                                        visible=False)
                grounding_doc = gr.Textbox(label="Grounding Document (optional)", placeholder="Document to ground the response...")
            run_pipeline_btn = gr.Button("Run SecKnowledge 2.0", variant="primary")
        with gr.Column(scale=1):
            rewritten_response = gr.Textbox(label="Rewritten Response", show_copy_button=True)
            preferred = gr.Textbox(label="Preferred Answer")
            factuality_score = gr.Slider(1, 10, step=1, label="Factuality Score")
            search_results_slider = gr.Slider(0, 0, step=1, label="Search Query Index")
            search_result = gr.Markdown(label="Search Results", show_copy_button=True)
            search_results_state = gr.State([])
    
    # When the prompt_file changes, show the preview
    prompt_file.change(preview_prompt, inputs=prompt_file, outputs=prompt_preview)

    # When user clicks the button, generate output
    generate_btn.click(
        generate_template,
        inputs=[model_id, prompt_file, task_description, examples_df],
        outputs=template
    )

    # When the dataset changes, update the task dropdown
    dataset.change(update_tasks, inputs=dataset, outputs=task)

    # When the dataset and task are selected, get a random example
    sample_btn.click(
        sample_examples,
        inputs=[dataset, task, num_examples],
        outputs=examples_df
    )

    # one-tap “select all”
    deselect_all_btn.click(
        deselect_all, inputs=examples_df, outputs=examples_df
    )

    examples_df.change(
        push_first_selected,
        inputs=examples_df,
        outputs=[instruction, response]
    )

    run_pipeline_btn.click(
        run_secknowledge_2,
        inputs=[
            dataset, task, task_description, template,
            instruction, response,
            is_search, grounding_doc, max_queries, limit_slider, summarize_checkbox,
            model_id
        ],
        outputs=[
            rewritten_response, preferred, factuality_score,
            search_results_state, search_results_slider, search_result
        ]
    )
    def my_task(idx, results):
        return results[idx]
    # When the search results slider changes, update the search result display
    search_results_slider.change(
        my_task,
        inputs=[search_results_slider, search_results_state],
        outputs=[search_result]
    )

    def _toggle_search_opts(use_search: bool):
        return (
            gr.update(visible=use_search),   # max_queries
            gr.update(visible=use_search),   # limit_slider
            gr.update(visible=use_search),   # summarize_checkbox
        )

    # wire it up
    is_search.change(
        _toggle_search_opts,
        inputs=is_search,
        outputs=[max_queries, limit_slider, summarize_checkbox]
    )

demo.launch()
