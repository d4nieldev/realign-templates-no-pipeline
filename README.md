# ReAlign Template Generator

## Setup

### Environment

Create a `.env` file with the following variables:

```env
# OpenAI LLM
OPENAI_API_KEY=
```

### Dependencies

Install dependencies:
```bash
pip install uv
uv venv
uv sync
```

## UI

To activate the UI run: `uv run src/ui.py`

Open the UI: http://127.0.0.1:7860/

> NOTE: Running the pipeline by clicking the "Re-Align" button will not work, because we did not yet release the code for the pipeline, but the code for running it exists as a preperation for the future.