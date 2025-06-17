# ReAlign Template Generator

## Setup

### Environment

Create a `.env` file with the following variables:

```env
# WatsonX LLM
WATSONX_URL=
WATSONX_APIKEY=
WATSONX_PROJECT_ID=

# OpenAI LLM
OPENAI_API_KEY=
```

### Dependencies

```bash
pip install uv
uv venv
uv sync
```

## UI

To activate the UI run: `uv run src/ui.py`

Open the UI: http://127.0.0.1:7860/