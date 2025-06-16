# ReAlign Template Generator

## Setup

### Environment

Create a `.env` file with the following variables:

```env
WATSONX_URL=
WATSONX_APIKEY=
WATSONX_PROJECT_ID=
```

### Dependencies

```bash
pip install uv
uv venv
uv sync
```

## UI

To activate the UI run: `uv run src/ui.py`