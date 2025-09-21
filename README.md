# Format Generation Framework UI

![Q&A example from SecKnowledge (green), and our improved answer (orange).](assets/example_without_task.png)

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

### Usage

The system operates in three main stages, with the **Format Generation** and **Evaluation** stages forming an iterative loop. Users can repeat this loop until they are satisfied with the resulting format. Below is a step-by-step guide:

#### 1. Data Exploration & Example Selection
- Select a **category** and **task** from the menus.  
- Use the sliding bar to select up to **10 examples** to sample randomly.  
- This stage allows you to:
  - Explore *SecKnowledge* by reviewing the range of questions and answers.  
  - Identify representative examples that will guide format generation and serve as pipeline inputs.  

#### 2. Format Generation
- Choose a **model** from the available options via the OpenAI API.  
- Select a suitable **prompt** to generate the initial candidate format.  
  - You can add your own prompt or modify existing ones in the `format-gen-prompts` directory.  
- Provide a **brief description of the task**.  
- (Optional) Select **relevant examples** from Stage 1 to support format generation.  
- Click **Generate Format** to create the candidate format.  

#### 3. Evaluation Through Pipeline Execution
- Choose any example to run through the pipeline (the first example is pre-filled by default).  
- Configure **hyper-parameters**, such as:  
  - Enable/disable web search.  
  - Set the number of queries and results per query.  
  - Decide whether to summarize retrieved content before including it in the rewriting context.  
- (Optional) Provide a **grounding document** to supplement or replace web search.  
- Run the pipeline by clicking **Run SecKnowledge 2.0**. The pipeline will execute using:  
  - The selected LLM.  
  - The generated (or edited) format.  
- The pipeline outputs:  
  - Rewritten responses.  
  - Quality assessment scores.  
  - Retrieved search results (if enabled).  

#### Iterative Workflow
Repeat **Format Generation** and **Evaluation** until the format meets your requirements. This iterative process ensures the system adapts to different tasks and produces high-quality results.

## Disclaimer

The pipeline as implemented here, is not the original **SecKnowledge 2.0**, it is merely a simplification of it. The full original pipeline supports additional configurations such as connecting to a vector store, more robust error handling, and smarter, more efficient utilization of the resources. We intend to release the full pipeline soon.