# Remember Why

Remember Why is a local-first AI memory assistant that helps people rediscover forgotten context behind saved screenshots.

Instead of searching through hundreds of screenshots by hand, ask questions such as:

- “What did I save about MCP?”
- “What did I save about PyTorch?”
- “Why did I save MCP_Architecture.png?”

Remember Why retrieves relevant screenshot memories through semantic search. For “why did I save this?” questions, it combines temporal proximity with semantic relationships to provide clearly labeled possible context.

## The problem
## Built for a Friend

I built Remember Why for a friend who regularly saves screenshots while learning and working, but later has trouble remembering what they saved and why it mattered.

The goal was to turn those forgotten screenshots into a searchable personal memory instead of another folder of files.

The project started from a simple question:

> What if you could ask your own saved screenshots what you were interested in at the time?

People save screenshots and digital information because something matters to them at the time. Later, they may forget what they saved or the context around it.

## Why Does Open Innovation Matter?

>**Open innovation mattered here because the data being searched is personal.**

The core memory pipeline can run locally:

```text
Screenshot
   ↓
OCR
   ↓
Embedding
   ↓
FAISS
   ↓
Local LLM

## The solution

Remember Why makes saved screenshots searchable with natural language. When asked about a specific screenshot, it can show nearby memories and semantically related memories as possible context.

### Grounding rule

- Retrieved OCR, filenames, timestamps, and similarity scores are evidence.
- **Possible context is an inference, not a fact.**
- Remember Why does not claim to know the user's true intent unless that intent is explicitly stored in the evidence.

## Demo

Ask: **“What did I save about MCP?”**

The UI displays the actual retrieved screenshot together with its filename, timestamp, similarity when available, and OCR evidence. Related screenshots are available in expanders.

**Video demo:** https://youtu.be/8UyR7Y0YTQ8

The demo shows:

1. Asking Remember Why about saved MCP memories.
2. Retrieving relevant memories using semantic search.
3. Displaying the original saved screenshot.
4. Exploring related memories.
5. Showing possible context.
6. Observing the LangGraph agent execution through Sentry.

## Architecture


text
User
  ↓
Streamlit UI
  ↓
LangGraph Agent ─────→ Ollama (qwen3:4b)
  ↓
MCP Server
  ↓
FAISS semantic memory
  ↓
OCR + local embeddings


## Technology stack

- Python
- Streamlit
- LangGraph
- MCP Python SDK
- Ollama / qwen3:4b
- Tesseract OCR
- Sentence Transformers / all-MiniLM-L6-v2
- FAISS
- Sentry Agent Tracing

## Sentry Agent Tracing

Sentry provides observability into actual agent execution, including LangGraph agent spans, model operations, tool execution, and HTTP operations.

Sentry is optional. When SENTRY_DSN is configured, scrubbed telemetry is sent to Sentry. Prompt capture is disabled. Local screenshot files and the full local memory store are not sent as Sentry payloads.

## Privacy

Core screenshot OCR, embeddings, FAISS retrieval, the MCP server, and Ollama model execution run locally. When Sentry is enabled, scrubbed telemetry is sent to the configured Sentry project; this README does not claim that absolutely everything remains on-device.

## Setup

Create and activate a virtual environment, then install the project dependencies:


powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


Install the Tesseract OCR engine separately. If it is not on PATH or in a standard Windows location, set its executable path:


powershell
$env:TESSERACT_CMD = 'C:\Path\To\tesseract.exe'


Install and start Ollama, then ensure the default model is available locally:


powershell
ollama pull qwen3:4b


The app does not automatically pull Ollama models. Supported environment variables:

- TESSERACT_CMD: full path to the Tesseract executable.
- REMEMBER_WHY_MODEL: Ollama model name; defaults to qwen3:4b.
- SENTRY_DSN: optional Sentry project DSN. If unset, tracing is disabled.
- SENTRY_TRACES_SAMPLE_RATE: trace sample rate from 0.0 to 1.0; defaults to 1.0 when a DSN is configured.

Example PowerShell configuration:


powershell
$env:REMEMBER_WHY_MODEL = 'qwen3:4b'
$env:SENTRY_DSN = 'https://<public-key>@<organization>.ingest.sentry.io/<project-id>'
$env:SENTRY_TRACES_SAMPLE_RATE = '1.0'


## Usage

Place PNG, JPG, JPEG, or WEBP screenshots in data/screenshots/, then build the local index, search, ask a question, or launch the UI:


powershell
python main.py --build
python main.py --search "MCP"
python main.py --ask "What did I save about MCP?"
python -m streamlit run app/ui/streamlit_app.py


## Testing

Run the complete test suite with:


powershell
python -m unittest discover -s tests -v


**Verified result: 79 tests passed.** The agent tests mock model and MCP calls; they do not require Ollama to be running.

## Limitations and future directions

The current MVP focuses on local screenshot memories. Direct integration with private X or LinkedIn saved posts is not included and would require appropriate authenticated integrations.

## Hacktoberfest partner technology

**Sentry Agent Tracing** is the partner technology used for real agent observability across LangGraph, model, tool, and HTTP operations.  2-3 pionts are missing