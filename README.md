# Remember Why

> **You saved it for a reason. I help you remember why.**

Remember Why is a local-first AI memory assistant that helps people rediscover forgotten context behind saved screenshots.

Instead of searching through hundreds of screenshots by hand, ask questions such as:

- "What did I save about MCP?"
- "What did I save about PyTorch?"
- "Why did I save MCP_Architecture.png?"

Remember Why retrieves relevant screenshot memories through semantic search. For "why did I save this?" questions, it combines temporal proximity with semantic relationships to provide clearly labeled possible context.

---

## The Problem

People save screenshots while learning, researching, or working because something seems useful at the time.

Weeks or months later, those screenshots become difficult to search through. You may remember that you saw something important, but not:

- what it was about
- when you saved it
- what other topics you were exploring
- why it seemed relevant at the time

The result is a folder full of information that is technically saved but practically forgotten.

---

## Built for a Friend

I built Remember Why for a friend who regularly saves screenshots while learning and working, but later has trouble remembering what they saved and why it mattered.

The goal was to turn those forgotten screenshots into a searchable personal memory instead of another folder of files.

The project started from a simple question:

> What if you could ask your own saved screenshots what you were interested in at the time?

---

## What I Built

Remember Why turns screenshots into a searchable local memory.

The system:

1. Scans saved screenshots.
2. Extracts text using local OCR.
3. Converts the extracted text into embeddings.
4. Stores the embeddings in a local FAISS index.
5. Uses semantic search to find relevant memories.
6. Uses an MCP server to expose memory tools to the agent.
7. Uses LangGraph to orchestrate the agent workflow.
8. Uses a local Qwen3 model through Ollama.
9. Displays retrieved screenshots and evidence in a Streamlit interface.

For questions such as:

> "What did I save about MCP?"

the system retrieves relevant memories.

For questions such as:

> "Why did I save MCP_Architecture.png?"

the system looks at nearby and semantically related memories and presents them as **possible context**, rather than pretending to know the user's true intention.

---

## Grounding Rule

Remember Why separates retrieved evidence from inference.

- Retrieved OCR, filenames, timestamps, and similarity scores are evidence.
- **Possible context is an inference, not a fact.**
- Remember Why does not claim to know the user's true intent unless that intent is explicitly stored in the evidence.

This is important because an AI system should not invent a reason for why someone saved something.

---

## Demo

Ask:

> **"What did I save about MCP?"**

The UI displays the actual retrieved screenshot together with:

- filename
- timestamp
- semantic similarity
- OCR evidence
- related screenshots
- possible context

### Video Demo

[Watch the Remember Why demo](https://youtu.be/8UyR7Y0YTQ8)

The demo shows:

1. Asking Remember Why about saved MCP memories.
2. Retrieving relevant memories using semantic search.
3. Displaying the original saved screenshot.
4. Exploring related memories.
5. Showing possible context.
6. Observing the LangGraph agent execution through Sentry.

---

## Screenshots

### Remember Why UI

The application retrieves an actual saved screenshot and presents the evidence behind the result.

<!-- Add your Streamlit screenshot here if you upload it to the repository later -->

### Sentry Agent Tracing

Sentry shows the actual agent execution flow, including:

- LangGraph agent execution
- model operations
- MCP tool execution
- HTTP operations

<!-- Add your Sentry screenshot here if you upload it to the repository later -->

---

## How I Built It

The project is built around a fully local memory pipeline.

```text
Screenshots
    ↓
Tesseract OCR
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Index
    ↓
MCP Memory Tools
    ↓
LangGraph Agent
    ↓
Ollama / Qwen3 4B
    ↓
Grounded Response
```

### Agent Architecture

```text
User
  ↓
Streamlit UI
  ↓
LangGraph Agent ─────→ Ollama (Qwen3 4B)
  ↓
MCP Server
  ↓
FAISS Semantic Memory
  ↓
OCR + Local Embeddings
```

### MCP Tools

The MCP server exposes memory operations including:

- `search_screenshots`
- `get_screenshot_memory`
- `search_memories_by_date`
- `get_memory_context`

The agent can use these tools to retrieve evidence from the local memory store.

---

## Technology Stack

- **Python** — core application
- **Streamlit** — user interface
- **LangGraph** — agent orchestration
- **MCP Python SDK** — memory tool interface
- **Ollama** — local model runtime
- **Qwen3 4B** — local open-weight language model
- **Tesseract OCR** — local text extraction
- **Sentence Transformers** — local embeddings
- **all-MiniLM-L6-v2** — embedding model
- **FAISS** — local vector search
- **Sentry Agent Tracing** — agent observability

---

## Why Open Innovation Matters

> **Open innovation mattered here because the data being searched is personal.**

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
```

This approach provides several important benefits.

### Privacy

The core screenshot processing and memory retrieval happen locally.

The screenshots do not need to be uploaded to a third-party AI service just to perform semantic search.

### Model Choice

The application uses an open-weight Qwen3 model through Ollama.

The model can be changed without redesigning the entire application.

### Replaceable Components

The architecture is built from replaceable open components:

- local LLM
- embedding model
- vector database
- OCR engine
- agent framework
- MCP tools

This makes the system easier to experiment with and adapt.

### Local Development

The core application can be developed and tested without requiring a hosted AI API.

This makes experimentation accessible and keeps the core memory workflow under the user's control.

---

## Sentry Agent Tracing

Sentry Agent Tracing provides observability into the actual agent execution.

The trace can show operations such as:

```text
gen_ai.invoke_agent
        ↓
gen_ai.chat
        ↓
gen_ai.execute_tool
        ↓
http.client
        ↓
gen_ai.chat
```

Sentry is optional.

When `SENTRY_DSN` is configured, scrubbed telemetry is sent to the configured Sentry project.

Prompt capture is disabled.

Local screenshot files and the full local memory store are not sent as Sentry payloads.

---

## Privacy

Core screenshot OCR, embeddings, FAISS retrieval, the MCP server, and Ollama model execution run locally.

When Sentry is enabled, scrubbed telemetry is sent to the configured Sentry project.

Therefore, the project is described as **local-first**, rather than claiming that absolutely everything remains on-device.

---

## Setup

### 1. Create a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Install Tesseract OCR

Install the Tesseract OCR engine separately.

If Tesseract is not on PATH or in a standard Windows location, set its executable path:

```powershell
$env:TESSERACT_CMD = 'C:\Path\To\tesseract.exe'
```

### 3. Install Ollama

Install Ollama and pull the local model:

```powershell
ollama pull qwen3:4b
```

The application does not automatically download Ollama models.

### 4. Optional Sentry configuration

Sentry tracing is optional.

```powershell
$env:REMEMBER_WHY_MODEL = 'qwen3:4b'
$env:SENTRY_DSN = 'https://<public-key>@<organization>.ingest.sentry.io/<project-id>'
$env:SENTRY_TRACES_SAMPLE_RATE = '1.0'
```

Supported environment variables:

- `TESSERACT_CMD` — path to the Tesseract executable.
- `REMEMBER_WHY_MODEL` — Ollama model name. Defaults to `qwen3:4b`.
- `SENTRY_DSN` — optional Sentry project DSN.
- `SENTRY_TRACES_SAMPLE_RATE` — trace sample rate from `0.0` to `1.0`.

---

## Usage

Place your own PNG, JPG, JPEG, or WEBP screenshots inside:

```text
data/screenshots/
```

Build the local memory index:

```powershell
python main.py --build
```

Search memories:

```powershell
python main.py --search "MCP"
```

Ask a natural-language question:

```powershell
python main.py --ask "What did I save about MCP?"
```

Launch the Streamlit interface:

```powershell
python -m streamlit run app/ui/streamlit_app.py
```

---

## Testing

Run the complete test suite:

```powershell
python -m unittest discover -s tests -v
```

**Verified result: 79 tests passed.**

The agent tests mock model and MCP calls, so the test suite does not require Ollama to be running.

---

## Code

GitHub repository:

https://github.com/Rohith-Itagi7/remember-why

The repository contains:

- application source code
- MCP server
- LangGraph agent
- Streamlit UI
- retrieval system
- OCR pipeline
- tests
- setup instructions

Personal screenshots and generated local memory indexes are intentionally excluded from the public repository.

---

## Limitations

The current MVP focuses on local screenshot memories.

Direct integration with private X or LinkedIn saved posts is not included.

Such integrations would require appropriate authenticated APIs or connectors.

The current project therefore uses screenshots as the input source rather than claiming direct access to private social-media bookmarks.

---

## Future Directions

Possible future extensions include:

- authenticated connectors for supported services
- additional local memory sources
- richer temporal memory exploration
- user-provided goals and notes
- more advanced memory organization

These are not required for the current MVP.

---

## Hacktoberfest Partner Technology

**Sentry Agent Tracing**

Sentry is used for real agent observability across:

- LangGraph agent execution
- model operations
- MCP tool execution
- HTTP operations

The project uses Sentry as an observability layer without making it a dependency for the core local memory pipeline.

---

## Prize Categories

### Sentry Agent Tracing

Remember Why uses Sentry Agent Tracing to make the agent workflow observable.

The submitted demo includes an actual trace showing the LangGraph agent execution and downstream operations.

---

## Project Status

**MVP complete.**

Core functionality implemented:

- Local screenshot ingestion
- OCR
- Semantic embeddings
- FAISS vector search
- MCP memory tools
- LangGraph agent
- Local Qwen3 4B inference
- Context reconstruction
- Streamlit interface
- Sentry Agent Tracing
- Automated test suite

**79 tests passed.**

---

## License

Add your preferred open-source license here before publishing if you want the repository to be explicitly licensed for reuse.
