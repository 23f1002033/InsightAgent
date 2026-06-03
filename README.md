# InsightAgent 🔍

> **An Agentic RAG Knowledge Assistant** — ingest your documents, chat with them, get answers grounded in your own knowledge base with inline source citations.

InsightAgent is a production-grade, end-to-end **Retrieval-Augmented Generation (RAG)** system built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/). It lets you ingest PDFs, Markdown, text files, and web pages into a personal knowledge base, then converse with an agentic LLM that answers *only* from your documents — citing every source inline.

---

## ✨ Features

- **Multi-format ingestion** — PDF, `.txt`, `.md`, and web URLs via a single CLI command
- **Agentic routing** — LangGraph `StateGraph` decides per query: retrieve from KB → search the web → answer directly
- **Inline citations** — every answer shows the source filename/URL and chunk ID
- **Provider-agnostic** — swap LLM (Gemini / OpenAI / Anthropic) and embeddings with one env var change
- **Conversational memory** — `MemorySaver` checkpointer keeps context across turns, keyed by `thread_id`
- **Swappable vector store** — Chroma locally; Pinecone or Qdrant in production (one-line swap)
- **REST API** — FastAPI with `/ingest`, `/chat`, `/health` endpoints
- **Chat UI** — Streamlit frontend with citation cards
- **Observability** — LangSmith tracing, gated by env var (zero overhead when off)
- **Evaluation** — RAGAS metrics (faithfulness, relevancy, context precision/recall) with a golden Q&A dataset
- **Containerised** — Dockerfile + docker-compose for one-command deployment

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User (Browser / CLI)                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │  HTTP
┌──────────────────────────────▼──────────────────────────────────┐
│               Streamlit Chat UI  (Phase 6)                      │
│           chat bubbles · citation cards · file upload           │
└──────────────────────────────┬──────────────────────────────────┘
                               │  REST
┌──────────────────────────────▼──────────────────────────────────┐
│                  FastAPI  (Phase 6)                              │
│        POST /ingest  ·  POST /chat  ·  GET /health              │
└──────┬────────────────────────────────────────┬─────────────────┘
       │                                        │
┌──────▼────────────────────┐   ┌───────────────▼───────────────┐
│   Ingestion Pipeline      │   │   LangGraph Agent             │
│   (Phase 2)               │   │   (Phases 3–5)                │
│                           │   │                               │
│  Loaders (PDF/txt/md/URL) │   │  StateGraph                   │
│       ↓                   │   │    ├─ agent node (LLM)        │
│  RecursiveCharSplitter    │   │    ├─ tools node              │
│       ↓                   │   │    │   ├─ retrieve_docs()     │
│  Embeddings               │   │    │   └─ web_search()        │
│  (Gemini / OpenAI / HF)   │   │    └─ back to agent           │
│       ↓                   │   │                               │
│  Chroma (local disk)  ◄───┤   │  MemorySaver checkpointer     │
│  (→ Pinecone in prod)     │   │  (thread_id per session)      │
└───────────────────────────┘   └───────────────────────────────┘
                                              │
                              ┌───────────────▼──────────────────┐
                              │  LangSmith Tracing  (Phase 7)    │
                              │  RAGAS Evaluation   (Phase 7)    │
                              └──────────────────────────────────┘
```

---

## Project Structure

```
insightagent/
├── app/
│   ├── config.py           # pydantic-settings: all env vars, type-safe
│   ├── llm.py              # provider-agnostic LLM factory (Gemini/OpenAI/Anthropic)
│   ├── logging_config.py   # centralised structured logging
│   └── main.py             # FastAPI app (Phase 6)
│
├── ingestion/              # Phase 2 ✅
│   ├── loaders.py          # PDF, txt, md, URL → List[Document]
│   ├── splitter.py         # RecursiveCharacterTextSplitter + chunk metadata
│   ├── embeddings.py       # Embedding factory (Gemini / OpenAI / HuggingFace)
│   ├── vector_store.py     # Chroma abstraction behind VectorStore interface
│   └── cli.py              # `python -m ingestion.cli ingest ./docs/`
│
├── agent/                  # Phases 3–5
│   ├── graph.py            # LangGraph StateGraph definition
│   ├── tools.py            # retrieve_documents, web_search tools
│   ├── prompts.py          # system prompt with grounding + citation rules
│   └── memory.py           # checkpointer factory (MemorySaver / PostgresSaver)
│
├── eval/                   # Phase 7
│   ├── ragas_eval.py
│   └── golden_dataset.json
│
├── frontend/               # Phase 6
│   └── streamlit_app.py
│
├── tests/
│   ├── test_config.py      # Phase 1 ✅  (11 tests)
│   ├── test_ingestion.py   # Phase 2 ✅  (18 tests)
│   ├── test_agent.py       # Phases 3–5
│   └── test_api.py         # Phase 6
│
├── scripts/
│   └── hello_llm.py        # smoke test — verifies LLM factory end-to-end
│
├── data/                   # gitignored — Chroma DB lives here at runtime
│
├── .env.example            # copy to .env and fill in your keys
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── Dockerfile              # Phase 8
├── docker-compose.yml      # Phase 8
└── README.md
```

---

## Quick Start

### 1. Clone & set up the environment

```bash
git clone https://github.com/23f1002033/InsightAgent.git
cd InsightAgent

python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure your API keys

```bash
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY)
```

Get a free Gemini key at → [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 3. Verify the LLM is wired up

```bash
# Single response
python scripts/hello_llm.py --provider gemini

# Streaming tokens
python scripts/hello_llm.py --provider gemini --stream

# Try OpenAI or Anthropic instead
python scripts/hello_llm.py --provider openai
```

### 4. Ingest documents into the knowledge base

```bash
# Ingest a folder (PDF, txt, md supported)
python -m ingestion.cli ingest ./docs/

# Ingest a single file
python -m ingestion.cli ingest ./docs/my_report.pdf

# Ingest a web page
python -m ingestion.cli ingest https://en.wikipedia.org/wiki/Retrieval-augmented_generation

# Query the KB (sanity check)
python -m ingestion.cli query "What is RAG?"

# Check how many chunks are stored
python -m ingestion.cli stats
```

### 5. Run the test suite

```bash
# All unit tests (no API keys needed — everything is mocked)
pytest -v

# Include the local integration test (HuggingFace embeddings, no API key)
pytest -v -m integration
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `openai`, or `anthropic` |
| `GEMINI_API_KEY` | — | Your Google AI Studio key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini chat model |
| `GEMINI_EMBEDDING_MODEL` | `models/text-embedding-004` | Gemini embedding model (768-dim) |
| `OPENAI_API_KEY` | — | Your OpenAI key |
| `ANTHROPIC_API_KEY` | — | Your Anthropic key |
| `EMBEDDING_PROVIDER` | `gemini` | `gemini`, `openai`, or `huggingface` |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Where the vector store is saved |
| `TAVILY_API_KEY` | — | For live web search (Phase 4) |
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to enable LangSmith tracing |

See [`.env.example`](.env.example) for the full reference.

---

## LangChain / LangGraph Concepts

| Concept | File | Why it matters |
|---|---|---|
| `BaseChatModel` | `app/llm.py` | Single interface for Gemini, OpenAI, Anthropic — swap with one env var |
| `HumanMessage` / `AIMessage` | everywhere | Universal message schema all LangChain models understand |
| `BaseDocumentLoader` | `ingestion/loaders.py` | Uniform `.load()` API across PDF, txt, URL — add new formats in one place |
| `RecursiveCharacterTextSplitter` | `ingestion/splitter.py` | Splits `\n\n` → `\n` → ` ` — preserves paragraph semantics over blind character cuts |
| `Embeddings` interface | `ingestion/embeddings.py` | Swap embedding model without touching any vector store code |
| `VectorStore` / `Chroma` | `ingestion/vector_store.py` | Chroma locally; returns abstract `VectorStore` type so Pinecone/Qdrant drops in |
| `@tool` decorator | `agent/tools.py` | Turns a Python function into a tool the LLM can call with structured args |
| `StateGraph` | `agent/graph.py` | Explicit state machine: agent → tools → agent until `END` |
| `MemorySaver` | `agent/memory.py` | In-process conversation checkpointer keyed by `thread_id` |
| `pydantic-settings` | `app/config.py` | Type-safe env var loading — no `os.environ` calls in business logic |
| LangSmith tracing | `app/llm.py` | Every LLM call, tool call, token captured — toggled by env var, zero cost when off |
| RAGAS | `eval/ragas_eval.py` | Automated RAG quality metrics against a golden Q&A dataset |

---

## Build Status

| Phase | Status | What it builds |
|---|---|---|
| 1 — Scaffold | ✅ Done | Repo structure, venv, deps, config, LLM factory, smoke test |
| 2 — Ingestion | ✅ Done | Load → split → embed → Chroma; ingestion CLI |
| 3 — Basic RAG | 🔨 In progress | Retrieve → stuff context → answer with citations |
| 4 — Agentic LangGraph | ⏳ | StateGraph + retrieve / web\_search tools + routing |
| 5 — Memory | ⏳ | Checkpointer + thread\_id per conversation |
| 6 — API + Frontend | ⏳ | FastAPI endpoints + Streamlit chat UI |
| 7 — Observability + Eval | ⏳ | LangSmith tracing + RAGAS golden dataset |
| 8 — Docker | ⏳ | Dockerfile + docker-compose + full README |

---

## Contributing

1. Fork the repo and create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes with type hints and docstrings
3. Run `pytest -v` — all tests must pass before opening a PR
4. Open a pull request with a clear description of what changed and why

---

## License

MIT — see [LICENSE](LICENSE) for details.
