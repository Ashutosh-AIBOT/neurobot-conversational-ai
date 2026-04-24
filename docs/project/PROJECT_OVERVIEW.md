# NeuroBot Project Overview

NeuroBot is a compact end-to-end RAG assistant for technical documents. It supports:
- multi-session chat in Streamlit
- PDF upload and chunked document retrieval
- arXiv discovery and paper ingestion
- web-search fallback when local document context is weak
- optional response auditing when grounded context exists
- MCP-backed tool execution with a local default server

## What Makes This Version Stronger
- runtime configuration is centralized in `src/neurobot_settings.py`
- upload and prompt validation are explicit
- uploaded document indices persist in `runtime/vector_store/` instead of living only in memory
- the LangGraph flow now has a real fallback search path
- UI metrics are tied to actual state instead of fixed percentages
- dependency ranges are constrained for better reproducibility
- a small FastAPI backend exists for `/v1/chat` and `/v1/documents`
- benchmark cases exist under `data/benchmark/`
- tenant IDs namespace runtime artifacts and checkpoint files
- the agent no longer calls the core tools directly; it talks to the local MCP server instead

## Key Files
| File | Purpose |
|------|---------|
| `api/main.py` | FastAPI backend service |
| `app.py` | Streamlit UI and session control |
| `src/neurobot_graph.py` | LangGraph orchestration and recovery logic |
| `src/neurobot_service.py` | Shared service layer for app/API |
| `src/neurobot_rag.py` | PDF parsing, chunking, embedding, vector persistence |
| `src/neurobot_benchmark.py` | Benchmark loader and summary helpers |
| `src/neurobot_tools.py` | MCP-backed client wrappers for search, retrieval, audit, and paper-download tools |
| `src/neurobot_mcp_server.py` | Local MCP server exposing the core tools |
| `src/neurobot_tool_impl.py` | Direct tool implementations used by the MCP server |
| `src/neurobot_eval.py` | grounded response audit |
| `src/neurobot_settings.py` | runtime paths and environment-driven settings |
| `src/neurobot_validation.py` | input validation helpers |

## Run Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
make streamlit
```

## Intended Scope
This is a serious portfolio project, not a large platform. The design goal is to show strong applied GenAI engineering without adding unnecessary enterprise boilerplate.
