# Low-Level Design

The low-level structure is centered around:
- `app.py` for the UI flow
- `api/main.py` for HTTP entry points
- `src/neurobot_service.py` for shared orchestration
- `src/neurobot_graph.py` for LangGraph workflow logic
- `src/neurobot_rag.py` for ingestion and retrieval
- `src/neurobot_quality.py` for latency and quality reporting
