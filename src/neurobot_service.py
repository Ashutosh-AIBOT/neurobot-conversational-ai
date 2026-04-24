import time

from langchain_core.messages import AIMessage, HumanMessage

from src.neurobot_graph import compile_brain
from src.neurobot_quality import build_quality_report
from src.neurobot_rag import get_doc_metadata, ingest_pdf
from src.neurobot_settings import get_settings

settings = get_settings()
_BRAIN_CACHE = {}


def build_session_key(tenant_id: str | None, session_id: str) -> str:
    return settings.session_namespace(tenant_id, session_id)


def get_brain(tenant_id: str | None = None):
    key = tenant_id or settings.default_tenant_id
    if key not in _BRAIN_CACHE:
        _BRAIN_CACHE[key] = compile_brain(key)
    return _BRAIN_CACHE[key]


def ingest_document(file_bytes: bytes, tenant_id: str, session_id: str, filename: str) -> dict:
    return ingest_pdf(file_bytes, build_session_key(tenant_id, session_id), filename)


def chat_turn(tenant_id: str, session_id: str, history: list[HumanMessage]) -> dict:
    brain = get_brain(tenant_id)
    scoped_session_key = build_session_key(tenant_id, session_id)
    config = {"configurable": {"thread_id": scoped_session_key}}
    response_text = ""
    tool_events = []
    started = time.perf_counter()

    for chunk in brain.stream({"messages": history}, config=config, stream_mode="messages"):
        msg = chunk[0] if isinstance(chunk, tuple) else chunk
        if isinstance(msg, AIMessage) and msg.content:
            response_text += str(msg.content)
        else:
            tool_events.append(msg)

    metadata = get_doc_metadata(scoped_session_key)
    latency_ms = (time.perf_counter() - started) * 1000
    quality_report = build_quality_report(
        answer_text=response_text,
        tool_events=tool_events,
        latency_ms=latency_ms,
        indexed_chunks=int(metadata.get("chunks", 0)),
    )

    return {
        "answer": response_text,
        "tool_events": tool_events,
        "metadata": metadata,
        "quality_report": quality_report,
    }
