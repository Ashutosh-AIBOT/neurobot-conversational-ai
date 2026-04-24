import re
from typing import Any

from langchain_core.messages import ToolMessage


def parse_audit_markdown(audit_text: str) -> dict[str, Any]:
    report = {
        "audit_status": "not_run",
        "faithfulness": None,
        "answer_relevancy": None,
    }

    if not audit_text or "### Response Audit" not in audit_text:
        return report

    status_match = re.search(r"- Status:\s*(.+)", audit_text)
    faith_match = re.search(r"- Faithfulness:\s*([0-9.]+)", audit_text)
    relevancy_match = re.search(r"- Answer relevancy:\s*([0-9.]+)", audit_text)

    if status_match:
        report["audit_status"] = status_match.group(1).strip()
    if faith_match:
        report["faithfulness"] = float(faith_match.group(1))
    if relevancy_match:
        report["answer_relevancy"] = float(relevancy_match.group(1))

    return report


def count_sources(answer_text: str) -> int:
    if "**Sources:**" not in answer_text:
        return 0

    tail = answer_text.split("**Sources:**", 1)[1].strip()
    if not tail or tail == "General Knowledge":
        return 0
    return len([item for item in tail.split(",") if item.strip()])


def build_quality_report(
    answer_text: str,
    tool_events: list[Any],
    latency_ms: float,
    indexed_chunks: int = 0,
) -> dict[str, Any]:
    tool_names = [getattr(event, "name", "unknown") for event in tool_events if isinstance(event, ToolMessage)]
    audit_event = next(
        (
            event.content
            for event in reversed(tool_events)
            if isinstance(event, ToolMessage) and getattr(event, "name", "") == "response_audit"
        ),
        "",
    )
    audit = parse_audit_markdown(audit_event)

    return {
        "latency_ms": round(latency_ms, 2),
        "source_count": count_sources(answer_text),
        "indexed_chunks": indexed_chunks,
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "used_retrieval": "pdf_qa_tool" in tool_names,
        "used_web_search": "duckduckgo_search" in tool_names,
        "used_arxiv": "arxiv_search" in tool_names or "download_and_talk_to_paper" in tool_names,
        **audit,
    }


def format_quality_report(report: dict[str, Any]) -> str:
    faith = report.get("faithfulness")
    relevancy = report.get("answer_relevancy")

    return (
        "### Quality Report\n"
        f"- Latency ms: {report.get('latency_ms', 0):.2f}\n"
        f"- Sources used: {report.get('source_count', 0)}\n"
        f"- Indexed chunks available: {report.get('indexed_chunks', 0)}\n"
        f"- Tool count: {report.get('tool_count', 0)}\n"
        f"- Used retrieval: {str(report.get('used_retrieval', False)).lower()}\n"
        f"- Used web search: {str(report.get('used_web_search', False)).lower()}\n"
        f"- Used arXiv: {str(report.get('used_arxiv', False)).lower()}\n"
        f"- Audit status: {report.get('audit_status', 'not_run')}\n"
        f"- Faithfulness: {faith if faith is not None else 'n/a'}\n"
        f"- Answer relevancy: {relevancy if relevancy is not None else 'n/a'}"
    )
