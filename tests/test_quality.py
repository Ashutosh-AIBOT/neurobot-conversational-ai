from langchain_core.messages import ToolMessage

from src.neurobot_benchmark import score_answer_against_focus
from src.neurobot_quality import build_quality_report, parse_audit_markdown


def test_parse_audit_markdown_extracts_scores():
    report = parse_audit_markdown(
        "### Response Audit\n- Status: strongly grounded\n- Faithfulness: 0.91\n- Answer relevancy: 0.88"
    )
    assert report["audit_status"] == "strongly grounded"
    assert report["faithfulness"] == 0.91
    assert report["answer_relevancy"] == 0.88


def test_build_quality_report_counts_tools_and_sources():
    report = build_quality_report(
        answer_text="Answer\n\n**Sources:** arxiv, pdf",
        tool_events=[ToolMessage(content="### Response Audit\n- Status: skipped", name="response_audit", tool_call_id="1")],
        latency_ms=123.4,
        indexed_chunks=10,
    )
    assert report["source_count"] == 2
    assert report["tool_count"] == 1
    assert report["audit_status"] == "skipped"


def test_score_answer_against_focus_passes_on_overlap():
    result = score_answer_against_focus(
        "The system uses tenant namespacing and checkpoint files for isolation.",
        "tenant namespacing and checkpoint files",
    )
    assert result["pass"] is True
