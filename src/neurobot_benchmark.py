import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.neurobot_settings import get_settings

settings = get_settings()


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    tenant_id: str
    session_id: str
    question: str
    expected_focus: str
    tags: list[str]


def score_answer_against_focus(answer: str, expected_focus: str) -> dict[str, Any]:
    answer_lower = answer.lower()
    expected_terms = [
        term.strip().lower()
        for term in expected_focus.replace("-", " ").replace("/", " ").split()
        if len(term.strip()) >= 4
    ]
    matched_terms = [term for term in expected_terms if term in answer_lower]
    coverage = 0.0 if not expected_terms else len(matched_terms) / len(expected_terms)

    return {
        "coverage": round(coverage, 2),
        "matched_terms": matched_terms,
        "pass": coverage >= 0.4,
    }


def load_cases(path: Path | None = None) -> list[BenchmarkCase]:
    source = path or (settings.benchmark_dir / "mini_benchmark.jsonl")
    cases: list[BenchmarkCase] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(BenchmarkCase(**raw))
    return cases


def summarize_cases(cases: list[BenchmarkCase]) -> dict:
    tag_counts: dict[str, int] = {}
    for case in cases:
        for tag in case.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return {
        "count": len(cases),
        "tags": tag_counts,
        "questions": [case.question for case in cases],
    }


def score_cases(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for item in results if item["score"]["pass"])
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": 0.0 if not results else round(passed / len(results), 2),
    }
