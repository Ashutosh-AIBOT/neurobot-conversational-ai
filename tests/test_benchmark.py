from src.neurobot_benchmark import BenchmarkCase, load_cases, summarize_cases


def test_benchmark_loader_reads_cases():
    cases = load_cases()
    assert len(cases) >= 5
    assert all(isinstance(case, BenchmarkCase) for case in cases)


def test_benchmark_summary_counts_tags():
    cases = load_cases()
    summary = summarize_cases(cases)
    assert summary["count"] == len(cases)
    assert "rag" in summary["tags"]
