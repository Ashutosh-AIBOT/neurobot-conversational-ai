import argparse
import json
from pathlib import Path

from src.neurobot_benchmark import load_cases, score_answer_against_focus, score_cases, summarize_cases
from src.neurobot_service import chat_turn
from langchain_core.messages import HumanMessage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=str, default="", help="Path to a JSON file mapping benchmark id to answer text.")
    parser.add_argument("--live", action="store_true", help="Run live benchmark requests against the local service layer.")
    args = parser.parse_args()

    cases = load_cases()
    summary = summarize_cases(cases)

    if not args.answers and not args.live:
        print(f"Loaded {summary['count']} benchmark cases")
        print("Tags:", summary["tags"])
        for question in summary["questions"]:
            print("-", question)
        return

    answers_map = {}
    if args.answers:
        answers_map = json.loads(Path(args.answers).read_text(encoding="utf-8"))

    results = []
    for case in cases:
        if args.live:
            service_result = chat_turn(
                case.tenant_id,
                case.session_id,
                [HumanMessage(content=case.question)],
            )
            answer = service_result["answer"]
            quality = service_result["quality_report"]
        else:
            answer = answers_map.get(case.id, "")
            quality = {}

        score = score_answer_against_focus(answer, case.expected_focus)
        results.append(
            {
                "id": case.id,
                "question": case.question,
                "expected_focus": case.expected_focus,
                "score": score,
                "quality": quality,
            }
        )

    overall = score_cases(results)
    print(json.dumps({"summary": overall, "results": results}, indent=2))


if __name__ == "__main__":
    main()
