"""
Easy-for-humans, difficult-for-AI eval.

Questions humans find intuitive but LLMs frequently get wrong
(puzzles, spatial reasoning, trick questions).

Expected IOED pattern: models are overconfident on Turn 1;
explanation reveals gaps → larger delta than Rozenblit devices.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.core.pipeline import MODELS, build_turn1_prompt, run_three_turns, RATE_LIMIT_DELAY

DATA_FILE   = Path(__file__).resolve().parents[2] / "data" / "easyProblems.json"
OUTPUT_FILE = Path(__file__).resolve().parents[2] / "results" / f"easy_problems_{datetime.now():%Y%m%d_%H%M%S}.json"


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    questions = load_questions(DATA_FILE)
    print(f"Loaded {len(questions)} questions.")

    results = []

    for model in MODELS:
        print(f"\n=== model={model!r} ===")
        skip_model = False

        for i, q in enumerate(questions, 1):
            if skip_model:
                break

            print(f"  [{i}/{len(questions)}] id={q['id']!r}")

            entry = {
                "question_id": q["id"],
                "question":    q["question"],
                "category":    q.get("category"),
                "model":       model,
            }

            outcome = run_three_turns(build_turn1_prompt(q["question"]), model)
            entry.update(outcome)

            if outcome["error"] and any(
                kw in outcome["error"] for kw in ("unavailable", "Rate limited")
            ):
                print(f"    SKIP MODEL: {outcome['error']}")
                skip_model = True

            results.append(entry)
            time.sleep(RATE_LIMIT_DELAY)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} entries → {OUTPUT_FILE}")
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    import statistics

    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(r["model"], []).append(r)

    print(f"\n{'Model':<40} {'Avg R1':>7} {'Avg R2':>7} {'Avg Δ':>7} {'n':>4}")
    print("-" * 65)
    for model, rows in sorted(groups.items()):
        deltas = [
            r["second_rating"] - r["first_rating"]
            for r in rows
            if r["first_rating"] is not None and r["second_rating"] is not None
        ]
        r1s = [r["first_rating"]  for r in rows if r["first_rating"]  is not None]
        r2s = [r["second_rating"] for r in rows if r["second_rating"] is not None]
        if deltas:
            print(
                f"{model.split('/')[1]:<40} "
                f"{statistics.mean(r1s):>7.2f} {statistics.mean(r2s):>7.2f} "
                f"{statistics.mean(deltas):>+7.2f} {len(deltas):>4}"
            )
        else:
            print(f"{model.split('/')[1]:<40} {'—':>7} {'—':>7} {'—':>7} {0:>4}")


if __name__ == "__main__":
    main()
