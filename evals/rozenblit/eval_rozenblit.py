"""
Rozenblit & Keil (2002) replication eval.

Tests IOED across three question categories from the original paper:
  - devices        (IOED expected: humans overestimate understanding)
  - procedures     (no IOED expected: humans are well-calibrated)
  - natural_phenomena (IOED expected)

Key comparison: delta on devices vs. delta on procedures.
If models mirror human IOED, devices should show larger drops than procedures.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.core.pipeline import MODELS, build_turn1_prompt, run_three_turns, RATE_LIMIT_DELAY

DATA_FILE   = Path(__file__).resolve().parents[2] / "data" / "rozenblit_dataset.json"
OUTPUT_FILE = Path(__file__).resolve().parents[2] / "results" / f"rozenblit_{datetime.now():%Y%m%d_%H%M%S}.json"


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        studies = json.load(f)

    questions = []
    for study in studies:
        for item in study["items"]:
            questions.append({
                "id":       item["id"],
                "question": item["question"],
                "category": item["category"],
                "study":    study["study"],   # devices | procedures | natural_phenomena
            })
    return questions


def main():
    questions = load_questions(DATA_FILE)
    print(f"Loaded {len(questions)} questions from {len(set(q['study'] for q in questions))} studies.")

    results = []
    total   = len(questions) * len(MODELS)

    for model in MODELS:
        print(f"\n=== model={model!r} ===")
        skip_model = False

        for i, q in enumerate(questions, 1):
            if skip_model:
                break

            print(f"  [{i}/{len(questions)}] study={q['study']!r} id={q['id']!r}")

            entry = {
                "question_id": q["id"],
                "question":    q["question"],
                "category":    q["category"],
                "study":       q["study"],
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

    # Group by study x model
    groups: dict[tuple, list] = {}
    for r in results:
        key = (r["study"], r["model"])
        groups.setdefault(key, []).append(r)

    print(f"\n{'Study':<22} {'Model':<35} {'Avg R1':>7} {'Avg R2':>7} {'Avg Δ':>7} {'n':>4}")
    print("-" * 80)
    for (study, model), rows in sorted(groups.items()):
        deltas = [
            r["second_rating"] - r["first_rating"]
            for r in rows
            if r["first_rating"] is not None and r["second_rating"] is not None
        ]
        r1s = [r["first_rating"]  for r in rows if r["first_rating"]  is not None]
        r2s = [r["second_rating"] for r in rows if r["second_rating"] is not None]
        if deltas:
            print(
                f"{study:<22} {model.split('/')[1]:<35} "
                f"{statistics.mean(r1s):>7.2f} {statistics.mean(r2s):>7.2f} "
                f"{statistics.mean(deltas):>+7.2f} {len(deltas):>4}"
            )
        else:
            print(f"{study:<22} {model.split('/')[1]:<35} {'—':>7} {'—':>7} {'—':>7} {0:>4}")


if __name__ == "__main__":
    main()
