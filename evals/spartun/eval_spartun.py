"""
SPARTUN spatial reasoning eval.

Questions require reasoning over spatial relationships described in short stories.
Has ground-truth answers, so we track both accuracy and confidence delta.

Grading fix: previous eval compared first_selected (str) to correct_answer (list),
which always returned False. Fixed by checking membership in the list.

For YN questions: candidate answers are provided, model selects one.
For FR questions: free-response, graded by exact string match (case-insensitive).

Expected IOED pattern: models are overconfident on spatial tasks;
explanation may surface reasoning errors → larger confidence drop.
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.core.pipeline import (
    MODELS, run_three_turns, extract_rating,
    RATE_LIMIT_DELAY, call_openrouter,
)

DATA_FILE   = Path(__file__).resolve().parents[2] / "data" / "spartun_100_samples.json"
OUTPUT_FILE = Path(__file__).resolve().parents[2] / "results" / f"spartun_{datetime.now():%Y%m%d_%H%M%S}.json"


def build_turn1_prompt(story: str, question: str, candidates: list[str]) -> str:
    options = " / ".join(candidates)
    return (
        "Read the following story carefully, then answer the question by selecting "
        f"one of the options provided.\n\n"
        f"Story: {story}\n\n"
        f"Question: {question}\n"
        f"Options: {options}\n\n"
        "State your selected answer exactly as it appears in the options. "
        "Then rate your confidence 0-10, with 10 being the most confident. "
        "Only provide the answer and the number."
    )


def extract_selection(text: str, candidates: list[str]) -> str | None:
    """Return the candidate that appears in the model's reply, or None."""
    text_lower = text.lower()
    for candidate in candidates:
        if candidate.lower() in text_lower:
            return candidate
    return None


def is_correct(selected: str | None, correct_answers: list[str]) -> bool | None:
    """Check if selected answer is in the correct answers list (case-insensitive)."""
    if selected is None:
        return None
    return selected.strip().lower() in [a.strip().lower() for a in correct_answers]


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data[0]["items"]


def main():
    questions = load_questions(DATA_FILE)
    print(f"Loaded {len(questions)} SPARTUN questions.")

    results = []

    for model in MODELS:
        print(f"\n=== model={model!r} ===")
        skip_model = False

        for i, q in enumerate(questions, 1):
            if skip_model:
                break

            print(f"  [{i}/{len(questions)}] id={q['id']!r} qtype={q['qtype']!r}")

            candidates     = q["candidateanswers"]
            correct        = q["answer"]
            turn1_prompt   = build_turn1_prompt(q["story"], q["question"], candidates)

            entry = {
                "question_id":      q["id"],
                "question":         q["question"],
                "story":            q["story"],
                "qtype":            q["qtype"],
                "correct_answer":   correct,
                "candidate_answers": candidates,
                "model":            model,
                "first_answer":     None,
                "first_selected":   None,
                "first_correct":    None,
                "first_rating":     None,
                "explanation":      None,
                "second_selected":  None,
                "second_correct":   None,
                "second_rating":    None,
                "error":            None,
            }

            outcome = run_three_turns(turn1_prompt, model)

            entry["first_answer"]  = outcome["first_answer"]
            entry["first_rating"]  = outcome["first_rating"]
            entry["explanation"]   = outcome["explanation"]
            entry["second_rating"] = outcome["second_rating"]
            entry["error"]         = outcome["error"]

            if outcome["first_answer"]:
                sel = extract_selection(outcome["first_answer"], candidates)
                entry["first_selected"] = sel
                entry["first_correct"]  = is_correct(sel, correct)  # fixed

            # Extract second selection from explanation text if present
            if outcome["explanation"]:
                sel2 = extract_selection(outcome["explanation"], candidates)
                entry["second_selected"] = sel2
                entry["second_correct"]  = is_correct(sel2, correct)

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

    print(f"\n{'Model':<40} {'Acc1':>6} {'Acc2':>6} {'Avg R1':>7} {'Avg R2':>7} {'Avg Δ':>7} {'n':>4}")
    print("-" * 80)
    for model, rows in sorted(groups.items()):
        graded1 = [r for r in rows if r["first_correct"] is not None]
        graded2 = [r for r in rows if r["second_correct"] is not None]
        acc1    = sum(r["first_correct"]  for r in graded1) / len(graded1) if graded1 else float("nan")
        acc2    = sum(r["second_correct"] for r in graded2) / len(graded2) if graded2 else float("nan")
        deltas  = [
            r["second_rating"] - r["first_rating"]
            for r in rows
            if r["first_rating"] is not None and r["second_rating"] is not None
        ]
        r1s = [r["first_rating"]  for r in rows if r["first_rating"]  is not None]
        r2s = [r["second_rating"] for r in rows if r["second_rating"] is not None]
        if deltas:
            print(
                f"{model.split('/')[1]:<40} "
                f"{acc1:>6.0%} {acc2:>6.0%} "
                f"{statistics.mean(r1s):>7.2f} {statistics.mean(r2s):>7.2f} "
                f"{statistics.mean(deltas):>+7.2f} {len(deltas):>4}"
            )
        else:
            print(f"{model.split('/')[1]:<40} {'—':>6} {'—':>6} {'—':>7} {'—':>7} {'—':>7} {0:>4}")


if __name__ == "__main__":
    main()
