"""
SPaRTQA spatial reasoning eval.

Questions require reasoning over spatial relationships described in short stories.
Has ground-truth answers, so we track both accuracy and confidence delta.

Three question types:
  YN — yes/no; candidates are implicitly ["Yes", "No"]
  FR — fill-in relationship (left/right/above/etc.); answer field contains indices
       into candidate_answers that must be resolved to strings on load
  FB — fill-in block (A/B/C); candidates and answers are already strings

Expected IOED pattern: models are overconfident on spatial tasks;
explanation may surface reasoning errors → larger confidence drop.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.core.pipeline import (
    MODELS, run_three_turns,
    RATE_LIMIT_DELAY, format_eta,
)
from evals.core.text_analysis import print_lexical_summary

DATA_FILE   = Path(__file__).resolve().parents[2] / "data" / "spartqa_human_train_dataset.json"
OUTPUT_DIR  = Path(__file__).resolve().parents[2] / "results"
OUTPUT_FILE = OUTPUT_DIR / "spartqa_results.jsonl"


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

    questions = []
    for story_idx, story_entry in enumerate(data["data"]):
        story_text = " ".join(story_entry["story"])
        for q in story_entry["questions"]:
            candidates = q["candidate_answers"]

            # YN questions have no explicit candidates
            if not candidates:
                candidates = ["Yes", "No"]

            # FR answers are indices into candidate_answers; convert to strings
            answer = q["answer"]
            if q["q_type"] == "FR" and answer and isinstance(answer[0], int):
                answer = [candidates[i] for i in answer]

            questions.append({
                "id":               f"s{story_idx}_q{q['q_id']}",
                "question":         q["question"],
                "story":            story_text,
                "qtype":            q["q_type"],
                "candidateanswers": candidates,
                "answer":           answer,
                "reasoning_type":   q.get("reasoning_type", []),
            })
    return questions


def load_saved_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def save(entry: dict) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    questions  = load_questions(DATA_FILE)
    total      = len(questions) * len(MODELS)
    print(f"Loaded {len(questions)} SPaRTQA questions × {len(MODELS)} models = {total} entries.")
    print(f"Output → {OUTPUT_FILE}\n")

    results        = load_saved_results(OUTPUT_FILE)
    completed_keys = {(r["question_id"], r["model"]) for r in results}
    completed      = len(results)
    start_time     = time.time()

    if results:
        print(f"Resuming: {completed} entries already saved.")

    for model in MODELS:
        print(f"\n=== model={model!r} ===")
        skip_model = False

        for q in questions:
            if skip_model:
                break

            if (q["id"], model) in completed_keys:
                continue

            candidates   = q["candidateanswers"]
            correct      = q["answer"]
            turn1_prompt = build_turn1_prompt(q["story"], q["question"], candidates)

            print(f"  [{completed+1}/{total}] id={q['id']!r} qtype={q['qtype']!r}  {format_eta(start_time, completed, total)}")

            entry = {
                "question_id":        q["id"],
                "question":           q["question"],
                "story":              q["story"],
                "qtype":              q["qtype"],
                "reasoning_type":     q["reasoning_type"],
                "correct_answer":     correct,
                "candidate_answers":  candidates,
                "model":              model,
                "first_answer":       None,
                "first_selected":     None,
                "first_correct":      None,
                "first_rating":       None,
                "explanation":        None,
                "explanation_scores": None,
                "second_selected":    None,
                "second_correct":     None,
                "second_rating":      None,
                "error":              None,
            }

            outcome = run_three_turns(turn1_prompt, model)

            entry["first_answer"]       = outcome["first_answer"]
            entry["first_rating"]       = outcome["first_rating"]
            entry["explanation"]        = outcome["explanation"]
            entry["explanation_scores"] = outcome["explanation_scores"]
            entry["second_rating"]      = outcome["second_rating"]
            entry["error"]              = outcome["error"]

            if outcome["first_answer"]:
                sel = extract_selection(outcome["first_answer"], candidates)
                entry["first_selected"] = sel
                entry["first_correct"]  = is_correct(sel, correct)
                print(f"    correct={correct}  selected={sel}  graded={entry['first_correct']}")

            if outcome["explanation"]:
                sel2 = extract_selection(outcome["explanation"], candidates)
                entry["second_selected"] = sel2
                entry["second_correct"]  = is_correct(sel2, correct)

            results.append(entry)
            completed += 1
            completed_keys.add((q["id"], model))
            save(entry)

            if outcome["error"] and any(
                kw in outcome["error"] for kw in ("unavailable", "Rate limited")
            ):
                print(f"    SKIP MODEL: {outcome['error']}")
                skip_model = True

            time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone. {len(results)} entries saved → {OUTPUT_FILE}")
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

    print_lexical_summary(results, group_by="model")


if __name__ == "__main__":
    main()
