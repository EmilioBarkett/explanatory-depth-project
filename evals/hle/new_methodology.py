"""
Humanity's Last Exam (HLE) eval — difficult for both humans and AI.

These are expert-level questions across 100+ subjects. Models likely
already have moderate-to-low confidence, so IOED signal may be smaller.
This serves as a calibration baseline: if delta here is similar to
Rozenblit devices, it suggests model drops are a general response to
being asked to explain rather than genuine IOED.

Data format: JSONL, each line has {id, question, category}.
No ground-truth answers available in this subset.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.core.pipeline import MODELS, build_turn1_prompt, run_three_turns, RATE_LIMIT_DELAY, format_eta, call_openrouter, extract_rating

DATA_FILE   = Path(__file__).resolve().parents[2] / "data" / "hle_100_sample.jsonl"
OUTPUT_DIR  = Path(__file__).resolve().parents[2] / "results" 
OUTPUT_FILE = OUTPUT_DIR / "new_sample_HLE.jsonl"

# HLE sample has 100 questions.
# BEGIN_IDX is 1-based and inclusive.
BEGIN_IDX = 0
MAX_QUESTIONS = 100

MODELS = [ 
    ## Reasoning
    "deepseek/deepseek-r1", 
    "openai/o3-mini", 
    "qwen/qwq-32b", 
    "anthropic/claude-opus-4", 
    "google/gemini-2.5-pro",
    "x-ai/grok-3",

    ## Non-reasoning:
    "openai/gpt-4o", 
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.0-flash-001", 
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mixtral-8x22b-instruct",
    "cohere/command-a"
]

def load_questions(path: Path, begin_idx: int, limit: int) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx < begin_idx:
                continue
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
            if len(questions) >= limit:
                break
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


def new_3_turn(question: str, model: str) -> dict:
    """
    Execute the new three-turn protocol:
      Turn 1 — Ask question, get confidence rating only (0-10), NO explanation/answer
      Turn 2 — Ask for explanation and answer
      Turn 3 — Get fresh confidence rating (0-10)
    
    Returns a dict with keys:
        first_confidence,
        answer,
        explanation,
        second_confidence,
        error
    """
    result = {
        "first_confidence": None,
        "answer": None,
        "explanation": None,
        "second_confidence": None,
        "error": None,
    }

    try:
        # Turn 1: Ask for confidence only (0-10), no explanation or answer
        turn1_prompt = (
            f"How confident are you that you can answer this question? "
            f"Rate your confidence 0-10, with 0 being not confident at all and 10 being extremely confident. "
            f"Respond with only a number.\n\n"
            f"Question: {question}"
        )
        conversation = [{"role": "user", "content": turn1_prompt}]
        first_reply = call_openrouter(conversation, model)
        result["first_confidence"] = extract_rating(first_reply)
        print(f"    T1 confidence: {first_reply[:80].strip()!r}  R1={result['first_confidence']}")
        time.sleep(RATE_LIMIT_DELAY)

        # Turn 2: Ask for explanation and answer
        turn2_prompt = (
            "Please explain your answer step-by-step and provide your answer to the question."
        )
        conversation.append({"role": "assistant", "content": first_reply})
        conversation.append({"role": "user", "content": turn2_prompt})
        second_reply = call_openrouter(conversation, model)
        result["answer"] = second_reply
        result["explanation"] = second_reply
        print(f"    T2 answer: {len(second_reply.split())} words")
        time.sleep(RATE_LIMIT_DELAY)

        # Turn 3: Fresh confidence rating (0-10)
        turn3_prompt = (
            "Now that you have explained your answer, how confident are you in it? "
            "Rate your confidence 0-10, with 0 being not confident at all and 10 being extremely confident. "
            "Respond with only a number."
        )
        conversation.append({"role": "assistant", "content": second_reply})
        conversation.append({"role": "user", "content": turn3_prompt})
        third_reply = call_openrouter(conversation, model)
        result["second_confidence"] = extract_rating(third_reply)

        delta = (
            result["second_confidence"] - result["first_confidence"]
            if result["first_confidence"] is not None and result["second_confidence"] is not None
            else None
        )
        delta_str = f"{delta:+d}" if delta is not None else "n/a"
        print(f"    R1={result['first_confidence']}  →  R2={result['second_confidence']}  Δ={delta_str}")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"    ERROR: {exc}")

    return result


def main():
    questions  = load_questions(DATA_FILE, BEGIN_IDX, MAX_QUESTIONS)
    total      = len(questions) * len(MODELS)
    end_idx    = BEGIN_IDX + len(questions) - 1
    if questions:
        print(
            f"Loaded HLE questions {BEGIN_IDX}-{end_idx} "
            f"(capped at {MAX_QUESTIONS}) × {len(MODELS)} models = {total} entries."
        )
    else:
        print(
            f"Loaded 0 HLE questions starting at {BEGIN_IDX} "
            f"(capped at {MAX_QUESTIONS}) × {len(MODELS)} models = {total} entries."
        )
    print(f"Output → {OUTPUT_FILE}\n")

    results    = load_saved_results(OUTPUT_FILE)
    completed  = 0
    start_time = time.time()

    if results:
        print(f"Loaded {len(results)} existing results from {OUTPUT_FILE}")

    for model in MODELS:
        print(f"\n=== model={model!r} ===")
        skip_model = False

        for q in questions:
            if skip_model:
                break

            print(f"  [{completed+1}/{total}] id={q['id']!r} category={q.get('category','?')!r}  {format_eta(start_time, completed, total)}")

            entry = {
                "question_id": q["id"],
                "question":    q["question"],
                "category":    q.get("category"),
                "model":       model,
            }

            outcome = new_3_turn(q["question"], model)
            entry.update(outcome)
            results.append(entry)
            completed += 1
            save(entry)

            error_message = outcome["error"] or ""
            if error_message and any(
                kw in error_message for kw in ("unavailable", "Rate limited")
            ):
                print(f"    SKIP MODEL: {error_message}")
                skip_model = True

            time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone. {len(results)} entries saved → {OUTPUT_FILE}")
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
            r["second_confidence"] - r["first_confidence"]
            for r in rows
            if r["first_confidence"] is not None and r["second_confidence"] is not None
        ]
        r1s = [r["first_confidence"]  for r in rows if r["first_confidence"]  is not None]
        r2s = [r["second_confidence"] for r in rows if r["second_confidence"] is not None]
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
