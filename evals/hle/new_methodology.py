"""
Humanity's Last Exam (HLE) eval — difficult for both humans and AI.

This version uses the Rozenblit protocol machinery:
- anchored confidence scales
- structured JSON on rating turns
- explanation and control arms
- five replications per (model, question, arm)

Data format: JSONL, each line has {id, question, category}.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.core.anchors import ROZENBLIT_ANCHOR_SYSTEM
from evals.core.pipeline import MODELS, RATE_LIMIT_DELAY, format_eta
from evals.core.structured import (
    ModelNotFoundError,
    ModelRateLimitError,
    call_openrouter,
    extract_answer_and_confidence,
    extract_confidence,
)
from evals.core.text_analysis import score_explanation


DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "hle_100_sample.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results"
OUTPUT_FILE = OUTPUT_DIR / "new_method_sampled_HLE.jsonl"

# HLE sample has 100 questions.
# BEGIN_IDX is 1-based and inclusive.

BEGIN_IDX = int(os.getenv("HLE_BEGIN_IDX", "1"))
MAX_QUESTIONS = int(os.getenv("HLE_MAX_QUESTIONS", "65"))
TARGET_QUESTION_ID = os.getenv("HLE_TARGET_QUESTION_ID")

# Replicate the Rozenblit-style arms on every item.
K_SAMPLES = int(os.getenv("HLE_K_SAMPLES", "5"))
TEMPERATURE = float(os.getenv("HLE_TEMPERATURE", "0.7"))

# Max number of models to query in parallel at once. Tune to avoid 429s.
MAX_PARALLEL_MODELS = int(os.getenv("HLE_MAX_PARALLEL_MODELS", "3"))

# Max number of questions to process in parallel per model worker.
MAX_PARALLEL_QUESTIONS = int(os.getenv("HLE_MAX_PARALLEL_QUESTIONS", "1"))

ARMS = ("explanation", "control")
SYSTEM_MESSAGE = {"role": "system", "content": ROZENBLIT_ANCHOR_SYSTEM}


def load_questions(
    path: Path,
    begin_idx: int,
    limit: int,
    target_question_id: str | None = None,
) -> list[dict]:
    questions: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if target_question_id is not None:
                if str(row.get("id")) == target_question_id:
                    return [row]
                continue
            if idx < begin_idx:
                continue
            questions.append(row)
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


def load_completed(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()

    keys: set[tuple[str, str, str, int]] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("error"):
                    continue
                question_id = str(row["question_id"])
                model = str(row["model"])
                arm = str(row.get("arm", "explanation"))
                sample_index = int(row.get("sample_index", 0))
                keys.add((question_id, model, arm, sample_index))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return keys


def save(entry: dict, path: Path | None = None) -> None:
    out = path if path is not None else OUTPUT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _t1_prompt(question: str) -> str:
    return (
        f"Question: {question}\n\n"
        "Give a brief one-sentence answer, then rate your confidence on the "
        "0-10 scale described in the system message.\n\n"
        'Reply with valid JSON only, in this exact form:\n'
        '{"answer": "<your brief answer>", "confidence": <integer 0-10>}'
    )


T2_EXPLANATION_PROMPT = (
    "Now please explain step-by-step how you arrived at your answer. "
    "Walk through the reasoning in as much causal detail as you can, "
    "without skipping intermediate steps. Reply in plain prose; no JSON."
)


T3_RATING_PROMPT = (
    "Given everything you have written so far, how well do you now feel "
    "you understand the original question? Use the same 0-10 scale from "
    "the system message.\n\n"
    'Reply with valid JSON only, in this exact form:\n'
    '{"confidence": <integer 0-10>}'
)


def _delta(first_confidence: int | None, second_confidence: int | None) -> int | None:
    if first_confidence is None or second_confidence is None:
        return None
    return second_confidence - first_confidence


def _missing_keys_for_question(
    question_id: str,
    model: str,
    completed_keys: set[tuple[str, str, str, int]],
) -> list[tuple[str, str, str, int]]:
    return [
        (question_id, model, arm, sample_index)
        for arm in ARMS
        for sample_index in range(K_SAMPLES)
        if (question_id, model, arm, sample_index) not in completed_keys
    ]
def _safe_call(
    messages: list[dict],
    model: str,
    temperature: float,
    json_object: bool,
) -> tuple[str | None, str | None]:
    try:
        text = call_openrouter(
            messages,
            model,
            temperature=temperature,
            json_object=json_object,
        )
        return text, None
    except (ModelNotFoundError, ModelRateLimitError, RuntimeError) as exc:
        return None, str(exc)


def _base_row(item: dict, model: str, arm: str, sample_index: int, temperature: float) -> dict:
    return {
        "question_id": item["id"],
        "question": item["question"],
        "category": item.get("category"),
        "model": model,
        "arm": arm,
        "sample_index": sample_index,
        "temperature": temperature,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "first_answer": None,
        "first_confidence": None,
        "first_rating_src": None,
        "first_raw_text": None,
        "answer": None,
        "explanation": None,
        "explanation_scores": None,
        "second_confidence": None,
        "second_rating_src": None,
        "second_raw_text": None,
        "first_rating": None,
        "second_rating": None,
        "delta": None,
        "error": None,
    }


def _run_t1(question: str, model: str, temperature: float) -> tuple[list[dict], dict]:
    conversation = [SYSTEM_MESSAGE, {"role": "user", "content": _t1_prompt(question)}]
    turn = {
        "raw_text": None,
        "answer": None,
        "confidence": None,
        "rating_src": None,
        "error": None,
    }

    text, err = _safe_call(conversation, model, temperature, json_object=True)
    if err:
        turn["error"] = err
        return conversation, turn

    turn["raw_text"] = text
    answer, confidence, src = extract_answer_and_confidence(text)
    turn["answer"] = answer
    turn["confidence"] = confidence
    turn["rating_src"] = src
    conversation.append({"role": "assistant", "content": text})
    return conversation, turn


def _run_explanation_sample(item: dict, model: str, sample_index: int, temperature: float) -> dict:
    row = _base_row(item, model, "explanation", sample_index, temperature)
    conversation, t1 = _run_t1(item["question"], model, temperature)
    row["first_answer"] = t1["answer"]
    row["first_confidence"] = t1["confidence"]
    row["first_rating_src"] = t1["rating_src"]
    row["first_raw_text"] = t1["raw_text"]
    row["answer"] = t1["answer"]
    row["first_rating"] = t1["confidence"]
    if t1["error"]:
        row["error"] = t1["error"]
        return row

    print(
        f"    [{row['arm']:11s} k={sample_index}] T1 answer: {str(t1['raw_text'])[:80].strip()!r} "
        f"R1={t1['confidence']}"
    )
    time.sleep(RATE_LIMIT_DELAY)

    conversation.append({"role": "user", "content": T2_EXPLANATION_PROMPT})
    explanation, err = _safe_call(conversation, model, temperature, json_object=False)
    if err:
        row["error"] = err
        return row
    row["explanation"] = explanation
    row["explanation_scores"] = score_explanation(explanation)
    conversation.append({"role": "assistant", "content": explanation})
    print(
        f"    [{row['arm']:11s} k={sample_index}] T2 explanation: {len(explanation.split())} words  "
        f"[unc={(row['explanation_scores'] or {}).get('uncertainty', float('nan')):.2f} "
        f"con={(row['explanation_scores'] or {}).get('confidence', float('nan')):.2f} "
        f"net={(row['explanation_scores'] or {}).get('net_epistemic', float('nan')):+.2f}]"
    )
    time.sleep(RATE_LIMIT_DELAY)

    conversation.append({"role": "user", "content": T3_RATING_PROMPT})
    final_text, err = _safe_call(conversation, model, temperature, json_object=True)
    if err:
        row["error"] = err
        return row

    row["second_raw_text"] = final_text
    row["second_confidence"], row["second_rating_src"] = extract_confidence(final_text)
    row["second_rating"] = row["second_confidence"]
    row["delta"] = _delta(row["first_confidence"], row["second_confidence"])
    print(
        f"    [{row['arm']:11s} k={sample_index}] R1={row['first_confidence']}  →  R2={row['second_confidence']}  Δ={row['delta'] if row['delta'] is not None else 'n/a'}"
    )
    return row


def _run_control_sample(item: dict, model: str, sample_index: int, temperature: float) -> dict:
    row = _base_row(item, model, "control", sample_index, temperature)
    conversation, t1 = _run_t1(item["question"], model, temperature)
    row["first_answer"] = t1["answer"]
    row["first_confidence"] = t1["confidence"]
    row["first_rating_src"] = t1["rating_src"]
    row["first_raw_text"] = t1["raw_text"]
    row["answer"] = t1["answer"]
    row["first_rating"] = t1["confidence"]
    if t1["error"]:
        row["error"] = t1["error"]
        return row

    print(
        f"    [{row['arm']:11s} k={sample_index}] T1 answer: {str(t1['raw_text'])[:80].strip()!r} "
        f"R1={t1['confidence']}"
    )
    time.sleep(RATE_LIMIT_DELAY)

    conversation.append({"role": "user", "content": T3_RATING_PROMPT})
    final_text, err = _safe_call(conversation, model, temperature, json_object=True)
    if err:
        row["error"] = err
        return row

    row["second_raw_text"] = final_text
    row["second_confidence"], row["second_rating_src"] = extract_confidence(final_text)
    row["second_rating"] = row["second_confidence"]
    row["delta"] = _delta(row["first_confidence"], row["second_confidence"])
    print(
        f"    [{row['arm']:11s} k={sample_index}] R1={row['first_confidence']}  →  R2={row['second_confidence']}  Δ={row['delta'] if row['delta'] is not None else 'n/a'}"
    )
    return row


def _run_model_for_question(
    item: dict,
    model: str,
    completed_keys: set[tuple[str, str, str, int]] | None = None,
) -> list[dict]:
    sample_rows: list[dict] = []
    try:
        for arm in ARMS:
            for sample_index in range(K_SAMPLES):
                if completed_keys is not None:
                    key = (str(item["id"]), model, arm, sample_index)
                    if key in completed_keys:
                        continue
                time.sleep(random.uniform(0, 0.25))
                if arm == "explanation":
                    row = _run_explanation_sample(item, model, sample_index, TEMPERATURE)
                else:
                    row = _run_control_sample(item, model, sample_index, TEMPERATURE)
                sample_rows.append(row)

                error_message = str(row.get("error") or "")
                if error_message and any(
                    kw in error_message for kw in ("unavailable", "Rate limited", "Rate limited for")
                ):
                    print(f"    SKIP MODEL {model!r}: {error_message}")
                    return sample_rows
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        sample_rows.append(
            {
                "question_id": item["id"],
                "question": item["question"],
                "category": item.get("category"),
                "model": model,
                "arm": "explanation",
                "sample_index": 0,
                "temperature": TEMPERATURE,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "first_answer": None,
                "first_confidence": None,
                "first_rating_src": None,
                "first_raw_text": None,
                "answer": None,
                "explanation": None,
                "explanation_scores": None,
                "second_confidence": None,
                "second_rating_src": None,
                "second_raw_text": None,
                "first_rating": None,
                "second_rating": None,
                "delta": None,
                "error": f"Thread exception: {exc}",
            }
        )
    return sample_rows


def _is_skip_error(msg: str | None) -> bool:
    if not msg:
        return False
    return any(kw in msg for kw in ("unavailable", "Rate limited", "Rate limited for"))


def main(models: list[str] | None = None, output_file: Path | None = None) -> None:
    model_list = models if models is not None else MODELS
    out_path = output_file if output_file is not None else OUTPUT_FILE

    questions = load_questions(DATA_FILE, BEGIN_IDX, MAX_QUESTIONS, TARGET_QUESTION_ID)
    total = len(questions) * len(model_list) * len(ARMS) * K_SAMPLES
    end_idx = BEGIN_IDX + len(questions) - 1
    if questions:
        print(
            f"Loaded HLE questions {BEGIN_IDX}-{end_idx} "
            f"(capped at {MAX_QUESTIONS}) × {len(model_list)} models × {len(ARMS)} arms × {K_SAMPLES} samples = {total} entries."
        )
    else:
        print(
            f"Loaded 0 HLE questions starting at {BEGIN_IDX} "
            f"(capped at {MAX_QUESTIONS}) × {len(model_list)} models × {len(ARMS)} arms × {K_SAMPLES} samples = {total} entries."
        )
    print(f"Output → {out_path}\n")
    results = load_saved_results(out_path)
    completed_keys = load_completed(out_path)
    start_time = time.time()
    state_lock = Lock()
    progress_lock = Lock()
    progress = {"completed": 0}

    if results:
        print(f"Loaded {len(results)} existing results from {out_path}")

    def _record_row(row: dict) -> int:
        key = (
            str(row["question_id"]),
            str(row["model"]),
            str(row.get("arm", "explanation")),
            int(row.get("sample_index", 0)),
        )
        with state_lock:
            results.append(row)
            completed_keys.add(key)
            save(row, path=out_path)
        with progress_lock:
            progress["completed"] += 1
            return progress["completed"]

    def _run_model_worker(model: str) -> None:
        model_completed = {key for key in completed_keys if key[1] == model}
        pending_questions: list[tuple[dict, list[tuple[str, str, str, int]]]] = []
        pending_rows = 0

        with progress_lock:
            progress["completed"] = len(model_completed)

        for item in questions:
            missing_keys = _missing_keys_for_question(str(item["id"]), model, model_completed)
            if missing_keys:
                pending_questions.append((item, missing_keys))
                pending_rows += len(missing_keys)

        if not pending_questions:
            print(f"  [{model}] already complete: no missing entries")
            return

        print(
            f"  [{model}] resuming with {len(pending_questions)} questions and {pending_rows} missing rows"
        )

        def _process_question(q: dict) -> None:
            with progress_lock:
                current_completed = progress["completed"]
            print(
                f"  [{current_completed + 1}/{total}] id={q['id']!r} model={model!r} "
                f"category={q.get('category','?')!r}  {format_eta(start_time, current_completed, total)}"
            )

            new_rows = _run_model_for_question(q, model, completed_keys=model_completed)
            for row in new_rows:
                _record_row(row)
                model_completed.add(
                    (
                        str(row["question_id"]),
                        str(row["model"]),
                        str(row.get("arm", "explanation")),
                        int(row.get("sample_index", 0)),
                    )
                )

                if _is_skip_error(row.get("error")):
                    print(f"    SKIP MODEL {model!r}: {row.get('error')}")
                    continue

            time.sleep(RATE_LIMIT_DELAY)

        if MAX_PARALLEL_QUESTIONS <= 1 or len(pending_questions) == 1:
            for q, _missing_keys in pending_questions:
                _process_question(q)
        else:
            max_question_workers = min(MAX_PARALLEL_QUESTIONS, len(pending_questions))
            print(f"  [{model}] question-parallelism = {max_question_workers}")
            with ThreadPoolExecutor(max_workers=max_question_workers) as ex:
                futures = [ex.submit(_process_question, q) for q, _missing_keys in pending_questions]
                for fut in as_completed(futures):
                    fut.result()

    max_workers = min(MAX_PARALLEL_MODELS, len(model_list))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_model_worker, model): model for model in model_list}
        for fut in as_completed(futures):
            fut.result()

    print(f"\nDone. {len(results)} entries saved → {out_path}")
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    groups: dict[str, list[dict]] = {}
    for row in results:
        if row.get("error"):
            continue
        groups.setdefault(str(row["model"]), []).append(row)

    print(f"\n{'Model':<40} {'Δ expl':>8} {'Δ ctrl':>8} {'IOED':>8} {'n_e':>5} {'n_c':>5}")
    print("-" * 65)
    for model, model_rows in sorted(groups.items()):
        expl = [row["delta"] for row in model_rows if row.get("arm") == "explanation" and row.get("delta") is not None]
        ctrl = [row["delta"] for row in model_rows if row.get("arm") == "control" and row.get("delta") is not None]

        if expl and ctrl:
            ioed = statistics.mean(expl) - statistics.mean(ctrl)
            print(
                f"{model.split('/')[1]:<40} "
                f"{statistics.mean(expl):>8.2f} {statistics.mean(ctrl):>8.2f} {ioed:>8.2f} "
                f"{len(expl):>5} {len(ctrl):>5}"
            )
        else:
            print(f"{model.split('/')[1]:<40} {'—':>8} {'—':>8} {'—':>8} {len(expl):>5} {len(ctrl):>5}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HLE Rozenblit-style eval — resumable JSONL; use distinct -o per parallel worker.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"JSONL output path (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated OpenRouter slugs for this run only (default: evals.core.pipeline.MODELS).",
    )
    parser.add_argument(
        "--summary-only",
        type=Path,
        nargs="+",
        metavar="JSONL",
        help="Print summary tables from existing JSONL file(s) and exit.",
    )
    args = parser.parse_args()

    if args.summary_only:
        rows = []
        for p in args.summary_only:
            if not p.exists():
                continue
            rows.extend(load_saved_results(p))
        _print_summary(rows)
        sys.exit(0)

    selected_output_path = args.output if args.output is not None else OUTPUT_FILE
    model_subset = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else None
    )
    main(models=model_subset, output_file=selected_output_path)
