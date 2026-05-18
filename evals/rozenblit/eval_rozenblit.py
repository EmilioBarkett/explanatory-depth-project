"""
Rozenblit & Keil (2002) replication eval.

For every (model, item) we run two arms:
  - explanation arm (T1 → explanation → diagnostic → re-rate)
  - control arm     (T1 → re-rate, with no intervening explanation)

Each arm is replicated K_SAMPLES times so we can compute per-item variance
and a clean IOED estimate:

    IOED = mean Δ_explanation - mean Δ_control

The diagnostic phase fires only on items that carry a `diagnostic_question`
in the dataset (devices + natural_phenomena), matching Rozenblit's protocol.

Output is JSONL, append-only and resumable: rerunning the script picks up
where it left off using the (question_id, model, arm, sample_index) key.

Parallel batch runs: use a different --output per process (same file will corrupt
if two writers append at once). Example:

  python evals/rozenblit/eval_rozenblit.py -o results/rozenblit_r1.jsonl \\
    --models deepseek/deepseek-r1
  python evals/rozenblit/eval_rozenblit.py -o results/rozenblit_gpt4o.jsonl \\
    --models openai/gpt-4o

Then merge and summarise:

  python evals/rozenblit/eval_rozenblit.py --summary-only results/rozenblit_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.core.pipeline import MODELS  # reuse the curated model list
from evals.rozenblit.protocol import (
    K_SAMPLES,
    TEMPERATURE,
    run_arm_replicated,
    is_skip_error,
)

DATA_FILE   = Path(__file__).resolve().parents[2] / "data" / "rozenblit_dataset.json"
OUTPUT_DIR  = Path(__file__).resolve().parents[2] / "results"
OUTPUT_FILE = OUTPUT_DIR / "rozenblit_results.jsonl"

ARMS = ("explanation", "control")


# ── Loaders / IO ──────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        studies = json.load(f)
    out: list[dict] = []
    for study in studies:
        for item in study["items"]:
            out.append({
                "id":                  item["id"],
                "question":            item["question"],
                "category":            item["category"],
                "study":               study["study"],
                "is_test_item":        item.get("is_test_item", False),
                "diagnostic_question": item.get("diagnostic_question"),
            })
    return out


def load_completed(path: Path) -> set[tuple[str, str, str, int]]:
    """Return the set of (question_id, model, arm, sample_index) keys already saved."""
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
                keys.add((row["question_id"], row["model"], row["arm"], row["sample_index"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Driver ────────────────────────────────────────────────────────────────────

def main(models: list[str] | None = None, output_file: Path | None = None) -> None:
    model_list = models if models is not None else MODELS
    out_path = output_file if output_file is not None else OUTPUT_FILE

    questions = load_questions(DATA_FILE)
    completed = load_completed(out_path)

    # Total expected entries:
    #   per question: K_SAMPLES * 2 arms (explanation + control)
    total_expected = len(questions) * len(model_list) * len(ARMS) * K_SAMPLES
    todo_estimate  = total_expected - len(completed)

    print(
        f"Loaded {len(questions)} items × {len(model_list)} models × {len(ARMS)} arms "
        f"× {K_SAMPLES} samples = {total_expected} entries.\n"
        f"Already completed: {len(completed)}.  Remaining estimate: {todo_estimate}.\n"
        f"Temperature: {TEMPERATURE}.  Output → {out_path}\n"
    )
    if datetime.now():  # informational header line
        print(f"Run started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    start_time = time.time()
    completed_now = 0

    for model in model_list:
        print(f"\n=== model={model!r} ===")
        skip_model = False

        for q in questions:
            if skip_model:
                break

            for arm in ARMS:
                if skip_model:
                    break

                # Skip samples already on disk.
                missing_indices = [
                    k for k in range(K_SAMPLES)
                    if (q["id"], model, arm, k) not in completed
                ]
                if not missing_indices:
                    continue

                print(
                    f"  [{q['study']:<17s} {q['id']:<7s}  arm={arm:<11s}] "
                    f"need {len(missing_indices)} samples"
                )

                samples = run_arm_replicated(
                    item=q,
                    model=model,
                    arm=arm,
                    k_samples=len(missing_indices),
                    temperature=TEMPERATURE,
                )

                # The protocol numbers samples 0..len(missing)-1; remap to actual
                # global indices so keys stay stable across resumes.
                for local_k, sample in zip(missing_indices, samples):
                    sample.sample_index = local_k
                    row = {
                        "question_id":         q["id"],
                        "question":            q["question"],
                        "category":            q["category"],
                        "study":               q["study"],
                        "is_test_item":        q["is_test_item"],
                        "has_diagnostic":      q["diagnostic_question"] is not None,
                        "model":               model,
                        "arm":                 arm,
                        "sample_index":        local_k,
                        "temperature":         TEMPERATURE,
                        "timestamp":           datetime.utcnow().isoformat() + "Z",
                        **sample.to_dict(),
                    }
                    append_jsonl(out_path, row)
                    completed.add((q["id"], model, arm, local_k))
                    completed_now += 1

                    if is_skip_error(sample.error):
                        print(f"    SKIP MODEL: {sample.error}")
                        skip_model = True

                if completed_now > 0:
                    elapsed = time.time() - start_time
                    rate = elapsed / completed_now
                    remaining = max(0, todo_estimate - completed_now)
                    eta_min = int((rate * remaining) // 60)
                    print(f"    progress: {completed_now}/{todo_estimate}  ETA ~{eta_min}m")

    print(f"\nDone. Results written to {out_path}")
    print_summary(out_path)


# ── Summary ───────────────────────────────────────────────────────────────────

def _delta(row: dict) -> int | None:
    t1 = row.get("t1") or {}
    final = row.get("final") or {}
    r1 = t1.get("confidence")
    r2 = final.get("confidence")
    if r1 is None or r2 is None:
        return None
    return r2 - r1


def _load_jsonl_paths(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def print_summary(path_or_paths: Path | list[Path]) -> None:
    paths = [path_or_paths] if isinstance(path_or_paths, Path) else path_or_paths
    rows = _load_jsonl_paths(paths)
    if not rows:
        print("No results rows to summarise (missing or empty JSONL).")
        return

    # group by (study, model, arm)
    grouped: dict[tuple, list[int]] = {}
    for r in rows:
        if r.get("error"):
            continue
        d = _delta(r)
        if d is None:
            continue
        key = (r["study"], r["model"], r["arm"])
        grouped.setdefault(key, []).append(d)

    print("\n────────────────────────────────────────────────────────────────────────")
    print("Per-arm mean Δ (R2 − R1).  IOED = Δ_explanation − Δ_control.")
    print("────────────────────────────────────────────────────────────────────────")
    header = f"{'Study':<18} {'Model':<32} {'Δ expl':>8} {'Δ ctrl':>8} {'IOED':>8} {'n_e':>5} {'n_c':>5}"
    print(header)
    print("-" * len(header))

    studies = sorted({k[0] for k in grouped})
    models  = sorted({k[1] for k in grouped})

    for study in studies:
        for model in models:
            expl = grouped.get((study, model, "explanation"), [])
            ctrl = grouped.get((study, model, "control"),     [])
            if not expl and not ctrl:
                continue
            mean_e = statistics.mean(expl) if expl else float("nan")
            mean_c = statistics.mean(ctrl) if ctrl else float("nan")
            ioed   = (mean_e - mean_c) if expl and ctrl else float("nan")
            short_model = model.split("/", 1)[-1]
            def f(x: float) -> str:
                return f"{x:+.2f}" if x == x else "  —  "  # NaN check
            print(
                f"{study:<18} {short_model:<32} {f(mean_e):>8} {f(mean_c):>8} "
                f"{f(ioed):>8} {len(expl):>5} {len(ctrl):>5}"
            )

    # Devices vs procedures within each model — Rozenblit's central contrast.
    print("\nDevices-vs-procedures contrast (explanation arm only):")
    print(f"{'Model':<32} {'Δ devices':>10} {'Δ procs':>10} {'gap':>8}")
    print("-" * 64)
    for model in models:
        d_dev = grouped.get(("devices",    model, "explanation"), [])
        d_pro = grouped.get(("procedures", model, "explanation"), [])
        if not d_dev or not d_pro:
            continue
        m_dev = statistics.mean(d_dev)
        m_pro = statistics.mean(d_pro)
        print(
            f"{model.split('/', 1)[-1]:<32} "
            f"{m_dev:+10.2f} {m_pro:+10.2f} {m_dev - m_pro:+8.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rozenblit IOED eval — resumable JSONL; use distinct -o per parallel worker.",
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
        print_summary(list(args.summary_only))
        sys.exit(0)

    out_path = args.output if args.output is not None else OUTPUT_FILE
    model_subset = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else None
    )
    main(models=model_subset, output_file=out_path)
