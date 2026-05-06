#!/usr/bin/env python3
"""
Launch one resumable Rozenblit eval subprocess per model (distinct JSONL per worker).

Models are taken from evals.core.pipeline.MODELS unless --models overrides.

Examples (from repo root):

  python3 scripts/run_rozenblit_parallel.py
  python3 scripts/run_rozenblit_parallel.py --jobs 0
  python3 scripts/run_rozenblit_parallel.py --dry-run

After all workers finish, summarise shards:

  python3 evals/rozenblit/eval_rozenblit.py --summary-only results/rozenblit_worker_*.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "evals" / "rozenblit" / "eval_rozenblit.py"
DEFAULT_RESULTS = ROOT / "results"


def _shard_filename(model: str) -> str:
    safe = model.replace("/", "_").replace(":", "_").replace(" ", "_")
    return f"rozenblit_worker_{safe}.jsonl"


def _load_models(override: str | None) -> list[str]:
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    sys.path.insert(0, str(ROOT))
    from evals.core.pipeline import MODELS

    return list(MODELS)


def _run_one(model: str, output_dir: Path) -> tuple[str, int, Path]:
    out = output_dir / _shard_filename(model)
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "-o",
        str(out),
        "--models",
        model,
    ]
    print(f"[start] {model}\n        -> {out}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    print(f"[exit {proc.returncode}] {model}", flush=True)
    return model, proc.returncode, out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Rozenblit eval with one subprocess per model (separate JSONL per model).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Max concurrent model workers. Use 0 to run all models at once (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Directory for worker JSONL files (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated OpenRouter slugs (default: pipeline.MODELS).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not spawn subprocesses.",
    )
    args = parser.parse_args()

    if not EVAL_SCRIPT.is_file():
        print(f"Missing {EVAL_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    models = _load_models(args.models)
    if not models:
        print("No models to run.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_workers = len(models) if args.jobs == 0 else min(max(args.jobs, 1), len(models))

    if args.dry_run:
        for m in models:
            out = args.output_dir / _shard_filename(m)
            print(
                f"Would run: {sys.executable} {EVAL_SCRIPT} -o {out} --models {m}",
                flush=True,
            )
        print(f"\nDry run: {len(models)} workers, max concurrent = {n_workers}.")
        return

    failures: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, m, args.output_dir): m for m in models}
        for fut in as_completed(futures):
            model, code, _out = fut.result()
            if code != 0:
                failures.append((model, code))

    print("\nAll workers finished.")
    if failures:
        print("Non-zero exits:", file=sys.stderr)
        for m, c in failures:
            print(f"  {m}  exit {c}", file=sys.stderr)
        sys.exit(1)

    shard_glob = args.output_dir / "rozenblit_worker_*.jsonl"
    print(
        "\nSummarise combined shards:\n"
        f"  python3 evals/rozenblit/eval_rozenblit.py --summary-only {shard_glob}"
    )


if __name__ == "__main__":
    main()
