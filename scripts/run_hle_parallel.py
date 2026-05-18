#!/usr/bin/env python3
"""
Launch one resumable HLE eval subprocess per model (distinct JSONL per worker).

Models are taken from evals.core.pipeline.MODELS unless --models overrides.

Examples (from repo root):

  python3 scripts/run_hle_parallel.py
  python3 scripts/run_hle_parallel.py --jobs 0
  python3 scripts/run_hle_parallel.py --dry-run

After all workers finish, summarise shards:

  python3 evals/hle/new_methodology.py --summary-only results/HLE/hle_worker_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "evals" / "hle" / "new_methodology.py"
DEFAULT_RESULTS = ROOT / "results" / "HLE"


def _shard_filename(model: str) -> str:
    safe = model.replace("/", "_").replace(":", "_").replace(" ", "_")
    return f"hle_worker_{safe}.jsonl"


def _load_models(override: str | None) -> list[str]:
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    sys.path.insert(0, str(ROOT))
    from evals.core.pipeline import MODELS

    return list(MODELS)


def _safe_shard_name(model: str) -> str:
    return _shard_filename(model)


def _load_completed_keys(path: Path) -> set[tuple[str, str, str, int]]:
    keys: set[tuple[str, str, str, int]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("error"):
                    continue
                qid = str(row.get("question_id"))
                model = str(row.get("model"))
                arm = str(row.get("arm", "explanation"))
                sample_index = int(row.get("sample_index", 0))
                keys.add((qid, model, arm, sample_index))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return keys


def split_global_to_shards(global_file: Path, output_dir: Path) -> None:
    """Split a global JSONL into per-model shard files in output_dir.

    Avoid writing duplicate (question_id, model, arm, sample_index) entries.
    """
    if not global_file.exists():
        print(f"Split source missing: {global_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache existing keys per shard to avoid duplicates
    shard_keys: dict[str, set[tuple[str, str, str, int]]] = {}

    with open(global_file, encoding="utf-8") as gf:
        for line in gf:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = str(row.get("model") or "unknown")
            shard_name = _safe_shard_name(model)
            shard_path = output_dir / shard_name

            if shard_name not in shard_keys:
                shard_keys[shard_name] = _load_completed_keys(shard_path)

            key = (str(row.get("question_id")), model, str(row.get("arm", "explanation")), int(row.get("sample_index", 0)))
            if key in shard_keys[shard_name]:
                continue

            # append row to shard
            with open(shard_path, "a", encoding="utf-8") as out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            shard_keys[shard_name].add(key)



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
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    print(f"[exit {proc.returncode}] {model}", flush=True)
    return model, proc.returncode, out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HLE eval with one subprocess per model (separate JSONL per model).",
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
    parser.add_argument(
        "--split-from",
        type=Path,
        default=None,
        help="Optional global JSONL file to split into per-model shards before running (default: none).",
    )
    args = parser.parse_args()

    if not EVAL_SCRIPT.is_file():
        print(f"Missing {EVAL_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    # If requested, split a global JSONL into per-model shards first
    if args.split_from:
        print(f"Splitting {args.split_from} → {args.output_dir}")
        split_global_to_shards(args.split_from, args.output_dir)

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

    shard_glob = args.output_dir / "hle_worker_*.jsonl"
    print(
        "\nSummarise combined shards:\n"
        f"  python3 evals/hle/new_methodology.py --summary-only {shard_glob}"
    )


if __name__ == "__main__":
    main()
