"""Run the HLE methodology for the DeepSeek worker slice.

This is a thin wrapper around ``evals.hle.new_methodology`` that pins the
target question and writes to the DeepSeek worker JSONL file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("HLE_TARGET_QUESTION_ID", "67343cda5b69aac4188acc6e")
os.environ.setdefault("HLE_MAX_PARALLEL_MODELS", "1")
os.environ.setdefault("HLE_MAX_PARALLEL_QUESTIONS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


MODEL = "deepseek/deepseek-r1"
OUTPUT_FILE = (
	Path(__file__).resolve().parents[2]
	/ "results"
	/ "HLE"
	/ "new_method"
	/ "hle_worker_deepseek_deepseek-r1.jsonl"
)


def main() -> None:
	from evals.hle.new_methodology import main as run_hle

	run_hle(models=[MODEL], output_file=OUTPUT_FILE)


if __name__ == "__main__":
	main()
