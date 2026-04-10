"""Analyze HLE result deltas by model and category.

The main signal in this project is the change in confidence between
the first and second ratings:

    delta = second_rating - first_rating

Negative deltas mean the model became less confident after explaining
itself, which is the IOED-style effect this project studies.

This script reads the saved HLE JSONL results and prints grouped
summary tables for:
  - overall dataset statistics
  - per-model statistics
  - per-category statistics
  - model x category statistics

Usage: put in the input for INPUT_FILE, OUTPUT_FILE
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from contextlib import redirect_stdout
from pathlib import Path


INPUT_FILE = Path(__file__).resolve().parents[2] / "results" / "HLE_results.jsonl"
OUTPUT_FILE = Path(__file__).resolve().parents[2] / "analysis" / "HLE_first_run"

REASONING_MODELS = {
    "deepseek/deepseek-r1",
    "openai/o3-mini",
    "qwen/qwq-32b",
    "anthropic/claude-opus-4",
    "google/gemini-2.5-pro",
    "x-ai/grok-3",
}


@dataclass(frozen=True)
class Row:
    """Normalized view of one JSONL entry."""

    question_id: str
    model: str
    category: str
    first_rating: int
    second_rating: int

    @property
    def delta(self) -> int:
        return self.second_rating - self.first_rating


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc

            first_rating = record.get("first_rating")
            second_rating = record.get("second_rating")
            question_id = record.get("question_id") or record.get("id") or "unknown"
            model = record.get("model") or "unknown"
            category = record.get("category") or "unknown"

            if first_rating is None or second_rating is None:
                continue

            rows.append(
                Row(
                    question_id=str(question_id),
                    model=str(model),
                    category=str(category),
                    first_rating=int(first_rating),
                    second_rating=int(second_rating),
                )
            )

    return rows


def summarize(rows: list[Row]) -> dict[str, float | int]:
    deltas = [row.delta for row in rows]
    first_ratings = [row.first_rating for row in rows]
    second_ratings = [row.second_rating for row in rows]

    return {
        "n": len(rows),
        "avg_first": statistics.mean(first_ratings) if first_ratings else float("nan"),
        "avg_second": statistics.mean(second_ratings) if second_ratings else float("nan"),
        "avg_delta": statistics.mean(deltas) if deltas else float("nan"),
        "median_delta": statistics.median(deltas) if deltas else float("nan"),
        "var_delta": statistics.pvariance(deltas) if deltas else float("nan"),
        "drop_rate": (sum(1 for delta in deltas if delta < 0) / len(deltas)) if deltas else float("nan"),
        "rise_rate": (sum(1 for delta in deltas if delta > 0) / len(deltas)) if deltas else float("nan"),
        "flat_rate": (sum(1 for delta in deltas if delta == 0) / len(deltas)) if deltas else float("nan"),
    }


def group_rows(rows: list[Row], key_fn) -> dict[str, list[Row]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def format_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def format_num(value: float) -> str:
    return f"{value:7.2f}"


def display_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _group_label_width(grouped: dict[str, list[Row]], minimum: int) -> int:
    widest = max((len(name) for name in grouped), default=minimum)
    return max(minimum, widest)


def print_table(title: str, grouped: dict[str, list[Row]], *, include_category: bool = False) -> None:
    print(f"\n{title}")
    label_width = _group_label_width(grouped, 42 if include_category else 30)
    if include_category:
        header = f"{'Group':<{label_width}} {'n':>5} {'R1':>7} {'R2':>7} {'Δ':>7} {'medΔ':>7} {'varΔ':>7} {'drop%':>8} {'rise%':>8} {'flat%':>8}"
    else:
        header = f"{'Group':<{label_width}} {'n':>5} {'R1':>7} {'R2':>7} {'Δ':>7} {'medΔ':>7} {'varΔ':>7} {'drop%':>8} {'rise%':>8} {'flat%':>8}"
    print(header)
    print("-" * len(header))

    for group_name in sorted(grouped):
        summary = summarize(grouped[group_name])
        print(
            f"{group_name:<{label_width}} "
            f"{summary['n']:>5} "
            f"{format_num(summary['avg_first']):>7} "
            f"{format_num(summary['avg_second']):>7} "
            f"{format_num(summary['avg_delta']):>7} "
            f"{format_num(summary['median_delta']):>7} "
            f"{format_num(summary['var_delta']):>7} "
            f"{format_pct(summary['drop_rate']):>8} "
            f"{format_pct(summary['rise_rate']):>8} "
            f"{format_pct(summary['flat_rate']):>8}"
        )


def print_model_table(grouped: dict[str, list[Row]]) -> None:
    title = "\nBy Model"
    print(title)
    label_width = max((len(display_model_name(name)) for name in grouped), default=30)
    label_width = max(label_width, len("Group"))
    header = (
        f"{'Group':<{label_width}} {'reasoning':>14} {'n':>5} {'R1':>7} {'R2':>7} "
        f"{'Δ':>7} {'medΔ':>7} {'varΔ':>7} {'drop%':>8} {'rise%':>8} {'flat%':>8}"
    )
    print(header)
    print("-" * len(header))

    for model_name in sorted(grouped):
        summary = summarize(grouped[model_name])
        reasoning_label = "Y" if model_name in REASONING_MODELS else "N"
        shown_name = display_model_name(model_name)
        print(
            f"{shown_name:<{label_width}} "
            f"{reasoning_label:>14} "
            f"{summary['n']:>5} "
            f"{format_num(summary['avg_first']):>7} "
            f"{format_num(summary['avg_second']):>7} "
            f"{format_num(summary['avg_delta']):>7} "
            f"{format_num(summary['median_delta']):>7} "
            f"{format_num(summary['var_delta']):>7} "
            f"{format_pct(summary['drop_rate']):>8} "
            f"{format_pct(summary['rise_rate']):>8} "
            f"{format_pct(summary['flat_rate']):>8}"
        )


def print_model_category_table(rows: list[Row]) -> None:
    grouped = group_rows(rows, lambda row: (row.model, row.category))
    model_width = max((len(display_model_name(model)) for model, _ in grouped), default=5)
    category_width = max((len(category) for _, category in grouped), default=8)
    model_width = max(model_width, len("Model"))
    category_width = max(category_width, len("Category"))

    print("\nBy Model x Question Category")
    header = (
        f"{'Model':<{model_width}}  "
        f"{'Category':<{category_width}}  "
        f"{'n':>5} {'R1':>7} {'R2':>7} {'Δ':>7} {'medΔ':>7} {'varΔ':>7} {'drop%':>8} {'rise%':>8} {'flat%':>8}"
    )
    print(header)
    print("-" * len(header))

    for model, category in sorted(grouped):
        summary = summarize(grouped[(model, category)])
        shown_name = display_model_name(model)
        print(
            f"{shown_name:<{model_width}}  "
            f"{category:<{category_width}}  "
            f"{summary['n']:>5} "
            f"{format_num(summary['avg_first']):>7} "
            f"{format_num(summary['avg_second']):>7} "
            f"{format_num(summary['avg_delta']):>7} "
            f"{format_num(summary['median_delta']):>7} "
            f"{format_num(summary['var_delta']):>7} "
            f"{format_pct(summary['drop_rate']):>8} "
            f"{format_pct(summary['rise_rate']):>8} "
            f"{format_pct(summary['flat_rate']):>8}"
        )


def print_reasoning_category_table(rows: list[Row]) -> None:
    grouped = group_rows(
        rows,
        lambda row: "reasoning_model" if row.model in REASONING_MODELS else "non_reasoning",
    )
    print_table("\nBy Reasoning Category", grouped)


def print_overall(rows: list[Row]) -> None:
    summary = summarize(rows)
    print("Overall")
    print(f"  n            : {summary['n']}")
    print(f"  avg first    : {summary['avg_first']:.2f}")
    print(f"  avg second   : {summary['avg_second']:.2f}")
    print(f"  avg delta    : {summary['avg_delta']:+.2f}")
    print(f"  median delta : {summary['median_delta']:+.2f}")
    print(f"  var delta    : {summary['var_delta']:.2f}")
    print(f"  drop rate    : {summary['drop_rate'] * 100:.1f}%")
    print(f"  rise rate    : {summary['rise_rate'] * 100:.1f}%")
    print(f"  flat rate    : {summary['flat_rate'] * 100:.1f}%")


def print_extremes(rows: list[Row], limit: int = 10) -> None:
    sorted_rows = sorted(rows, key=lambda row: (row.delta, row.model, row.category))

    model_width = max((len(display_model_name(row.model)) for row in rows), default=35)
    model_width = max(model_width, len("model"))

    print("\nLargest confidence drops (lowest deltas)")
    print(f"{'model':<{model_width}} {'category':<26} {'question_id':<25} {'R1':>4} {'R2':>4} {'Δ':>4}")
    print("-" * (model_width + 42))
    for row in sorted_rows[:limit]:
        shown_name = display_model_name(row.model)
        print(f"{shown_name:<{model_width}} {row.category:<26} {row.question_id:<25} {row.first_rating:>4} {row.second_rating:>4} {row.delta:>4}")

    print("\nLargest confidence increases (highest deltas)")
    print(f"{'model':<{model_width}} {'category':<26} {'question_id':<25} {'R1':>4} {'R2':>4} {'Δ':>4}")
    print("-" * (model_width + 42))
    for row in sorted_rows[-limit:][::-1]:
        shown_name = display_model_name(row.model)
        print(f"{shown_name:<{model_width}} {row.category:<26} {row.question_id:<25} {row.first_rating:>4} {row.second_rating:>4} {row.delta:>4}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HLE first/second rating deltas.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE, help="Path to HLE_results.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Path to write the text report")
    parser.add_argument("--top-n", type=int, default=10, help="How many extreme rows to show")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)

    if not rows:
        raise SystemExit(f"No usable rows found in {args.input}")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"Loaded {len(rows)} scored rows from {args.input}")
        print_overall(rows)

        by_model = group_rows(rows, lambda row: row.model)
        by_category = group_rows(rows, lambda row: row.category)
        print_model_table(by_model)
        print_reasoning_category_table(rows)
        print_table("\nBy Question Category", by_category)
        print_model_category_table(rows)
        print_extremes(rows, limit=args.top_n)

    report = buffer.getvalue()
    print(report, end="")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()