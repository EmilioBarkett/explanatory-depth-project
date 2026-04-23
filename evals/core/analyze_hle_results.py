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
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from contextlib import redirect_stdout
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


INPUT_FILE = Path(__file__).resolve().parents[2] / "results" / "HLE_results.jsonl"
OUTPUT_FILE = Path(__file__).resolve().parents[2] / "analysis" / "HLE"

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


@dataclass(frozen=True)
class InferentialStats:
    """Inferential metrics for a paired-rating delta series."""

    n: int
    mean_delta: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_t: float
    wilcoxon_stat: float
    p_wilcoxon: float
    p_sign: float
    cohens_dz: float


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


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values)


def _mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")

    mean_val = statistics.mean(values)
    std_val = _sample_std(values)
    se = std_val / math.sqrt(len(values))
    alpha = 1 - confidence

    if scipy_stats is not None:
        critical = scipy_stats.t.ppf(1 - alpha / 2, df=len(values) - 1)
    else:
        critical = NormalDist().inv_cdf(1 - alpha / 2)

    margin = critical * se
    return mean_val - margin, mean_val + margin


def _one_sample_t(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")

    if scipy_stats is not None:
        result = scipy_stats.ttest_1samp(values, popmean=0.0)
        return float(result.statistic), float(result.pvalue)

    # Normal approximation fallback if SciPy is unavailable.
    mean_val = statistics.mean(values)
    std_val = _sample_std(values)
    if std_val == 0 or math.isnan(std_val):
        return float("nan"), float("nan")
    z = mean_val / (std_val / math.sqrt(len(values)))
    p_value = 2 * (1 - NormalDist().cdf(abs(z)))
    return z, p_value


def _wilcoxon_signed_rank(values: list[float]) -> tuple[float, float]:
    if scipy_stats is None:
        return float("nan"), float("nan")

    non_zero = [value for value in values if value != 0]
    if len(non_zero) < 2:
        return float("nan"), float("nan")

    result = scipy_stats.wilcoxon(non_zero, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def _two_sided_sign_test_p(values: list[float]) -> float:
    pos = sum(1 for value in values if value > 0)
    neg = sum(1 for value in values if value < 0)
    n = pos + neg
    if n == 0:
        return float("nan")

    # Exact two-sided binomial sign test with p=0.5 under H0.
    k = min(pos, neg)
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * cdf)


def _cohens_dz(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    std_val = _sample_std(values)
    if std_val == 0 or math.isnan(std_val):
        return float("nan")
    return statistics.mean(values) / std_val


def inferential_stats(rows: list[Row]) -> InferentialStats:
    deltas = [float(row.delta) for row in rows]
    if not deltas:
        return InferentialStats(
            n=0,
            mean_delta=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            t_stat=float("nan"),
            p_t=float("nan"),
            wilcoxon_stat=float("nan"),
            p_wilcoxon=float("nan"),
            p_sign=float("nan"),
            cohens_dz=float("nan"),
        )

    ci_low, ci_high = _mean_ci(deltas, confidence=0.95)
    t_stat, p_t = _one_sample_t(deltas)
    wilcoxon_stat, p_wilcoxon = _wilcoxon_signed_rank(deltas)

    return InferentialStats(
        n=len(deltas),
        mean_delta=statistics.mean(deltas),
        ci_low=ci_low,
        ci_high=ci_high,
        t_stat=t_stat,
        p_t=p_t,
        wilcoxon_stat=wilcoxon_stat,
        p_wilcoxon=p_wilcoxon,
        p_sign=_two_sided_sign_test_p(deltas),
        cohens_dz=_cohens_dz(deltas),
    )


def summarize(rows: list[Row]) -> dict[str, float | int]:
    deltas = [row.delta for row in rows]
    first_ratings = [row.first_rating for row in rows]
    second_ratings = [row.second_rating for row in rows]
    inferential = inferential_stats(rows)

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
        "ci95_low": inferential.ci_low,
        "ci95_high": inferential.ci_high,
        "t_stat": inferential.t_stat,
        "p_t": inferential.p_t,
        "wilcoxon_stat": inferential.wilcoxon_stat,
        "p_wilcoxon": inferential.p_wilcoxon,
        "p_sign": inferential.p_sign,
        "cohens_dz": inferential.cohens_dz,
    }


def group_rows(rows: list[Row], key_fn: Callable[[Row], Any]) -> dict[Any, list[Row]]:
    groups: dict[Any, list[Row]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def format_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def format_num(value: float) -> str:
    return f"{value:7.2f}"


def format_p(value: float) -> str:
    if math.isnan(value):
        return "   nan"
    if value < 0.001:
        return "<0.001"
    return f"{value:6.3f}"


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


def print_inferential_table(title: str, grouped: dict[str, list[Row]]) -> None:
    print(f"\n{title}")
    label_width = _group_label_width(grouped, 30)
    header = (
        f"{'Group':<{label_width}} {'n':>5} {'meanΔ':>8} {'CI low':>8} {'CI high':>8} "
        f"{'t':>8} {'p(t)':>8} {'W':>8} {'p(W)':>8} {'p(sign)':>9} {'dz':>8}"
    )
    print(header)
    print("-" * len(header))

    for group_name in sorted(grouped):
        summary = summarize(grouped[group_name])
        print(
            f"{group_name:<{label_width}} "
            f"{summary['n']:>5} "
            f"{format_num(summary['avg_delta']):>8} "
            f"{format_num(summary['ci95_low']):>8} "
            f"{format_num(summary['ci95_high']):>8} "
            f"{format_num(summary['t_stat']):>8} "
            f"{format_p(summary['p_t']):>8} "
            f"{format_num(summary['wilcoxon_stat']):>8} "
            f"{format_p(summary['p_wilcoxon']):>8} "
            f"{format_p(summary['p_sign']):>9} "
            f"{format_num(summary['cohens_dz']):>8}"
        )


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
    print(f"  95% CI (Δ)   : [{summary['ci95_low']:+.2f}, {summary['ci95_high']:+.2f}]")
    print(f"  t-stat (Δ=0) : {summary['t_stat']:.2f}")
    print(f"  p-value (t)  : {format_p(summary['p_t']).strip()}")
    print(f"  Wilcoxon W   : {summary['wilcoxon_stat']:.2f}")
    print(f"  p-value (W)  : {format_p(summary['p_wilcoxon']).strip()}")
    print(f"  p-value sign : {format_p(summary['p_sign']).strip()}")
    print(f"  Cohen's dz   : {summary['cohens_dz']:.2f}")


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
        if scipy_stats is None:
            print("Warning: scipy not installed; t-test uses normal approximation and Wilcoxon p-values are unavailable.")
        print_overall(rows)

        by_model = group_rows(rows, lambda row: row.model)
        by_category = group_rows(rows, lambda row: row.category)
        print_model_table(by_model)
        print_reasoning_category_table(rows)
        print_table("\nBy Question Category", by_category)
        print_inferential_table("Inferential Stats by Model", by_model)
        print_inferential_table("Inferential Stats by Question Category", by_category)
        print_model_category_table(rows)
        print_extremes(rows, limit=args.top_n)

    report = buffer.getvalue()
    print(report, end="")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()