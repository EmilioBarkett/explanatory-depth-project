"""Generate publication-style plots for HLE confidence-shift analysis.

This script reads HLE JSONL rows with first_rating and second_rating,
computes delta = second_rating - first_rating, and writes figures to disk.

Usage:
  python evals/core/plot_hle_results.py
  python evals/core/plot_hle_results.py --input results/HLE_results.jsonl --output-dir analysis/hle_figures
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


INPUT_FILE = Path(__file__).resolve().parents[2] / "results" / "HLE_results.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "analysis" / "hle_figures"


@dataclass(frozen=True)
class Row:
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


def display_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")

    mean_val = statistics.mean(values)
    if len(values) < 2:
        return mean_val, float("nan"), float("nan")

    std_val = statistics.stdev(values)
    se = std_val / math.sqrt(len(values))
    alpha = 1 - confidence
    if scipy_stats is not None:
        critical = scipy_stats.t.ppf(1 - alpha / 2, df=len(values) - 1)
    else:
        critical = NormalDist().inv_cdf(1 - alpha / 2)

    margin = critical * se
    return mean_val, mean_val - margin, mean_val + margin


def plot_delta_hist(rows: list[Row], output_dir: Path, dpi: int) -> None:
    deltas = [row.delta for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = range(min(deltas) - 1, max(deltas) + 2)
    ax.hist(deltas, bins=bins, color="#2a9d8f", alpha=0.85, edgecolor="black")
    ax.axvline(0, color="#d62828", linestyle="--", linewidth=2, label="No change")
    ax.set_title("Distribution of Confidence Shift (Delta = R2 - R1)")
    ax.set_xlabel("Delta")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "delta_histogram.png", dpi=dpi)
    plt.close(fig)


def plot_mean_delta_by_group(
    grouped_values: dict[str, list[float]],
    title: str,
    xlabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    labels = sorted(grouped_values)
    means: list[float] = []
    err_low: list[float] = []
    err_high: list[float] = []

    for label in labels:
        mean_val, ci_low, ci_high = mean_ci(grouped_values[label])
        means.append(mean_val)
        if math.isnan(ci_low) or math.isnan(ci_high):
            err_low.append(0.0)
            err_high.append(0.0)
        else:
            err_low.append(mean_val - ci_low)
            err_high.append(ci_high - mean_val)

    # Order by mean shift for a cleaner forest-style view.
    order = sorted(range(len(labels)), key=lambda idx: means[idx])
    labels = [labels[idx] for idx in order]
    means = [means[idx] for idx in order]
    err_low = [err_low[idx] for idx in order]
    err_high = [err_high[idx] for idx in order]

    fig_height = max(4.0, 0.5 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_positions = list(range(len(labels)))
    ax.errorbar(
        means,
        y_positions,
        xerr=[err_low, err_high],
        fmt="o",
        color="#264653",
        ecolor="#457b9d",
        capsize=4,
    )
    ax.axvline(0, color="#d62828", linestyle="--", linewidth=1.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_rating_change(rows: list[Row], output_dir: Path, dpi: int) -> None:
    first_vals = [float(row.first_rating) for row in rows]
    second_vals = [float(row.second_rating) for row in rows]

    mean_first, low_first, high_first = mean_ci(first_vals)
    mean_second, low_second, high_second = mean_ci(second_vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = [0, 1]
    means = [mean_first, mean_second]
    labels = ["First rating (R1)", "Second rating (R2)"]

    yerr_low = [0.0 if math.isnan(low_first) else mean_first - low_first, 0.0 if math.isnan(low_second) else mean_second - low_second]
    yerr_high = [0.0 if math.isnan(high_first) else high_first - mean_first, 0.0 if math.isnan(high_second) else high_second - mean_second]

    ax.errorbar(x, means, yerr=[yerr_low, yerr_high], fmt="o-", capsize=5, color="#1d3557", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean confidence")
    ax.set_title("Average Confidence Before and After Explanation")
    ax.set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig(output_dir / "rating_change_overall.png", dpi=dpi)
    plt.close(fig)


def plot_delta_box_by_category(rows: list[Row], output_dir: Path, dpi: int) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.category].append(float(row.delta))

    categories = sorted(grouped)
    values = [grouped[category] for category in categories]

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.5 * len(categories) + 2)))
    ax.boxplot(values, labels=categories, vert=False, patch_artist=True)
    ax.axvline(0, color="#d62828", linestyle="--", linewidth=1.5)
    ax.set_title("Delta Distribution by Question Category")
    ax.set_xlabel("Delta (R2 - R1)")
    ax.set_ylabel("Category")
    fig.tight_layout()
    fig.savefig(output_dir / "delta_boxplot_by_category.png", dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HLE confidence-shift plots.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE, help="Path to HLE_results.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to save plots")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No usable rows found in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_grouped: dict[str, list[float]] = defaultdict(list)
    category_grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        model_grouped[display_model_name(row.model)].append(float(row.delta))
        category_grouped[row.category].append(float(row.delta))

    plot_delta_hist(rows, args.output_dir, args.dpi)
    plot_mean_delta_by_group(
        model_grouped,
        title="Mean Delta by Model (95% CI)",
        xlabel="Mean delta (R2 - R1)",
        output_path=args.output_dir / "mean_delta_by_model.png",
        dpi=args.dpi,
    )
    plot_mean_delta_by_group(
        category_grouped,
        title="Mean Delta by Question Category (95% CI)",
        xlabel="Mean delta (R2 - R1)",
        output_path=args.output_dir / "mean_delta_by_category.png",
        dpi=args.dpi,
    )
    plot_rating_change(rows, args.output_dir, args.dpi)
    plot_delta_box_by_category(rows, args.output_dir, args.dpi)

    print(f"Saved figures to {args.output_dir}")
    if scipy_stats is None:
        print("Note: scipy not installed; confidence intervals use normal approximation.")


if __name__ == "__main__":
    main()
