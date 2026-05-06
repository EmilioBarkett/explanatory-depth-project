"""Statistical significance analysis for HLE confidence-shift results.

Analyzes whether observed confidence shifts (deltas) are statistically significant
using multiple statistical tests: t-test, Wilcoxon signed-rank, sign test, and
effect sizes (Cohen's dz).

Usage:
  python evals/hle/stat_sig.py
  python evals/hle/stat_sig.py --input results/new_sample_HLE.jsonl --output analysis/stat_sig_report.txt
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


INPUT_FILE = Path(__file__).resolve().parents[2] / "results" / "new_sample_HLE.jsonl"
OUTPUT_FILE = Path(__file__).resolve().parents[2] / "analysis" / "stat_sig_report.txt"


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
class SignificanceResult:
    """Result of a statistical significance test."""

    group_name: str
    n: int
    mean_delta: float
    median_delta: float
    std_delta: float
    t_stat: float
    p_t: float
    wilcoxon_stat: float
    p_wilcoxon: float
    p_sign: float
    cohens_dz: float
    ci_low: float
    ci_high: float
    direction: str  # "increases", "decreases", or "no change"


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

            first_rating = record.get("first_rating") or record.get("first_confidence")
            second_rating = record.get("second_rating") or record.get("second_confidence")
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


def test_significance(rows: list[Row], group_name: str = "Overall") -> SignificanceResult:
    """Run all significance tests on a group of rows."""
    deltas = [float(row.delta) for row in rows]
    
    if not deltas:
        return SignificanceResult(
            group_name=group_name,
            n=0,
            mean_delta=float("nan"),
            median_delta=float("nan"),
            std_delta=float("nan"),
            t_stat=float("nan"),
            p_t=float("nan"),
            wilcoxon_stat=float("nan"),
            p_wilcoxon=float("nan"),
            p_sign=float("nan"),
            cohens_dz=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            direction="no change",
        )

    mean_delta = statistics.mean(deltas)
    median_delta = statistics.median(deltas)
    std_delta = _sample_std(deltas)
    ci_low, ci_high = _mean_ci(deltas, confidence=0.95)
    t_stat, p_t = _one_sample_t(deltas)
    wilcoxon_stat, p_wilcoxon = _wilcoxon_signed_rank(deltas)
    p_sign = _two_sided_sign_test_p(deltas)
    cohens_dz = _cohens_dz(deltas)

    # Determine direction
    if mean_delta > 0:
        direction = "increases"
    elif mean_delta < 0:
        direction = "decreases"
    else:
        direction = "no change"

    return SignificanceResult(
        group_name=group_name,
        n=len(deltas),
        mean_delta=mean_delta,
        median_delta=median_delta,
        std_delta=std_delta,
        t_stat=t_stat,
        p_t=p_t,
        wilcoxon_stat=wilcoxon_stat,
        p_wilcoxon=p_wilcoxon,
        p_sign=p_sign,
        cohens_dz=cohens_dz,
        ci_low=ci_low,
        ci_high=ci_high,
        direction=direction,
    )


def group_rows(rows: list[Row], key_fn: Callable[[Row], Any]) -> dict[Any, list[Row]]:
    """Group rows by a key function."""
    groups: defaultdict[Any, list[Row]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def format_p(value: float, threshold: float = 0.05) -> str:
    """Format p-value with significance marker."""
    if math.isnan(value):
        return "    nan"
    if value < 0.001:
        marker = "***"
    elif value < 0.01:
        marker = "** "
    elif value < 0.05:
        marker = "*  "
    else:
        marker = "   "
    return f"{value:6.3f}{marker}"


def format_effect(value: float) -> str:
    """Format Cohen's dz with interpretation."""
    if math.isnan(value):
        return "   nan"
    abs_val = abs(value)
    if abs_val < 0.2:
        interpretation = "(negligible)"
    elif abs_val < 0.5:
        interpretation = "(small)"
    elif abs_val < 0.8:
        interpretation = "(medium)"
    else:
        interpretation = "(large)"
    return f"{value:7.3f} {interpretation}"


def print_header() -> None:
    print("=" * 130)
    print("STATISTICAL SIGNIFICANCE ANALYSIS OF HLE CONFIDENCE SHIFTS")
    print("=" * 130)
    print("\nInterpretation guide:")
    print("  * p < 0.05  (marked with *  )")
    print("  ** p < 0.01 (marked with ** )")
    print("  *** p < 0.001 (marked with ***)")
    print("\nEffect sizes (Cohen's dz):")
    print("  negligible: |dz| < 0.2")
    print("  small: 0.2 <= |dz| < 0.5")
    print("  medium: 0.5 <= |dz| < 0.8")
    print("  large: |dz| >= 0.8")
    print()


def print_significance_summary(results: list[SignificanceResult]) -> None:
    """Print a summary table of significance results."""
    print("\n" + "=" * 130)
    print("SIGNIFICANCE SUMMARY")
    print("=" * 130)
    print(
        f"{'Group':<40} {'n':>5} {'Mean Δ':>8} {'95% CI':>18} {'p-value (t)':>15} {'Effect size':>18} {'Direction':>12}"
    )
    print("-" * 130)

    for result in sorted(results, key=lambda r: r.p_t if not math.isnan(r.p_t) else 1.0):
        ci_str = f"[{result.ci_low:+.2f}, {result.ci_high:+.2f}]"
        effect_str = format_effect(result.cohens_dz)
        print(
            f"{result.group_name:<40} {result.n:>5} {result.mean_delta:>8.3f} {ci_str:>18} "
            f"{format_p(result.p_t):>15} {effect_str:>18} {result.direction:>12}"
        )


def print_detailed_results(results: list[SignificanceResult]) -> None:
    """Print detailed statistics for each group."""
    print("\n" + "=" * 130)
    print("DETAILED RESULTS BY GROUP")
    print("=" * 130)

    for result in sorted(results, key=lambda r: r.group_name):
        print(f"\n{result.group_name}")
        print("-" * 80)
        print(f"  Sample size (n)         : {result.n}")
        print(f"  Mean delta              : {result.mean_delta:+.3f}")
        print(f"  Median delta            : {result.median_delta:+.3f}")
        print(f"  Std deviation           : {result.std_delta:.3f}")
        print(f"  95% Confidence interval : [{result.ci_low:+.3f}, {result.ci_high:+.3f}]")
        print(f"\n  Parametric test (t-test, assumes normality):")
        print(f"    t-statistic           : {result.t_stat:+.3f}")
        print(f"    p-value               : {format_p(result.p_t)}")
        if result.p_t < 0.05:
            print(f"    Result                : SIGNIFICANT at α=0.05")
        else:
            print(f"    Result                : NOT significant at α=0.05")

        print(f"\n  Non-parametric tests (distribution-free):")
        print(f"    Wilcoxon W-statistic  : {result.wilcoxon_stat:.1f}")
        print(f"    p-value (Wilcoxon)    : {format_p(result.p_wilcoxon)}")
        print(f"    p-value (sign test)   : {format_p(result.p_sign)}")

        print(f"\n  Effect size:")
        print(f"    Cohen's dz            : {format_effect(result.cohens_dz)}")
        print(f"\n  Practical interpretation:")
        print(f"    Direction             : Confidence {result.direction}")
        if result.p_t < 0.05:
            print(f"    Significance          : YES (p < 0.05)")
        else:
            print(f"    Significance          : NO (p >= 0.05)")


def display_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HLE statistical significance.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE, help="Path to HLE JSONL file")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Path to write report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)

    if not rows:
        raise SystemExit(f"No usable rows found in {args.input}")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_header()

        # Overall significance
        overall_result = test_significance(rows, "Overall")

        # By model
        by_model = group_rows(rows, lambda row: row.model)
        model_results = [
            test_significance(by_model[model], display_model_name(model))
            for model in sorted(by_model)
        ]

        # By category
        by_category = group_rows(rows, lambda row: row.category)
        category_results = [
            test_significance(by_category[cat], cat) for cat in sorted(by_category)
        ]

        # Combine all results
        all_results = [overall_result] + model_results + category_results

        print_significance_summary(all_results)
        print_detailed_results(all_results)

        # Additional insights
        print("\n" + "=" * 130)
        print("KEY FINDINGS")
        print("=" * 130)

        significant_results = [r for r in all_results if r.p_t < 0.05 and r.n > 0]
        if significant_results:
            print(f"\nSignificant effects (p < 0.05): {len(significant_results)}")
            for result in sorted(significant_results, key=lambda r: abs(r.mean_delta), reverse=True):
                print(
                    f"  • {result.group_name}: Δ = {result.mean_delta:+.3f} "
                    f"(p = {result.p_t:.4f}, dz = {result.cohens_dz:.3f})"
                )
        else:
            print("\nNo significant effects found at α=0.05")

        largest_increases = sorted(model_results + category_results, key=lambda r: r.mean_delta, reverse=True)[:5]
        print(f"\nTop 5 largest confidence increases:")
        for i, result in enumerate(largest_increases, 1):
            sig = "***" if result.p_t < 0.001 else "**" if result.p_t < 0.01 else "*" if result.p_t < 0.05 else ""
            print(f"  {i}. {result.group_name}: Δ = {result.mean_delta:+.3f} {sig}")

        largest_decreases = sorted(model_results + category_results, key=lambda r: r.mean_delta)[:5]
        print(f"\nTop 5 largest confidence decreases:")
        for i, result in enumerate(largest_decreases, 1):
            sig = "***" if result.p_t < 0.001 else "**" if result.p_t < 0.01 else "*" if result.p_t < 0.05 else ""
            print(f"  {i}. {result.group_name}: Δ = {result.mean_delta:+.3f} {sig}")

    report = buffer.getvalue()
    print(report, end="")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
