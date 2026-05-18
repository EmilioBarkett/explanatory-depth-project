"""Statistical significance analysis for HLE confidence-shift results.

Aggregates worker JSONL files and tests whether deltas (R2 - R1) differ from 0
for overall data, per-model groups, per-category groups, and model x category.

Includes explicit group sizes so uneven sample counts are visible.
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
from typing import Callable

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "HLE"
DEFAULT_PATTERN = "hle_worker_*.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "analysis" / "HLE_new_sample" / "stat_sig_report.txt"


@dataclass(frozen=True)
class Row:
    source_file: str
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
    group_name: str
    n: int
    n_questions: int
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
    direction: str


def display_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _preferred_input_dir(input_dir: Path) -> Path:
    nested = input_dir / "new_method"
    return nested if nested.is_dir() else input_dir


def discover_input_files(inputs: list[Path] | None, input_dir: Path, pattern: str) -> list[Path]:
    files: list[Path] = []
    if inputs:
        for p in inputs:
            if p.is_dir():
                files.extend(sorted(_preferred_input_dir(p).glob(pattern)))
            elif p.exists():
                files.append(p)
        seen: set[Path] = set()
        out: list[Path] = []
        for p in files:
            rp = p.resolve()
            if rp not in seen:
                out.append(p)
                seen.add(rp)
        return out

    if not input_dir.exists():
        return []
    return sorted(_preferred_input_dir(input_dir).glob(pattern))


def load_rows(paths: list[Path]) -> tuple[list[Row], dict[str, int]]:
    rows: list[Row] = []
    invalid_json = 0
    missing_ratings = 0
    errored_rows = 0

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue

                if record.get("error"):
                    errored_rows += 1

                first_rating = record.get("first_rating")
                if first_rating is None:
                    first_rating = record.get("first_confidence")
                second_rating = record.get("second_rating")
                if second_rating is None:
                    second_rating = record.get("second_confidence")

                if first_rating is None or second_rating is None:
                    missing_ratings += 1
                    continue

                try:
                    row = Row(
                        source_file=path.name,
                        question_id=str(record.get("question_id") or record.get("id") or "unknown"),
                        model=str(record.get("model") or "unknown"),
                        category=str(record.get("category") or "unknown"),
                        first_rating=int(first_rating),
                        second_rating=int(second_rating),
                    )
                except (TypeError, ValueError):
                    missing_ratings += 1
                    continue

                rows.append(row)

    diagnostics = {
        "input_files": len(paths),
        "loaded_rows": len(rows),
        "invalid_json": invalid_json,
        "missing_ratings": missing_ratings,
        "errored_rows": errored_rows,
    }
    return rows, diagnostics


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
    deltas = [float(row.delta) for row in rows]

    if not deltas:
        return SignificanceResult(
            group_name=group_name,
            n=0,
            n_questions=0,
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

    if mean_delta > 0:
        direction = "increases"
    elif mean_delta < 0:
        direction = "decreases"
    else:
        direction = "no change"

    return SignificanceResult(
        group_name=group_name,
        n=len(deltas),
        n_questions=len({row.question_id for row in rows}),
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


def group_rows(rows: list[Row], key_fn: Callable[[Row], str]) -> dict[str, list[Row]]:
    groups: defaultdict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def format_p(value: float) -> str:
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


def print_header(input_files: list[Path], diagnostics: dict[str, int]) -> None:
    print("=" * 140)
    print("STATISTICAL SIGNIFICANCE ANALYSIS OF HLE CONFIDENCE SHIFTS")
    print("=" * 140)
    print(f"Input files loaded    : {diagnostics['input_files']}")
    print(f"Scored rows loaded    : {diagnostics['loaded_rows']}")
    print(f"Rows with error field : {diagnostics['errored_rows']}")
    print(f"Rows missing ratings  : {diagnostics['missing_ratings']}")
    print(f"Invalid JSON rows     : {diagnostics['invalid_json']}")
    if scipy_stats is None:
        print("Warning: scipy not installed; Wilcoxon p-values unavailable, t-test fallback uses normal approximation.")

    print("\nInput files:")
    for p in input_files:
        print(f"  - {p}")

    print("\nInterpretation guide:")
    print("  * p < 0.05  (marked with *  )")
    print("  ** p < 0.01 (marked with ** )")
    print("  *** p < 0.001 (marked with ***)")


def print_significance_summary(results: list[SignificanceResult]) -> None:
    print("\n" + "=" * 140)
    print("SIGNIFICANCE SUMMARY")
    print("=" * 140)
    print(
        f"{'Group':<44} {'n':>6} {'qids':>6} {'Mean Δ':>8} {'95% CI':>18} {'p-value (t)':>15} {'Effect size':>18} {'Direction':>12}"
    )
    print("-" * 140)

    for result in sorted(results, key=lambda r: r.p_t if not math.isnan(r.p_t) else 1.0):
        ci_str = f"[{result.ci_low:+.2f}, {result.ci_high:+.2f}]"
        effect_str = format_effect(result.cohens_dz)
        print(
            f"{result.group_name:<44} {result.n:>6} {result.n_questions:>6} {result.mean_delta:>8.3f} {ci_str:>18} "
            f"{format_p(result.p_t):>15} {effect_str:>18} {result.direction:>12}"
        )


def print_significant_findings(results: list[SignificanceResult], min_n: int) -> None:
    print("\n" + "=" * 140)
    print("SIGNIFICANT FINDINGS")
    print("=" * 140)

    eligible = [r for r in results if r.n >= min_n]
    significant = [r for r in eligible if not math.isnan(r.p_t) and r.p_t < 0.05]

    print(f"Eligible groups (n >= {min_n}): {len(eligible)}")
    print(f"Significant groups (p_t < 0.05): {len(significant)}")

    if not significant:
        print("No statistically significant groups under the configured threshold.")
        return

    print("\nTop significant decreases (most negative mean delta):")
    decreases = sorted([r for r in significant if r.mean_delta < 0], key=lambda r: r.mean_delta)
    if decreases:
        for r in decreases[:10]:
            print(
                f"  - {r.group_name}: n={r.n}, qids={r.n_questions}, Δ={r.mean_delta:+.3f}, p={r.p_t:.4g}, dz={r.cohens_dz:.3f}"
            )
    else:
        print("  - none")

    print("\nTop significant increases (most positive mean delta):")
    increases = sorted([r for r in significant if r.mean_delta > 0], key=lambda r: r.mean_delta, reverse=True)
    if increases:
        for r in increases[:10]:
            print(
                f"  - {r.group_name}: n={r.n}, qids={r.n_questions}, Δ={r.mean_delta:+.3f}, p={r.p_t:.4g}, dz={r.cohens_dz:.3f}"
            )
    else:
        print("  - none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HLE statistical significance from worker files.")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit JSONL file(s) and/or directories. If omitted, uses --input-dir + --pattern.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing worker JSONL files")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="Glob pattern used under --input-dir")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write report")
    parser.add_argument(
        "--min-group-n",
        type=int,
        default=20,
        help="Minimum n required for a group to be included in highlighted findings",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = discover_input_files(args.inputs, args.input_dir, args.pattern)
    if not input_files:
        raise SystemExit("No input files found. Check --inputs or --input-dir/--pattern.")

    rows, diagnostics = load_rows(input_files)
    if not rows:
        raise SystemExit("No usable scored rows found in selected inputs.")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_header(input_files, diagnostics)

        overall_result = test_significance(rows, "Overall")

        by_model = group_rows(rows, lambda row: row.model)
        model_results = [
            test_significance(by_model[model], f"Model: {display_model_name(model)}")
            for model in sorted(by_model)
        ]

        by_category = group_rows(rows, lambda row: row.category)
        category_results = [
            test_significance(by_category[cat], f"Category: {cat}")
            for cat in sorted(by_category)
        ]

        by_model_category = group_rows(rows, lambda row: f"{display_model_name(row.model)} || {row.category}")
        model_category_results = [
            test_significance(by_model_category[key], f"Model×Category: {key}")
            for key in sorted(by_model_category)
        ]

        all_results = [overall_result] + model_results + category_results + model_category_results

        print_significance_summary(all_results)
        print_significant_findings(all_results, min_n=args.min_group_n)

    report = buffer.getvalue()
    print(report, end="")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
