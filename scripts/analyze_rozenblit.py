"""
Aggregate + plot Rozenblit IOED replication trials.

Loads every `rozenblit_worker_*.jsonl` under results/rozenblit_trials/, ignores
files prefixed with `ignore_`, drops rows with API errors, and reports:

  - per (model, study, arm) mean Δ (R2 − R1)
  - per (model, study) IOED  = Δ_explanation − Δ_control
  - devices-vs-procedures contrast (Hypothesis 2 in the README)
  - overall summary across all models

Figures are written to results/rozenblit_trials/figures/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRIALS_DIR = ROOT / "results" / "rozenblit_trials"
FIG_DIR = TRIALS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(TRIALS_DIR.glob("rozenblit_worker_*.jsonl")):
        if path.name.startswith("ignore_"):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error"):
                    continue
                t1 = r.get("t1") or {}
                final = r.get("final") or {}
                r1 = t1.get("confidence")
                r2 = final.get("confidence")
                if r1 is None or r2 is None:
                    continue
                rows.append({
                    "question_id":  r["question_id"],
                    "study":        r["study"],
                    "category":     r.get("category"),
                    "model":        r["model"],
                    "model_short":  r["model"].split("/", 1)[-1],
                    "arm":          r["arm"],
                    "sample_index": r["sample_index"],
                    "r1":           r1,
                    "r2":           r2,
                    "delta":        r2 - r1,
                })
    return pd.DataFrame(rows)


# ── Aggregations ──────────────────────────────────────────────────────────────

def delta_table(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["study", "model_short", "arm"])["delta"]
           .agg(["mean", "std", "count"])
           .reset_index())
    return g


def ioed_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (study, model): mean Δ per arm and IOED."""
    pivot = (df.pivot_table(index=["study", "model_short"],
                            columns="arm",
                            values="delta",
                            aggfunc="mean")
               .reset_index())
    pivot.columns.name = None
    pivot = pivot.rename(columns={"explanation": "delta_expl",
                                  "control":     "delta_ctrl"})
    if "delta_expl" not in pivot.columns:
        pivot["delta_expl"] = float("nan")
    if "delta_ctrl" not in pivot.columns:
        pivot["delta_ctrl"] = float("nan")
    pivot["ioed"] = pivot["delta_expl"] - pivot["delta_ctrl"]
    return pivot


def devices_vs_procedures(df: pd.DataFrame) -> pd.DataFrame:
    expl = df[df["arm"] == "explanation"]
    g = (expl.groupby(["model_short", "study"])["delta"]
             .mean()
             .unstack("study"))
    g["gap_dev_minus_proc"] = g.get("devices") - g.get("procedures")
    return g.reset_index()


# ── Plots ────────────────────────────────────────────────────────────────────

STUDIES = ["devices", "procedures", "natural_phenomena"]
STUDY_COLORS = {"devices": "#1f77b4",
                "procedures": "#ff7f0e",
                "natural_phenomena": "#2ca02c"}


def plot_ioed_per_model(ioed: pd.DataFrame, out: Path) -> None:
    models = sorted(ioed["model_short"].unique())
    studies = [s for s in STUDIES if s in ioed["study"].unique()]
    x = np.arange(len(models))
    width = 0.8 / len(studies)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, study in enumerate(studies):
        ys = []
        for m in models:
            row = ioed[(ioed["study"] == study) & (ioed["model_short"] == m)]
            ys.append(float(row["ioed"].iloc[0]) if len(row) else float("nan"))
        ax.bar(x + (i - (len(studies) - 1) / 2) * width, ys, width,
               label=study, color=STUDY_COLORS.get(study, None))

    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("IOED  (Δ explanation − Δ control)")
    ax.set_title("Rozenblit IOED per model and study")
    ax.legend(title="study")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_delta_arms(delta_tbl: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart: mean Δ per arm, averaged across models, by study."""
    studies = [s for s in STUDIES if s in delta_tbl["study"].unique()]
    arms = ["explanation", "control"]
    means = {a: [] for a in arms}
    errs = {a: [] for a in arms}
    for s in studies:
        for a in arms:
            sub = delta_tbl[(delta_tbl["study"] == s) & (delta_tbl["arm"] == a)]
            vals = sub["mean"].values
            means[a].append(vals.mean() if len(vals) else float("nan"))
            errs[a].append(vals.std(ddof=1) / math.sqrt(len(vals))
                           if len(vals) > 1 else 0.0)

    x = np.arange(len(studies))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, means["explanation"], width,
           yerr=errs["explanation"], capsize=4, label="explanation",
           color="#4c78a8")
    ax.bar(x + width/2, means["control"], width,
           yerr=errs["control"], capsize=4, label="control",
           color="#bab0ac")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(studies)
    ax.set_ylabel("mean Δ confidence (R2 − R1)")
    ax.set_title("Δ confidence by arm and study (averaged across models)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_devices_vs_procedures(gap_tbl: pd.DataFrame, out: Path) -> None:
    sub = gap_tbl.dropna(subset=["devices", "procedures"]).copy()
    sub = sub.sort_values("gap_dev_minus_proc")
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(sub))
    width = 0.38
    ax.bar(x - width/2, sub["devices"],    width,
           label="devices",    color=STUDY_COLORS["devices"])
    ax.bar(x + width/2, sub["procedures"], width,
           label="procedures", color=STUDY_COLORS["procedures"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model_short"], rotation=35, ha="right")
    ax.set_ylabel("mean Δ confidence (explanation arm)")
    ax.set_title("Devices vs. procedures — Rozenblit replication (explanation arm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_overall_ioed(ioed: pd.DataFrame, out: Path) -> None:
    """Mean IOED across models, by study, with 95% CI bars."""
    studies = [s for s in STUDIES if s in ioed["study"].unique()]
    means, lo, hi = [], [], []
    for s in studies:
        vals = ioed[ioed["study"] == s]["ioed"].dropna().values
        if len(vals) == 0:
            means.append(float("nan")); lo.append(0); hi.append(0); continue
        m = vals.mean()
        if len(vals) > 1:
            se = vals.std(ddof=1) / math.sqrt(len(vals))
            ci = 1.96 * se
        else:
            ci = 0.0
        means.append(m); lo.append(ci); hi.append(ci)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(studies))
    ax.bar(x, means, yerr=[lo, hi], capsize=6,
           color=[STUDY_COLORS[s] for s in studies])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(studies)
    ax.set_ylabel("IOED  (Δ explanation − Δ control)")
    ax.set_title("Aggregate IOED across models, by study  (mean ± 95% CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ── Report ───────────────────────────────────────────────────────────────────

def hr(c: str = "─", n: int = 76) -> None:
    print(c * n)


def main() -> None:
    df = load_rows()
    print(f"Loaded {len(df):,} usable rows "
          f"({df['model_short'].nunique()} models, "
          f"{df['question_id'].nunique()} items, "
          f"{df['study'].nunique()} studies).")
    print("Per-model rows:")
    counts = (df.groupby("model_short")
                .size()
                .sort_values(ascending=False))
    for m, n in counts.items():
        print(f"  {m:<40s} {n:>6d}")
    hr()

    delta_tbl = delta_table(df)
    ioed = ioed_table(df)
    gap = devices_vs_procedures(df)

    # ── per-(study, model) Δ + IOED table ──
    print("\nPer-arm mean Δ (R2 − R1).  IOED = Δ_explanation − Δ_control.")
    hr()
    print(f"{'Study':<20} {'Model':<35} {'Δ expl':>8} {'Δ ctrl':>8} {'IOED':>8}")
    hr()
    for _, row in ioed.sort_values(["study", "model_short"]).iterrows():
        e = row["delta_expl"]; c = row["delta_ctrl"]; i = row["ioed"]
        f = lambda v: f"{v:+.2f}" if pd.notna(v) else "  —  "
        print(f"{row['study']:<20} {row['model_short']:<35} "
              f"{f(e):>8} {f(c):>8} {f(i):>8}")

    # ── across-model summary per study ──
    print("\nAcross-model summary (each model = one observation):")
    hr()
    print(f"{'Study':<20} {'mean Δ expl':>12} {'mean Δ ctrl':>12} {'mean IOED':>10} {'n_models':>9}")
    hr()
    for s in STUDIES:
        sub = ioed[ioed["study"] == s].dropna(subset=["delta_expl", "delta_ctrl"])
        if len(sub) == 0:
            continue
        print(f"{s:<20} "
              f"{sub['delta_expl'].mean():>+12.3f} "
              f"{sub['delta_ctrl'].mean():>+12.3f} "
              f"{sub['ioed'].mean():>+10.3f} "
              f"{len(sub):>9d}")

    # ── devices vs procedures (Hypothesis 2) ──
    print("\nDevices vs. procedures (explanation-arm Δ):")
    hr()
    print(f"{'Model':<35} {'Δ devices':>10} {'Δ procs':>10} {'gap':>8}")
    hr()
    for _, row in gap.dropna(subset=["devices", "procedures"]) \
                     .sort_values("gap_dev_minus_proc").iterrows():
        print(f"{row['model_short']:<35} "
              f"{row['devices']:>+10.2f} "
              f"{row['procedures']:>+10.2f} "
              f"{row['gap_dev_minus_proc']:>+8.2f}")

    # ── plots ──
    plot_ioed_per_model(ioed,    FIG_DIR / "ioed_per_model.png")
    plot_delta_arms(delta_tbl,    FIG_DIR / "delta_by_arm_and_study.png")
    plot_devices_vs_procedures(gap, FIG_DIR / "devices_vs_procedures.png")
    plot_overall_ioed(ioed,       FIG_DIR / "ioed_overall.png")

    # ── csv dumps ──
    ioed.to_csv(FIG_DIR / "ioed_per_model.csv", index=False)
    gap.to_csv(FIG_DIR / "devices_vs_procedures.csv", index=False)
    delta_tbl.to_csv(FIG_DIR / "delta_table.csv", index=False)

    print(f"\nFigures + CSVs → {FIG_DIR}")


if __name__ == "__main__":
    main()
