"""
Dictionary-based lexical analysis for IOED explanation texts.

Scores each explanation across four epistemic categories:
  - uncertainty    : hedging / doubt language
  - confidence     : assertion / certainty language
  - self_correction: signs the model is catching or revising its own reasoning
  - complexity     : acknowledgment of shallow or incomplete understanding

All scores are normalised to hits per 100 words so explanations of different
lengths are comparable.  

A derived `net_epistemic` score (uncertainty minus
confidence) summarises the overall epistemic stance: positive = more hedged,
negative = more assertive.
"""

import re

# Lexicon

LEXICON: dict[str, list[str]] = {
    # Hedging, qualification, and doubt
    "uncertainty": [
        r"\bi think\b",
        r"\bi believe\b",
        r"\bi('m| am) not (sure|certain|entirely sure|fully sure|completely sure)\b",
        r"\bi('m| am) unsure\b",
        r"\bnot (entirely |fully |completely )?sure\b",
        r"\bprobably\b",
        r"\blikely\b",
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\bmaybe\b",
        r"\bmight\b",
        r"\bseems?\b",
        r"\bappears?\b",
        r"\buncertain\b",
        r"\bI('m| am) not (entirely |fully )?(confident|positive)\b",
        r"\bI would guess\b",
        r"\bI suspect\b",
        r"\bnot (entirely |fully |100%) (clear|sure|certain)\b",
        r"\bsomething like\b",
        r"\bapproximately\b",
        r"\broughly\b",
    ],

    # Assertive certainty language
    "confidence": [
        r"\bclearly\b",
        r"\bdefinitely\b",
        r"\bcertainly\b",
        r"\bobviously\b",
        r"\bwithout (a )?doubt\b",
        r"\bmust be\b",
        r"\bmust have\b",
        r"\balways\b",
        r"\bnever\b",
        r"\babsolutely\b",
        r"\bI('m| am) (confident|certain|sure|positive)\b",
        r"\bI know\b",
        r"\bI('m| am) (fully |completely |100% )(confident|sure|certain)\b",
        r"\bprecisely\b",
        r"\bexactly\b",
        r"\bguaranteed\b",
        r"\bwithout question\b",
    ],

    # Model catching or revising its own reasoning mid-explanation
    "self_correction": [
        r"\bwait\b",
        r"\bI was wrong\b",
        r"\bI made an? (error|mistake)\b",
        r"\bI need to correct\b",
        r"\bcorrection[:\s]",
        r"\bI initially (said|thought|stated|assumed)\b",
        r"\bon second thought\b",
        r"\bI (now |do )realize\b",
        r"\bI realize now\b",
        r"\bI('m| am) reconsidering\b",
        r"\bactually[,\s]+(I|this|that|my|the)\b",  # "actually I..." not just "actually"
        r"\bI should (have said|clarify|correct)\b",
        r"\blet me (reconsider|correct|revise|rethink)\b",
        r"\bI ('ve|have) reconsidered\b",
        r"\bupon reflection\b",
    ],

    # Acknowledging that understanding is shallow or incomplete
    "complexity": [
        r"\bcomplex\b",
        r"\bcomplicated\b",
        r"\bintricate\b",
        r"\bI don'?t (fully|completely|entirely) (understand|know|grasp|follow)\b",
        r"\bdifficult to (explain|understand|say|describe|articulate)\b",
        r"\bhard to (say|explain|tell|describe|articulate)\b",
        r"\bit'?s (not |un)?clear\b",
        r"\bit('s| is) (not |un)?clear\b",
        r"\bunclear\b",
        r"\bnot (entirely |fully |completely )?(clear|understood|explained)\b",
        r"\bbeyond (my|a simple)\b",
        r"\bI('m| am) not (entirely |fully |completely )?(familiar with|aware of|sure (about|how))\b",
        r"\blimited (understanding|knowledge)\b",
    ],
}

# Core scoring function
def score_explanation(text: str) -> dict:
    """
    Score a single explanation string against the IOED lexicon.

    Args:
        text: The model's step-by-step explanation (Turn 2 output).

    Returns:
        Dict with keys:
            uncertainty, confidence, self_correction, complexity
                → float, hits per 100 words
            net_epistemic
                → float, uncertainty minus confidence (positive = more hedged)
            raw_counts
                → dict of raw integer hit counts per category
            word_count
                → int
    """
    if not text or not text.strip():
        return {
            "uncertainty":     0.0,
            "confidence":      0.0,
            "self_correction": 0.0,
            "complexity":      0.0,
            "net_epistemic":   0.0,
            "raw_counts":      {k: 0 for k in LEXICON},
            "word_count":      0,
        }

    text_lower = text.lower()
    word_count = max(len(text_lower.split()), 1)

    raw_counts = {}
    rates      = {}
    for category, patterns in LEXICON.items():
        hits = sum(len(re.findall(p, text_lower)) for p in patterns)
        raw_counts[category] = hits
        rates[category]      = round(hits / word_count * 100, 4)

    rates["net_epistemic"] = round(rates["uncertainty"] - rates["confidence"], 4)
    rates["raw_counts"]    = raw_counts
    rates["word_count"]    = word_count
    return rates


# Batch helper
def score_results(results: list[dict], explanation_key: str = "explanation") -> list[dict]:
    """
    Add `explanation_scores` to each result dict in-place.

    Args:
        results:         List of result dicts (as saved to JSON by any eval).
        explanation_key: Field name holding the explanation text.
                         Defaults to "explanation" (core pipeline); use
                         "second_explanation" for the SPARTQA pipeline.

    Returns:
        The same list, mutated in-place, for convenience.
    """
    for r in results:
        r["explanation_scores"] = score_explanation(r.get(explanation_key) or "")
    return results


# Summary printer
def print_lexical_summary(results: list[dict], group_by: str = "model") -> None:
    """
    Print a grouped summary table of mean lexical scores.

    Args:
        results:  Result dicts that already have `explanation_scores` populated.
        group_by: Key to group rows by. Common values: "model", "category",
                  "study", "q_type".
    """
    from collections import defaultdict
    import statistics

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        key = r.get(group_by, "unknown")
        if r.get("explanation_scores") and r["explanation_scores"]["word_count"] > 0:
            groups[key].append(r["explanation_scores"])

    cats = ["uncertainty", "confidence", "self_correction", "complexity", "net_epistemic"]
    header = (
        f"{'Group':<40} "
        + "  ".join(f"{c[:6]:>8}" for c in cats)
        + f"  {'words':>7}"
    )
    print(f"\nLexical scores (per 100 words) — grouped by {group_by!r}")
    print(header)
    print("-" * len(header))

    for name, scores_list in sorted(groups.items()):
        def mean(key):
            vals = [s[key] for s in scores_list if s[key] is not None]
            return statistics.mean(vals) if vals else float("nan")

        row = f"{name:<40} "
        row += "  ".join(f"{mean(c):>8.3f}" for c in cats)
        row += f"  {mean('word_count'):>7.0f}"
        print(row)
