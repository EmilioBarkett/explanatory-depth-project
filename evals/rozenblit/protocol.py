"""
Rozenblit & Keil (2002) replication protocol — full version.

Two arms run for every (model, item) pair:

  Explanation arm  (mirrors Rozenblit's Phases 2-4)
     T1  initial answer + R1                 [JSON: {answer, confidence}]
     T2  step-by-step explanation             [free text]
     T3  diagnostic question (if available)   [free text]
     T4  re-rated confidence (R2)             [JSON: {confidence}]

  Control arm     (no-explanation control to isolate "drift from re-prompting")
     T1  initial answer + R1                 [JSON: {answer, confidence}]
     T2  re-rated confidence (R2)            [JSON: {confidence}]

The diagnostic phase fires only for items that carry a `diagnostic_question`
field — devices and natural_phenomena in our dataset, matching Rozenblit's
own protocol. Procedure items skip it (also matching the paper).

Each (model, item, arm) is run K_SAMPLES times so we have a per-item
variance estimate. Default temperature is 0.7 to give meaningful variance;
set TEMPERATURE=0.0 and K_SAMPLES=1 for deterministic single-shot runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Optional

from evals.core.anchors import ROZENBLIT_ANCHOR_SYSTEM
from evals.core.structured import (
    call_openrouter,
    extract_answer_and_confidence,
    extract_confidence,
    ModelNotFoundError,
    ModelRateLimitError,
)
from evals.core.text_analysis import score_explanation


# ── Knobs ─────────────────────────────────────────────────────────────────────

K_SAMPLES        = 5      # replications per (model, item, arm)
TEMPERATURE      = 0.7    # 0.0 + K_SAMPLES=1 for deterministic single-shot
RATE_LIMIT_DELAY = 1      # seconds between API calls


# ── Prompts ───────────────────────────────────────────────────────────────────

def _t1_prompt(question: str) -> str:
    return (
        f"Question: {question}\n\n"
        "Give a brief one-sentence answer, then rate your confidence on the "
        "0-10 scale described in the system message.\n\n"
        'Reply with valid JSON only, in this exact form:\n'
        '{"answer": "<your brief answer>", "confidence": <integer 0-10>}'
    )


T2_EXPLANATION_PROMPT = (
    "Now please explain step-by-step how you arrived at your answer. "
    "Walk through the mechanism in as much causal detail as you can, "
    "without skipping intermediate steps. Reply in plain prose; no JSON."
)


def _t3_diagnostic_prompt(diagnostic_question: str) -> str:
    return (
        "One more question on the same topic, probing a specific detail of "
        f"the mechanism:\n\n{diagnostic_question}\n\n"
        "Answer in plain prose; no JSON."
    )


T_FINAL_RATING_PROMPT = (
    "Given everything you have written so far, how well do you now feel "
    "you understand the original question? Use the same 0-10 scale from "
    "the system message.\n\n"
    'Reply with valid JSON only, in this exact form:\n'
    '{"confidence": <integer 0-10>}'
)


# ── Data shape ────────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    raw_text:    str | None = None
    confidence:  int | None = None
    rating_src:  str | None = None    # "json" | "json_field" | "last_int" | "missing"
    answer:      str | None = None    # T1 only
    error:       str | None = None


@dataclass
class SampleRecord:
    sample_index:        int
    arm:                 str               # "explanation" | "control"
    t1:                  TurnRecord        = field(default_factory=TurnRecord)
    explanation:         TurnRecord | None = None
    explanation_scores:  dict | None       = None
    diagnostic:          TurnRecord | None = None
    final:               TurnRecord        = field(default_factory=TurnRecord)
    error:               str | None        = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop None-valued nested turns to keep records compact.
        for k in ("explanation", "diagnostic"):
            if d[k] is None:
                d.pop(k)
        return d


# ── Helpers ───────────────────────────────────────────────────────────────────

def _system_msg() -> dict:
    return {"role": "system", "content": ROZENBLIT_ANCHOR_SYSTEM}


def _delta(t1: TurnRecord, final: TurnRecord) -> int | None:
    if t1.confidence is None or final.confidence is None:
        return None
    return final.confidence - t1.confidence


def _safe_call(messages, model, temperature, json_object) -> tuple[str | None, str | None]:
    """Wrap call_openrouter so callers get a (text, error) tuple."""
    try:
        text = call_openrouter(
            messages,
            model,
            temperature=temperature,
            json_object=json_object,
        )
        return text, None
    except (ModelNotFoundError, ModelRateLimitError, RuntimeError) as exc:
        return None, str(exc)


# ── Arms ──────────────────────────────────────────────────────────────────────

def _run_t1(question: str, model: str, temperature: float) -> tuple[list, TurnRecord]:
    """Send T1 once, return (conversation_so_far, TurnRecord)."""
    rec = TurnRecord()
    convo = [_system_msg(), {"role": "user", "content": _t1_prompt(question)}]

    text, err = _safe_call(convo, model, temperature, json_object=True)
    if err:
        rec.error = err
        return convo, rec

    rec.raw_text = text
    answer, conf, src = extract_answer_and_confidence(text)
    rec.answer       = answer
    rec.confidence   = conf
    rec.rating_src   = src
    convo.append({"role": "assistant", "content": text})
    return convo, rec


def run_explanation_sample(
    item: dict,
    model: str,
    sample_index: int,
    temperature: float = TEMPERATURE,
) -> SampleRecord:
    """One sample of the full explanation arm."""
    sample = SampleRecord(sample_index=sample_index, arm="explanation")
    convo, sample.t1 = _run_t1(item["question"], model, temperature)
    if sample.t1.error:
        sample.error = sample.t1.error
        return sample

    # T2: explanation
    convo.append({"role": "user", "content": T2_EXPLANATION_PROMPT})
    expl_text, err = _safe_call(convo, model, temperature, json_object=False)
    sample.explanation = TurnRecord()
    if err:
        sample.explanation.error = err
        sample.error = err
        return sample
    sample.explanation.raw_text = expl_text
    sample.explanation_scores   = score_explanation(expl_text)
    convo.append({"role": "assistant", "content": expl_text})
    time.sleep(RATE_LIMIT_DELAY)

    # T3: diagnostic question (only if dataset provides one)
    diagnostic_q = item.get("diagnostic_question")
    if diagnostic_q:
        convo.append({"role": "user", "content": _t3_diagnostic_prompt(diagnostic_q)})
        diag_text, err = _safe_call(convo, model, temperature, json_object=False)
        sample.diagnostic = TurnRecord()
        if err:
            sample.diagnostic.error = err
            sample.error = err
            return sample
        sample.diagnostic.raw_text = diag_text
        convo.append({"role": "assistant", "content": diag_text})
        time.sleep(RATE_LIMIT_DELAY)

    # Final: re-rating
    convo.append({"role": "user", "content": T_FINAL_RATING_PROMPT})
    final_text, err = _safe_call(convo, model, temperature, json_object=True)
    if err:
        sample.final.error = err
        sample.error = err
        return sample
    sample.final.raw_text = final_text
    conf, src = extract_confidence(final_text)
    sample.final.confidence = conf
    sample.final.rating_src = src
    return sample


def run_control_sample(
    item: dict,
    model: str,
    sample_index: int,
    temperature: float = TEMPERATURE,
) -> SampleRecord:
    """One sample of the no-explanation control arm.

    T1 is identical to the explanation arm. The model is then asked to
    re-rate its confidence with no intervening explanation or diagnostic.
    Any drop here is rating drift, not IOED; the IOED estimate is
    delta(explanation) - delta(control).
    """
    sample = SampleRecord(sample_index=sample_index, arm="control")
    convo, sample.t1 = _run_t1(item["question"], model, temperature)
    if sample.t1.error:
        sample.error = sample.t1.error
        return sample

    convo.append({"role": "user", "content": T_FINAL_RATING_PROMPT})
    final_text, err = _safe_call(convo, model, temperature, json_object=True)
    if err:
        sample.final.error = err
        sample.error = err
        return sample
    sample.final.raw_text = final_text
    conf, src = extract_confidence(final_text)
    sample.final.confidence = conf
    sample.final.rating_src = src
    return sample


# ── Multi-sample runners ──────────────────────────────────────────────────────

def run_arm_replicated(
    item: dict,
    model: str,
    arm: str,
    k_samples: int = K_SAMPLES,
    temperature: float = TEMPERATURE,
) -> list[SampleRecord]:
    runner = run_explanation_sample if arm == "explanation" else run_control_sample
    out: list[SampleRecord] = []
    for k in range(k_samples):
        rec = runner(item, model, sample_index=k, temperature=temperature)
        out.append(rec)
        # Show a one-line trace for the operator.
        r1 = rec.t1.confidence
        rf = rec.final.confidence
        d  = _delta(rec.t1, rec.final)
        d_str = f"{d:+d}" if d is not None else "n/a"
        diag = "+diag" if (rec.diagnostic is not None) else ""
        err  = f"  err={rec.error[:60]}" if rec.error else ""
        print(f"      [{arm:11s} k={k}{diag:5s}] R1={r1} R2={rf} Δ={d_str}{err}")
        time.sleep(RATE_LIMIT_DELAY)
    return out


def is_skip_error(msg: str | None) -> bool:
    if not msg:
        return False
    return any(kw in msg for kw in ("unavailable", "Rate limited"))
