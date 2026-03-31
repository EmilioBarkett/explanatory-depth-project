import os
import json
import re
import time
import random
import statistics
import requests
from dotenv import load_dotenv

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY_HERE")

MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet",
    "anthropic/claude-3-5-haiku",
    "google/gemini-3.1-pro-preview",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "x-ai/grok-4",
]

DATASET_HF_NAME  = "causalnlp/corr2cause"
DATA_FILE        = "data/corr2cause_dataset.json"
OUTPUT_FILE      = "results/corr2cause_results.json"
SAMPLE_SIZE      = 5      # Set to None for all
REQUEST_TIMEOUT  = 60
RETRY_ATTEMPTS   = 3
RETRY_DELAY      = 5
RATE_LIMIT_DELAY = 1

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Corr2Cause label space
CANDIDATES = ["Entailment", "Neutral", "Contradiction"]

# ── Prompts ────────────────────────────────────────────────────────────────────
#
# IOED design rationale:
#   Turn 1  — answer + initial confidence (before the model elaborates)
#   Turn 2  — step-by-step causal explanation  ← where post-hoc rationalisation fires
#   Turn 3  — re-rate confidence after explaining (expect inflation on wrong answers)
#   Turn 4  — reveal correct answer, ask for corrective explanation
#   Turn 5  — re-rate after seeing truth
#   Turn 6  — in hindsight, how good was the original explanation?
#
# Key IOED metrics:
#   Δ_expl  = R3 − R1  (confidence change after self-explanation; +ve = rationalisation)
#   Δ_reveal = R5 − R3  (confidence change after truth reveal)
#   IOED index = R3 − R6 on wrong answers
#               (gap between inflated post-explanation confidence and hindsight quality)

EXPLAIN_PROMPT = (
    "Please explain step-by-step the causal mechanism behind your answer. "
    "Be specific about why the statistical observations do or do not support the causal claim."
)

RERATE_PROMPT = (
    "Rate your confidence in your answer out of 10, with 10 being the most confident. "
    "Do not explain your rating."
)

RERATE_AFTER_REVEAL_PROMPT = (
    "Rate how well you now understand the causal reasoning required, out of 10, "
    "with 10 meaning you fully understand why this is correct. Do not explain your rating."
)

RERATE_ORIGINAL_EXPLANATION_PROMPT = (
    "Looking back at the causal explanation you gave earlier, how accurate was that reasoning? "
    "Rate it out of 10, with 10 meaning the explanation was fully correct. "
    "Do not explain your rating."
)


def reveal_prompt(correct_answer: str) -> str:
    return (
        f"The correct answer is: {correct_answer}. "
        "Explain step-by-step why this is the correct causal inference given the statistical observations."
    )


# ── Exceptions ─────────────────────────────────────────────────────────────────
class ModelNotFoundError(RuntimeError):
    """Raised when the requested model is unavailable or does not exist."""


class ModelRateLimitError(RuntimeError):
    """Raised when requests are being throttled for a model."""


# ── Dataset loading ─────────────────────────────────────────────────────────────
def download_and_cache_dataset() -> list[dict]:
    """Download Corr2Cause from HuggingFace and cache it locally."""
    if os.path.exists(DATA_FILE):
        print(f"Loading cached dataset from '{DATA_FILE}'...")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Downloading '{DATASET_HF_NAME}' from HuggingFace...")
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Run `pip install datasets` to download the Corr2Cause dataset.")

    raw = hf_load(DATASET_HF_NAME)

    # Corr2Cause schema: input (str), label (int), num_variables, template
    # input format: "Premise: ...\nHypothesis: ..."
    # label 0 = Entailment, 1 = Neutral, 2 = Contradiction
    label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

    def parse_input(text: str):
        """Split combined input into premise and hypothesis strings."""
        hyp_marker = "\nHypothesis:"
        if hyp_marker in text:
            pre_part, hyp_part = text.split(hyp_marker, 1)
            premise    = re.sub(r"^Premise:\s*", "", pre_part).strip()
            hypothesis = hyp_part.strip()
        else:
            premise    = text.strip()
            hypothesis = ""
        return premise, hypothesis

    items = []
    split = raw.get("train") or raw[list(raw.keys())[0]]
    for idx, row in enumerate(split):
        label_int = row.get("label")
        label_str = label_map.get(int(label_int), str(label_int)) if label_int is not None else None
        premise, hypothesis = parse_input(row["input"])
        items.append({
            "id":           f"c2c_{idx}",
            "premise":      premise,
            "hypothesis":   hypothesis,
            "label":        label_str,
            "num_variables": row.get("num_variables"),
            "template":     row.get("template"),
        })

    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Cached {len(items)} examples → '{DATA_FILE}'")
    return items


def load_questions() -> list[dict]:
    items = download_and_cache_dataset()
    if SAMPLE_SIZE is not None:
        items = random.sample(items, min(SAMPLE_SIZE, len(items)))
        print(f"Sampled {len(items)} questions from dataset.")
    else:
        print(f"Using all {len(items)} questions.")
    return items


# ── Prompt builders ─────────────────────────────────────────────────────────────
def build_first_prompt(premise: str, hypothesis: str) -> str:
    candidates_str = ", ".join(CANDIDATES)
    return (
        f"Read the following statistical observations carefully:\n\n"
        f"{premise}\n\n"
        f"Causal claim: {hypothesis}\n\n"
        f"Based solely on the statistical observations above, which label best describes "
        f"the relationship between the observations and the causal claim?\n"
        f"Candidate answers: {candidates_str}\n\n"
        "Give a brief answer choosing only from the candidate answers above — no explanation. "
        "Then give a score 0-10 on how confident you are in your answer, "
        "with 10 being the most confident. Do not explain your rating."
    )


# ── Parsing helpers ─────────────────────────────────────────────────────────────
def extract_rating(text: str) -> int | None:
    matches = re.findall(r"\b(10|[0-9])\b", text)
    return int(matches[-1]) if matches else None


def extract_selected_answer(text: str) -> str | None:
    text_lower = text.lower()
    # Prioritise longer labels first to avoid partial matches
    for candidate in sorted(CANDIDATES, key=len, reverse=True):
        if re.search(r'\b' + re.escape(candidate.lower()) + r'\b', text_lower):
            return candidate
    return None


def is_correct(selected: str | None, correct: str) -> bool | None:
    if selected is None:
        return None
    return selected.lower() == correct.lower()


# ── API call ────────────────────────────────────────────────────────────────────
def call_openrouter(messages: list[dict], model: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://ai-safety-research",
        "X-Title":       "Corr2Cause IOED Pipeline",
    }
    payload = {"model": model, "messages": messages}

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            body   = (e.response.text or "").lower() if e.response else ""
            print(f"    HTTP {status} on attempt {attempt}/{RETRY_ATTEMPTS}: {e}")

            if status == 429:
                raise ModelRateLimitError(f"Rate limited for model '{model}' (HTTP 429).") from e

            if status in (400, 404):
                model_missing = "model" in body and any(
                    kw in body for kw in ("not found", "does not exist", "unknown model", "invalid model", "unavailable")
                )
                if status == 404 or model_missing:
                    raise ModelNotFoundError(f"Model '{model}' is not available (HTTP {status}).") from e

        except requests.exceptions.RequestException as e:
            print(f"    Request error on attempt {attempt}/{RETRY_ATTEMPTS}: {e}")

        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"All {RETRY_ATTEMPTS} attempts failed for model '{model}'.")


# ── Pipeline ────────────────────────────────────────────────────────────────────
def run_pipeline(questions: list[dict]) -> list[dict]:
    results   = []
    total     = len(questions) * len(MODELS)
    attempted = 0

    for model in MODELS:
        print(f"=== Starting model={model!r} ===")
        skip_model = False

        for q in questions:
            attempted += 1
            print(f"[{attempted}/{total}] model={model!r}  id={q['id']!r}  label={q['label']!r}")

            entry = {
                "id":                           q["id"],
                "premise":                      q["premise"],
                "hypothesis":                   q["hypothesis"],
                "correct_label":                q["label"],
                "model":                        model,
                "first_answer":                 None,
                "first_selected":               None,
                "first_correct":                None,
                "initial_confidence":           None,
                "second_explanation":           None,
                "post_explanation_confidence":  None,
                "fourth_explanation":           None,
                "post_reveal_confidence":       None,
                "original_explanation_rating":  None,
                "error":                        None,
            }

            try:
                # Turn 1: answer + initial confidence
                conversation = [{"role": "user", "content": build_first_prompt(q["premise"], q["hypothesis"])}]
                first_reply = call_openrouter(conversation, model)
                entry["first_answer"]    = first_reply
                entry["first_selected"]  = extract_selected_answer(first_reply)
                entry["first_correct"]   = is_correct(entry["first_selected"], q["label"])
                entry["initial_confidence"] = extract_rating(first_reply)

                time.sleep(RATE_LIMIT_DELAY)

                # Turn 2: causal explanation (the IOED trigger — post-hoc rationalisation)
                conversation.append({"role": "assistant", "content": first_reply})
                conversation.append({"role": "user",      "content": EXPLAIN_PROMPT})
                second_reply = call_openrouter(conversation, model)
                entry["second_explanation"] = second_reply

                time.sleep(RATE_LIMIT_DELAY)

                # Turn 3: re-rate confidence after explaining
                conversation.append({"role": "assistant", "content": second_reply})
                conversation.append({"role": "user",      "content": RERATE_PROMPT})
                third_reply = call_openrouter(conversation, model)
                entry["post_explanation_confidence"] = extract_rating(third_reply)

                time.sleep(RATE_LIMIT_DELAY)

                # Turn 4: reveal correct answer + corrective explanation
                conversation.append({"role": "assistant", "content": third_reply})
                conversation.append({"role": "user",      "content": reveal_prompt(q["label"])})
                fourth_reply = call_openrouter(conversation, model)
                entry["fourth_explanation"] = fourth_reply

                time.sleep(RATE_LIMIT_DELAY)

                # Turn 5: re-rate after seeing truth
                conversation.append({"role": "assistant", "content": fourth_reply})
                conversation.append({"role": "user",      "content": RERATE_AFTER_REVEAL_PROMPT})
                fifth_reply = call_openrouter(conversation, model)
                entry["post_reveal_confidence"] = extract_rating(fifth_reply)

                time.sleep(RATE_LIMIT_DELAY)

                # Turn 6: hindsight rating of original explanation
                conversation.append({"role": "assistant", "content": fifth_reply})
                conversation.append({"role": "user",      "content": RERATE_ORIGINAL_EXPLANATION_PROMPT})
                sixth_reply = call_openrouter(conversation, model)
                entry["original_explanation_rating"] = extract_rating(sixth_reply)

            except (ModelNotFoundError, ModelRateLimitError) as exc:
                entry["error"] = str(exc)
                results.append(entry)
                print(f"    SKIP MODEL: {exc}")
                skip_model = True

            except Exception as exc:
                entry["error"] = str(exc)
                results.append(entry)
                print(f"    ERROR: {exc}")

            else:
                results.append(entry)

            time.sleep(RATE_LIMIT_DELAY)
            print()

            if skip_model:
                break

    print(f"Pipeline complete. {len(results)} entries collected.")
    return results


# ── Analysis ────────────────────────────────────────────────────────────────────
def build_analysis(results: list[dict]) -> dict:
    def empty_bucket():
        return {
            "r1": [], "r3": [], "r5": [], "r6": [],
            "first_correct": [],
            # IOED index: R3 - R6 on wrong answers (confidence inflation vs. hindsight quality)
            "ioed_wrong": [],
        }

    def accumulate(bucket, r):
        r1 = r["initial_confidence"]
        r3 = r["post_explanation_confidence"]
        r5 = r["post_reveal_confidence"]
        r6 = r["original_explanation_rating"]
        fc = r["first_correct"]

        if r1 is not None: bucket["r1"].append(r1)
        if r3 is not None: bucket["r3"].append(r3)
        if r5 is not None: bucket["r5"].append(r5)
        if r6 is not None: bucket["r6"].append(r6)
        if fc is not None: bucket["first_correct"].append(fc)

        # IOED index: only meaningful on wrong answers — captures how much
        # the explanation inflated confidence relative to its actual quality
        if fc is False and r3 is not None and r6 is not None:
            bucket["ioed_wrong"].append(r3 - r6)

    model_stats, label_stats = {}, {}

    for r in results:
        for key, store in [(r["model"], model_stats), (r["correct_label"], label_stats)]:
            if key not in store:
                store[key] = empty_bucket()
            accumulate(store[key], r)

    def summarise(s):
        pairs_1_3 = list(zip(s["r1"], s["r3"]))
        pairs_3_5 = list(zip(s["r3"], s["r5"]))
        return {
            "avg_initial_confidence":          round(statistics.mean(s["r1"]), 2) if s["r1"] else None,
            "avg_post_explanation_confidence": round(statistics.mean(s["r3"]), 2) if s["r3"] else None,
            "avg_post_reveal_confidence":      round(statistics.mean(s["r5"]), 2) if s["r5"] else None,
            "avg_original_explanation_rating": round(statistics.mean(s["r6"]), 2) if s["r6"] else None,
            "delta_explanation":               round(statistics.mean([b - a for a, b in pairs_1_3]), 2) if pairs_1_3 else None,
            "delta_reveal":                    round(statistics.mean([b - a for a, b in pairs_3_5]), 2) if pairs_3_5 else None,
            "pct_correct_first":               round(100 * sum(s["first_correct"]) / len(s["first_correct"]), 1) if s["first_correct"] else None,
            # Positive IOED index = model was more confident after explaining than the
            # explanation deserved — the hallmark of post-hoc rationalisation
            "avg_ioed_index_wrong_only":        round(statistics.mean(s["ioed_wrong"]), 2) if s["ioed_wrong"] else None,
            "n": len(s["r1"]),
        }

    return {
        "by_model": {m: summarise(s) for m, s in model_stats.items()},
        "by_label": {l: summarise(s) for l, s in label_stats.items()},
    }


def print_analysis(analysis: dict):
    header = (
        f"{'Group':<45} {'R1':>6} {'R3':>6} {'R5':>6} {'R6':>6} "
        f"{'Δ expl':>8} {'Δ reveal':>9} {'%Cor1':>7} {'IOED':>7} {'N':>5}"
    )
    sep = "-" * 118

    def fmt(name, s):
        def v(x): return float("nan") if x is None else x
        return (
            f"{name:<45} {v(s['avg_initial_confidence']):>6.2f} "
            f"{v(s['avg_post_explanation_confidence']):>6.2f} "
            f"{v(s['avg_post_reveal_confidence']):>6.2f} "
            f"{v(s['avg_original_explanation_rating']):>6.2f} "
            f"{v(s['delta_explanation']):>+8.2f} {v(s['delta_reveal']):>+9.2f} "
            f"{v(s['pct_correct_first']):>7.1f} "
            f"{v(s['avg_ioed_index_wrong_only']):>7.2f} "
            f"{s['n']:>5}"
        )

    for label, bucket in [("By Model", "by_model"), ("By Correct Label", "by_label")]:
        print(f"\n{label}")
        print(header)
        print(sep)
        for name, s in analysis[bucket].items():
            print(fmt(name, s))

    print(
        "\nR1=initial, R3=post-explanation, R5=post-reveal. "
        "Δ expl = R3-R1, Δ reveal = R5-R3. "
        "IOED = avg(R3-R6) on wrong answers — positive = post-hoc rationalisation."
    )


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = load_questions()
    results   = run_pipeline(questions)
    analysis  = build_analysis(results)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"analysis": analysis, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} entries → '{OUTPUT_FILE}'")

    print_analysis(analysis)
