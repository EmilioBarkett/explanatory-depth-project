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
    "openai/gpt-5.4",
    "anthropic/claude-3-5-sonnet",
    "anthropic/claude-3-5-haiku",
    "google/gemini-3.1-pro-preview",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "x-ai/grok-2",
]

QUESTIONS_FILE   = "spartqa_human_train_dataset.json"
OUTPUT_FILE      = "spartqa_results.json"
SAMPLE_SIZE      = 70    # Set to an int to randomly sample questions, or None for all
REQUEST_TIMEOUT  = 60
RETRY_ATTEMPTS   = 3
RETRY_DELAY      = 5
RATE_LIMIT_DELAY = 1

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models that support image input
VISION_MODELS = {
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-5.4",
    "anthropic/claude-3-5-sonnet",
    "anthropic/claude-3-5-haiku",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-2",
}

EXPLAIN_PROMPT = (
    "Please explain step-by-step how you reasoned about the spatial scene to reach your answer."
)

RERATE_PROMPT = (
    "Rate your confidence in your answer out of 10, with 10 being the most confident. "
    "Do not explain your rating."
)

RERATE_AFTER_REVEAL_PROMPT = (
    "Rate how well you now understand the spatial reasoning required, out of 10, "
    "with 10 meaning you fully understand why this is correct. Do not explain your rating."
)

RERATE_ORIGINAL_EXPLANATION_PROMPT = (
    "Looking back at the explanation you gave earlier, how confident are you that that reasoning was correct? "
    "Rate it out of 10, with 10 meaning you are fully confident the explanation was correct. "
    "Do not explain your rating."
)


def reveal_prompt(correct_answer):
    answer_str = ", ".join(str(a) for a in correct_answer)
    return (
        f"The correct answer is: {answer_str}. "
        "Explain step-by-step why this is the correct answer based on the spatial scene."
    )


# ── Exceptions ─────────────────────────────────────────────────────────────────
class ModelNotFoundError(RuntimeError):
    """Raised when the requested model is unavailable or does not exist."""


class ModelRateLimitError(RuntimeError):
    """Raised when requests are being throttled for a model."""


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_questions(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    items = []
    for story_entry in dataset["data"]:
        story_text = " ".join(story_entry["story"])
        for q in story_entry["questions"]:
            # YN questions have no explicit candidate list — fill it in
            candidates = q["candidate_answers"]
            if not candidates:
                candidates = ["Yes", "No"]

            # FR answers are indices into candidate_answers — resolve to strings
            if q["q_type"] == "FR":
                answer_strings = [candidates[i] for i in q["answer"] if isinstance(i, int) and i < len(candidates)]
            else:
                answer_strings = q["answer"]

            items.append({
                "id":             f"q{q['q_id']}",
                "story":          story_text,
                "question":       q["question"],
                "q_type":         q["q_type"],
                "reasoning_type": [r.strip() for r in q.get("reasoning_type", [])],
                "answer":         answer_strings,
                "candidates":     candidates,
                "image_url":      q.get("image") or None,
            })

    if SAMPLE_SIZE is not None:
        items = random.sample(items, SAMPLE_SIZE)
        print(f"Loaded {len(dataset['data'])} stories → flattened, sampled {SAMPLE_SIZE} questions.")
    else:
        print(f"Loaded {len(dataset['data'])} stories → {len(items)} questions total.")

    return items


def build_first_prompt(story: str, question: str, candidates: list, image_url: str | None = None):
    candidates_str = ", ".join(str(c) for c in candidates)
    text = (
        f"Read the following scene description carefully:\n\n"
        f"{story}\n\n"
        f"Question: {question}\n"
        f"Candidate answers: {candidates_str}\n\n"
        "Give a brief answer choosing only from the candidate answers above — no explanation. "
        "Then give a score 0-10 on how confident you are in your answer, "
        "with 10 being the most confident. Do not explain your rating."
    )
    if image_url:
        return [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": text},
        ]
    return text


def extract_rating(text: str) -> int | None:
    matches = re.findall(r"\b(10|[0-9])\b", text)
    return int(matches[-1]) if matches else None


def extract_selected_answers(text: str, candidates: list) -> list[str]:
    """Return all candidate answers found in the model's response."""
    text_lower = text.lower()
    found = []
    for candidate in candidates:
        if re.search(r'\b' + re.escape(str(candidate).lower()) + r'\b', text_lower):
            found.append(str(candidate))
    return found


def is_correct(selected: list[str], correct: list[str]) -> bool | None:
    if not selected:
        return None
    return set(s.lower() for s in selected) == set(c.lower() for c in correct)


def call_openrouter(messages: list[dict], model: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://ai-safety-research",
        "X-Title":       "SpaRTQA Spatial Reasoning Pipeline",
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


# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(questions):
    results  = []
    total    = len(questions) * len(MODELS)
    attempted = 0

    for model in MODELS:
        print(f"=== Starting model={model!r} ===")
        skip_model = False

        for q_entry in questions:
            q_id            = q_entry["id"]
            q_text          = q_entry["question"]
            q_story         = q_entry["story"]
            q_type          = q_entry["q_type"]
            reasoning_type  = q_entry["reasoning_type"]
            q_answer        = q_entry["answer"]
            q_candidates    = q_entry["candidates"]
            image_url       = q_entry.get("image_url")
            is_vision_model = model in VISION_MODELS
            has_image       = bool(image_url)

            attempted += 1
            print(f"[{attempted}/{total}] model={model!r}  question={q_id!r}  type={q_type!r}  image={'yes' if has_image else 'no'}")

            entry = {
                "question_id":        q_id,
                "question":           q_text,
                "story":              q_story,
                "q_type":             q_type,
                "reasoning_type":     reasoning_type,
                "correct_answer":     q_answer,
                "candidate_answers":  q_candidates,
                "image_url":          image_url,
                "model":              model,
                "image_skipped":      has_image and not is_vision_model,
                "first_answer":        None,
                "first_selected":      None,
                "first_correct":       None,
                "initial_confidence":        None,
                "second_explanation":  None,
                "post_explanation_confidence":        None,
                "fourth_explanation":              None,
                "post_reveal_confidence":          None,
                "original_explanation_rating":     None,
                "error":                           None,
            }

            # Skip questions with images for non-vision models
            if has_image and not is_vision_model:
                print(f"    SKIP (image question, model lacks vision support)")
                results.append(entry)
                print()
                continue

            first_prompt = build_first_prompt(q_story, q_text, q_candidates,
                                              image_url if is_vision_model else None)

            try:
                # Turn 1: answer + initial confidence rating
                conversation = [{"role": "user", "content": first_prompt}]
                first_reply = call_openrouter(conversation, model)
                entry["first_answer"]   = first_reply
                entry["first_selected"] = extract_selected_answers(first_reply, q_candidates)
                entry["first_correct"]  = is_correct(entry["first_selected"], q_answer)
                entry["initial_confidence"]   = extract_rating(first_reply)

                time.sleep(RATE_LIMIT_DELAY)

                # Turn 2: explanation only (no re-evaluation)
                conversation.append({"role": "assistant", "content": first_reply})
                conversation.append({"role": "user",      "content": EXPLAIN_PROMPT})
                second_reply = call_openrouter(conversation, model)
                entry["second_explanation"] = second_reply

                time.sleep(RATE_LIMIT_DELAY)

                # Turn 3: clean confidence re-rating
                conversation.append({"role": "assistant", "content": second_reply})
                conversation.append({"role": "user",      "content": RERATE_PROMPT})
                third_reply = call_openrouter(conversation, model)
                entry["post_explanation_confidence"] = extract_rating(third_reply)

                time.sleep(RATE_LIMIT_DELAY)

                # Turn 4: reveal correct answer + explanation
                conversation.append({"role": "assistant", "content": third_reply})
                conversation.append({"role": "user",      "content": reveal_prompt(q_answer)})
                fourth_reply = call_openrouter(conversation, model)
                entry["fourth_explanation"] = fourth_reply

                time.sleep(RATE_LIMIT_DELAY)

                # Turn 5: re-rating after reveal
                conversation.append({"role": "assistant", "content": fourth_reply})
                conversation.append({"role": "user",      "content": RERATE_AFTER_REVEAL_PROMPT})
                fifth_reply = call_openrouter(conversation, model)
                entry["post_reveal_confidence"] = extract_rating(fifth_reply)

                time.sleep(RATE_LIMIT_DELAY)

                # Turn 6: rate accuracy of original explanation in hindsight
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


# ── Analysis ───────────────────────────────────────────────────────────────────
def build_analysis(results):
    def empty_bucket():
        return {"r1": [], "r3": [], "r5": [], "r6": [], "first_correct": []}

    def accumulate(bucket, r):
        if r["initial_confidence"] is not None:
            bucket["r1"].append(r["initial_confidence"])
        if r["post_explanation_confidence"] is not None:
            bucket["r3"].append(r["post_explanation_confidence"])
        if r["post_reveal_confidence"] is not None:
            bucket["r5"].append(r["post_reveal_confidence"])
        if r["original_explanation_rating"] is not None:
            bucket["r6"].append(r["original_explanation_rating"])
        if r["first_correct"] is not None:
            bucket["first_correct"].append(r["first_correct"])

    model_stats, qtype_stats, reasoning_stats = {}, {}, {}

    for r in results:
        for key, store in [(r["model"], model_stats), (r["q_type"], qtype_stats)]:
            if key not in store:
                store[key] = empty_bucket()
            accumulate(store[key], r)

        for rt in r.get("reasoning_type", []):
            if rt not in reasoning_stats:
                reasoning_stats[rt] = empty_bucket()
            accumulate(reasoning_stats[rt], r)

    def summarise(s):
        pairs_1_3 = [(a, b) for a, b in zip(s["r1"], s["r3"])]
        pairs_3_5 = [(a, b) for a, b in zip(s["r3"], s["r5"])]
        return {
            "avg_initial_confidence":          round(statistics.mean(s["r1"]), 2) if s["r1"] else None,
            "avg_post_explanation_confidence": round(statistics.mean(s["r3"]), 2) if s["r3"] else None,
            "avg_post_reveal_confidence":      round(statistics.mean(s["r5"]), 2) if s["r5"] else None,
            "avg_original_explanation_rating": round(statistics.mean(s["r6"]), 2) if s["r6"] else None,
            "delta_explanation":               round(statistics.mean([b - a for a, b in pairs_1_3]), 2) if pairs_1_3 else None,
            "delta_reveal":                    round(statistics.mean([b - a for a, b in pairs_3_5]), 2) if pairs_3_5 else None,
            "pct_correct_first":               round(100 * sum(s["first_correct"]) / len(s["first_correct"]), 1) if s["first_correct"] else None,
            "n": len(s["r1"]),
        }

    return {
        "by_model":          {m:  summarise(s) for m,  s in model_stats.items()},
        "by_qtype":          {qt: summarise(s) for qt, s in qtype_stats.items()},
        "by_reasoning_type": {rt: summarise(s) for rt, s in reasoning_stats.items()},
    }


def print_analysis(analysis):
    header = f"{'Group':<45} {'R1':>6} {'R3':>6} {'R5':>6} {'R6':>6} {'Δ expl':>8} {'Δ reveal':>9} {'%Cor1':>7} {'N':>5}"
    sep    = "-" * 108

    def fmt(name, s):
        def v(x): return float('nan') if x is None else x
        return (f"{name:<45} {v(s['avg_initial_confidence']):>6.2f} {v(s['avg_post_explanation_confidence']):>6.2f} "
                f"{v(s['avg_post_reveal_confidence']):>6.2f} {v(s['avg_original_explanation_rating']):>6.2f} "
                f"{v(s['delta_explanation']):>+8.2f} {v(s['delta_reveal']):>+9.2f} "
                f"{v(s['pct_correct_first']):>7.1f} {s['n']:>5}")

    for label, bucket in [("By Model", "by_model"),
                           ("By QType", "by_qtype"),
                           ("By Reasoning Type", "by_reasoning_type")]:
        print(f"\n{label}")
        print(header)
        print(sep)
        for name, s in analysis[bucket].items():
            print(fmt(name, s))

    print("\nR1=initial, R3=post-explanation, R5=post-reveal. Δ expl = R3-R1, Δ reveal = R5-R3.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = load_questions(QUESTIONS_FILE)
    results   = run_pipeline(questions)
    analysis  = build_analysis(results)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"analysis": analysis, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} entries → '{OUTPUT_FILE}'")

    print_analysis(analysis)
