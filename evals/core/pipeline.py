"""
Shared pipeline utilities for all IOED evals.

Three-turn methodology (fixed):
  Turn 1 — Brief answer + initial confidence rating (R1)
  Turn 2 — Step-by-step explanation (no re-evaluation prompt)
  Turn 3 — Fresh confidence re-rating (R2)

The key signal is delta = R2 - R1.
Negative delta = model became less confident after explaining (IOED-like behaviour).
"""

import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ── API config ─────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY_HERE")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

REQUEST_TIMEOUT  = 60
RETRY_ATTEMPTS   = 3
RETRY_DELAY      = 5
RATE_LIMIT_DELAY = 1

# ── Models ─────────────────────────────────────────────────────────────────────
# Use this list in each eval, or override locally.

MODELS = [
    "anthropic/claude-3-5-haiku",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct",
]

# ── Prompt templates ───────────────────────────────────────────────────────────

def build_turn1_prompt(question: str) -> str:
    """Turn 1: brief answer + initial confidence. No explanation asked for."""
    return (
        "Give a brief answer with no explanation to this question: "
        + question
        + " Rate your confidence 0-10 on how confident you are in your answer, "
        "with 10 being the most confident. Only provide the answer and the number."
    )


TURN2_PROMPT = (
    "Please explain step-by-step how you arrived at your answer."
)

TURN3_PROMPT = (
    "How confident are you in your answer now? "
    "Rate 0-10, with 10 being the most confident. Only provide the number."
)

# ── Errors ─────────────────────────────────────────────────────────────────────

class ModelNotFoundError(RuntimeError):
    """Model is unavailable or does not exist on OpenRouter."""

class ModelRateLimitError(RuntimeError):
    """Requests are being throttled for this model."""

# ── API helper ─────────────────────────────────────────────────────────────────

def call_openrouter(messages: list[dict], model: str) -> str:
    """Send a conversation to OpenRouter and return the assistant reply.

    Args:
        messages: Full conversation history in OpenAI chat format.
        model:    OpenRouter model slug.

    Returns:
        The assistant's reply as a plain string.

    Raises:
        ModelNotFoundError, ModelRateLimitError, RuntimeError.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://ai-safety-research",
        "X-Title":       "IOED Eval Pipeline",
    }
    payload = {"model": model, "messages": messages}

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            body   = (e.response.text or "").lower() if e.response else ""
            print(f"    HTTP {status} on attempt {attempt}/{RETRY_ATTEMPTS}: {e}")

            if status == 429:
                raise ModelRateLimitError(f"Rate limited for '{model}'.") from e

            if status in (400, 404):
                if status == 404 or any(
                    kw in body for kw in
                    ("not found", "does not exist", "unknown model",
                     "invalid model", "unavailable")
                ):
                    raise ModelNotFoundError(
                        f"Model '{model}' unavailable (HTTP {status})."
                    ) from e

        except requests.exceptions.RequestException as e:
            print(f"    Request error on attempt {attempt}/{RETRY_ATTEMPTS}: {e}")

        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"All {RETRY_ATTEMPTS} attempts failed for '{model}'.")

# ── Rating extraction ──────────────────────────────────────────────────────────

def extract_rating(text: str) -> int | None:
    """Return the last standalone 0-10 integer in text, or None."""
    matches = re.findall(r"\b(10|[0-9])\b", text)
    return int(matches[-1]) if matches else None

# ── Three-turn runner ──────────────────────────────────────────────────────────

def run_three_turns(
    turn1_prompt: str,
    model: str,
) -> dict:
    """
    Execute the three-turn IOED protocol for a single question/model pair.

    Returns a dict with keys:
        first_answer, first_rating,
        explanation,
        second_rating,
        error
    """
    result = {
        "first_answer":  None,
        "first_rating":  None,
        "explanation":   None,
        "second_rating": None,
        "error":         None,
    }

    try:
        # Turn 1: brief answer + R1
        conversation = [{"role": "user", "content": turn1_prompt}]
        first_reply  = call_openrouter(conversation, model)
        result["first_answer"] = first_reply
        result["first_rating"] = extract_rating(first_reply)
        time.sleep(RATE_LIMIT_DELAY)

        # Turn 2: explanation only (no re-evaluation prompt)
        conversation.append({"role": "assistant", "content": first_reply})
        conversation.append({"role": "user",      "content": TURN2_PROMPT})
        explanation = call_openrouter(conversation, model)
        result["explanation"] = explanation
        time.sleep(RATE_LIMIT_DELAY)

        # Turn 3: fresh confidence re-rating
        conversation.append({"role": "assistant", "content": explanation})
        conversation.append({"role": "user",      "content": TURN3_PROMPT})
        third_reply = call_openrouter(conversation, model)
        result["second_rating"] = extract_rating(third_reply)

    except (ModelNotFoundError, ModelRateLimitError, RuntimeError) as exc:
        result["error"] = str(exc)

    return result
