"""
Structured-output helpers for IOED evals.

Replaces the regex-based rating extractor with model-emitted JSON. Falls
back to regex if a provider ignores the JSON instruction, but flags the
row so we know when the structured path failed.

Two payload shapes are used by the Rozenblit protocol:

  T1 (answer + initial rating):
      {"answer": "<brief answer>", "confidence": <int 0-10>}

  Subsequent rating turns (after explanation, after diagnostic, control re-rate):
      {"confidence": <int 0-10>}
"""

import json
import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = os.getenv(
    "OPENROUTER_API_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)

REQUEST_TIMEOUT  = 90
RETRY_ATTEMPTS   = 3
RETRY_DELAY      = 5


def openrouter_requests_kwargs() -> dict:
    """Extra kwargs for requests to OpenRouter.

    Set OPENROUTER_NO_PROXY=1 if HTTP(S)_PROXY sends traffic through a proxy that
    returns HTML 404 for api.openrouter (common on school/corporate networks).
    Override base URL with OPENROUTER_API_URL if OpenRouter documents a change.
    """
    out: dict = {}
    if os.getenv("OPENROUTER_NO_PROXY", "").strip().lower() in ("1", "true", "yes"):
        out["proxies"] = {"http": None, "https": None}
    return out


class ModelNotFoundError(RuntimeError):
    """Model is unavailable or does not exist on OpenRouter."""


class ModelRateLimitError(RuntimeError):
    """Requests are being throttled for this model."""


def call_openrouter(
    messages: list[dict],
    model: str,
    temperature: float | None = 0.0,
    json_object: bool = False,
    seed: int | None = None,
) -> str:
    """Send messages to OpenRouter; return assistant reply as a string.

    Args:
        messages:    OpenAI-style chat history.
        model:       OpenRouter model slug.
        temperature: 0.0 for deterministic, higher for sampling. Some
                     reasoning models silently ignore temperature; we
                     pass it anyway and let the provider decide.
        json_object: If True, request response_format json_object. Many
                     providers honour this; if not, parsing falls back to
                     a regex on the raw string.
        seed:        Optional integer seed; honoured by some providers.

    Raises:
        ModelNotFoundError, ModelRateLimitError, RuntimeError.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://ai-safety-research",
        "X-Title":       "IOED Eval Pipeline",
    }
    payload: dict = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if seed is not None:
        payload["seed"] = seed
    if json_object:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
                **openrouter_requests_kwargs(),
            )
            resp.raise_for_status()

            data = resp.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                raise RuntimeError(f"Malformed response for '{model}': missing choices.")

            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(f"Empty or invalid content returned for '{model}'.")

            return content.strip()

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            raw    = (e.response.text or "") if e.response is not None else ""
            body   = raw.lower()
            print(f"    HTTP {status} on attempt {attempt}/{RETRY_ATTEMPTS}: {e}")
            if status == 404 and raw.strip():
                snippet = raw.strip().replace("\n", " ")[:240]
                print(f"    Body snippet: {snippet!r}")

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

                # response_format not supported by this provider — retry without it.
                if json_object and "response_format" in body:
                    print("    Provider rejected response_format; retrying without it.")
                    payload.pop("response_format", None)
                    json_object = False
                    continue

        except requests.exceptions.RequestException as e:
            print(f"    Request error on attempt {attempt}/{RETRY_ATTEMPTS}: {e}")

        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"All {RETRY_ATTEMPTS} attempts failed for '{model}'.")


# ── JSON parsing ──────────────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RATING_FALLBACK_RE = re.compile(r'"confidence"\s*:\s*(10|[0-9])(?!\d)')
_LAST_INT_RE = re.compile(r"\b(10|[0-9])\b")


def _strip_json_fences(text: str) -> str:
    """If the reply is wrapped in ```json … ``` fences, return the inside."""
    m = _JSON_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def parse_json_payload(text: str) -> dict | None:
    """Best-effort JSON parse. Returns None if the text isn't valid JSON."""
    candidate = _strip_json_fences(text)
    # Some models prepend prose; try to slice from first { to last }.
    if not candidate.startswith("{"):
        first = candidate.find("{")
        last  = candidate.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = candidate[first:last + 1]
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def extract_confidence(text: str) -> tuple[int | None, str]:
    """Pull a 0-10 confidence out of `text`.

    Returns (rating, source) where source is one of:
        "json"        - cleanly parsed from a {"confidence": N} object
        "json_field"  - regex hit on a "confidence": N substring
        "last_int"    - fallback: last 0-10 integer in the text
        "missing"     - nothing found
    """
    payload = parse_json_payload(text)
    if isinstance(payload, dict) and "confidence" in payload:
        try:
            v = int(payload["confidence"])
            if 0 <= v <= 10:
                return v, "json"
        except (TypeError, ValueError):
            pass

    m = _RATING_FALLBACK_RE.search(text)
    if m:
        return int(m.group(1)), "json_field"

    matches = _LAST_INT_RE.findall(text)
    if matches:
        return int(matches[-1]), "last_int"

    return None, "missing"


def extract_answer_and_confidence(text: str) -> tuple[str | None, int | None, str]:
    """Pull (answer, confidence, source) for T1-style replies.

    `source` mirrors `extract_confidence`. If JSON parsing succeeds we
    return both fields; otherwise the answer is the full raw text and
    confidence falls back through the same chain.
    """
    payload = parse_json_payload(text)
    if isinstance(payload, dict) and "confidence" in payload:
        try:
            v = int(payload["confidence"])
            if 0 <= v <= 10:
                ans = payload.get("answer")
                if not isinstance(ans, str):
                    ans = text
                return ans, v, "json"
        except (TypeError, ValueError):
            pass

    rating, source = extract_confidence(text)
    return text, rating, source
