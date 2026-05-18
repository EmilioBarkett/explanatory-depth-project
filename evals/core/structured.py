from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv

from evals.core.pipeline import ModelNotFoundError, ModelRateLimitError

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv(
    "OPENROUTER_API_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)
REQUEST_TIMEOUT = 60
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5


def openrouter_requests_kwargs() -> dict[str, Any]:
    return {}


def _extract_content(data: dict[str, Any], model: str) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise RuntimeError(f"Malformed response for '{model}': missing choices.")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Empty or invalid content returned for '{model}'.")
    return content.strip()


def call_openrouter(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    json_object: bool = False,
) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-safety-research",
        "X-Title": "HLE Rozenblit-style Eval Pipeline",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
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
            return _extract_content(data, model)

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else None
            body = (exc.response.text or "").lower() if exc.response else ""
            print(f"    HTTP {status} on attempt {attempt}/{RETRY_ATTEMPTS}: {exc}")

            if status == 429:
                raise ModelRateLimitError(f"Rate limited for '{model}'.") from exc

            if status in (400, 404):
                if status == 404 or any(
                    kw in body
                    for kw in (
                        "not found",
                        "does not exist",
                        "unknown model",
                        "invalid model",
                        "unavailable",
                    )
                ):
                    raise ModelNotFoundError(
                        f"Model '{model}' unavailable (HTTP {status})."
                    ) from exc

        except requests.exceptions.RequestException as exc:
            print(f"    Request error on attempt {attempt}/{RETRY_ATTEMPTS}: {exc}")

        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"All {RETRY_ATTEMPTS} attempts failed for '{model}'.")


def _extract_jsonish_text(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def extract_answer_and_confidence(text: str) -> tuple[str | None, int | None, str]:
    parsed = _extract_jsonish_text(text)
    if isinstance(parsed, dict):
        answer = parsed.get("answer")
        confidence = parsed.get("confidence")
        try:
            confidence_int = int(confidence) if confidence is not None else None
        except Exception:
            confidence_int = None
        return (str(answer) if answer is not None else None, confidence_int, "json")

    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    confidence_match = re.search(r'"confidence"\s*:\s*(10|[0-9])', text)
    answer = answer_match.group(1) if answer_match else None
    confidence = int(confidence_match.group(1)) if confidence_match else None
    return answer, confidence, "regex"


def extract_confidence(text: str) -> tuple[int | None, str]:
    parsed = _extract_jsonish_text(text)
    if isinstance(parsed, dict) and "confidence" in parsed:
        try:
            return int(parsed["confidence"]), "json"
        except Exception:
            pass

    match = re.findall(r"\b(10|[0-9])\b", text)
    return (int(match[-1]), "regex") if match else (None, "regex")
