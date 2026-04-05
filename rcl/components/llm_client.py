"""Unified LLM client with provider/model routing.

Model format: "provider/model-name"
  - google/gemini-3-flash-preview
  - anthropic/claude-opus-4-6
  - openai/gpt-5.4-mini

Supported providers out of the box: gemini, anthropic, openai.
"""

import json
import os
import queue
import random
import re
import threading
import time
from typing import Any, Callable, Optional

# Lazy imports — SDKs are only imported when their provider is used.
_anthropic = None
_openai = None
_genai = None
_genai_types = None


def _get_anthropic():
    global _anthropic
    if _anthropic is None:
        import anthropic
        _anthropic = anthropic
    return _anthropic


def _get_openai():
    global _openai
    if _openai is None:
        import openai
        _openai = openai
    return _openai


def _get_genai():
    global _genai, _genai_types
    if _genai is None:
        from google import genai
        from google.genai import types
        _genai = genai
        _genai_types = types
    return _genai, _genai_types


def _create_genai_client(genai, model_name: str = ""):
    """Create a google-genai client, auto-detecting Vertex AI vs API key mode.

    Uses Vertex AI if GOOGLE_CLOUD_PROJECT is set, otherwise uses GEMINI_API_KEY.

    Env vars:
        GEMINI_API_KEY: API key for the public Gemini API.
        GOOGLE_CLOUD_PROJECT: GCP project for Vertex AI.
        GOOGLE_CLOUD_LOCATION: GCP region for Vertex AI (default: us-central1).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project and not api_key:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        return genai.Client(vertexai=True, project=project, location=location)
    return genai.Client(api_key=api_key)


# ── Retry / error classification ─────────────────────────────

_RETRYABLE_ERROR_MARKERS = (
    "429", "rate limit", "resource_exhausted", "too many requests",
    "overloaded", "quota", "temporarily unavailable", "retry later",
    "throttle", "empty content", "empty response",
    "timeout", "timed out", "readtimeout", "read timeout",
    "connecttimeout", "connect timeout",
    "remoteprotocolerror", "remote protocol error",
    "connectionerror", "connection error", "api_connection_error",
    "connection reset", "connection aborted", "connection closed",
    "server disconnected",
)


def is_retryable_llm_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_ERROR_MARKERS)


def is_content_filtered_llm_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "content filtering policy" in text
        or "blocked by content filtering policy" in text
    )


def _retry_with_backoff(fn):
    max_retries = int(os.environ.get("RCL_LLM_MAX_RETRIES", "5"))
    base_delay = float(os.environ.get("RCL_LLM_RETRY_BASE_DELAY", "1.0"))
    max_delay = float(os.environ.get("RCL_LLM_RETRY_MAX_DELAY", "30.0"))

    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries or not is_retryable_llm_error(exc):
                raise
            sleep_s = min(max_delay, delay) * (0.8 + 0.4 * random.random())
            print(
                f"LLM transient error ({type(exc).__name__}); retrying in {sleep_s:.1f}s "
                f"[attempt {attempt + 1}/{max_retries}]",
                flush=True,
            )
            time.sleep(sleep_s)
            delay = min(max_delay, delay * 2)


def _call_with_timeout(fn, timeout_s: float, label: str):
    """Run fn() in a thread with a hard timeout."""
    if timeout_s <= 0:
        return fn()

    result_queue: "queue.Queue[tuple[bool, object]]" = queue.Queue(maxsize=1)

    def _target():
        try:
            result_queue.put((True, fn()))
        except BaseException as exc:
            result_queue.put((False, exc))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_s)

    if thread.is_alive():
        raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s")

    ok, payload = result_queue.get_nowait()
    if ok:
        return payload
    raise payload


# ── Provider/model parsing ────────────────────────────────────

def parse_model(model: str) -> tuple[str, str]:
    """Parse 'provider/model-name' into (provider, model_name).

    Model format must be 'provider/model-name' (e.g. 'openai/gpt-5.4-nano',
    'anthropic/claude-opus-4-6', 'google/gemini-3-flash-preview').

    Returns (provider, model_name) where provider is one of
    'gemini', 'anthropic', 'openai'.
    """
    if "/" not in model:
        raise ValueError(
            f"Model must be in 'provider/model-name' format, got '{model}'. "
            f"Examples: 'openai/gpt-5.4-nano', 'anthropic/claude-opus-4-6', "
            f"'google/gemini-3-flash-preview'."
        )
    provider, name = model.split("/", 1)
    provider = provider.lower()
    if provider not in ("openai", "anthropic", "google"):
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: openai, anthropic, google."
        )
    return provider, name


# ── Provider implementations ─────────────────────────────────

def _create_openai_generate(
    model_name: str,
    temperature: Optional[float],
    max_output_tokens: int,
    thinking: Optional[str],
) -> Callable[[str], str]:
    openai = _get_openai()
    client = openai.OpenAI()
    # Reasoning models use max_completion_tokens and support reasoning_effort.
    # Update this list when new OpenAI reasoning model families are released.
    is_reasoning = any(model_name.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))

    def generate(prompt: str) -> str:
        def _call():
            kwargs = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}
            if is_reasoning:
                kwargs["max_completion_tokens"] = max_output_tokens
                if thinking and thinking.lower() != "none":
                    kwargs["reasoning_effort"] = thinking.lower()
            else:
                kwargs["max_tokens"] = max_output_tokens
                if temperature is not None:
                    kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        return _retry_with_backoff(_call)

    return generate


def _create_anthropic_generate(
    model_name: str,
    temperature: Optional[float],
    max_output_tokens: int,
    thinking: Optional[str],
) -> Callable[[str], str]:
    anthropic = _get_anthropic()
    client = anthropic.Anthropic()
    timeout_s = float(os.environ.get("RCL_CLAUDE_CALL_TIMEOUT_S", "300"))
    thinking_level = str(thinking).lower() if thinking else "none"
    use_thinking = thinking_level and thinking_level != "none"

    supports_adaptive = model_name.startswith("claude-opus-4") or model_name.startswith("claude-sonnet-4")

    def generate(prompt: str) -> str:
        def _call():
            kwargs = dict(
                model=model_name,
                max_tokens=max_output_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if use_thinking:
                budget_map = {"low": 2048, "medium": 8192, "high": 16384}
                budget = budget_map.get(thinking_level, 16384)
                kwargs["temperature"] = 1
                if supports_adaptive:
                    effort_map_opus = {"low": "low", "medium": "medium", "high": "max"}
                    effort_map_sonnet = {"low": "low", "medium": "medium", "high": "high"}
                    effort_map = effort_map_opus if "opus" in model_name else effort_map_sonnet
                    kwargs["thinking"] = {"type": "adaptive"}
                    kwargs["output_config"] = {"effort": effort_map.get(thinking_level, "high")}
                    kwargs["max_tokens"] = max(max_output_tokens, budget + max_output_tokens)
                else:
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                    kwargs["max_tokens"] = max(max_output_tokens, budget + max_output_tokens)
            elif temperature is not None:
                kwargs["temperature"] = temperature

            def _invoke():
                if use_thinking:
                    with client.messages.stream(**kwargs) as stream:
                        for _ in stream:
                            pass
                    return stream.get_final_message()
                return client.messages.create(**kwargs)

            response = _call_with_timeout(_invoke, timeout_s, f"Claude {model_name}")

            text_blocks = []
            for block in getattr(response, "content", []) or []:
                if getattr(block, "type", None) == "text" and getattr(block, "text", ""):
                    text_blocks.append(block.text)

            combined = "".join(text_blocks).strip()
            if combined:
                return combined
            raise RuntimeError(f"empty content from Claude response (model={model_name})")

        return _retry_with_backoff(_call)

    return generate


def _create_gemini_generate(
    model_name: str,
    temperature: Optional[float],
    max_output_tokens: int,
    thinking: Optional[str],
) -> Callable[[str], str]:
    genai, types = _get_genai()
    client = _create_genai_client(genai, model_name)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    def generate(prompt: str) -> str:
        def _call():
            response = client.models.generate_content(
                model=model_name, contents=prompt, config=config,
            )
            try:
                return response.text
            except Exception:
                if response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
                raise RuntimeError(f"empty content from Gemini response (model={model_name})")
        return _retry_with_backoff(_call)

    return generate


# ── Public API ────────────────────────────────────────────────

_PROVIDER_FACTORIES = {
    "google": _create_gemini_generate,
    "anthropic": _create_anthropic_generate,
    "openai": _create_openai_generate,
}


def create_generate_fn(
    model: str,
    temperature: Optional[float] = None,
    max_output_tokens: int = 8192,
    thinking: Optional[str] = None,
) -> Callable[[str], str]:
    """Create a generate function for the given model.

    Model format: "provider/model-name" (e.g. "anthropic/claude-opus-4-6").

    Returns a callable(prompt: str) -> str.
    """
    provider, model_name = parse_model(model)
    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {', '.join(sorted(_PROVIDER_FACTORIES))}. "
            f"Use 'provider/model-name' format."
        )
    return factory(model_name, temperature, max_output_tokens, thinking)


# ── JSON extraction utility ──────────────────────────────────

def extract_json_from_response(text: str) -> Optional[Any]:
    """Extract JSON from an LLM response, handling nested backticks."""
    match = re.search(r'```json\s*(.*)\s*```', text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        parsed = _try_parse_bracketed(candidate)
        if parsed is not None:
            return parsed
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    parsed = _try_parse_bracketed(text)
    if parsed is not None:
        return parsed

    return None


def _try_parse_bracketed(text: str) -> Optional[Any]:
    """Find and parse the outermost JSON array or object in text."""
    candidates = [('[', ']'), ('{', '}')]
    candidates.sort(key=lambda pair: (text.find(pair[0]) if text.find(pair[0]) != -1 else float('inf')))
    for open_ch, close_ch in candidates:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
    return None
