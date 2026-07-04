"""
gBrain / Hermes multi-LLM failover router.

Priority chain (configured in llm_router_config.json):
    1. OpenAI Codex   (primary  — your linked account/credits)
    2. Claude         (secondary — kicks in when OpenAI quota/rate limit is hit)
    3. Gemini         (tertiary — kicks in when Claude also fails)

Public entry point:
    call_llm_with_fallback(prompt, system=None, max_tokens=None, ...) -> LLMResponse

Design notes
------------
* Each provider is tried in ascending `priority`. A *failover trigger* error
  (rate limit / quota / server overload — see config) moves to the next
  provider. A *fatal* error (e.g. a malformed request / 400) is NOT failed
  over, because the next provider would reject it identically; it is raised.
* Transient triggers (429 / 5xx) are retried on the SAME provider first,
  honoring any Retry-After header. Hard quota/billing triggers skip straight
  to the next provider (retrying would just burn time).
* Every served request is logged as one JSON line (provider, model, tokens,
  latency, estimated cost) for cost tracking.
* SDKs are imported lazily, so a provider you have disabled does not need its
  SDK installed.

Install (only what you enable):
    pip install openai anthropic google-genai
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("gbrain.llm_router")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

CONFIG_PATH = os.environ.get("LLM_ROUTER_CONFIG", "llm_router_config.json")


# --------------------------------------------------------------------------- #
# Normalized result / error types
# --------------------------------------------------------------------------- #
@dataclass
class LLMResponse:
    """Provider-agnostic result. The three provider SDKs return wildly
    different shapes; everything is normalized into this."""
    text: str
    provider: str          # "openai" | "anthropic" | "gemini"
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    finish_reason: str
    estimated_cost_usd: float
    attempts: list[dict[str, Any]] = field(default_factory=list)  # audit trail of failed hops
    raw: Any = None        # original SDK response object


class AllProvidersFailedError(RuntimeError):
    """Raised when every enabled provider in the chain failed."""
    def __init__(self, attempts: list[dict[str, Any]]):
        self.attempts = attempts
        summary = " -> ".join(f"{a['provider']}:{a['error_kind']}" for a in attempts)
        super().__init__(f"All LLM providers failed: {summary}")


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_config(path: str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    # providers sorted by priority so the chain order is authoritative
    cfg["providers"] = sorted(
        (p for p in cfg["providers"] if p.get("enabled", True)),
        key=lambda p: p["priority"],
    )
    return cfg


# --------------------------------------------------------------------------- #
# Error classification (the heart of the failover logic)
# --------------------------------------------------------------------------- #
def _extract_status_and_message(exc: Exception) -> tuple[Optional[int], str]:
    """Pull an HTTP status code and message out of any of the three SDKs'
    exceptions without importing them. All three expose either
    `.status_code` / `.code` or embed the code in the message."""
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "status", None)
        or getattr(exc, "code", None)
    )
    if isinstance(status, str) and status.isdigit():
        status = int(status)
    if not isinstance(status, int):
        status = None
    # google-genai wraps the code inside .response / a dict sometimes
    resp = getattr(exc, "response", None)
    if status is None and resp is not None:
        status = getattr(resp, "status_code", None)
    return status, str(exc).lower()


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Honor a Retry-After header if the SDK surfaced one."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if headers:
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def classify_error(exc: Exception, failover_cfg: dict[str, Any]) -> str:
    """Return one of:
        'hard_quota'  -> skip straight to next provider (no same-provider retry)
        'transient'   -> retry same provider first, then fail over
        'fatal'       -> do NOT fail over (bad request, auth, etc.); raise
    """
    status, msg = _extract_status_and_message(exc)
    substrings = [s.lower() for s in failover_cfg["trigger_error_substrings"]]
    hard = [s.lower() for s in failover_cfg["retry_same_provider"]["hard_quota_substrings"]]

    # Hard quota/billing exhaustion — the whole point of the chain.
    if any(s in msg for s in hard):
        return "hard_quota"

    matched_substring = any(s in msg for s in substrings)
    matched_status = status in failover_cfg["trigger_status_codes"]

    if matched_status or matched_substring:
        # 429 can be either a soft rate limit (retry) or hard quota (already
        # caught above). Treat remaining 429/5xx as transient.
        return "transient"

    # 401/403 (bad key), 400 (bad request), 404 (bad model) — failing over
    # will not help; surface it.
    return "fatal"


# --------------------------------------------------------------------------- #
# Per-provider callers — each normalizes into LLMResponse
# --------------------------------------------------------------------------- #
def _cost(prov: dict[str, Any], p_tok: int, c_tok: int) -> float:
    return round(
        p_tok / 1000 * prov.get("cost_per_1k_input_usd", 0.0)
        + c_tok / 1000 * prov.get("cost_per_1k_output_usd", 0.0),
        6,
    )


def _call_openai(prov, prompt, system, max_tokens, temperature) -> LLMResponse:
    from openai import OpenAI  # lazy import

    client = OpenAI(
        api_key=os.environ[prov["api_key_env"]],
        timeout=prov.get("timeout_seconds", 60),
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=prov["model"],
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    latency = time.monotonic() - t0

    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = resp.usage
    p_tok = getattr(usage, "prompt_tokens", 0) or 0
    c_tok = getattr(usage, "completion_tokens", 0) or 0
    return LLMResponse(
        text=text,
        provider="openai",
        model=prov["model"],
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        latency_seconds=round(latency, 3),
        finish_reason=choice.finish_reason or "stop",
        estimated_cost_usd=_cost(prov, p_tok, c_tok),
        raw=resp,
    )


def _call_anthropic(prov, prompt, system, max_tokens, temperature) -> LLMResponse:
    import anthropic  # lazy import

    client = anthropic.Anthropic(
        api_key=os.environ[prov["api_key_env"]],
        timeout=prov.get("timeout_seconds", 60),
    )
    kwargs: dict[str, Any] = {
        "model": prov["model"],
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if prov.get("adaptive_thinking"):
        # Opus 4.8 / Sonnet 5: adaptive thinking. Note: temperature is NOT
        # accepted alongside thinking on these models, so it is omitted here.
        kwargs["thinking"] = {"type": "adaptive"}
    else:
        # temperature is accepted when thinking is not enabled.
        kwargs["temperature"] = temperature

    t0 = time.monotonic()
    resp = client.messages.create(**kwargs)
    latency = time.monotonic() - t0

    # A safety refusal is a 200 with stop_reason "refusal" and (often) empty
    # content — treat it as a provider failure so the chain falls through.
    if resp.stop_reason == "refusal":
        raise RuntimeError("anthropic refusal stop_reason (usage limit or safety)")

    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    p_tok = resp.usage.input_tokens
    c_tok = resp.usage.output_tokens
    return LLMResponse(
        text=text,
        provider="anthropic",
        model=prov["model"],
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        latency_seconds=round(latency, 3),
        finish_reason=resp.stop_reason or "end_turn",
        estimated_cost_usd=_cost(prov, p_tok, c_tok),
        raw=resp,
    )


def _call_gemini(prov, prompt, system, max_tokens, temperature) -> LLMResponse:
    from google import genai  # lazy import  (pip install google-genai)
    from google.genai import types

    client = genai.Client(api_key=os.environ[prov["api_key_env"]])
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        system_instruction=system or None,
        http_options=types.HttpOptions(timeout=int(prov.get("timeout_seconds", 60) * 1000)),
    )

    t0 = time.monotonic()
    resp = client.models.generate_content(
        model=prov["model"],
        contents=prompt,
        config=config,
    )
    latency = time.monotonic() - t0

    text = resp.text or ""
    um = getattr(resp, "usage_metadata", None)
    p_tok = getattr(um, "prompt_token_count", 0) or 0
    c_tok = getattr(um, "candidates_token_count", 0) or 0
    finish = "stop"
    if getattr(resp, "candidates", None):
        finish = str(getattr(resp.candidates[0], "finish_reason", "stop"))
    return LLMResponse(
        text=text,
        provider="gemini",
        model=prov["model"],
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        latency_seconds=round(latency, 3),
        finish_reason=finish,
        estimated_cost_usd=_cost(prov, p_tok, c_tok),
        raw=resp,
    )


_DISPATCH = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
}


# --------------------------------------------------------------------------- #
# Cost / audit logging
# --------------------------------------------------------------------------- #
def _log_served(cfg: dict[str, Any], result: LLMResponse) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "served",
        "provider": result.provider,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_seconds": result.latency_seconds,
        "estimated_cost_usd": result.estimated_cost_usd,
        "failed_over_from": [a["provider"] for a in result.attempts],
    }
    logger.info(
        "SERVED by %s/%s | %d+%d tok | %.2fs | $%.6f%s",
        result.provider, result.model,
        result.prompt_tokens, result.completion_tokens,
        result.latency_seconds, result.estimated_cost_usd,
        f" | after failover from {record['failed_over_from']}" if result.attempts else "",
    )
    _append_jsonl(cfg, record)


def _append_jsonl(cfg: dict[str, Any], record: dict[str, Any]) -> None:
    path = cfg["router"].get("cost_log_path")
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # never let logging break the request path
        logger.warning("Could not write cost log: %s", exc)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def call_llm_with_fallback(
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    config: Optional[dict[str, Any]] = None,
) -> LLMResponse:
    """Send `prompt` through the failover chain and return the first success.

    Raises AllProvidersFailedError if every provider fails (unless
    router.fail_open is true, in which case a synthetic error LLMResponse is
    returned so Hermes can degrade gracefully).
    """
    cfg = config or load_config()
    router = cfg["router"]
    failover = cfg["failover"]
    attempts: list[dict[str, Any]] = []

    for prov in cfg["providers"]:
        name = prov["name"]
        caller = _DISPATCH.get(name)
        if caller is None:
            logger.warning("No caller implemented for provider '%s' — skipping", name)
            continue

        if prov["api_key_env"] not in os.environ:
            logger.warning("%s: env var %s not set — skipping provider", name, prov["api_key_env"])
            attempts.append({"provider": name, "error_kind": "missing_api_key",
                             "detail": prov["api_key_env"]})
            continue

        mt = max_tokens or prov.get("max_tokens") or router["default_max_tokens"]
        temp = temperature if temperature is not None else prov.get(
            "temperature", router["default_temperature"])

        max_same = failover["retry_same_provider"]["max_attempts"]
        backoff = failover["retry_same_provider"]["backoff_seconds"]

        for attempt_i in range(max_same):
            try:
                result = caller(prov, prompt, system, mt, temp)
                result.attempts = attempts  # audit trail of everyone before us
                _log_served(cfg, result)
                return result
            except Exception as exc:  # noqa: BLE001 — we classify below
                kind = classify_error(exc, failover)
                record = {
                    "provider": name, "model": prov["model"],
                    "error_kind": kind, "detail": str(exc)[:300],
                    "attempt": attempt_i + 1,
                }
                logger.warning("%s attempt %d failed [%s]: %s",
                               name, attempt_i + 1, kind, str(exc)[:200])

                if kind == "fatal":
                    # Not a quota/rate issue — failing over won't help.
                    attempts.append(record)
                    raise

                if kind == "hard_quota":
                    # Credits/usage exhausted: don't retry this provider,
                    # go straight to the next one in the chain.
                    attempts.append(record)
                    break

                # transient: retry same provider, honoring Retry-After
                if attempt_i < max_same - 1:
                    delay = _retry_after_seconds(exc)
                    if delay is None:
                        delay = backoff[min(attempt_i, len(backoff) - 1)]
                    logger.info("%s transient error — retrying in %.1fs", name, delay)
                    time.sleep(delay)
                else:
                    attempts.append(record)  # exhausted retries -> fail over

    # Every provider failed.
    if router.get("fail_open"):
        logger.error("All providers failed and fail_open=true — returning error response")
        return LLMResponse(
            text="", provider="none", model="none",
            prompt_tokens=0, completion_tokens=0, latency_seconds=0.0,
            finish_reason="all_providers_failed", estimated_cost_usd=0.0,
            attempts=attempts, raw=None,
        )
    raise AllProvidersFailedError(attempts)


# --------------------------------------------------------------------------- #
# CLI smoke test:  python llm_router.py "your prompt here"
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    test_prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello and name which model you are."
    try:
        out = call_llm_with_fallback(test_prompt)
        print(f"\n[served by {out.provider} / {out.model}]  "
              f"({out.prompt_tokens}+{out.completion_tokens} tok, "
              f"{out.latency_seconds}s, ${out.estimated_cost_usd})\n")
        print(out.text)
    except AllProvidersFailedError as e:
        print("ALL PROVIDERS FAILED:")
        print(json.dumps(e.attempts, indent=2))
        sys.exit(1)
