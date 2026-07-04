# gBrain / Hermes — Multi-LLM Failover Chain

Automatic provider failover for the Hermes orchestration layer (NousResearch,
WSL2 v0.15.2).

**Priority order**

| Tier | Provider | Kicks in when… |
|------|----------|----------------|
| 1 (primary) | **OpenAI Codex** | default — uses your linked account/credits |
| 2 (secondary) | **Claude** (`claude-opus-4-8`) | OpenAI credits / usage hours are exhausted |
| 3 (tertiary) | **Gemini** (`gemini-2.5-pro`) | Claude's usage limit is also reached |

Files:

- `llm_router.py` — the wrapper (`call_llm_with_fallback`)
- `llm_router_config.json` — providers, keys, priorities, failover triggers
- `.env.example` — API-key env vars
- `requirements-llm-router.txt` — SDKs (install only what you enable)

---

## 1. Where this lives in your 6-layer architecture

```
WhatsApp
   │
Hermes fuzzy-router      ← ★ the failover chain lives HERE
   │                        (llm_router.py, called by the router)
Paperclip governance
   │
~32 Python worker scripts
   │
ERP systems
```

Put the router **inside the Hermes layer**, as the single choke point every
LLM-bound request passes through. Concretely:

- Drop `llm_router.py` + `llm_router_config.json` into the Hermes package.
- Anywhere Hermes currently calls OpenAI directly, replace that call with
  `call_llm_with_fallback(prompt, ...)`.
- The **worker scripts do not each talk to a provider.** They ask Hermes for a
  completion; Hermes owns the provider choice and the failover. That keeps
  keys, quota logic, and cost logging in one place instead of scattered across
  32 scripts — and lets Paperclip governance sit *above* a provider-agnostic
  result, so a policy rule doesn't have to know whether OpenAI or Gemini
  answered.

> Rule of thumb: routing = policy, and policy belongs in the router layer.
> Workers stay dumb; Hermes decides who answers.

---

## 2. Failover logic — how a provider failure is detected

`classify_error()` in `llm_router.py` sorts every SDK exception into three buckets:

| Bucket | Meaning | What the router does |
|--------|---------|----------------------|
| `hard_quota` | Credits/usage exhausted (`insufficient_quota`, `credit balance is too low`, `usage limit`, hard `quota`) | **Skip straight to the next provider** — retrying is pointless |
| `transient` | Temporary rate limit / server overload (HTTP 408/429/500/502/503/504/529, "rate limit", "overloaded", "capacity") | **Retry the same provider** (honoring `Retry-After`), then fail over |
| `fatal` | Bad request / bad key / bad model (400/401/403/404) | **Raise** — the next provider would reject it identically |

### Per-provider signals it keys on

| Provider | Quota exhausted (→ fail over) | Transient rate limit (→ retry then fail over) |
|----------|-------------------------------|-----------------------------------------------|
| **OpenAI** | HTTP 429 with body `error.type: insufficient_quota` (billing/credits gone) | HTTP 429 `rate_limit_exceeded`; 500/503 server errors |
| **Claude** | HTTP 400 `invalid_request_error` "credit balance is too low"; message contains "usage limit" | HTTP 429 `rate_limit_error` (has `Retry-After`); HTTP 529 `overloaded_error` |
| **Gemini** | HTTP 429 `RESOURCE_EXHAUSTED` (quota); message "quota" | HTTP 429 rate-limit; 500/503 `UNAVAILABLE` |

The `_comment` blocks in `llm_router_config.json` list every status code and
substring — edit those to tune behavior; no code changes needed.

Note: a **Claude safety refusal** (HTTP 200, `stop_reason: "refusal"`) is also
treated as a provider failure so the chain falls through to Gemini rather than
returning an empty answer.

---

## 3. API differences the wrapper normalizes for you

You do **not** have to handle these at the call site — `LLMResponse` hides them —
but they are the reason a naive "just swap the base URL" approach breaks:

| Concern | OpenAI | Claude | Gemini |
|---------|--------|--------|--------|
| **Text location** | `resp.choices[0].message.content` | `resp.content[]` — a list of blocks; concatenate `block.text` where `type=="text"` | `resp.text` |
| **System prompt** | a `{"role":"system"}` message | top-level `system=` kwarg (not in `messages`) | `system_instruction` in `GenerateContentConfig` |
| **Max output tokens** | `max_tokens` | `max_tokens` (**required**) | `max_output_tokens` |
| **Prompt tokens** | `usage.prompt_tokens` | `usage.input_tokens` | `usage_metadata.prompt_token_count` |
| **Output tokens** | `usage.completion_tokens` | `usage.output_tokens` | `usage_metadata.candidates_token_count` |
| **Rate-limit error** | `openai.RateLimitError`, 429, `Retry-After` header | `anthropic.RateLimitError`, 429/529, `Retry-After` header | `google.genai.errors.ClientError`, 429 `RESOURCE_EXHAUSTED` |
| **Timeout unit** | seconds | seconds | **milliseconds** (`HttpOptions(timeout=...)`) — the wrapper converts |
| **`temperature` + reasoning** | always allowed | **rejected** when adaptive thinking is on (wrapper omits it in that case) | allowed |
| **"Refusal"** | finish_reason `content_filter` | dedicated `stop_reason: "refusal"` (HTTP 200) | `finish_reason: SAFETY` on the candidate |

---

## 4. Usage

```python
from llm_router import call_llm_with_fallback, AllProvidersFailedError

try:
    r = call_llm_with_fallback(
        "Summarize this ERP invoice batch and flag anomalies.",
        system="You are Hermes, a terse operations assistant.",
        max_tokens=1024,
    )
    print(r.provider, r.model, r.text)   # e.g. 'openai gpt-5.1 ...'
except AllProvidersFailedError as e:
    ...  # every provider down — let Paperclip governance decide
```

Every served request appends one line to `logs/llm_cost_log.jsonl`:

```json
{"ts":"2026-07-04T...","event":"served","provider":"anthropic","model":"claude-opus-4-8","prompt_tokens":812,"completion_tokens":240,"latency_seconds":3.4,"estimated_cost_usd":0.01006,"failed_over_from":["openai"]}
```

`failed_over_from` tells you exactly when the primary was down — the data you
need for cost attribution and quota planning.

---

## 5. Setup checklist

1. `pip install -r requirements-llm-router.txt` (or just the SDKs for the
   providers you enable).
2. `cp .env.example .env` and fill in `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
   `GEMINI_API_KEY`; load it into the Hermes process env.
3. In `llm_router_config.json`, set each provider's `model` to the exact model
   string your account serves (`gpt-5.1` / `claude-opus-4-8` / `gemini-2.5-pro`)
   and adjust `cost_per_1k_*` to your billed rates.
4. Smoke-test the chain: `python llm_router.py "hello, which model are you?"` —
   confirm it reports `served by openai`.
5. Force a failover to verify tiers 2 and 3: temporarily set an invalid
   `OPENAI_API_KEY` (or `"enabled": false` on OpenAI) and re-run — it should
   report `served by anthropic`; disable Claude too and confirm Gemini answers.
6. Wire Hermes: replace the direct OpenAI call in the fuzzy-router with
   `call_llm_with_fallback(...)`; leave the ~32 workers calling Hermes, not
   providers.
7. Point log rotation / your cost dashboard at `logs/llm_cost_log.jsonl`.
