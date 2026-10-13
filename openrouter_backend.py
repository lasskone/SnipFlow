"""OpenRouter BYOK backend for the three SaaSShorts AI operations.

SaaSShorts researches a product, analyses it and writes video scripts. All
three ran on Gemini exclusively; this module adds OpenRouter as an optional
bring-your-own-key provider so a self-hoster can run the whole shorts pipeline
without a Google key, and so a user can opt into a model of their choice.

OpenRouter speaks the OpenAI chat-completions dialect over
``POST https://openrouter.ai/api/v1/chat/completions``:

* Structured output — the catalogue only lists models advertising
  ``response_format`` in ``supported_parameters``. We ask for ``json_object``
  first and fall back to no ``response_format`` at all for a model that rejects
  the field, mirroring ``llm_backend.py``'s ladder. Whatever comes back is
  fence-stripped and bracket-sliced exactly like the Gemini path, so the three
  operations see one shape regardless of provider.
* Web research — enabled per request by suffixing the model id with
  ``:online``. It is billed per search, separately from tokens, so a model
  entry carries ``web_search_price`` alongside the token prices.
* Citations — online models return ``choices[0].message.annotations`` with
  ``type: "url_citation"`` objects. These are mapped onto the same
  ``{"title", "url"}`` list Gemini's grounding metadata produces, so
  ``analyze_saas`` and the dashboard need no provider branch.

Errors raise :class:`OpenRouterError` with a ``kind`` the API layer maps onto
an HTTP status. Messages are built from the status and a short, key-scrubbed
excerpt of the body: the key is a secret and never appears in a response, a
log line or an exception.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

BASE_URL = (os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")

DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_TIMEOUT = 240.0  # research + a 3-script batch can take a while on slower models
DEFAULT_MAX_TOKENS = 8192
_MODELS_TTL = 600.0  # the catalogue / pricing shifts; refetch at most every 10 min

_MIN_CONTEXT = 32_768
_MIN_OUTPUT = 8_192


# ───────────────────────────────────────────────────────────────────────────
# Curated defaults
# ───────────────────────────────────────────────────────────────────────────
# Every entry below was verified live against GET /models with
# ``response_format`` (and ``structured_outputs``) advertised, at least 128k
# context and enough max_completion_tokens for the script batch — i.e. each one
# *advertises* support for all three SaaSShorts operations (research, analysis,
# scripts). ``name`` and the prices are what the API returned at freeze time;
# ``list_models`` overrides both whenever a key is available, and
# ``catalogue`` marks these recommended.
#
# Advertising JSON support is not the same as running the pipeline end to end, so
# the two are tracked separately: catalogue metadata gets a model *into*
# ``CURATED_MODELS``; only a real end-to-end run gets it into ``VERIFIED_MODELS``
# (see ``verify_openrouter_models.py``). ``catalogue`` surfaces the difference as
# ``verified`` so the picker never passes a metadata check off as proof.
#
# ``web_search_price`` is the per-search-call cost, reported separately from the
# token prices because OpenRouter bills it per search, not per token:
# * models whose provider has native search (Google, OpenAI, Anthropic) carry the
#   passthrough price the API returned in ``pricing.web_search``;
# * the rest fall back to the Exa engine, which OpenRouter documents at
#   $0.007/request (auto mode) — ``:online`` is model-agnostic, so a model
#   without a native price still does the research step.
CURATED_MODELS: List[Dict[str, Any]] = [
    {"id": "google/gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash Lite",
     "context_length": 1_048_576, "max_completion_tokens": 65_536,
     "prompt_price": 0.00000025, "completion_price": 0.0000015, "web_search_price": 0.014},
    {"id": "google/gemini-3.7-flash", "name": "Gemini 3.7 Flash",
     "context_length": 1_048_576, "max_completion_tokens": 65_536,
     "prompt_price": 0.00000075, "completion_price": 0.00000375, "web_search_price": 0.014},
    {"id": "anthropic/claude-haiku-5.5", "name": "Claude Haiku 5.5",
     "context_length": 1_000_000, "max_completion_tokens": 128_000,
     "prompt_price": 0.0000001, "completion_price": 0.0000005, "web_search_price": 0.01},
    {"id": "openai/gpt-4.1-mini", "name": "GPT-4.1 mini",
     "context_length": 1_047_576, "max_completion_tokens": 32_768,
     "prompt_price": 0.0000004, "completion_price": 0.0000016, "web_search_price": 0.01},
    {"id": "openai/gpt-4.1", "name": "GPT-4.1",
     "context_length": 1_047_576, "max_completion_tokens": 32_768,
     "prompt_price": 0.000002, "completion_price": 0.000008, "web_search_price": 0.01},
    {"id": "openai/gpt-5.1", "name": "GPT-5.1",
     "context_length": 400_000, "max_completion_tokens": 128_000,
     "prompt_price": 0.00000125, "completion_price": 0.00001, "web_search_price": 0.01},
    {"id": "openai/gpt-5.2", "name": "GPT-5.2",
     "context_length": 400_000, "max_completion_tokens": 128_000,
     "prompt_price": 0.00000175, "completion_price": 0.000014, "web_search_price": 0.01},
    {"id": "openai/gpt-oss-120b", "name": "gpt-oss-120b",
     "context_length": 131_072, "max_completion_tokens": 117_964,
     "prompt_price": 0.000000037, "completion_price": 0.00000017,
     # No native search; :online routes this to the Exa engine ($0.007/request).
     "web_search_price": 0.007},
    {"id": "openai/gpt-oss-20b", "name": "gpt-oss-20b",
     "context_length": 131_072, "max_completion_tokens": 32_768,
     "prompt_price": 0.000000018, "completion_price": 0.00000009,
     "web_search_price": 0.007},  # Exa engine, as above
]


# Models proven against the *complete* SaaSShorts pipeline — research with
# citations, marketing-analysis JSON, UGC-script JSON arrays — on a real chat
# completion, not just their catalogue metadata.
#
# This starts empty on purpose. A model can advertise ``response_format`` and
# still emit a malformed array in practice, so membership is only ever granted by
# an actual run of ``verify_openrouter_models.py`` against a key that was
# explicitly configured and authorised for the test spend. Until that exists, the
# picker still offers the curated models (they are genuinely metadata-qualified)
# but every one reports ``verified=False`` — metadata is evidence of capability,
# not of compatibility, and the two must not be conflated.
VERIFIED_MODELS: Set[str] = set()


class OpenRouterError(RuntimeError):
    """An OpenRouter request failed. ``message`` never contains the API key."""

    def __init__(self, message: str, *, status: Optional[int] = None, kind: str = "server"):
        super().__init__(message)
        self.status = status
        self.kind = kind  # auth | not_found | rate_limit | empty | server


# In-process cache for the live catalogue (per-process; tests clear it).
_MODELS_CACHE: Dict[str, Any] = {"at": 0.0, "models": None, "error": None}



def _scrub(text: str, key: Optional[str]) -> str:
    """Remove the key from anything that may be shown to the user or logged."""
    if not text:
        return ""
    out = str(text)
    if key:
        out = out.replace(key, "[REDACTED]")
    # An OpenRouter key is ``sk-or-v1-<64 hex>``; catch a partially-pasted one too.
    return re.sub(r"sk-or-[A-Za-z0-9\-_]{8,}", "[REDACTED]", out)


def _timeout() -> float:
    try:
        return float(os.environ.get("OPENROUTER_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    timeout = kwargs.pop("timeout", None) or _timeout()
    return httpx.Client(timeout=timeout, **kwargs)


def _headers(api_key: str) -> dict:
    # X-Title is an optional ranking attribute OpenRouter shows in the user's
    # own dashboard; it carries no secret.
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "SnipFlow SaaSShorts",
    }


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_object" in body or "json_schema" in body


def _error(resp: httpx.Response, *, model: str, key: str) -> OpenRouterError:
    excerpt = _scrub(resp.text, key)[:200]
    if resp.status_code in (401, 403):
        return OpenRouterError(
            "OpenRouter rejected the API key (401/403). Check the OpenRouter key in AI Settings.",
            status=resp.status_code, kind="auth")
    if resp.status_code == 404:
        return OpenRouterError(
            f"OpenRouter has no model {model!r} (404). Pick a different model in AI Settings.",
            status=404, kind="not_found")
    if resp.status_code == 429:
        return OpenRouterError(
            "OpenRouter rate limit or credit quota exceeded (429). Wait a moment, add credits, "
            "or choose a cheaper/free model in AI Settings.",
            status=429, kind="rate_limit")
    return OpenRouterError(
        f"OpenRouter returned {resp.status_code}: {excerpt}",
        status=resp.status_code, kind="server")


def _parse_content(content: Any) -> str:
    """Flatten an OpenAI-style ``content`` (string or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def _extract_json(text: str, expect: str) -> Any:
    """Same defensive parse the Gemini path uses: strip fences, slice to the JSON."""
    if not text or not text.strip():
        raise OpenRouterError("OpenRouter returned an empty response body.", kind="empty")
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    open_ch, close_ch = ("{", "}") if expect == "object" else ("[", "]")
    start, end = t.find(open_ch), t.rfind(close_ch)
    if start != -1 and end != -1 and end > start:
        t = t[start:end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        raise OpenRouterError(f"Failed to parse OpenRouter JSON: {e}", kind="empty")


def _extract_sources(message: Dict[str, Any]) -> List[Dict[str, str]]:
    """Map OpenRouter citations onto Gemini's ``{"title", "url"}`` grounding shape.

    Online models annotate the message with ``url_citation`` objects; the older
    shape was a bare ``citations`` list of URLs. Title falls back to the URL.
    """
    sources: List[Dict[str, str]] = []
    seen = set()

    def _add(title: Optional[str], url: Optional[str]) -> None:
        if not url or url in seen:
            return
        seen.add(url)
        sources.append({"title": (title or url).strip() or url, "url": url})

    for ann in message.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        if ann.get("type") == "url_citation":
            cite = ann.get("url_citation") or {}
            if isinstance(cite, dict):
                _add(cite.get("title"), cite.get("url"))
        elif ann.get("type") == "source" and isinstance(ann.get("source"), dict):
            # Some providers wrap the same data in a ``source`` annotation.
            src = ann["source"]
            _add(src.get("title"), src.get("url"))
    for url in message.get("citations") or []:
        if isinstance(url, str):
            _add(None, url)
    return sources


def chat_json(
    prompt: str,
    api_key: str,
    model: str,
    *,
    expect: str = "object",
    web_search: bool = False,
    system: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.2,
    timeout: Optional[float] = None,
) -> Tuple[Any, List[Dict[str, str]], Dict[str, Any]]:
    """One chat completion that must come back as JSON.

    Returns ``(parsed, sources, cost)`` where ``sources`` is the
    ``[{"title", "url"}]`` list (empty unless ``web_search``) and ``cost`` has
    the token/pricing fields the pipeline reports. Raises
    :class:`OpenRouterError` on any HTTP failure, an empty body or unparseable
    JSON — it never silently falls back to another provider.
    """
    if not api_key:
        raise OpenRouterError("Missing OpenRouter API key.", status=401, kind="auth")
    if not model:
        raise OpenRouterError("No OpenRouter model selected.", kind="not_found")

    # Web search is a per-request plugin: ``:online`` on the model id.
    online_model = model if model.endswith(":online") else (f"{model}:online" if web_search else model)

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    base_body: Dict[str, Any] = {
        "model": online_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        "stream": False,
    }

    url = f"{BASE_URL}/chat/completions"
    last_excerpt: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    with _client(timeout=timeout) as client:
        # json_object first (broadly supported), then no response_format for a
        # model that rejects the field entirely.
        for fmt in ({"type": "json_object"}, None):
            body = dict(base_body)
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=_headers(api_key))
            if resp.status_code >= 400:
                if fmt is not None and _is_format_rejection(resp):
                    last_excerpt = _scrub(resp.text, api_key)[:200]
                    continue
                raise _error(resp, model=online_model, key=api_key)
            data = resp.json()
            break
        else:
            raise OpenRouterError(
                f"OpenRouter rejected json_object response_format: {last_excerpt}", kind="server")

    if data is None:
        raise OpenRouterError("OpenRouter returned no data.", kind="empty")

    choices = data.get("choices") or []
    message: Dict[str, Any] = (choices[0].get("message") or {}) if choices else {}
    parsed = _extract_json(_parse_content(message.get("content")), expect)
    sources = _extract_sources(message) if web_search else []

    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    pricing = _pricing_for(online_model.split(":online")[0])
    cost = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "thinking_tokens": 0,
        "input_cost": round(prompt_tokens * pricing["prompt"], 6),
        "output_cost": round(completion_tokens * pricing["completion"], 6),
        "total_cost": round(prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"], 6),
        "model": online_model,
        "price_estimated": True,
        "local": False,
        # Web search is billed per search call, not per token — reported so the
        # UI can show the two cost lines separately.
        "web_search_price_per_call": pricing.get("web_search"),
    }
    return parsed, sources, cost


# ───────────────────────────────────────────────────────────────────────────
# Catalogue
# ───────────────────────────────────────────────────────────────────────────

def _pricing_for(model_id: str) -> Dict[str, Any]:
    """Per-token USD prices for ``model_id``: live cache, then curated, else zero."""
    for m in CURATED_MODELS:
        if m["id"] == model_id:
            return {"prompt": m["prompt_price"], "completion": m["completion_price"],
                    "web_search": m["web_search_price"]}
    for m in (_MODELS_CACHE.get("models") or []):
        if m.get("id") == model_id:
            return {"prompt": m.get("prompt_price") or 0.0,
                    "completion": m.get("completion_price") or 0.0,
                    "web_search": m.get("web_search_price")}
    return {"prompt": 0.0, "completion": 0.0, "web_search": None}


def _model_from_api(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sp = m.get("supported_parameters") or []
    if "response_format" not in sp:
        return None
    ctx = int(m.get("context_length") or 0)
    maxout = int((m.get("top_provider") or {}).get("max_completion_tokens") or 0)
    if ctx < _MIN_CONTEXT or maxout < _MIN_OUTPUT:
        return None
    p = m.get("pricing") or {}
    prompt_price = float(p.get("prompt") or 0.0)
    completion_price = float(p.get("completion") or 0.0)
    web_search_price = p.get("web_search")
    return {
        "id": m["id"],
        "name": m.get("name") or m["id"],
        "context_length": ctx,
        "max_completion_tokens": maxout,
        "free": prompt_price == 0.0 and completion_price == 0.0,
        "prompt_price": prompt_price,
        "completion_price": completion_price,
        "web_search_price": float(web_search_price) if web_search_price is not None else None,
        "supports_web_search": web_search_price is not None,
        "structured_outputs": "structured_outputs" in sp,
    }


def list_models(api_key: str, *, force: bool = False) -> List[Dict[str, Any]]:
    """Live catalogue (cached for ``_MODELS_TTL``): models that can do JSON.

    Raises :class:`OpenRouterError` on auth/network failure so the caller can
    surface "bad key" instead of an empty dropdown.
    """
    now = time.time()
    if not force and _MODELS_CACHE["models"] is not None and now - _MODELS_CACHE["at"] < _MODELS_TTL:
        return _MODELS_CACHE["models"]

    if not api_key:
        raise OpenRouterError("Missing OpenRouter API key.", status=401, kind="auth")

    with _client() as client:
        resp = client.get(f"{BASE_URL}/models", headers=_headers(api_key))
    if resp.status_code >= 400:
        raise _error(resp, model="models", key=api_key)
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        raise OpenRouterError("OpenRouter /models returned no JSON.", kind="empty")

    models: List[Dict[str, Any]] = []
    for m in payload.get("data") or []:
        try:
            entry = _model_from_api(m)
        except (TypeError, ValueError):
            entry = None
        if entry:
            models.append(entry)
    recommended = {m["id"] for m in CURATED_MODELS}
    for entry in models:
        entry["recommended"] = entry["id"] in recommended
    models.sort(key=lambda e: (not e["recommended"], not e["free"], e["completion_price"]))
    _MODELS_CACHE.update(at=now, models=models, error=None)
    return models


def catalogue(api_key: Optional[str] = None) -> Dict[str, Any]:
    """The model picker's data: curated defaults, enriched by the live list.

    Never empty — the curated set ships in-repo — so the picker works before the
    first successful live fetch. ``live`` tells the UI whether prices/availability
    are current or the frozen curated snapshot.
    """
    curated = []
    for m in CURATED_MODELS:
        curated.append({
            "id": m["id"], "name": m["name"],
            "context_length": m["context_length"],
            "max_completion_tokens": m["max_completion_tokens"],
            "free": m["prompt_price"] == 0.0 and m["completion_price"] == 0.0,
            "prompt_price": m["prompt_price"], "completion_price": m["completion_price"],
            "web_search_price": m["web_search_price"],
            "supports_web_search": m["web_search_price"] is not None,
            "structured_outputs": True,
            "recommended": True,
            "verified": m["id"] in VERIFIED_MODELS,
        })

    if not api_key:
        return {"live": False, "models": curated, "error": None}

    try:
        live = list_models(api_key)
    except OpenRouterError as e:
        return {"live": False, "models": curated, "error": str(e)}

    by_id = {m["id"]: m for m in live}
    merged: List[Dict[str, Any]] = []
    seen = set()
    # Curated first (in curated order), with live availability/prices when present.
    for entry in curated:
        live_entry = by_id.get(entry["id"])
        merged_entry = {**entry, **(live_entry or {})}
        # A live entry without ``pricing.web_search`` (no native provider search)
        # still runs ``:online`` through the Exa engine, so the curated fallback
        # price is kept rather than wiped out by a null.
        if merged_entry.get("web_search_price") is None and entry.get("web_search_price") is not None:
            merged_entry["web_search_price"] = entry["web_search_price"]
            merged_entry["supports_web_search"] = True
        merged_entry["recommended"] = True
        merged.append(merged_entry)
        seen.add(entry["id"])
    # Then any other live model that qualifies.
    for entry in live:
        if entry["id"] not in seen:
            merged.append({**entry, "recommended": False,
                           "verified": entry["id"] in VERIFIED_MODELS})
            seen.add(entry["id"])
    return {"live": True, "models": merged, "error": None}
