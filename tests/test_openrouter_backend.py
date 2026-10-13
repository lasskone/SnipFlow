"""The OpenRouter BYOK backend (openrouter_backend.py) + its saasshorts wiring.

OpenRouter is an optional second provider for the three SaaSShorts AI steps
(web research, marketing analysis, UGC scripts). These tests pin the contract
the rest of the app relies on: the same (parsed, sources, cost) shapes the
Gemini path produces, the ``:online`` web-research mode, the response_format
fallback ladder, error kinds the API layer maps to HTTP statuses, and that the
default Gemini path is untouched by the new provider argument.
"""
import json
import types

import httpx
import pytest

import openrouter_backend
import saasshorts


KEY = "sk-or-v1-" + "a" * 64


@pytest.fixture(autouse=True)
def _clear_models_cache():
    """The catalogue cache is module-global; never let one test warm it for another."""
    openrouter_backend._MODELS_CACHE.update(at=0.0, models=None, error=None)
    yield
    openrouter_backend._MODELS_CACHE.update(at=0.0, models=None, error=None)


def _serve(handler, monkeypatch):
    """Route openrouter_backend's httpx client through an in-process handler."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(openrouter_backend, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _completion(payload, usage=None, message_extra=None):
    message = {"role": "assistant", "content": json.dumps(payload)}
    if message_extra:
        message.update(message_extra)
    return httpx.Response(200, json={
        "choices": [{"message": message}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 50},
    })


# --- chat_json: the contract the three AI steps depend on -------------------

def test_parses_object_and_reports_cost(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return _completion({"product_name": "Acme", "pain_points": []})

    _serve(handler, monkeypatch)
    parsed, sources, cost = openrouter_backend.chat_json(
        "research this", KEY, "openai/gpt-4.1-mini")

    assert seen["url"].endswith("/chat/completions")
    assert seen["body"]["model"] == "openai/gpt-4.1-mini"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["stream"] is False
    assert seen["auth"] == f"Bearer {KEY}"
    assert parsed["product_name"] == "Acme"
    # Sources are only present for the web-research step.
    assert sources == []
    # Cost shape mirrors the Gemini stage, priced off the curated gpt-4.1-mini entry.
    assert cost["input_tokens"] == 100 and cost["output_tokens"] == 50
    assert cost["model"] == "openai/gpt-4.1-mini"
    assert cost["price_estimated"] is True and cost["local"] is False
    assert cost["input_cost"] == round(100 * 0.0000004, 6)
    assert cost["output_cost"] == round(50 * 0.0000016, 6)
    assert cost["web_search_price_per_call"] == 0.01



# --- web research: :online + citations -------------------------------------

def test_web_search_appends_online_suffix_and_maps_citations(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"product_name": "Acme"}, message_extra={
            "annotations": [
                {"type": "url_citation", "url_citation": {"title": "G2 review", "url": "https://g2.com/x"}},
                # A missing title falls back to the URL.
                {"type": "url_citation", "url_citation": {"title": None, "url": "https://reddit.com/r/x"}},
                # A repeated URL is de-duplicated.
                {"type": "url_citation", "url_citation": {"url": "https://g2.com/x"}},
                # The older "source"-wrapped shape is still understood.
                {"type": "source", "source": {"title": "HN", "url": "https://news.ycombinator.com/x"}},
            ],
        })

    _serve(handler, monkeypatch)
    _parsed, sources, cost = openrouter_backend.chat_json(
        "research", KEY, "openai/gpt-4.1-mini", web_search=True)

    assert seen["body"]["model"] == "openai/gpt-4.1-mini:online"
    assert cost["model"] == "openai/gpt-4.1-mini:online"
    assert sources == [
        {"title": "G2 review", "url": "https://g2.com/x"},
        {"title": "https://reddit.com/r/x", "url": "https://reddit.com/r/x"},
        {"title": "HN", "url": "https://news.ycombinator.com/x"},
    ]


def test_online_suffix_is_not_doubled(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"ok": True})

    _serve(handler, monkeypatch)
    openrouter_backend.chat_json("p", KEY, "some/model:online", web_search=True)
    assert seen["body"]["model"] == "some/model:online"


def test_citations_ignored_without_web_search(monkeypatch):
    def handler(request):
        return _completion({}, message_extra={
            "annotations": [{"type": "url_citation", "url_citation": {"title": "t", "url": "https://x.com"}}]})

    _serve(handler, monkeypatch)
    _parsed, sources, _cost = openrouter_backend.chat_json("p", KEY, "openai/gpt-4.1-mini")
    assert sources == []


def test_legacy_bare_citations_list(monkeypatch):
    def handler(request):
        return _completion({}, message_extra={"citations": ["https://legacy.example.com/a"]})

    _serve(handler, monkeypatch)
    _parsed, sources, _cost = openrouter_backend.chat_json(
        "p", KEY, "openai/gpt-4.1-mini", web_search=True)
    assert sources == [{"title": "https://legacy.example.com/a", "url": "https://legacy.example.com/a"}]


# --- JSON parsing: same defensive ladder as the Gemini path -----------------

def test_parses_array_and_strips_fences(monkeypatch):
    def handler(request):
        content = "Sure!\n```json\n[{\"title\": \"a\"}, {\"title\": \"b\"}]\n```\n"
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        })

    _serve(handler, monkeypatch)
    parsed, _sources, _cost = openrouter_backend.chat_json(
        "scripts", KEY, "openai/gpt-4.1-mini", expect="array")
    assert [s["title"] for s in parsed] == ["a", "b"]


def test_content_may_arrive_as_a_part_list(monkeypatch):
    def handler(request):
        content = [{"type": "text", "text": "{\"a\": 1}"}]
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {},
        })

    _serve(handler, monkeypatch)
    parsed, _s, _cost = openrouter_backend.chat_json("p", KEY, "openai/gpt-4.1-mini")
    assert parsed == {"a": 1}


def test_falls_back_when_json_object_is_rejected(monkeypatch):
    formats = []

    def handler(request):
        fmt = (json.loads(request.content).get("response_format") or {}).get("type")
        formats.append(fmt)
        if fmt == "json_object":
            return httpx.Response(
                400, json={"error": {"message": "response_format json_object unsupported"}})
        return _completion({"ok": True})

    _serve(handler, monkeypatch)
    parsed, _s, _cost = openrouter_backend.chat_json("p", KEY, "some/model")
    assert formats == ["json_object", None]
    assert parsed == {"ok": True}


def test_non_format_400_is_not_treated_as_a_format_rejection(monkeypatch):
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "content too long"}})

    _serve(handler, monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.chat_json("p", KEY, "some/model")


# --- errors: kinds the API layer maps onto HTTP statuses --------------------

def test_missing_key_raises_auth():
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.chat_json("p", "", "openai/gpt-4.1-mini")
    assert ei.value.kind == "auth"
    assert ei.value.status == 401


def test_missing_model_raises_not_found():
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.chat_json("p", KEY, "")
    assert ei.value.kind == "not_found"


def test_invalid_model_is_not_found(monkeypatch):
    def handler(request):
        return httpx.Response(404, json={"error": {"message": "No such model: nope/nonsense"}})

    _serve(handler, monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.chat_json("p", KEY, "nope/nonsense")
    assert ei.value.kind == "not_found"
    assert "nope/nonsense" in str(ei.value)


def test_bad_key_is_auth_and_never_leaks_the_key(monkeypatch):
    def handler(request):
        return httpx.Response(401, content=f"unauthorized for key {KEY}")

    _serve(handler, monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.chat_json("p", KEY, "openai/gpt-4.1-mini")

    assert ei.value.kind == "auth"
    # The key is a secret: it must never appear in a message a user might see.
    assert KEY not in str(ei.value)


def test_server_error_body_is_scrubbed_of_the_key(monkeypatch):
    def handler(request):
        return httpx.Response(500, content=f"internal error for key {KEY}")

    _serve(handler, monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.chat_json("p", KEY, "openai/gpt-4.1-mini")

    # The server path embeds the response body, so this is where a leak would show up.
    assert ei.value.kind == "server"
    assert KEY not in str(ei.value)
    assert "[REDACTED]" in str(ei.value)


def test_rate_limit_kind(monkeypatch):
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "rate limit"}})

    _serve(handler, monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.chat_json("p", KEY, "openai/gpt-4.1-mini")
    assert ei.value.kind == "rate_limit"


def test_server_error_kind(monkeypatch):
    def handler(request):
        return httpx.Response(500, content="boom")

    _serve(handler, monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.chat_json("p", KEY, "openai/gpt-4.1-mini")
    assert ei.value.kind == "server"
    assert "500" in str(ei.value)


def test_empty_body(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    _serve(handler, monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.chat_json("p", KEY, "openai/gpt-4.1-mini")
    assert ei.value.kind == "empty"


def test_unparseable_json(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json at all"}}]})

    _serve(handler, monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.chat_json("p", KEY, "openai/gpt-4.1-mini")
    assert ei.value.kind == "empty"


# --- catalogue: the model picker's data -------------------------------------

def _models_payload():
    """A live /models response with one of each kind the filter must handle."""
    return httpx.Response(200, json={"data": [
        {"id": "openai/gpt-4.1-mini", "name": "GPT-4.1 mini", "context_length": 1047576,
         "top_provider": {"max_completion_tokens": 32768},
         "supported_parameters": ["tools", "response_format", "structured_outputs"],
         "pricing": {"prompt": "0.0000004", "completion": "0.0000016", "web_search": "0.01"}},
        # Too little context — filtered out.
        {"id": "weird/tiny-context", "name": "Tiny", "context_length": 8192,
         "top_provider": {"max_completion_tokens": 32768},
         "supported_parameters": ["response_format"],
         "pricing": {"prompt": "0", "completion": "0"}},
        # No response_format — filtered out.
        {"id": "no/json-format", "name": "NoJson", "context_length": 200000,
         "top_provider": {"max_completion_tokens": 32768},
         "supported_parameters": ["tools"],
         "pricing": {"prompt": "0", "completion": "0"}},
        # Free and qualifying — kept, and surfaces in the picker.
        {"id": "free/qualifying", "name": "Freebie", "context_length": 131072,
         "top_provider": {"max_completion_tokens": 32768},
         "supported_parameters": ["response_format", "structured_outputs"],
         "pricing": {"prompt": "0", "completion": "0"}},
    ]})


def test_list_models_filters_to_json_capable(monkeypatch):
    _serve(lambda r: _models_payload(), monkeypatch)
    models = openrouter_backend.list_models(KEY)

    # Recommended (curated) first, then free, then by price.
    assert [m["id"] for m in models] == ["openai/gpt-4.1-mini", "free/qualifying"]
    assert models[0]["recommended"] is True
    assert models[0]["structured_outputs"] is True
    assert models[0]["supports_web_search"] is True
    assert models[1]["recommended"] is False
    assert models[1]["free"] is True


def test_list_models_caches_within_ttl(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return _models_payload()

    _serve(handler, monkeypatch)
    openrouter_backend.list_models(KEY)
    openrouter_backend.list_models(KEY)
    assert len(calls) == 1
    openrouter_backend.list_models(KEY, force=True)
    assert len(calls) == 2


def test_list_models_bad_key(monkeypatch):
    _serve(lambda r: httpx.Response(401, content="nope"), monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        openrouter_backend.list_models(KEY)
    assert ei.value.kind == "auth"


def test_catalogue_without_key_is_the_curated_set():
    out = openrouter_backend.catalogue(None)
    assert out["live"] is False
    assert out["error"] is None
    assert len(out["models"]) == len(openrouter_backend.CURATED_MODELS)
    assert all(m["recommended"] for m in out["models"])


def test_catalogue_never_infers_verified_from_metadata(monkeypatch):
    """``verified`` is end-to-end evidence, never derived from catalogue metadata.

    The curated set is *chosen* from metadata (JSON mode advertised, enough
    context and output budget) — necessary, but the task is explicit that it is
    not proof of end-to-end compatibility. So ``verified`` tracks a separate set,
    populated only by a real run of ``verify_openrouter_models.py``, and is
    reported independently of ``recommended``. With no evidence recorded every
    curated model is ``recommended=True, verified=False``; the two must never be
    derived from one another.
    """
    monkeypatch.setattr(openrouter_backend, "VERIFIED_MODELS", set())
    out = openrouter_backend.catalogue(None)

    assert out["models"]
    for m in out["models"]:
        # Recommended (metadata-qualified) but not verified (not run end to end).
        assert m["recommended"] is True
        assert m["verified"] is False

    # Only recorded evidence flips the flag.
    proven = openrouter_backend.CURATED_MODELS[0]["id"]
    monkeypatch.setattr(openrouter_backend, "VERIFIED_MODELS", {proven})
    by_id = {m["id"]: m for m in openrouter_backend.catalogue(None)["models"]}
    assert by_id[proven]["verified"] is True
    assert sum(m["verified"] for m in by_id.values()) == 1


def test_live_only_models_carry_the_verified_flag(monkeypatch):
    """A live-only model reports ``verified`` from the same evidence set."""
    _serve(lambda r: _models_payload(), monkeypatch)
    monkeypatch.setattr(openrouter_backend, "VERIFIED_MODELS", {"free/qualifying"})

    by_id = {m["id"]: m for m in openrouter_backend.catalogue(KEY)["models"]}

    assert by_id["free/qualifying"]["verified"] is True
    # Curated models are unaffected: membership is the only route to verified.
    assert by_id[openrouter_backend.CURATED_MODELS[0]["id"]]["verified"] is False


def test_catalogue_merges_curated_with_live(monkeypatch):
    _serve(lambda r: _models_payload(), monkeypatch)
    out = openrouter_backend.catalogue(KEY)

    assert out["live"] is True
    ids = [m["id"] for m in out["models"]]
    # Curated entries come first, in curated order, then live-only qualifiers.
    assert ids[0] == openrouter_backend.CURATED_MODELS[0]["id"]
    assert "free/qualifying" in ids
    assert "no/json-format" not in ids
    assert "weird/tiny-context" not in ids
    curated_n = len(openrouter_backend.CURATED_MODELS)
    assert all(m["recommended"] for m in out["models"][:curated_n])


def test_catalogue_survives_a_bad_key(monkeypatch):
    """A rejected key must not break the picker — it degrades to the curated list."""
    _serve(lambda r: httpx.Response(401, content="nope"), monkeypatch)
    out = openrouter_backend.catalogue(KEY)

    assert out["live"] is False
    assert "401" in out["error"]
    assert len(out["models"]) == len(openrouter_backend.CURATED_MODELS)


def test_curated_models_meet_the_json_output_budget():
    """Every curated id carries the token/output budget the JSON steps need.

    These are the metadata invariants a curated entry is *chosen* on — context
    for the research step, max-completion for the script batch, and populated
    prices for the picker. They say the model advertises the capability; they do
    not by themselves prove it runs the pipeline (see ``VERIFIED_MODELS``).
    """
    for m in openrouter_backend.CURATED_MODELS:
        assert m["context_length"] >= openrouter_backend._MIN_CONTEXT, m["id"]
        assert m["max_completion_tokens"] >= openrouter_backend._MIN_OUTPUT, m["id"]
        assert m["prompt_price"] is not None and m["completion_price"] is not None


def test_curated_models_all_carry_a_web_search_price():
    """A curated id needs a per-search price to run the research step.

    Research is routed through ``:online``, billed per search separately from
    tokens, so every curated entry must carry ``web_search_price`` alongside the
    JSON/output budget. Metadata invariants only — not end-to-end proof.
    """
    for m in openrouter_backend.CURATED_MODELS:
        assert m["web_search_price"] is not None, m["id"]
        assert m["web_search_price"] > 0, m["id"]
        assert m["max_completion_tokens"] >= openrouter_backend._MIN_OUTPUT, m["id"]


def test_catalogue_keeps_the_web_search_fallback_price(monkeypatch):
    """A live entry with no native ``pricing.web_search`` keeps the curated fallback.

    ``:online`` is model-agnostic: a model whose provider has no native search
    still runs the research step through the Exa engine. The live API reports no
    search price for it, and that null must not erase the curated per-search cost.
    """
    payload = httpx.Response(200, json={"data": [
        {"id": "openai/gpt-oss-20b", "name": "gpt-oss-20b", "context_length": 131072,
         "top_provider": {"max_completion_tokens": 32768},
         "supported_parameters": ["response_format", "structured_outputs"],
         # No web_search key at all — the Exa case.
         "pricing": {"prompt": "0.000000018", "completion": "0.00000009"}},
    ]})
    _serve(lambda r: payload, monkeypatch)
    out = openrouter_backend.catalogue(KEY)

    entry = next(m for m in out["models"] if m["id"] == "openai/gpt-oss-20b")
    assert out["live"] is True
    assert entry["recommended"] is True
    assert entry["supports_web_search"] is True
    assert entry["web_search_price"] == 0.007


def test_web_search_cost_is_reported_separately_from_tokens(monkeypatch):
    """The cost line keeps per-search billing apart from per-token billing."""
    _serve(lambda r: _completion({"product_name": "Acme"}), monkeypatch)
    _parsed, _sources, cost = openrouter_backend.chat_json(
        "research this", KEY, "openai/gpt-oss-20b", web_search=True)

    # gpt-oss-20b is not natively priced for search, so the curated Exa rate applies.
    assert cost["web_search_price_per_call"] == 0.007
    assert cost["input_cost"] == round(100 * 0.000000018, 6)
    assert cost["output_cost"] == round(50 * 0.00000009, 6)
    # total_cost is the rounded sum of the raw token costs (not the sum of the
    # rounded parts), and web search is billed on top of it, per call.
    assert cost["total_cost"] == round(100 * 0.000000018 + 50 * 0.00000009, 6)
    assert cost["total_cost"] > 0


def test_no_silent_fallback_to_another_provider(monkeypatch):
    """A failed OpenRouter call must surface, not quietly retry on Gemini."""

    def handler(request):
        return httpx.Response(404, json={"error": {"message": "no model"}})

    _serve(handler, monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError):
        openrouter_backend.chat_json("p", KEY, "nope/nonsense", web_search=True)



# --- saasshorts routing: Gemini untouched, OpenRouter wired in --------------

class _FakeGenai:
    """Stands in for google.genai so the Gemini path can be exercised offline."""

    def __init__(self):
        self.calls = []
        self.response = _GenaiResponse('{"product_name": "Acme"}')

    def Client(self, api_key):
        self.calls.append(("client", api_key))
        return self

    @property
    def models(self):
        return self

    def generate_content(self, model, contents, config=None):
        self.calls.append(("generate", model, contents, config))
        return self.response


class _GenaiResponse:
    def __init__(self, text, grounding_chunks=None):
        self.text = text
        metadata = None
        if grounding_chunks:
            metadata = types.SimpleNamespace(
                grounding_chunks=grounding_chunks, web_search_queries=["q"])
        self.candidates = [types.SimpleNamespace(grounding_metadata=metadata)]


@pytest.fixture
def fake_genai(monkeypatch):
    genai = pytest.importorskip("google.genai")
    fake = _FakeGenai()
    # saasshorts resolves `from google import genai` at call time, so patching
    # the attribute on the imported module is enough.
    monkeypatch.setattr(genai, "Client", fake.Client)
    return fake


def _scraped():
    return {
        "url": "https://acme.io",
        "title": "Acme",
        "meta_description": "",
        "headings": [],
        "main_content": "",
        "additional_pages": "",
    }


def test_gemini_path_is_the_default_and_untouched(fake_genai, monkeypatch):
    def handler(request):
        raise AssertionError("OpenRouter must not be called on the default Gemini path")

    _serve(handler, monkeypatch)
    out = saasshorts.analyze_saas(_scraped(), "gkey")

    assert out == {"product_name": "Acme"}
    assert fake_genai.calls[0] == ("client", "gkey")
    assert fake_genini_call(fake_genai, "model") == saasshorts.GEMINI_MODEL


def fake_genini_call(fake, key):
    gen = [c for c in fake.calls if c[0] == "generate"]
    return gen[-1][1] if gen else None


def test_openrouter_path_never_builds_a_gemini_client(fake_genai, monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"product_name": "Acme", "pain_points": [{"pain": "p"}]})

    _serve(handler, monkeypatch)
    out = saasshorts.analyze_saas(
        _scraped(), "gkey",
        llm_provider="openrouter", llm_model="openai/gpt-4.1-mini", openrouter_key=KEY)

    assert out["product_name"] == "Acme"
    assert fake_genai.calls == []
    assert seen["body"]["model"] == "openai/gpt-4.1-mini"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_openrouter_analysis_keeps_the_web_sources(fake_genai, monkeypatch):
    def handler(request):
        return _completion({"product_name": "Acme"})

    _serve(handler, monkeypatch)
    out = saasshorts.analyze_saas(
        _scraped(), "gkey", web_research={"grounding_sources": [{"title": "t", "url": "https://t.io"}]},
        llm_provider="openrouter", llm_model="openai/gpt-4.1-mini", openrouter_key=KEY)

    # Same key the Gemini path attaches, so the UI needs no provider branch.
    assert out["_web_sources"] == [{"title": "t", "url": "https://t.io"}]


def test_openrouter_research_maps_sources_and_fills_defaults(fake_genai, monkeypatch):
    def handler(request):
        # Web research must run online.
        assert json.loads(request.content)["model"].endswith(":online")
        return _completion({"what_it_does": "a thing"}, message_extra={
            "annotations": [{"type": "url_citation",
                             "url_citation": {"title": "Reddit", "url": "https://reddit.com/r/x"}}]})

    _serve(handler, monkeypatch)
    out = saasshorts.research_saas_online(
        "https://acme.io", "gkey",
        llm_provider="openrouter", llm_model="google/gemini-3.1-flash-lite", openrouter_key=KEY)

    assert out["grounding_sources"] == [{"title": "Reddit", "url": "https://reddit.com/r/x"}]
    assert out["product_name"] == "acme.io"
    assert out["website_url"] == "https://acme.io"
    assert fake_genai.calls == []


def test_openrouter_scripts(fake_genai, monkeypatch):
    def handler(request):
        return _completion([{"title": "a", "segments": []}])

    _serve(handler, monkeypatch)
    scripts = saasshorts.generate_scripts(
        {"product_name": "Acme"}, "gkey", 3, "ugc", "en", "female",
        llm_provider="openrouter", llm_model="openai/gpt-4.1-mini", openrouter_key=KEY)
    assert scripts == [{"title": "a", "segments": []}]


def test_openrouter_failure_propagates_without_falling_back(fake_genai, monkeypatch):
    _serve(lambda r: httpx.Response(404, json={"error": {"message": "no model"}}), monkeypatch)
    with pytest.raises(openrouter_backend.OpenRouterError) as ei:
        saasshorts.generate_scripts(
            {"product_name": "Acme"}, "gkey", 3, "ugc", "en", "female",
            llm_provider="openrouter", llm_model="nope/nonsense", openrouter_key=KEY)
    assert ei.value.kind == "not_found"
    assert fake_genai.calls == []  # never fell back to Gemini


def test_gemini_research_still_reads_grounding_metadata(fake_genai, monkeypatch):
    fake_genai.response = _GenaiResponse(
        '{"product_name": "Acme"}',
        grounding_chunks=[types.SimpleNamespace(
            web=types.SimpleNamespace(title="G2", uri="https://g2.com/x"))],
    )
    out = saasshorts.research_saas_online("https://acme.io", "gkey")
    assert out["grounding_sources"] == [{"title": "G2", "url": "https://g2.com/x"}]
