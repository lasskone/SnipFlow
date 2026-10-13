"""OpenRouter is strictly bring-your-own-key: the caller's key, and nothing else.

Task #2.1. The OpenRouter path is caller-paid, so there must be no server-side
default key it can silently fall back to. An earlier revision fell back to the
``OPENROUTER_API_KEY`` env var when the ``X-OpenRouter-Key`` header was absent —
which is exactly the wrong failure mode for a paid path: it would let one
self-hoster's pre-set key spend credits on another self-hoster's requests, and
would turn a "the user forgot to paste their key" bug into real money spent.

The contract these tests lock in:

* the resolve helper reads *only* the request header, never the environment;
* a missing key surfaces as a clear 400 on the endpoint, never a 500 and never a
  silent detour onto Gemini or a server-side OpenRouter key; and
* the model catalogue still returns the curated list when no key is present, so
  the dashboard can render the dropdown before the user has pasted anything.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import app, resolve_openrouter

client = TestClient(app)

BYOK_ENDPOINT = "/api/saasshorts/analyze"
MODELS_ENDPOINT = "/api/saasshorts/openrouter/models"


def _request(headers=None):
    """A minimal Starlette request, scoped the way FastAPI builds it."""
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(scope={"type": "http", "headers": raw})


def _resolve(headers=None):
    """Run the async resolve helper on a fresh request."""
    return asyncio.run(resolve_openrouter(_request(headers)))


class TestStrictByok:
    """The only OpenRouter key the backend will ever use is the caller's."""

    @pytest.fixture(autouse=True)
    def no_env_key(self, monkeypatch):
        # A stray key in the developer's shell must never leak into these tests,
        # and must never become a fallback the resolve helper can use.
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        yield

    def test_header_key_is_returned_verbatim(self):
        assert _resolve({"X-OpenRouter-Key": "sk-or-v1-test"}) == "sk-or-v1-test"

    def test_missing_key_resolves_to_none(self):
        assert _resolve() is None

    def test_env_var_is_not_used_as_a_fallback(self, monkeypatch):
        # The env var is set and the header is not. The old code returned the env
        # var here; strict BYOK must return None instead.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-should-not-be-used")

        assert _resolve() is None

    def test_empty_header_is_not_a_usable_key(self):
        # An explicitly empty header is not a key. resolve_openrouter returns it
        # verbatim, but it is falsy, so every consumer (the 400 guard, the model
        # catalogue) treats it exactly like a missing key.
        assert not _resolve({"X-OpenRouter-Key": ""})

    def test_analyze_without_key_is_a_clear_400(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-should-not-be-used")
        response = client.post(
            BYOK_ENDPOINT,
            json={
                "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "llm_provider": "openrouter",
                "llm_model": "google/gemini-3.1-flash-lite",
            },
        )

        # 400 = the caller was told what to fix. Not 500 (crash), not 200
        # (silent success on someone else's dime), and not a Gemini detour.
        assert response.status_code == 400
        assert "X-OpenRouter-Key" in response.json()["detail"]

    def test_models_endpoint_lists_curated_models_without_a_key(self, monkeypatch):
        # The dropdown needs its options before the user has pasted a key, so the
        # curated list is returned with live=false. A stray env var must not turn
        # this into a live (and possibly failing) catalogue probe.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-should-not-be-used")
        response = client.get(MODELS_ENDPOINT)

        assert response.status_code == 200
        body = response.json()
        assert body["live"] is False
        # The endpoint still returns every curated model so the picker can offer
        # them for selection; the curated set is metadata-qualified, so it is
        # selectable even without recorded end-to-end evidence — the UI labels
        # each such model "Not end-to-end tested" rather than verified (#2.3).
        assert len(body["models"]) == 9
        # The flag the UI partitions on is always present and, with no evidence
        # recorded, honest: nothing is claimed proven it cannot prove.
        assert all("verified" in m for m in body["models"])
        assert all(m["verified"] is False for m in body["models"])
        # No key means no probe was attempted, hence no key error to surface.
        assert not body["error"]
