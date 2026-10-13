#!/usr/bin/env python3
"""End-to-end verification for the curated OpenRouter SaaSShorts models.

``openrouter_backend.CURATED_MODELS`` is frozen from catalogue metadata — each
model *advertises* ``response_format``, enough context and enough output budget
to run the three SaaSShorts AI steps. Advertising support is not the same as
being compatible, though: a model can accept a ``json_object`` request and still
emit a malformed array, or answer ``:online`` research without usable citations.
The only real proof is running the pipeline.

So this script drives the *actual* pipeline functions (``saasshorts`` research →
analysis → scripts, the identical prompts and parsing the app uses) once per
curated model on deliberately tiny inputs, and asserts the three contracts the
dashboard depends on:

* research returns a JSON object whose ``grounding_sources`` is a list of
  ``{"title", "url"}`` citation objects;
* analysis returns a JSON object with the expected top-level keys; and
* scripts returns a JSON *array* of scene objects.

Models that pass all three are printed as a ready-to-paste ``VERIFIED_MODELS``
literal for ``openrouter_backend.py``.

It spends money, so it will not run without both:

* ``OPENROUTER_TEST_KEY`` in the environment (deliberately *not*
  ``OPENROUTER_API_KEY`` — that name would let a production key be spent by
  accident, which is exactly the failure mode strict BYOK exists to prevent); and
* an explicit confirmation on stdin, or ``--yes`` to skip it.

Usage::

    OPENROUTER_TEST_KEY=sk-or-v1-... python3 verify_openrouter_models.py [--yes]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openrouter_backend
import saasshorts

# Where the structured record of a run is written (see requirement 5: record
# pass/fail and the exact model ID for every candidate). Local artefact only —
# gitignored so a verification log is never committed to a public repo.
RESULTS_PATH = Path(__file__).with_name("openrouter_verification_results.json")

# A stable, tiny, public target. Research builds its prompt from the domain and
# lets ``:online`` do the searching; it does not scrape this page, so the choice
# only needs to resolve to something real for the search queries to be sane.
TEST_URL = "https://example.com"
NUM_SCRIPTS = 2  # the array contract is what matters, not the batch size

# Minimal but shape-complete scrape, so ``analyze_saas`` reads every field it
# interpolates instead of tripping on a missing key unrelated to the model tested.
FAKE_SCRAPE: Dict[str, Any] = {
    "url": TEST_URL,
    "title": "Example Domain",
    "meta_description": "Example domain for use in illustrative examples.",
    "headings": ["Example Domain"],
    "main_content": "This domain is for use in illustrative examples in documents.",
    "additional_pages": "",
}


def _short(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _estimate_cost_usd() -> float:
    """A generous upper bound on the spend, printed before anything is sent.

    Tokens are negligible at these input sizes; the dominant term is the per-call
    web-search price the research step pays once per model.
    """
    total = 0.0
    for m in openrouter_backend.CURATED_MODELS:
        total += (m.get("web_search_price") or 0.0)
        # ~2k prompt + ~2k completion tokens per model, at the frozen prices.
        total += 2000 * m["prompt_price"] + 2000 * m["completion_price"]
    return total


def _verify_research(key: str, model: str) -> Tuple[bool, str]:
    try:
        result = saasshorts.research_saas_online(
            TEST_URL, gemini_key=None,
            llm_provider="openrouter", llm_model=model, openrouter_key=key,
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the sweep
        return False, f"research raised: {type(exc).__name__}: {_short(str(exc))}"
    if not isinstance(result, dict):
        return False, f"research returned {type(result).__name__}, expected a JSON object"
    sources = result.get("grounding_sources")
    if not isinstance(sources, list):
        return False, "research JSON has no 'grounding_sources' list"
    # The citation contract: every entry is a {title, url} pair the UI renders.
    bad = [s for s in sources if not isinstance(s, dict) or not s.get("url")]
    if bad:
        return False, f"{len(bad)} citation(s) missing a url"
    return True, f"ok ({len(sources)} citation(s))"


def _verify_analysis(key: str, model: str) -> Tuple[bool, str]:
    try:
        analysis = saasshorts.analyze_saas(
            FAKE_SCRAPE, gemini_key=None, web_research=None,
            llm_provider="openrouter", llm_model=model, openrouter_key=key,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"analysis raised: {type(exc).__name__}: {_short(str(exc))}"
    if not isinstance(analysis, dict):
        return False, f"analysis returned {type(analysis).__name__}, expected a JSON object"
    # The keys the script writer and the dashboard read downstream.
    missing = [k for k in ("product_name", "pain_points", "emotional_hooks")
               if k not in analysis]
    if missing:
        return False, f"analysis JSON missing expected keys: {missing}"
    if not isinstance(analysis["pain_points"], list):
        return False, "'pain_points' is not a list"
    return True, "ok"


def _verify_scripts(key: str, model: str) -> Tuple[bool, str]:
    analysis = {
        "product_name": "Example",
        "description": FAKE_SCRAPE["meta_description"],
        "value_proposition": "Illustrative examples.",
        "target_audience": "writers",
        "key_features": ["examples"],
        "pain_points": ["documents need examples"],
        "tone": "casual and authentic",
    }
    try:
        scripts = saasshorts.generate_scripts(
            analysis, gemini_key=None, num_scripts=NUM_SCRIPTS,
            style="ugc", language="en", actor_gender="female",
            llm_provider="openrouter", llm_model=model, openrouter_key=key,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"scripts raised: {type(exc).__name__}: {_short(str(exc))}"
    # The contract that differs most from the other two: this step must be a
    # JSON *array*, not an object — the renderer maps over it directly.
    if not isinstance(scripts, list):
        return False, f"scripts returned {type(scripts).__name__}, expected a JSON array"
    if not scripts:
        return False, "scripts array is empty"
    if any(not isinstance(s, dict) for s in scripts):
        return False, "scripts array contains non-objects"
    return True, f"ok ({len(scripts)} script(s))"


def verify_model(model_id: str, key: str) -> List[Tuple[str, bool, str]]:
    checks = [
        ("research (citations)", _verify_research),
        ("analysis (json object)", _verify_analysis),
        ("scripts (json array)", _verify_scripts),
    ]
    results: List[Tuple[str, bool, str]] = []
    for label, fn in checks:
        ok, note = fn(key, model_id)
        results.append((label, ok, note))
        if not ok:
            break  # a later step depends on the earlier output; stop the bleed
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify curated OpenRouter models end to end.")
    parser.add_argument("--yes", action="store_true",
                        help="skip the spend confirmation")
    args = parser.parse_args(argv)

    key = os.environ.get("OPENROUTER_TEST_KEY")
    if not key:
        print("Refusing to run: set OPENROUTER_TEST_KEY to a key authorised for "
              "this test spend.\nIt is deliberately not OPENROUTER_API_KEY, so a "
              "production key cannot be spent by accident.",
              file=sys.stderr)
        return 2

    estimate = _estimate_cost_usd()
    models = openrouter_backend.CURATED_MODELS
    print(f"Verifying {len(models)} curated models against the real pipeline.")
    print(f"Each runs research + analysis + {NUM_SCRIPTS} scripts on tiny inputs.")
    print(f"Estimated spend: at most ~${estimate:.3f} (dominated by the per-model "
          f"web-search call). Actual cost will be lower.\n")

    if not args.yes:
        try:
            answer = input("Proceed and spend against OPENROUTER_TEST_KEY? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted; nothing was sent.")
            return 130

    passed: List[str] = []
    failed: List[str] = []
    # Structured per-candidate record: exact model ID + the outcome and note of
    # every check, so a run is auditable after the terminal scrolls away.
    record: Dict[str, Any] = {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": Path(sys.argv[0]).name,
        "num_candidates": len(models),
        "checks": ["research (citations)", "analysis (json object)", "scripts (json array)"],
        "candidates": [],
    }
    for m in models:
        model_id = m["id"]
        print(f"─ {model_id}")
        results = verify_model(model_id, key)
        model_ok = all(ok for _, ok, _ in results)
        for label, ok, note in results:
            print(f"    {'PASS' if ok else 'FAIL'}  {label}: {note}")
        (passed if model_ok else failed).append(model_id)
        record["candidates"].append({
            "id": model_id,
            "name": m.get("name"),
            "passed": model_ok,
            "results": [
                {"check": label, "passed": ok, "note": note} for label, ok, note in results
            ],
        })
        time.sleep(1.0)  # be gentle on the shared rate limit between models

    record["passed"] = sorted(passed)
    record["failed"] = sorted(failed)
    RESULTS_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nPer-candidate record written to {RESULTS_PATH.name} (never contains the key).")

    print("\n" + "=" * 60)
    print(f"Passed end-to-end: {len(passed)}/{len(models)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("\nPaste into openrouter_backend.py to record the evidence:")
    rendered = ", ".join(sorted(passed))
    print(f"VERIFIED_MODELS: Set[str] = {{{rendered}}}")
    print("\nThe picker reports `verified` from this set. Curated models stay")
    print("selectable without it — they are metadata-qualified — but are labelled")
    print("\"Not end-to-end tested\", never \"verified\"; only this set grants that.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
