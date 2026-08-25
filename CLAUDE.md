# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**HIDL — Hierarchical Image Description Language**: a schema, a controlled vocabulary, and a thin annotator that drives a vision backend and parses the result into typed objects. It turns images into compact, consistent, hierarchical JSON so very large photo sets (10k–100k) can be scanned, clustered and sampled without opening every frame. It is not a vision model, a labelling GUI, an image store, or a clustering library — it produces the representation those things consume. The core library has **zero runtime dependencies**; `openai` is an optional extra for the reference backend. Research prototype: interfaces may still change.

## Commands

```bash
pip install -e ".[dev]"        # dev install (pytest, pytest-cov); add [openai] for the reference backend
pytest                          # run all tests (config in pyproject.toml: testpaths=tests, -v)
pytest tests/test_parser.py     # one file
pytest -k test_name             # one test
pytest --cov=hidl --cov-fail-under=86   # what CI runs — coverage is a ratchet at the measured baseline
ruff check .                    # lint (CI job; config in pyproject.toml: line-length 99, target py39)
```

CI (`.github/workflows/ci.yml`) runs the coverage-gated pytest on Python 3.9 and 3.13 — the declared floor and current stable — plus `ruff check .`. Two consequences worth knowing before editing: raise the `--cov-fail-under` ratchet only in the change that adds the tests that earn it, and keep `from __future__ import annotations` at the top of modules so new-style unions (`X | None`) stay valid on 3.9.

## Architecture

Everything hangs off one pipeline in `hidl/annotator.py` (`HIDLAnnotator.annotate`):

1. **Resolve mode** (`hidl/modes.py`) — explicit `mode` wins; else a free-text `instruction` is keyword-matched by `infer_mode` (priority refinement > detailed > fast-scan); else `FAST_SCAN`. Modes select which levels are generated: `fast-scan` → L0+L1, `detailed` → +L2, `refinement` → +L3.
2. **Build prompts** (`hidl/prompts.py`) — `build_system_prompt` assembles the system prompt from a template, splicing in the L2/L3 schema sections only for the modes that need them; the controlled vocabulary is embedded in the prompt text.
3. **Call the backend** (`hidl/backends.py`) — `VisionBackend` is an ABC with a single `call(system_prompt, user_message, image)` method; `OpenAIVisionBackend` is the reference implementation (import guarded, since `openai` is optional). `ImageSource` is a path, raw `bytes`, or an HTTP(S) URL; `_encode_image`/`_detect_mime` handle base64 data URIs and magic-byte MIME detection.
4. **Parse** (`hidl/parser.py`) — `parse_hidl_response` extracts JSON from the raw model reply (markdown fences first, then first-`{`/last-`}` brace scan), fills a missing `id` from the caller's `image_id`, and deserialises via `HIDLAnnotation.from_dict`. All failures raise `ValueError` with a snippet of the offending text.
5. **Canonicalise** (`hidl/vocabulary.py`) — on by default (`canonicalize=True`); normalises L0/L1 scene/subject/signal labels and L2 labels/predicates against the vocabulary. This step is what makes two annotations of the same scene compare equal — turn it off only when raw backend labels are wanted.

The **schema** (`hidl/schema.py`) is plain dataclasses: `HIDLAnnotation` holds required `L0` (scene/subjects/signals) and `L1` (subjects with counts, attributes, quality flags) plus optional `L2` (objects with IDs and subject–predicate–object relations) and `L3` (ambiguity/composition/uniqueness). L2/L3 being `Optional` is the point — a partial annotation is a valid annotation, so cost scales with the mode.

The **vocabulary** (`hidl/vocabulary.py`) is frozensets of canonical labels plus alias maps (e.g. `pedestrian` → `person`). Canonicalisers return unknown labels normalised but *unchanged in meaning* rather than dropping them, so novel terms survive.

## Things to keep aligned

- The vocabulary lives in **two places**: the frozensets/aliases in `hidl/vocabulary.py` and the label lists written into the prompt template in `hidl/prompts.py`. A vocabulary change edits both, plus the tests, in the same commit — the schema/vocabulary tests are the specification, not incidental assertions (see `TestingStrategy.md`).
- Keep model calls out of the tests. The suite is pure — no network, no model, no runtime dependencies — which is why a red test always means the *language* changed, never that a vendor did. Testing mechanics, the per-file map of what each test file pins, and the rules for extending the suite are in `TestingStrategy.md`.
- New parser behaviour lands with both the accepting case and the rejecting case.
