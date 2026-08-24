# Testing Strategy

How HIDL is tested, and why the suite can afford to be strict.

## The governing principle

HIDL's promise is **compact, consistent, hierarchical JSON** — a language,
not a service. A language is testable in a way a service never is: zero
runtime dependencies, no network, no model calls, so every property of the
notation (schema, vocabulary, parsing, rendering modes, prompt assembly) can
be pinned exactly and run in a fraction of a second. The suite leans into
that: 92 tests across six files, all pure, all verified passing locally on
2026-08-24.

## The suite (`tests/`, pytest)

| File | Pins |
| --- | --- |
| `test_schema.py` | the structural contract of a HIDL document |
| `test_vocabulary.py` | the controlled vocabulary — the terms the language admits |
| `test_parser.py` | text → structure, including malformed input |
| `test_annotator.py` | the annotation pipeline that produces documents |
| `test_modes.py` | the rendering/detail modes and their invariants |
| `test_prompts.py` | the prompt templates handed to a vision model |

Run: `pytest` (config in `pyproject.toml`). The `openai` extra is optional
and nothing in the tests requires it — model integration stays out of the
suite by design, so a red test always means the *language* changed, never
that a vendor did.

## Extending

- A vocabulary or schema change is a **language change**: update the schema
  and vocabulary tests in the same commit, deliberately — they are the
  specification readers rely on, not incidental assertions.
- New parser behaviour lands with both the accepting case and the rejecting
  case; a parser tested only on well-formed input is half-tested.
- Keep model calls out. If an integration example is ever wanted, it belongs
  behind a marker (`-m integration`) that CI does not run by default.

## Known gaps (candidates for next)

- **No CI.** The suite is dependency-free and sub-second — the cheapest
  possible workflow (checkout, setup-python, `pip install -e ".[dev]"`,
  `pytest`, `ruff check .`) would make the 92 tests enforced instead of
  voluntary. `ruff` is already configured in `pyproject.toml` and
  `pytest-cov` already declared; neither currently runs anywhere.
- Round-trip properties (parse → render → parse is a fixed point) are a
  natural `hypothesis` target for a parser/renderer pair and would generalise
  the fixed-case tests.
- Coverage is not measured; once CI exists, ratchet it at the measured
  baseline.
