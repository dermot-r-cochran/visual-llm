# visual-llm

**HIDL — Hierarchical Image Description Language.** A structured annotation
system that turns images into compact, consistent, hierarchical JSON, so that
very large photo sets can be scanned, clustered and sampled without opening
every frame.

HIDL acts as an intermediate layer between raw images and decisions. It is
designed for datasets of 10k–100k images.

---

## The problem it solves

A vision model given a free-text prompt will describe the same photograph
differently on Tuesday than it did on Monday, and differently again from the
photograph beside it. That is fine for one image and useless for fifty
thousand, where the only useful questions are comparative:

- which of these are the same scene?
- which is the best of a near-identical burst?
- which two hundred should a human actually look at?

Free-text captions cannot answer those, because nothing lines up. HIDL fixes
the shape of the answer first — a controlled vocabulary and a fixed schema — so
annotations are comparable across images and across runs. That comparability is
what makes semantic clustering and coreset-style subset selection possible.

## What HIDL is, and is not

**It is** a schema, a controlled vocabulary, and a thin annotator that drives a
vision backend and parses the result into typed objects.

**It is not** a vision model, a labelling GUI, an image store, or a clustering
library. It produces the representation those things consume.

---

## Four levels

Annotation is layered, so cost scales with how much detail is actually needed.
L0 is cheap enough to run across an entire archive; L3 is for the shortlist.

| Level | Holds | Use | Proved by |
|---|---|---|---|
| **L0** | `scene`, `subjects`, `signals` | one-line abstraction — bulk scanning | `tests/test_schema.py::TestL0::test_to_dict_keys` |
| **L1** | subjects with counts | structured summary — grouping and filtering | `tests/test_schema.py::TestL1::test_to_dict_structure`, `::TestHIDLAnnotationDeserialisation::test_count_defaults_to_one` |
| **L2** | objects with attributes, and relations between them | detail — separating similar frames | `tests/test_schema.py::TestL2::test_objects_serialised`, `::TestL2::test_relation_serialised` |
| **L3** | `ambiguity`, `composition`, `uniqueness` | refinement — choosing between near-duplicates | `tests/test_schema.py::TestL3::test_to_dict` |

L2 and L3 are optional, so a partial annotation is still a valid annotation
(`tests/test_schema.py::TestHIDLAnnotationDeserialisation::test_l2_none_explicitly`,
`::TestHIDLAnnotationSerialisation::test_fast_scan_keys`).

## Three modes

Modes select which levels are generated:

| Mode | Produces | Proved by |
|---|---|---|
| `fast-scan` *(default)* | L0 + L1 | `tests/test_prompts.py::TestBuildSystemPrompt::test_fast_scan_excludes_l2_l3`; the default by `tests/test_annotator.py::TestAnnotatorModeResolution::test_no_mode_defaults_fast_scan` |
| `detailed` | L0 + L1 + L2 | `tests/test_prompts.py::TestBuildSystemPrompt::test_detailed_includes_l2`, `::TestBuildSystemPrompt::test_detailed_excludes_l3` |
| `refinement` | L0 + L1 + L2 + L3 | `tests/test_prompts.py::TestBuildSystemPrompt::test_refinement_includes_l2_and_l3` |

A mode can be passed explicitly, or **inferred from a free-text instruction** —
`"annotate in detailed mode"` resolves to `detailed` by keyword. This lets an
agent drive the annotator in natural language without threading a mode argument
through every call. (`tests/test_annotator.py::TestAnnotatorModeResolution::test_explicit_enum_mode`
and `::TestAnnotatorModeResolution::test_instruction_infers_detailed`;
`tests/test_modes.py::TestInferMode::test_refinement_takes_priority_over_detailed`.)

---

## Quick start

```bash
pip install -e ".[openai]"
```

```python
from hidl import HIDLAnnotator, AnnotationMode
from hidl.backends import OpenAIVisionBackend

backend   = OpenAIVisionBackend(api_key="sk-...")
annotator = HIDLAnnotator(backend)

annotation = annotator.annotate("image.jpg", image_id="IMG_0001")
print(annotation.to_json())
```

Images may be given as a path, as raw `bytes`, or as an HTTPS URL. (No test
yet: the backends are outside the suite, which never calls a model.)

### Batches

```python
annotations = annotator.annotate_batch(paths, mode=AnnotationMode.FAST_SCAN)
```

(`tests/test_annotator.py::TestAnnotatorBatch::test_batch_returns_list`,
`::TestAnnotatorBatch::test_batch_preserves_order`.)

### Canonicalisation

`HIDLAnnotator(backend, canonicalize=True)` is the default. It normalises L0 and
L1 fields against the controlled vocabulary after parsing, which is what makes
two annotations of the same scene actually compare equal. Turn it off only if
the backend's raw labels are wanted.
(`tests/test_annotator.py::TestAnnotatorCanonicalization::test_alias_canonicalised`,
`::TestAnnotatorCanonicalization::test_canonicalize_false_preserves_aliases`,
`::TestAnnotatorCanonicalization::test_alias_replies_share_l0_key`.)

`HIDLAnnotation.l0_key()` gives `(scene, subjects, signals)` with the lists
de-duplicated and sorted, so annotations can be grouped by their L0 summary
(`tests/test_schema.py::TestHIDLAnnotationL0Key::test_key_ignores_order_and_repeats`,
`::TestHIDLAnnotationL0Key::test_key_leaves_annotation_unchanged`):

```python
groups = {}
for ann in annotations:
    groups.setdefault(ann.l0_key(), []).append(ann)
```

---

## Backends

`VisionBackend` is a small abstract base class — implement `call()` and any
vision model can drive HIDL. `OpenAIVisionBackend` ships as the reference
implementation; `openai` is an **optional** dependency and the core library has
none. (The suite drives the annotator through its own `VisionBackend`, as in
`tests/test_annotator.py::TestAnnotatorOutput::test_l0_fields`;
`OpenAIVisionBackend` has no test yet; `.github/scripts/check_docs.py` fails
if `pyproject.toml` gains a runtime dependency or loses the `openai` extra.)

Every capability row or claim above names the test that proves it (file and
test name), or says it has no test yet (the README-proof convention, 10 October
2026). `.github/scripts/check_docs.py`, run in CI, also fails if a cited test
does not exist, a relative link resolves to nothing, a Markdown file has a
second front-matter block, or the four levels, three modes and the fields the
levels table names stop matching `hidl/schema.py` and `hidl/modes.py`.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

Requires Python 3.9+.

## Status

Research prototype. The schema, vocabulary and mode inference are settled and
covered by tests; the backend roster is deliberately thin. Interfaces may still
change.

## Licence

MIT — see [LICENSE](LICENSE).
