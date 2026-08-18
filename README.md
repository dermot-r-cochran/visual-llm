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

| Level | Holds | Use |
|---|---|---|
| **L0** | `scene`, `subjects`, `signals` | one-line abstraction — bulk scanning |
| **L1** | subjects with counts | structured summary — grouping and filtering |
| **L2** | objects with attributes, and relations between them | detail — separating similar frames |
| **L3** | `ambiguity`, `composition`, `uniqueness` | refinement — choosing between near-duplicates |

L2 and L3 are optional, so a partial annotation is still a valid annotation.

## Three modes

Modes select which levels are generated:

| Mode | Produces |
|---|---|
| `fast-scan` *(default)* | L0 + L1 |
| `detailed` | L0 + L1 + L2 |
| `refinement` | L0 + L1 + L2 + L3 |

A mode can be passed explicitly, or **inferred from a free-text instruction** —
`"annotate in detailed mode"` resolves to `detailed` by keyword. This lets an
agent drive the annotator in natural language without threading a mode argument
through every call.

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

Images may be given as a path, as raw `bytes`, or as an HTTPS URL.

### Batches

```python
annotations = annotator.annotate_batch(paths, mode=AnnotationMode.FAST_SCAN)
```

### Canonicalisation

`HIDLAnnotator(backend, canonicalize=True)` is the default. It normalises L0 and
L1 fields against the controlled vocabulary after parsing, which is what makes
two annotations of the same scene actually compare equal. Turn it off only if
the backend's raw labels are wanted.

---

## Backends

`VisionBackend` is a small abstract base class — implement `call()` and any
vision model can drive HIDL. `OpenAIVisionBackend` ships as the reference
implementation; `openai` is an **optional** dependency and the core library has
none.

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
