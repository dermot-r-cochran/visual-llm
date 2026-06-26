"""Tests for hidl.annotator – HIDLAnnotator integration tests using a mock backend."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from hidl.annotator import HIDLAnnotator, _canonicalize_annotation
from hidl.backends import VisionBackend, ImageSource
from hidl.modes import AnnotationMode
from hidl.schema import (
    HIDLAnnotation,
    L0,
    L1,
    L1Subject,
    L2,
    L2Object,
    L2Relation,
)


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockVisionBackend(VisionBackend):
    """A deterministic mock backend that returns a preset JSON response."""

    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def call(
        self,
        system_prompt: str,
        user_message: str,
        image: ImageSource,
    ) -> str:
        return json.dumps(self._response)


FAST_SCAN_RESPONSE = {
    "id": "IMG_0001",
    "l0": {
        "scene": "urban-street",
        "subjects": ["person", "car"],
        "signals": ["low-light"],
    },
    "l1": {
        "scene": "urban-street",
        "subjects": [
            {"label": "person", "count": 1},
            {"label": "car", "count": 2},
        ],
        "attributes": ["night", "streetlight"],
        "signals": ["low-light"],
        "quality_flags": [],
    },
}

DETAILED_RESPONSE = {
    **FAST_SCAN_RESPONSE,
    "l2": {
        "objects": [
            {"id": "p1", "label": "person", "attributes": ["standing"]},
            {"id": "car1", "label": "car", "attributes": []},
        ],
        "relations": [
            {"subject": "p1", "predicate": "near", "object": "car1"},
        ],
    },
}

REFINEMENT_RESPONSE = {
    **DETAILED_RESPONSE,
    "l3": {
        "ambiguity": ["possible-bicycle"],
        "composition": ["rule-of-thirds"],
        "uniqueness": [],
    },
}


# ---------------------------------------------------------------------------
# Annotate – mode resolution
# ---------------------------------------------------------------------------


class TestAnnotatorModeResolution:
    def test_explicit_enum_mode(self):
        annotator = HIDLAnnotator(MockVisionBackend(FAST_SCAN_RESPONSE))
        ann = annotator.annotate(b"fake", image_id="IMG_0001", mode=AnnotationMode.FAST_SCAN)
        assert ann.id == "IMG_0001"

    def test_explicit_string_mode(self):
        annotator = HIDLAnnotator(MockVisionBackend(FAST_SCAN_RESPONSE))
        ann = annotator.annotate(b"fake", image_id="IMG_0001", mode="fast-scan")
        assert ann.l2 is None

    def test_invalid_string_mode_raises(self):
        annotator = HIDLAnnotator(MockVisionBackend(FAST_SCAN_RESPONSE))
        with pytest.raises(ValueError):
            annotator.annotate(b"fake", mode="turbo-mode")

    def test_instruction_infers_detailed(self):
        annotator = HIDLAnnotator(MockVisionBackend(DETAILED_RESPONSE))
        ann = annotator.annotate(
            b"fake",
            image_id="IMG_0001",
            instruction="generate detailed annotations",
        )
        assert ann.l2 is not None

    def test_instruction_infers_refinement(self):
        annotator = HIDLAnnotator(MockVisionBackend(REFINEMENT_RESPONSE))
        ann = annotator.annotate(
            b"fake",
            image_id="IMG_0001",
            instruction="run refinement pass",
        )
        assert ann.l3 is not None

    def test_no_mode_defaults_fast_scan(self):
        annotator = HIDLAnnotator(MockVisionBackend(FAST_SCAN_RESPONSE))
        ann = annotator.annotate(b"fake", image_id="IMG_0001")
        assert ann.l2 is None


# ---------------------------------------------------------------------------
# Annotate – output correctness
# ---------------------------------------------------------------------------


class TestAnnotatorOutput:
    def test_l0_fields(self):
        annotator = HIDLAnnotator(MockVisionBackend(FAST_SCAN_RESPONSE))
        ann = annotator.annotate(b"fake", image_id="IMG_0001")
        assert ann.l0.scene == "urban-street"
        assert "person" in ann.l0.subjects
        assert "car" in ann.l0.subjects
        assert "low-light" in ann.l0.signals

    def test_l1_subject_counts(self):
        annotator = HIDLAnnotator(MockVisionBackend(FAST_SCAN_RESPONSE))
        ann = annotator.annotate(b"fake", image_id="IMG_0001")
        counts = {s.label: s.count for s in ann.l1.subjects}
        assert counts["person"] == 1
        assert counts["car"] == 2

    def test_l2_objects(self):
        annotator = HIDLAnnotator(MockVisionBackend(DETAILED_RESPONSE))
        ann = annotator.annotate(b"fake", image_id="IMG_0001", mode=AnnotationMode.DETAILED)
        assert ann.l2 is not None
        ids = [o.id for o in ann.l2.objects]
        assert "p1" in ids

    def test_l3_fields(self):
        annotator = HIDLAnnotator(MockVisionBackend(REFINEMENT_RESPONSE))
        ann = annotator.annotate(b"fake", image_id="IMG_0001", mode=AnnotationMode.REFINEMENT)
        assert ann.l3 is not None
        assert "possible-bicycle" in ann.l3.ambiguity

    def test_image_id_override(self):
        resp = dict(FAST_SCAN_RESPONSE)
        resp["id"] = "WRONG"
        annotator = HIDLAnnotator(MockVisionBackend(resp), canonicalize=False)
        ann = annotator.annotate(b"fake", image_id="CORRECT")
        # Parser uses LLM id when present; image_id is only a fallback
        assert ann.id == "WRONG"


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


class TestAnnotatorCanonicalization:
    def test_alias_canonicalised(self):
        resp = {
            "id": "X",
            "l0": {"scene": "street", "subjects": ["automobile"], "signals": []},
            "l1": {
                "scene": "street",
                "subjects": [{"label": "automobile", "count": 1}],
                "attributes": [],
                "signals": [],
                "quality_flags": [],
            },
        }
        annotator = HIDLAnnotator(MockVisionBackend(resp), canonicalize=True)
        ann = annotator.annotate(b"fake")
        assert ann.l0.scene == "urban-street"
        assert ann.l0.subjects == ["car"]
        assert ann.l1.subjects[0].label == "car"

    def test_canonicalize_false_preserves_aliases(self):
        resp = {
            "id": "X",
            "l0": {"scene": "street", "subjects": ["automobile"], "signals": []},
            "l1": {
                "scene": "street",
                "subjects": [{"label": "automobile", "count": 1}],
                "attributes": [],
                "signals": [],
                "quality_flags": [],
            },
        }
        annotator = HIDLAnnotator(MockVisionBackend(resp), canonicalize=False)
        ann = annotator.annotate(b"fake")
        assert ann.l0.subjects == ["automobile"]


# ---------------------------------------------------------------------------
# Batch annotation
# ---------------------------------------------------------------------------


class TestAnnotatorBatch:
    def test_batch_returns_list(self):
        annotator = HIDLAnnotator(MockVisionBackend(FAST_SCAN_RESPONSE))
        images = [("A", b"fake"), ("B", b"fake"), ("C", b"fake")]
        results = annotator.annotate_batch(images)
        assert len(results) == 3

    def test_batch_preserves_order(self):
        # Each call returns the same mock response; just verify length and types
        annotator = HIDLAnnotator(MockVisionBackend(FAST_SCAN_RESPONSE))
        images = [(f"IMG_{i:04d}", b"fake") for i in range(5)]
        results = annotator.annotate_batch(images)
        assert all(isinstance(r, HIDLAnnotation) for r in results)
