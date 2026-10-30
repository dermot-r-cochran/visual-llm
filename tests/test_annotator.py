"""Tests for hidl.annotator – HIDLAnnotator integration tests using a mock backend."""

from __future__ import annotations

import json
from typing import Any

import pytest

from hidl.annotator import BatchFailure, BatchResult, HIDLAnnotator
from hidl.backends import ImageSource, VisionBackend
from hidl.modes import AnnotationMode
from hidl.schema import (
    HIDLAnnotation,
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

    def test_mistyped_reply_raises_value_error_before_canonicalisation(self):
        # A list where a label belongs used to pass the parser and fail inside
        # canonicalisation with an AttributeError; it is a ValueError now, and
        # the message names the field.
        resp = json.loads(json.dumps(FAST_SCAN_RESPONSE))
        resp["l0"]["scene"] = ["urban-street"]
        annotator = HIDLAnnotator(MockVisionBackend(resp))
        with pytest.raises(ValueError, match=r"l0\.scene must be a string, got list"):
            annotator.annotate(b"fake", image_id="IMG_0001")

    def test_string_subjects_are_not_annotated_as_letters(self):
        resp = json.loads(json.dumps(FAST_SCAN_RESPONSE))
        resp["l0"]["subjects"] = "person"
        annotator = HIDLAnnotator(MockVisionBackend(resp))
        with pytest.raises(ValueError, match=r"l0\.subjects must be a list, got str"):
            annotator.annotate(b"fake", image_id="IMG_0001")


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

    def test_alias_replies_share_l0_key(self):
        def reply(subjects: list[str]) -> dict[str, Any]:
            return {
                "id": "X",
                "l0": {"scene": "street", "subjects": subjects, "signals": []},
                "l1": {
                    "scene": "street",
                    "subjects": [],
                    "attributes": [],
                    "signals": [],
                    "quality_flags": [],
                },
            }

        first = HIDLAnnotator(MockVisionBackend(reply(["car", "pedestrian"]))).annotate(b"fake")
        second = HIDLAnnotator(
            MockVisionBackend(reply(["person", "automobile", "man"]))
        ).annotate(b"fake")
        assert first.l0_key() == second.l0_key()
        assert first.l0_key() == ("urban-street", ("car", "person"), ())


# ---------------------------------------------------------------------------
# Batch annotation
# ---------------------------------------------------------------------------


class SequenceBackend(VisionBackend):
    """A mock backend that returns its preset raw replies one call at a time."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)

    def call(
        self,
        system_prompt: str,
        user_message: str,
        image: ImageSource,
    ) -> str:
        return next(self._replies)


# A reply that parses but is mistyped: a string where the subject list belongs.
MISTYPED_REPLY = json.dumps(
    {**FAST_SCAN_RESPONSE, "l0": {**FAST_SCAN_RESPONSE["l0"], "subjects": "car"}}
)
GOOD_REPLY = json.dumps({k: v for k, v in FAST_SCAN_RESPONSE.items() if k != "id"})


class TestAnnotatorBatch:
    def test_batch_returns_list(self):
        annotator = HIDLAnnotator(MockVisionBackend(FAST_SCAN_RESPONSE))
        images = [("A", b"fake"), ("B", b"fake"), ("C", b"fake")]
        batch = annotator.annotate_batch(images)
        assert isinstance(batch, BatchResult)
        assert len(batch.results) == 3
        assert batch.failures == []

    def test_batch_preserves_order(self):
        # A reply without an id takes the caller's image_id, so each result
        # carries the id of the image it came from and order is observable.
        no_id = {k: v for k, v in FAST_SCAN_RESPONSE.items() if k != "id"}
        annotator = HIDLAnnotator(MockVisionBackend(no_id))
        images = [(f"IMG_{i:04d}", b"fake") for i in (3, 0, 4, 1, 2)]
        batch = annotator.annotate_batch(images)
        assert all(isinstance(r, HIDLAnnotation) for r in batch.results)
        assert [r.id for r in batch.results] == [img_id for img_id, _ in images]

    def test_bad_reply_in_the_middle_keeps_the_others(self):
        annotator = HIDLAnnotator(SequenceBackend([GOOD_REPLY, MISTYPED_REPLY, GOOD_REPLY]))
        images = [("A", b"fake"), ("B", b"fake"), ("C", b"fake")]
        batch = annotator.annotate_batch(images)
        assert [r.id for r in batch.results] == ["A", "C"]
        assert len(batch.failures) == 1
        failure = batch.failures[0]
        assert isinstance(failure, BatchFailure)
        assert (failure.index, failure.image_id) == (1, "B")
        assert "subjects" in failure.error

    def test_all_bad_replies_are_all_recorded(self):
        annotator = HIDLAnnotator(SequenceBackend([MISTYPED_REPLY, "no json here"]))
        batch = annotator.annotate_batch([("A", b"fake"), ("B", b"fake")])
        assert batch.results == []
        assert [(f.index, f.image_id) for f in batch.failures] == [(0, "A"), (1, "B")]
        assert all(f.error for f in batch.failures)

    def test_unknown_mode_raises_before_any_call(self):
        annotator = HIDLAnnotator(SequenceBackend([]))
        with pytest.raises(ValueError):
            annotator.annotate_batch([("A", b"fake")], mode="no-such-mode")

    def test_backend_error_is_not_swallowed(self):
        class Broken(VisionBackend):
            def call(self, system_prompt, user_message, image):
                raise ConnectionError("unreachable")

        with pytest.raises(ConnectionError):
            HIDLAnnotator(Broken()).annotate_batch([("A", b"fake")])
