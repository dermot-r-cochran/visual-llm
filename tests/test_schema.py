"""Tests for hidl.schema – HIDL data schema serialisation/deserialisation."""

from __future__ import annotations

import json

import pytest

from hidl.schema import (
    L0,
    L1,
    L2,
    L3,
    HIDLAnnotation,
    L1Subject,
    L2Object,
    L2Relation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_fast_scan_annotation() -> HIDLAnnotation:
    return HIDLAnnotation(
        id="IMG_0001",
        l0=L0(
            scene="urban-street",
            subjects=["person", "car"],
            signals=["low-light"],
        ),
        l1=L1(
            scene="urban-street",
            subjects=[
                L1Subject(label="person", count=1),
                L1Subject(label="car", count=2),
            ],
            attributes=["night", "streetlight"],
            signals=["low-light"],
            quality_flags=[],
        ),
    )


def make_detailed_annotation() -> HIDLAnnotation:
    ann = make_fast_scan_annotation()
    ann.l2 = L2(
        objects=[
            L2Object(id="p1", label="person", attributes=["standing"]),
            L2Object(id="car1", label="car", attributes=["parked"]),
            L2Object(id="car2", label="car", attributes=[]),
        ],
        relations=[
            L2Relation(subject="p1", predicate="near", object="car1"),
        ],
    )
    return ann


def make_refinement_annotation() -> HIDLAnnotation:
    ann = make_detailed_annotation()
    ann.l3 = L3(
        ambiguity=["possible-bicycle"],
        composition=["rule-of-thirds"],
        uniqueness=["neon-sign"],
    )
    return ann


# ---------------------------------------------------------------------------
# L0 tests
# ---------------------------------------------------------------------------


class TestL0:
    def test_to_dict_keys(self):
        l0 = L0(scene="urban-street", subjects=["person"], signals=["low-light"])
        d = l0.to_dict()
        assert set(d.keys()) == {"scene", "subjects", "signals"}

    def test_to_dict_values(self):
        l0 = L0(scene="urban-street", subjects=["person", "car"], signals=[])
        d = l0.to_dict()
        assert d["scene"] == "urban-street"
        assert d["subjects"] == ["person", "car"]
        assert d["signals"] == []

    def test_default_empty_lists(self):
        l0 = L0(scene="indoor")
        assert l0.subjects == []
        assert l0.signals == []


# ---------------------------------------------------------------------------
# L1 tests
# ---------------------------------------------------------------------------


class TestL1:
    def test_to_dict_structure(self):
        l1 = L1(
            scene="urban-street",
            subjects=[L1Subject(label="person", count=3)],
            attributes=["day"],
            signals=[],
            quality_flags=["blurry"],
        )
        d = l1.to_dict()
        assert d["scene"] == "urban-street"
        assert d["subjects"] == [{"label": "person", "count": 3}]
        assert d["attributes"] == ["day"]
        assert d["quality_flags"] == ["blurry"]

    def test_default_empty_lists(self):
        l1 = L1(scene="indoor")
        assert l1.subjects == []
        assert l1.attributes == []
        assert l1.signals == []
        assert l1.quality_flags == []


# ---------------------------------------------------------------------------
# L2 tests
# ---------------------------------------------------------------------------


class TestL2:
    def test_objects_serialised(self):
        l2 = L2(
            objects=[L2Object(id="p1", label="person", attributes=["standing"])],
            relations=[],
        )
        d = l2.to_dict()
        assert d["objects"] == [{"id": "p1", "label": "person", "attributes": ["standing"]}]
        assert d["relations"] == []

    def test_relation_serialised(self):
        l2 = L2(
            objects=[],
            relations=[L2Relation(subject="p1", predicate="near", object="car1")],
        )
        d = l2.to_dict()
        assert d["relations"] == [{"subject": "p1", "predicate": "near", "object": "car1"}]


# ---------------------------------------------------------------------------
# L3 tests
# ---------------------------------------------------------------------------


class TestL3:
    def test_to_dict(self):
        l3 = L3(ambiguity=["maybe-truck"], composition=[], uniqueness=["graffiti"])
        d = l3.to_dict()
        assert d["ambiguity"] == ["maybe-truck"]
        assert d["composition"] == []
        assert d["uniqueness"] == ["graffiti"]


# ---------------------------------------------------------------------------
# HIDLAnnotation.to_dict / to_json
# ---------------------------------------------------------------------------


class TestHIDLAnnotationSerialisation:
    def test_fast_scan_keys(self):
        ann = make_fast_scan_annotation()
        d = ann.to_dict()
        assert set(d.keys()) == {"id", "l0", "l1"}

    def test_detailed_includes_l2(self):
        ann = make_detailed_annotation()
        d = ann.to_dict()
        assert "l2" in d
        assert "l3" not in d

    def test_refinement_includes_l2_and_l3(self):
        ann = make_refinement_annotation()
        d = ann.to_dict()
        assert "l2" in d
        assert "l3" in d

    def test_to_json_is_valid_json(self):
        ann = make_refinement_annotation()
        raw = ann.to_json()
        parsed = json.loads(raw)
        assert parsed["id"] == "IMG_0001"

    def test_to_json_field_order(self):
        ann = make_refinement_annotation()
        d = ann.to_dict()
        keys = list(d.keys())
        # id must come first, then l0, l1, l2, l3
        assert keys[0] == "id"
        assert keys[1] == "l0"
        assert keys[2] == "l1"


# ---------------------------------------------------------------------------
# HIDLAnnotation.l0_key
# ---------------------------------------------------------------------------


class TestHIDLAnnotationL0Key:
    def test_key_ignores_order_and_repeats(self):
        ann = make_fast_scan_annotation()
        ann.l0.subjects = ["person", "car", "person"]
        assert ann.l0_key() == ("urban-street", ("car", "person"), ("low-light",))

    def test_key_leaves_annotation_unchanged(self):
        ann = make_fast_scan_annotation()
        before = ann.to_json()
        hash(ann.l0_key())
        assert ann.to_json() == before
        assert ann.l0.subjects == ["person", "car"]


# ---------------------------------------------------------------------------
# HIDLAnnotation.from_dict round-trip
# ---------------------------------------------------------------------------


class TestHIDLAnnotationDeserialisation:
    def test_fast_scan_round_trip(self):
        ann = make_fast_scan_annotation()
        reconstructed = HIDLAnnotation.from_dict(ann.to_dict())
        assert reconstructed.id == ann.id
        assert reconstructed.l0.scene == ann.l0.scene
        assert reconstructed.l0.subjects == ann.l0.subjects
        assert reconstructed.l1.subjects[0].count == 1
        assert reconstructed.l2 is None
        assert reconstructed.l3 is None

    def test_detailed_round_trip(self):
        ann = make_detailed_annotation()
        reconstructed = HIDLAnnotation.from_dict(ann.to_dict())
        assert reconstructed.l2 is not None
        assert reconstructed.l2.objects[0].id == "p1"
        assert reconstructed.l2.relations[0].predicate == "near"

    def test_refinement_round_trip(self):
        ann = make_refinement_annotation()
        reconstructed = HIDLAnnotation.from_dict(ann.to_dict())
        assert reconstructed.l3 is not None
        assert reconstructed.l3.ambiguity == ["possible-bicycle"]
        assert reconstructed.l3.composition == ["rule-of-thirds"]

    def test_missing_id_raises(self):
        d = make_fast_scan_annotation().to_dict()
        del d["id"]
        with pytest.raises(KeyError):
            HIDLAnnotation.from_dict(d)

    def test_missing_l0_raises(self):
        d = make_fast_scan_annotation().to_dict()
        del d["l0"]
        with pytest.raises(KeyError):
            HIDLAnnotation.from_dict(d)

    def test_l2_none_explicitly(self):
        d = make_fast_scan_annotation().to_dict()
        d["l2"] = None
        ann = HIDLAnnotation.from_dict(d)
        assert ann.l2 is None

    def test_count_defaults_to_one(self):
        d = make_fast_scan_annotation().to_dict()
        # Remove count from first subject
        d["l1"]["subjects"][0].pop("count")
        ann = HIDLAnnotation.from_dict(d)
        assert ann.l1.subjects[0].count == 1


# ---------------------------------------------------------------------------
# HIDLAnnotation.from_dict type checks
# ---------------------------------------------------------------------------


class TestHIDLAnnotationTypes:
    """A field of the wrong type is rejected with a ValueError naming it.

    The rejecting cases are what a vision model actually returns when it
    strays from the schema; the accepting case is the fixed point that shows
    the checks admit every well-formed annotation.
    """

    def test_well_typed_annotation_round_trips_to_itself(self):
        ann = make_refinement_annotation()
        reconstructed = HIDLAnnotation.from_dict(json.loads(ann.to_json()))
        assert reconstructed == ann
        assert HIDLAnnotation.from_dict(json.loads(reconstructed.to_json())) == ann

    def test_string_where_list_belongs_is_not_split_into_letters(self):
        d = make_fast_scan_annotation().to_dict()
        d["l0"]["subjects"] = "person"
        with pytest.raises(ValueError, match=r"l0\.subjects must be a list, got str"):
            HIDLAnnotation.from_dict(d)

    def test_list_where_label_belongs_is_rejected(self):
        d = make_fast_scan_annotation().to_dict()
        d["l0"]["scene"] = ["urban-street"]
        with pytest.raises(ValueError, match=r"l0\.scene must be a string, got list"):
            HIDLAnnotation.from_dict(d)

    def test_null_where_label_belongs_is_rejected(self):
        d = make_fast_scan_annotation().to_dict()
        d["l1"]["scene"] = None
        with pytest.raises(ValueError, match=r"l1\.scene must be a string, got NoneType"):
            HIDLAnnotation.from_dict(d)

    def test_non_string_list_item_is_named_by_index(self):
        d = make_fast_scan_annotation().to_dict()
        d["l1"]["attributes"] = ["night", 3]
        with pytest.raises(ValueError, match=r"l1\.attributes\[1\] must be a string, got int"):
            HIDLAnnotation.from_dict(d)

    def test_subject_entry_must_be_an_object(self):
        d = make_fast_scan_annotation().to_dict()
        d["l1"]["subjects"] = ["person"]
        with pytest.raises(ValueError, match=r"l1\.subjects\[0\] must be an object, got str"):
            HIDLAnnotation.from_dict(d)

    def test_subjects_must_be_a_list_not_a_single_entry(self):
        d = make_fast_scan_annotation().to_dict()
        d["l1"]["subjects"] = {"label": "person", "count": 1}
        with pytest.raises(ValueError, match=r"l1\.subjects must be a list, got dict"):
            HIDLAnnotation.from_dict(d)

    def test_subject_label_must_be_a_string(self):
        d = make_fast_scan_annotation().to_dict()
        d["l1"]["subjects"][0]["label"] = ["person"]
        with pytest.raises(ValueError, match=r"l1\.subjects\[0\]\.label must be a string"):
            HIDLAnnotation.from_dict(d)

    def test_level_must_be_an_object(self):
        d = make_fast_scan_annotation().to_dict()
        d["l2"] = "none"
        with pytest.raises(ValueError, match=r"l2 must be an object, got str"):
            HIDLAnnotation.from_dict(d)

    def test_l2_object_attributes_must_be_a_list(self):
        d = make_detailed_annotation().to_dict()
        d["l2"]["objects"][0]["attributes"] = "standing"
        with pytest.raises(ValueError, match=r"l2\.objects\[0\]\.attributes must be a list"):
            HIDLAnnotation.from_dict(d)

    def test_l2_relation_fields_must_be_strings(self):
        d = make_detailed_annotation().to_dict()
        d["l2"]["relations"][0]["object"] = 1
        with pytest.raises(ValueError, match=r"l2\.relations\[0\]\.object must be a string"):
            HIDLAnnotation.from_dict(d)

    def test_l3_lists_must_be_lists(self):
        d = make_refinement_annotation().to_dict()
        d["l3"]["ambiguity"] = "maybe-truck"
        with pytest.raises(ValueError, match=r"l3\.ambiguity must be a list, got str"):
            HIDLAnnotation.from_dict(d)

    def test_id_must_be_a_string(self):
        d = make_fast_scan_annotation().to_dict()
        d["id"] = 7
        with pytest.raises(ValueError, match=r"^id must be a string, got int"):
            HIDLAnnotation.from_dict(d)

    def test_annotation_must_be_an_object(self):
        with pytest.raises(ValueError, match=r"annotation must be an object, got list"):
            HIDLAnnotation.from_dict([make_fast_scan_annotation().to_dict()])

    def test_missing_required_key_still_raises_key_error(self):
        # Type checks do not change what a missing required field raises.
        d = make_fast_scan_annotation().to_dict()
        del d["l1"]["scene"]
        with pytest.raises(KeyError):
            HIDLAnnotation.from_dict(d)
