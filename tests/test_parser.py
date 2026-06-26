"""Tests for hidl.parser – JSON extraction and HIDL response parsing."""

from __future__ import annotations

import json

import pytest

from hidl.parser import extract_json, parse_hidl_response
from hidl.schema import HIDLAnnotation


# ---------------------------------------------------------------------------
# Sample valid HIDL JSON payloads
# ---------------------------------------------------------------------------

FAST_SCAN_JSON = {
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

DETAILED_JSON = {
    **FAST_SCAN_JSON,
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


def _to_str(d: dict) -> str:
    return json.dumps(d)


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        s = _to_str(FAST_SCAN_JSON)
        result = extract_json(s)
        assert json.loads(result) == FAST_SCAN_JSON

    def test_json_with_markdown_fence(self):
        s = "Here is the annotation:\n```json\n" + _to_str(FAST_SCAN_JSON) + "\n```"
        result = extract_json(s)
        assert json.loads(result) == FAST_SCAN_JSON

    def test_json_with_plain_code_fence(self):
        s = "```\n" + _to_str(FAST_SCAN_JSON) + "\n```"
        result = extract_json(s)
        assert json.loads(result) == FAST_SCAN_JSON

    def test_json_surrounded_by_prose(self):
        s = "Sure, here it is: " + _to_str(FAST_SCAN_JSON) + " That's the result."
        result = extract_json(s)
        assert json.loads(result) == FAST_SCAN_JSON

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("There is no JSON here at all.")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            extract_json("{this is not valid json}")

    def test_nested_json_extracted(self):
        nested = {"id": "X", "l0": {"scene": "indoor", "subjects": [], "signals": []},
                  "l1": {"scene": "indoor", "subjects": [], "attributes": [],
                         "signals": [], "quality_flags": []}}
        s = _to_str(nested)
        assert json.loads(extract_json(s)) == nested


# ---------------------------------------------------------------------------
# parse_hidl_response
# ---------------------------------------------------------------------------


class TestParseHidlResponse:
    def test_parse_fast_scan(self):
        raw = _to_str(FAST_SCAN_JSON)
        ann = parse_hidl_response(raw)
        assert isinstance(ann, HIDLAnnotation)
        assert ann.id == "IMG_0001"
        assert ann.l0.scene == "urban-street"
        assert ann.l0.subjects == ["person", "car"]
        assert ann.l1.subjects[0].label == "person"
        assert ann.l1.subjects[0].count == 1
        assert ann.l1.subjects[1].count == 2
        assert ann.l2 is None
        assert ann.l3 is None

    def test_parse_detailed(self):
        raw = _to_str(DETAILED_JSON)
        ann = parse_hidl_response(raw)
        assert ann.l2 is not None
        assert ann.l2.objects[0].id == "p1"
        assert ann.l2.relations[0].predicate == "near"

    def test_fallback_image_id(self):
        d = dict(FAST_SCAN_JSON)
        d["id"] = ""
        raw = json.dumps(d)
        ann = parse_hidl_response(raw, image_id="FALLBACK_ID")
        assert ann.id == "FALLBACK_ID"

    def test_markdown_fence_parsed(self):
        raw = "```json\n" + _to_str(FAST_SCAN_JSON) + "\n```"
        ann = parse_hidl_response(raw)
        assert ann.id == "IMG_0001"

    def test_invalid_structure_raises(self):
        raw = json.dumps({"id": "X", "l0": {"scene": "street"}})
        with pytest.raises(ValueError):
            parse_hidl_response(raw)

    def test_non_json_raises(self):
        with pytest.raises(ValueError):
            parse_hidl_response("I cannot annotate this image.")
