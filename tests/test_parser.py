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

    def test_fence_with_invalid_json_falls_through_to_brace_scan(self):
        # A fenced block that is not JSON is skipped, and the object after it
        # is still found by the first-'{' / last-'}' scan.
        s = "```\nnot json at all\n```\n" + _to_str(FAST_SCAN_JSON)
        assert json.loads(extract_json(s)) == FAST_SCAN_JSON

    @pytest.mark.parametrize(
        "fenced", ['"I cannot annotate this image"', "42", "null", '["person", "car"]']
    )
    def test_fenced_json_that_is_not_an_object_is_not_returned(self, fenced):
        # Valid JSON, but not an object: there is no annotation in the reply.
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("```json\n" + fenced + "\n```")

    def test_fenced_non_object_falls_through_to_brace_scan(self):
        # The same fall-through as a fence that is not JSON: the object after
        # the fenced block is still found.
        s = "```\n42\n```\n" + _to_str(FAST_SCAN_JSON)
        assert json.loads(extract_json(s)) == FAST_SCAN_JSON


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

    def test_caller_image_id_beats_reply_id(self):
        raw = _to_str(FAST_SCAN_JSON)  # the reply says IMG_0001
        ann = parse_hidl_response(raw, image_id="IMG_0042")
        assert ann.id == "IMG_0042"

    def test_reply_id_kept_without_caller_image_id(self):
        ann = parse_hidl_response(_to_str(FAST_SCAN_JSON))
        assert ann.id == "IMG_0001"

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

    def test_string_where_list_belongs_raises_naming_the_field(self):
        d = json.loads(_to_str(FAST_SCAN_JSON))
        d["l0"]["subjects"] = "person"
        with pytest.raises(ValueError, match=r"invalid HIDL structure: l0\.subjects must be a list"):
            parse_hidl_response(json.dumps(d))

    def test_non_object_level_raises_value_error_not_attribute_error(self):
        d = json.loads(_to_str(FAST_SCAN_JSON))
        d["l2"] = "none"
        with pytest.raises(ValueError, match=r"l2 must be an object, got str"):
            parse_hidl_response(json.dumps(d))

    @pytest.mark.parametrize("fenced", ['"I cannot annotate this image"', "42", "null", "[]"])
    def test_fenced_non_object_raises_value_error_not_attribute_error(self, fenced):
        # A fenced reply that is JSON but not an object used to reach
        # ``data.get`` as a str, int, None or list and raise AttributeError.
        with pytest.raises(ValueError, match="No JSON object found"):
            parse_hidl_response("```json\n" + fenced + "\n```", image_id="IMG_0001")
