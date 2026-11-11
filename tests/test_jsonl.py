"""Tests for hidl.jsonl – writing batch results to JSON Lines and loading them back."""

from __future__ import annotations

import json

import pytest

from hidl.jsonl import load_jsonl, write_jsonl
from hidl.schema import HIDLAnnotation

FAST_SCAN = {
    "id": "A",
    "l0": {"scene": "urban-street", "subjects": ["person", "car"], "signals": ["low-light"]},
    "l1": {
        "scene": "urban-street",
        "subjects": [{"label": "person", "count": 1}, {"label": "car", "count": 2}],
        "attributes": ["night"],
        "signals": ["low-light"],
        "quality_flags": [],
    },
}

REFINEMENT = {
    **FAST_SCAN,
    "id": "B",
    "l2": {
        "objects": [{"id": "p1", "label": "person", "attributes": ["standing"]}],
        "relations": [{"subject": "p1", "predicate": "near", "object": "p1"}],
    },
    "l3": {"ambiguity": [], "composition": ["rule-of-thirds"], "uniqueness": ["neon-sign"]},
}


def _annotations() -> list[HIDLAnnotation]:
    return [HIDLAnnotation.from_dict(FAST_SCAN), HIDLAnnotation.from_dict(REFINEMENT)]


class TestJsonlRoundTrip:
    def test_load_returns_what_was_written(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        write_jsonl(_annotations(), path)
        assert load_jsonl(path) == _annotations()

    def test_one_annotation_per_line(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        write_jsonl(_annotations(), path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["id"] for line in lines] == ["A", "B"]

    def test_accepts_a_string_path(self, tmp_path):
        path = str(tmp_path / "batch.jsonl")
        write_jsonl(_annotations(), path)
        assert [a.id for a in load_jsonl(path)] == ["A", "B"]


class TestJsonlMalformed:
    def test_line_that_is_not_json_names_its_line(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        path.write_text(json.dumps(FAST_SCAN) + "\n{not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            load_jsonl(path)

    def test_line_that_is_not_an_object_names_its_line(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        path.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 1.*expected an object"):
            load_jsonl(path)

    def test_line_missing_a_level_names_its_line(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        broken = {k: v for k, v in FAST_SCAN.items() if k != "l1"}
        lines = [json.dumps(FAST_SCAN), json.dumps(FAST_SCAN), json.dumps(broken)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 3"):
            load_jsonl(path)
