"""Tests for hidl.prompts – system and user prompt builders."""

from __future__ import annotations

from hidl.modes import AnnotationMode
from hidl.prompts import build_system_prompt, build_user_message
from hidl.vocabulary import (
    ATTRIBUTE_LABELS,
    RELATION_PREDICATES,
    SCENE_LABELS,
    SIGNAL_LABELS,
    SUBJECT_LABELS,
)


class TestBuildSystemPrompt:
    def test_fast_scan_excludes_l2_l3(self):
        prompt = build_system_prompt(AnnotationMode.FAST_SCAN)
        assert "l2" not in prompt.lower().split("include")[1] if "include" in prompt else True
        # The L2 section header must not appear in fast-scan mode
        assert "objects (list)" not in prompt
        assert "relations (list)" not in prompt

    def test_fast_scan_excludes_l3(self):
        prompt = build_system_prompt(AnnotationMode.FAST_SCAN)
        assert "ambiguity" not in prompt
        assert "uniqueness" not in prompt

    def test_detailed_includes_l2(self):
        prompt = build_system_prompt(AnnotationMode.DETAILED)
        assert "objects (list)" in prompt
        assert "relations (list)" in prompt

    def test_detailed_excludes_l3(self):
        prompt = build_system_prompt(AnnotationMode.DETAILED)
        assert "ambiguity" not in prompt

    def test_refinement_includes_l2_and_l3(self):
        prompt = build_system_prompt(AnnotationMode.REFINEMENT)
        assert "objects (list)" in prompt
        assert "ambiguity" in prompt
        assert "uniqueness" in prompt

    def test_system_prompt_contains_vocabulary(self):
        prompt = build_system_prompt(AnnotationMode.FAST_SCAN)
        assert "urban-street" in prompt
        assert "person" in prompt
        assert "low-light" in prompt

    def test_system_prompt_contains_quality_rules(self):
        prompt = build_system_prompt(AnnotationMode.FAST_SCAN)
        assert "subjective" in prompt.lower() or "beautiful" in prompt.lower()

    def test_system_prompt_is_non_empty_string(self):
        for mode in AnnotationMode:
            assert isinstance(build_system_prompt(mode), str)
            assert len(build_system_prompt(mode)) > 100


class TestBuildUserMessage:
    def test_fast_scan_mentions_l0_l1(self):
        msg = build_user_message("IMG_0001", AnnotationMode.FAST_SCAN)
        assert "L0" in msg
        assert "L1" in msg
        assert "IMG_0001" in msg

    def test_detailed_mentions_l2(self):
        msg = build_user_message("IMG_0002", AnnotationMode.DETAILED)
        assert "L2" in msg

    def test_refinement_mentions_l3(self):
        msg = build_user_message("IMG_0003", AnnotationMode.REFINEMENT)
        assert "L3" in msg

    def test_user_message_requests_json_only(self):
        msg = build_user_message("IMG_0001", AnnotationMode.FAST_SCAN)
        assert "JSON" in msg

    def test_user_message_contains_image_id(self):
        for mode in AnnotationMode:
            msg = build_user_message("TEST_IMG", mode)
            assert "TEST_IMG" in msg


class TestPromptVocabulary:
    """The label lists the prompt shows the model are the ones the canonicaliser knows.

    The vocabulary lives in two places (hidl/vocabulary.py and the prompt
    template in hidl/prompts.py); CLAUDE.md asks that a change edit both.
    This pins it, so a label added to one and not the other is a red test.
    """

    @staticmethod
    def _listed(prompt: str, prefix: str) -> set[str]:
        lines = [ln for ln in prompt.splitlines() if ln.startswith(prefix)]
        assert len(lines) == 1, f"expected one line starting {prefix!r}, found {len(lines)}"
        return {token.strip() for token in lines[0][len(prefix):].split(",")}

    def test_scene_labels_match(self):
        prompt = build_system_prompt(AnnotationMode.FAST_SCAN)
        assert self._listed(prompt, "- Scene:") == set(SCENE_LABELS)

    def test_subject_labels_match(self):
        prompt = build_system_prompt(AnnotationMode.FAST_SCAN)
        assert self._listed(prompt, "- Subjects (singular):") == set(SUBJECT_LABELS)

    def test_signal_labels_match(self):
        prompt = build_system_prompt(AnnotationMode.FAST_SCAN)
        assert self._listed(prompt, "- Signals / quality_flags:") == set(SIGNAL_LABELS)

    def test_attribute_labels_match(self):
        prompt = build_system_prompt(AnnotationMode.FAST_SCAN)
        assert self._listed(prompt, "- Attributes:") == set(ATTRIBUTE_LABELS)

    def test_relation_predicates_match(self):
        prompt = build_system_prompt(AnnotationMode.FAST_SCAN)
        assert self._listed(prompt, "- Relation predicates:") == set(RELATION_PREDICATES)

    def test_l2_section_predicates_match(self):
        # The L2 schema section repeats the predicate list; it must agree too.
        prompt = build_system_prompt(AnnotationMode.DETAILED)
        listed = self._listed(prompt, "      predicate must be from:")
        assert listed == set(RELATION_PREDICATES)
