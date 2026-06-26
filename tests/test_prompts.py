"""Tests for hidl.prompts – system and user prompt builders."""

from __future__ import annotations

import pytest

from hidl.modes import AnnotationMode
from hidl.prompts import build_system_prompt, build_user_message


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
