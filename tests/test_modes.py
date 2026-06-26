"""Tests for hidl.modes – annotation mode inference."""

from __future__ import annotations

import pytest

from hidl.modes import AnnotationMode, infer_mode


class TestAnnotationMode:
    def test_values(self):
        assert AnnotationMode.FAST_SCAN.value == "fast-scan"
        assert AnnotationMode.DETAILED.value == "detailed"
        assert AnnotationMode.REFINEMENT.value == "refinement"

    def test_string_comparison(self):
        assert AnnotationMode.FAST_SCAN == "fast-scan"
        assert AnnotationMode.DETAILED == "detailed"


class TestInferMode:
    def test_fast_scan_keyword(self):
        assert infer_mode("fast-scan") == AnnotationMode.FAST_SCAN
        assert infer_mode("use fast scan mode") == AnnotationMode.FAST_SCAN
        assert infer_mode("quick annotation please") == AnnotationMode.FAST_SCAN

    def test_detailed_keyword(self):
        assert infer_mode("detailed") == AnnotationMode.DETAILED
        assert infer_mode("generate detailed annotations") == AnnotationMode.DETAILED
        assert infer_mode("full annotation with l2") == AnnotationMode.DETAILED
        assert infer_mode("include object attributes") == AnnotationMode.DETAILED

    def test_refinement_keyword(self):
        assert infer_mode("refinement") == AnnotationMode.REFINEMENT
        assert infer_mode("refine this image") == AnnotationMode.REFINEMENT
        assert infer_mode("add l3 notes") == AnnotationMode.REFINEMENT
        assert infer_mode("annotate with ambiguity info") == AnnotationMode.REFINEMENT

    def test_refinement_takes_priority_over_detailed(self):
        # An instruction mentioning both 'detailed' and 'refinement' should
        # resolve to refinement (higher priority).
        assert infer_mode("detailed refinement pass") == AnnotationMode.REFINEMENT

    def test_no_keyword_defaults_to_fast_scan(self):
        assert infer_mode("please annotate") == AnnotationMode.FAST_SCAN
        assert infer_mode("") == AnnotationMode.FAST_SCAN
        assert infer_mode("   ") == AnnotationMode.FAST_SCAN

    def test_case_insensitive(self):
        assert infer_mode("DETAILED") == AnnotationMode.DETAILED
        assert infer_mode("Refinement") == AnnotationMode.REFINEMENT
        assert infer_mode("Fast-Scan") == AnnotationMode.FAST_SCAN
