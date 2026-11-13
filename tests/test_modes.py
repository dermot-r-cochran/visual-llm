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


class TestInferModeWholeWords:
    """A keyword matches as a whole word, never as part of a longer one."""

    @pytest.mark.parametrize(
        "instruction",
        [
            "scan carefully",  # "full" in "carefully"
            "pick thoughtfully",  # "full" in "thoughtfully"
            "what is the objective",  # "object" in "objective"
            "check the correlation",  # "relation" in "correlation"
            "a quick look at the unrefined set",  # "refine" in "unrefined"
            "note any decomposition",  # "composition" in "decomposition"
            "use l20 and l30",  # "l2" in "l20", "l3" in "l30"
        ],
    )
    def test_keyword_inside_a_longer_word_does_not_match(self, instruction):
        assert infer_mode(instruction) == AnnotationMode.FAST_SCAN

    @pytest.mark.parametrize(
        "word",
        ["detail", "details", "detailed", "l2", "object", "objects", "relation",
         "relations", "relational", "relationship", "relationships", "attribute",
         "attributes",
         "full", "fully"],
    )
    def test_each_detailed_word_and_its_listed_forms_match(self, word):
        assert infer_mode(f"annotate with {word} please") == AnnotationMode.DETAILED

    @pytest.mark.parametrize(
        "word",
        ["refine", "refines", "refined", "refinement", "refinements", "l3",
         "ambiguity", "composition", "compositions", "compositional", "uniqueness"],
    )
    def test_each_refinement_word_and_its_listed_forms_match(self, word):
        assert infer_mode(f"annotate with {word} please") == AnnotationMode.REFINEMENT

    def test_keyword_next_to_punctuation_matches(self):
        assert infer_mode("detailed, please.") == AnnotationMode.DETAILED
        assert infer_mode("(refinement)") == AnnotationMode.REFINEMENT
        assert infer_mode("mode:L2") == AnnotationMode.DETAILED

    def test_keyword_joined_by_hyphen_or_underscore_matches(self):
        assert infer_mode("object-level annotation") == AnnotationMode.DETAILED
        assert infer_mode("l2_only") == AnnotationMode.DETAILED
        assert infer_mode("an L3-pass") == AnnotationMode.REFINEMENT
        assert infer_mode("fast_scan") == AnnotationMode.FAST_SCAN
