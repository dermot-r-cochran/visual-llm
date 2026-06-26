"""Tests for hidl.vocabulary – controlled vocabulary and canonicalization."""

from __future__ import annotations

import pytest

from hidl.vocabulary import (
    canonicalize_scene,
    canonicalize_subject,
    canonicalize_signal,
    canonicalize_predicate,
    SCENE_LABELS,
    SUBJECT_LABELS,
    SIGNAL_LABELS,
    RELATION_PREDICATES,
)


class TestCanonicalizeScene:
    def test_known_label_unchanged(self):
        assert canonicalize_scene("urban-street") == "urban-street"

    def test_alias_resolved(self):
        assert canonicalize_scene("street") == "urban-street"
        assert canonicalize_scene("city") == "urban-street"
        assert canonicalize_scene("freeway") == "highway"
        assert canonicalize_scene("park") == "outdoor-park"

    def test_case_insensitive(self):
        assert canonicalize_scene("Urban-Street") == "urban-street"
        assert canonicalize_scene("HIGHWAY") == "highway"

    def test_space_to_hyphen(self):
        assert canonicalize_scene("parking lot") == "parking-lot"

    def test_unknown_label_returned_lowercased(self):
        result = canonicalize_scene("rooftop")
        assert result == "rooftop"

    def test_all_canonical_scenes_unchanged(self):
        for label in SCENE_LABELS:
            assert canonicalize_scene(label) == label


class TestCanonicalizeSubject:
    def test_known_label_unchanged(self):
        assert canonicalize_subject("person") == "person"
        assert canonicalize_subject("car") == "car"

    def test_alias_resolved(self):
        assert canonicalize_subject("automobile") == "car"
        assert canonicalize_subject("pedestrian") == "person"
        assert canonicalize_subject("human") == "person"
        assert canonicalize_subject("motorbike") == "motorcycle"
        assert canonicalize_subject("aeroplane") == "airplane"
        assert canonicalize_subject("lorry") == "truck"

    def test_case_insensitive(self):
        assert canonicalize_subject("Car") == "car"
        assert canonicalize_subject("PERSON") == "person"

    def test_all_canonical_subjects_unchanged(self):
        for label in SUBJECT_LABELS:
            assert canonicalize_subject(label) == label


class TestCanonicalizeSignal:
    def test_known_label_unchanged(self):
        assert canonicalize_signal("low-light") == "low-light"
        assert canonicalize_signal("blurry") == "blurry"

    def test_case_insensitive(self):
        assert canonicalize_signal("Low-Light") == "low-light"

    def test_space_to_hyphen(self):
        assert canonicalize_signal("motion blur") == "motion-blur"

    def test_all_canonical_signals_unchanged(self):
        for label in SIGNAL_LABELS:
            assert canonicalize_signal(label) == label


class TestCanonicalizePredicate:
    def test_known_predicate_unchanged(self):
        assert canonicalize_predicate("near") == "near"
        assert canonicalize_predicate("in-front-of") == "in-front-of"

    def test_case_insensitive(self):
        assert canonicalize_predicate("Near") == "near"

    def test_space_to_hyphen(self):
        assert canonicalize_predicate("in front of") == "in-front-of"

    def test_all_canonical_predicates_unchanged(self):
        for pred in RELATION_PREDICATES:
            assert canonicalize_predicate(pred) == pred


class TestVocabularySets:
    def test_scene_labels_non_empty(self):
        assert len(SCENE_LABELS) > 0

    def test_subject_labels_non_empty(self):
        assert len(SUBJECT_LABELS) > 0

    def test_signal_labels_non_empty(self):
        assert len(SIGNAL_LABELS) > 0

    def test_relation_predicates_includes_standard(self):
        required = {"near", "behind", "in-front-of", "on", "under", "inside", "holding"}
        assert required.issubset(RELATION_PREDICATES)

    def test_subject_labels_are_singular(self):
        # No label should end with 's' (crude plural check for key labels)
        # Exception: some words like "bus" naturally end in 's'
        plural_patterns = {"persons", "cars", "trucks", "birds", "dogs", "cats"}
        for label in SUBJECT_LABELS:
            assert label not in plural_patterns, f"Plural label found: {label}"
