"""HIDL data-schema using Python dataclasses.

All fields use controlled vocabulary where possible.  Optional levels (L2, L3)
are represented as ``Optional`` fields to allow partial annotations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Type checks for data parsed from a model reply
# ---------------------------------------------------------------------------
#
# A vision model asked for JSON will sometimes return a string where the
# schema wants a list ("subjects": "person"), or a list or null where it wants
# a label.  Python's list() would split that string into letters, and a list
# label would survive until canonicalisation and fail there with an
# AttributeError.  Neither is a HIDL annotation, so each field is checked
# against the type its dataclass declares and the failure names the path.


def _wrong_type(where: str, expected: str, value: Any) -> ValueError:
    # ValueError rather than TypeError: ``from_dict`` documents ValueError for
    # a field of the wrong type, and ``parse_hidl_response`` re-raises it with
    # the parsed data attached, so a caller has one exception to catch.
    return ValueError(f"{where} must be {expected}, got {type(value).__name__}")


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _wrong_type(where, "an object", value)
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise _wrong_type(where, "a string", value)
    return value


def _string_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list):
        raise _wrong_type(where, "a list", value)
    return [_string(item, f"{where}[{i}]") for i, item in enumerate(value)]


def _object_list(value: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise _wrong_type(where, "a list", value)
    return [_object(item, f"{where}[{i}]") for i, item in enumerate(value)]


# ---------------------------------------------------------------------------
# Level 0 – one-line abstraction
# ---------------------------------------------------------------------------


@dataclass
class L0:
    """L0 annotation – lightweight, one-line summary of an image.

    Args:
        scene: Scene label from controlled vocabulary (e.g. ``"urban-street"``).
        subjects: Generic subject labels (e.g. ``["person", "car"]``).
        signals: Quality/condition signals (e.g. ``["low-light"]``).
    """

    scene: str
    subjects: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Level 1 – structured summary
# ---------------------------------------------------------------------------


@dataclass
class L1Subject:
    """A subject entry in an L1 annotation with a count.

    Args:
        label: Canonical subject label (e.g. ``"person"``).
        count: Number of instances visible in the image.
    """

    label: str
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class L1:
    """L1 annotation – structured summary with subject counts and attributes.

    Args:
        scene: Scene label (same controlled vocabulary as L0).
        subjects: List of :class:`L1Subject` entries.
        attributes: Context / lighting / weather tokens
            (e.g. ``["night", "streetlight"]``).
        signals: Quality signals (mirrors L0 signals).
        quality_flags: Flags about image quality defects
            (e.g. ``["blurry", "occluded"]``).
    """

    scene: str
    subjects: list[L1Subject] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "subjects": [s.to_dict() for s in self.subjects],
            "attributes": self.attributes,
            "signals": self.signals,
            "quality_flags": self.quality_flags,
        }


# ---------------------------------------------------------------------------
# Level 2 – object-level detail
# ---------------------------------------------------------------------------


@dataclass
class L2Object:
    """A single annotated object within an L2 annotation.

    Args:
        id: Short unique identifier within the image (e.g. ``"p1"``, ``"car1"``).
        label: Canonical subject label.
        attributes: Per-object attribute tokens (e.g. ``["standing", "adult"]``).
    """

    id: str
    label: str
    attributes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class L2Relation:
    """A spatial or semantic relation between two L2 objects.

    Args:
        subject: ID of the subject object (e.g. ``"p1"``).
        predicate: Canonical relation predicate (e.g. ``"near"``).
        object: ID of the object object (e.g. ``"car1"``).
    """

    subject: str
    predicate: str
    object: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class L2:
    """L2 annotation – per-object detail with IDs and relations.

    Args:
        objects: List of annotated :class:`L2Object` instances.
        relations: List of :class:`L2Relation` triples.
    """

    objects: list[L2Object] = field(default_factory=list)
    relations: list[L2Relation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": [o.to_dict() for o in self.objects],
            "relations": [r.to_dict() for r in self.relations],
        }


# ---------------------------------------------------------------------------
# Level 3 – refinement metadata
# ---------------------------------------------------------------------------


@dataclass
class L3:
    """L3 annotation – ambiguity, composition, and uniqueness notes.

    All fields are optional.  Prefer short tokens over prose.

    Args:
        ambiguity: Notes about uncertain labels or interpretations.
        composition: Composition hints (e.g. ``["rule-of-thirds"]``).
        uniqueness: Tokens describing unusual aspects of the image.
    """

    ambiguity: list[str] = field(default_factory=list)
    composition: list[str] = field(default_factory=list)
    uniqueness: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Top-level annotation
# ---------------------------------------------------------------------------


@dataclass
class HIDLAnnotation:
    """Top-level HIDL annotation for a single image.

    Args:
        id: Image identifier (e.g. ``"IMG_0001"``).
        l0: Required L0 summary.
        l1: Required L1 structured summary.
        l2: Optional L2 object-level detail (``None`` when not generated).
        l3: Optional L3 refinement metadata (``None`` when not generated).
    """

    id: str
    l0: L0
    l1: L1
    l2: L2 | None = None
    l3: L3 | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable ``dict`` representation."""
        result: dict[str, Any] = {
            "id": self.id,
            "l0": self.l0.to_dict(),
            "l1": self.l1.to_dict(),
        }
        if self.l2 is not None:
            result["l2"] = self.l2.to_dict()
        if self.l3 is not None:
            result["l3"] = self.l3.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialise the annotation to a JSON string.

        Args:
            indent: Indentation level for pretty-printing.

        Returns:
            A JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def l0_key(self) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        """Return a hashable key for grouping annotations by their L0 summary.

        The key is ``(scene, subjects, signals)`` with subjects and signals
        de-duplicated and sorted, so two annotations whose L0 lists differ
        only in order or repetition share a key.  The annotation itself is
        not changed.  Labels are compared as stored: run with
        canonicalisation on (the default) so aliases share a key.
        """
        return (
            self.l0.scene,
            tuple(sorted(set(self.l0.subjects))),
            tuple(sorted(set(self.l0.signals))),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HIDLAnnotation:
        """Deserialise a :class:`HIDLAnnotation` from a plain ``dict``.

        Args:
            data: A dictionary as produced by :meth:`to_dict` or parsed from
                JSON output of a vision LLM.

        Returns:
            A :class:`HIDLAnnotation` instance.

        Raises:
            KeyError: If required fields ``id``, ``l0``, or ``l1`` are missing.
            ValueError: If a field has the wrong type for the schema: a string
                where a list belongs, a list or ``None`` where a label belongs,
                a non-object level or entry.  The message names the field
                (``"l0.subjects must be a list, got str"``).  A ``count`` that
                ``int()`` cannot read also raises ``ValueError``.
        """
        _object(data, "annotation")
        image_id = _string(data["id"], "id")

        l0_data = _object(data["l0"], "l0")
        l0 = L0(
            scene=_string(l0_data["scene"], "l0.scene"),
            subjects=_string_list(l0_data.get("subjects", []), "l0.subjects"),
            signals=_string_list(l0_data.get("signals", []), "l0.signals"),
        )

        l1_data = _object(data["l1"], "l1")
        l1_subjects = [
            L1Subject(
                label=_string(s["label"], f"l1.subjects[{i}].label"),
                count=int(s.get("count", 1)),
            )
            for i, s in enumerate(_object_list(l1_data.get("subjects", []), "l1.subjects"))
        ]
        l1 = L1(
            scene=_string(l1_data["scene"], "l1.scene"),
            subjects=l1_subjects,
            attributes=_string_list(l1_data.get("attributes", []), "l1.attributes"),
            signals=_string_list(l1_data.get("signals", []), "l1.signals"),
            quality_flags=_string_list(l1_data.get("quality_flags", []), "l1.quality_flags"),
        )

        l2: L2 | None = None
        if "l2" in data and data["l2"] is not None:
            l2_data = _object(data["l2"], "l2")
            l2_objects = [
                L2Object(
                    id=_string(o["id"], f"l2.objects[{i}].id"),
                    label=_string(o["label"], f"l2.objects[{i}].label"),
                    attributes=_string_list(
                        o.get("attributes", []), f"l2.objects[{i}].attributes"
                    ),
                )
                for i, o in enumerate(_object_list(l2_data.get("objects", []), "l2.objects"))
            ]
            l2_relations = [
                L2Relation(
                    subject=_string(r["subject"], f"l2.relations[{i}].subject"),
                    predicate=_string(r["predicate"], f"l2.relations[{i}].predicate"),
                    object=_string(r["object"], f"l2.relations[{i}].object"),
                )
                for i, r in enumerate(
                    _object_list(l2_data.get("relations", []), "l2.relations")
                )
            ]
            l2 = L2(objects=l2_objects, relations=l2_relations)

        l3: L3 | None = None
        if "l3" in data and data["l3"] is not None:
            l3_data = _object(data["l3"], "l3")
            l3 = L3(
                ambiguity=_string_list(l3_data.get("ambiguity", []), "l3.ambiguity"),
                composition=_string_list(l3_data.get("composition", []), "l3.composition"),
                uniqueness=_string_list(l3_data.get("uniqueness", []), "l3.uniqueness"),
            )

        return cls(id=image_id, l0=l0, l1=l1, l2=l2, l3=l3)
