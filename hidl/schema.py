"""HIDL data-schema using Python dataclasses.

All fields use controlled vocabulary where possible.  Optional levels (L2, L3)
are represented as ``Optional`` fields to allow partial annotations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


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
    l2: Optional[L2] = None
    l3: Optional[L3] = None

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HIDLAnnotation":
        """Deserialise a :class:`HIDLAnnotation` from a plain ``dict``.

        Args:
            data: A dictionary as produced by :meth:`to_dict` or parsed from
                JSON output of a vision LLM.

        Returns:
            A :class:`HIDLAnnotation` instance.

        Raises:
            KeyError: If required fields ``id``, ``l0``, or ``l1`` are missing.
            ValueError: If data types are inconsistent.
        """
        image_id: str = data["id"]

        l0_data = data["l0"]
        l0 = L0(
            scene=l0_data["scene"],
            subjects=list(l0_data.get("subjects", [])),
            signals=list(l0_data.get("signals", [])),
        )

        l1_data = data["l1"]
        l1_subjects = [
            L1Subject(label=s["label"], count=int(s.get("count", 1)))
            for s in l1_data.get("subjects", [])
        ]
        l1 = L1(
            scene=l1_data["scene"],
            subjects=l1_subjects,
            attributes=list(l1_data.get("attributes", [])),
            signals=list(l1_data.get("signals", [])),
            quality_flags=list(l1_data.get("quality_flags", [])),
        )

        l2: Optional[L2] = None
        if "l2" in data and data["l2"] is not None:
            l2_data = data["l2"]
            l2_objects = [
                L2Object(
                    id=o["id"],
                    label=o["label"],
                    attributes=list(o.get("attributes", [])),
                )
                for o in l2_data.get("objects", [])
            ]
            l2_relations = [
                L2Relation(
                    subject=r["subject"],
                    predicate=r["predicate"],
                    object=r["object"],
                )
                for r in l2_data.get("relations", [])
            ]
            l2 = L2(objects=l2_objects, relations=l2_relations)

        l3: Optional[L3] = None
        if "l3" in data and data["l3"] is not None:
            l3_data = data["l3"]
            l3 = L3(
                ambiguity=list(l3_data.get("ambiguity", [])),
                composition=list(l3_data.get("composition", [])),
                uniqueness=list(l3_data.get("uniqueness", [])),
            )

        return cls(id=image_id, l0=l0, l1=l1, l2=l2, l3=l3)
