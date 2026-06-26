"""Annotation modes for the HIDL system."""

from __future__ import annotations

from enum import Enum
import re


class AnnotationMode(str, Enum):
    """Supported annotation modes.

    fast_scan   – generate L0 and L1 only (default)
    detailed    – include L2 (object-level attributes and relations)
    refinement  – include L3 (ambiguity notes, composition hints)
    """

    FAST_SCAN = "fast-scan"
    DETAILED = "detailed"
    REFINEMENT = "refinement"


# Keywords that trigger each mode when found in free-text instructions.
_MODE_KEYWORDS: dict[AnnotationMode, list[str]] = {
    AnnotationMode.REFINEMENT: [
        "refine",
        "refinement",
        "l3",
        "ambiguity",
        "composition",
        "uniqueness",
    ],
    AnnotationMode.DETAILED: [
        "detail",
        "detailed",
        "l2",
        "object",
        "relation",
        "attribute",
        "full",
    ],
    AnnotationMode.FAST_SCAN: [
        "fast",
        "scan",
        "fast-scan",
        "quick",
        "l0",
        "l1",
    ],
}


def infer_mode(instruction: str) -> AnnotationMode:
    """Infer the annotation mode from a free-text instruction string.

    Priority order: refinement > detailed > fast_scan.
    Falls back to :attr:`AnnotationMode.FAST_SCAN` when no keyword matches.

    Args:
        instruction: A natural-language instruction, e.g. ``"use detailed mode"``.

    Returns:
        The inferred :class:`AnnotationMode`.
    """
    lowered = instruction.lower()
    tokens = set(re.findall(r"[\w-]+", lowered))

    for mode in (
        AnnotationMode.REFINEMENT,
        AnnotationMode.DETAILED,
        AnnotationMode.FAST_SCAN,
    ):
        if any(kw in tokens or kw in lowered for kw in _MODE_KEYWORDS[mode]):
            return mode

    return AnnotationMode.FAST_SCAN
