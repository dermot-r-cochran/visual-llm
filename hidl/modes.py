"""Annotation modes for the HIDL system."""

from __future__ import annotations

import re
from enum import Enum


class AnnotationMode(str, Enum):
    """Supported annotation modes.

    fast_scan   – generate L0 and L1 only (default)
    detailed    – include L2 (object-level attributes and relations)
    refinement  – include L3 (ambiguity notes, composition hints)
    """

    FAST_SCAN = "fast-scan"
    DETAILED = "detailed"
    REFINEMENT = "refinement"


# Words that trigger each mode when found in free-text instructions. A word
# matches only as a whole word, never as part of a longer one ("full" is not
# found in "carefully"), so the inflected forms that should match are listed.
_MODE_KEYWORDS: dict[AnnotationMode, list[str]] = {
    AnnotationMode.REFINEMENT: [
        "refine",
        "refines",
        "refined",
        "refinement",
        "refinements",
        "l3",
        "ambiguity",
        "composition",
        "compositions",
        "compositional",
        "uniqueness",
    ],
    AnnotationMode.DETAILED: [
        "detail",
        "details",
        "detailed",
        "l2",
        "object",
        "objects",
        "relation",
        "relations",
        "relational",
        "relationship",
        "relationships",
        "attribute",
        "attributes",
        "full",
        "fully",
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
    A keyword matches as a whole word only: ``"scan carefully"`` is not
    ``detailed``, although "carefully" contains "full". Words joined by a
    hyphen or an underscore are also read one by one, so ``"object-level"``
    matches "object".
    Falls back to :attr:`AnnotationMode.FAST_SCAN` when no keyword matches.

    Args:
        instruction: A natural-language instruction, e.g. ``"use detailed mode"``.

    Returns:
        The inferred :class:`AnnotationMode`.
    """
    lowered = instruction.lower()
    # Each run of letters and digits, and each hyphenated compound whole.
    tokens = set(re.findall(r"[a-z0-9]+", lowered)) | set(re.findall(r"[a-z0-9-]+", lowered))

    for mode in (
        AnnotationMode.REFINEMENT,
        AnnotationMode.DETAILED,
        AnnotationMode.FAST_SCAN,
    ):
        if any(kw in tokens for kw in _MODE_KEYWORDS[mode]):
            return mode

    return AnnotationMode.FAST_SCAN
