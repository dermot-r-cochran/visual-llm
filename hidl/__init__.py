"""HIDL – Hierarchical Image Description Language.

A structured annotation system for large image datasets that produces
consistent, compact, hierarchical (L0–L3) JSON annotations suitable for
large-scale scanning and clustering.
"""

from hidl.annotator import HIDLAnnotator
from hidl.schema import (
    HIDLAnnotation,
    L0,
    L1,
    L2,
    L3,
    L1Subject,
    L2Object,
    L2Relation,
)
from hidl.modes import AnnotationMode

__all__ = [
    "HIDLAnnotator",
    "HIDLAnnotation",
    "AnnotationMode",
    "L0",
    "L1",
    "L2",
    "L3",
    "L1Subject",
    "L2Object",
    "L2Relation",
]
