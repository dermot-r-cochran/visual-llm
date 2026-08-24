"""HIDL – Hierarchical Image Description Language.

A structured annotation system for large image datasets that produces
consistent, compact, hierarchical (L0–L3) JSON annotations suitable for
large-scale scanning and clustering.
"""

from hidl.annotator import HIDLAnnotator
from hidl.modes import AnnotationMode
from hidl.schema import (
    L0,
    L1,
    L2,
    L3,
    HIDLAnnotation,
    L1Subject,
    L2Object,
    L2Relation,
)

__all__ = [
    "L0",
    "L1",
    "L2",
    "L3",
    "AnnotationMode",
    "HIDLAnnotation",
    "HIDLAnnotator",
    "L1Subject",
    "L2Object",
    "L2Relation",
]
