"""JSON Lines persistence for annotations.

Writes one :class:`~hidl.schema.HIDLAnnotation` per line and loads them back,
so the results of a batch can outlive the session that produced them.
Standard library only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from hidl.schema import HIDLAnnotation


def write_jsonl(annotations: Iterable[HIDLAnnotation], path: str | Path) -> None:
    """Write *annotations* to *path*, one compact JSON object per line.

    An existing file at *path* is replaced.
    """
    with Path(path).open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(ann.to_dict()) + "\n" for ann in annotations)


def load_jsonl(path: str | Path) -> list[HIDLAnnotation]:
    """Load the annotations written by :func:`write_jsonl`, in file order.

    Raises:
        ValueError: If a line is not JSON or not a valid HIDL annotation;
            the message names the line number (counted from 1).
    """
    annotations = []
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError(f"expected an object, got {type(data).__name__}")
            annotations.append(HIDLAnnotation.from_dict(data))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: line {number} is not a HIDL annotation: {exc}") from exc
    return annotations
