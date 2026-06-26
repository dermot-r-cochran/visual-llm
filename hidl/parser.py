"""JSON parsing and validation for HIDL LLM responses.

The parser extracts a JSON object from an LLM response (which may include
surrounding markdown fences or explanatory text) and deserialises it into a
:class:`~hidl.schema.HIDLAnnotation`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from hidl.schema import HIDLAnnotation


# Regex to strip markdown code fences (```json … ``` or ``` … ```)
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE,
)


def extract_json(text: str) -> str:
    """Extract the first JSON object from *text*.

    Handles:
    - Plain JSON strings.
    - JSON wrapped in markdown code fences.
    - Leading/trailing prose around the JSON object.

    Args:
        text: Raw string as returned by an LLM.

    Returns:
        The extracted JSON string.

    Raises:
        ValueError: If no JSON object can be located.
    """
    # 1. Try to strip markdown fences first.
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # 2. Find the first '{' and last '}' to extract a raw JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "No JSON object found in LLM response. "
            f"Response starts with: {text[:200]!r}"
        )
    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Extracted text is not valid JSON: {exc}. "
            f"Candidate starts with: {candidate[:200]!r}"
        ) from exc


def parse_hidl_response(
    raw: str,
    image_id: str | None = None,
) -> HIDLAnnotation:
    """Parse an LLM response string into a :class:`~hidl.schema.HIDLAnnotation`.

    Args:
        raw: Raw string as returned by the vision LLM.
        image_id: If supplied and the parsed ``id`` field is missing or empty,
            this value is used as a fallback.

    Returns:
        A :class:`~hidl.schema.HIDLAnnotation` instance.

    Raises:
        ValueError: If the JSON cannot be extracted or is structurally invalid.
    """
    json_str = extract_json(raw)
    data: dict[str, Any] = json.loads(json_str)

    # Ensure the id field is present.
    if not data.get("id") and image_id:
        data["id"] = image_id

    try:
        return HIDLAnnotation.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"LLM response has invalid HIDL structure: {exc}. "
            f"Parsed data: {data!r}"
        ) from exc
