"""Controlled vocabulary for HIDL annotations.

All labels in HIDL annotations **must** use the canonical forms defined
here to guarantee consistency across a dataset.  Aliases map common
alternative spellings to their canonical form.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical scene labels
# ---------------------------------------------------------------------------
SCENE_LABELS: frozenset[str] = frozenset(
    {
        "urban-street",
        "highway",
        "rural-road",
        "parking-lot",
        "indoor",
        "indoor-office",
        "indoor-home",
        "indoor-industrial",
        "outdoor",
        "outdoor-park",
        "outdoor-field",
        "aerial",
        "underwater",
        "unknown",
    }
)

# ---------------------------------------------------------------------------
# Canonical subject labels (singular)
# ---------------------------------------------------------------------------
SUBJECT_LABELS: frozenset[str] = frozenset(
    {
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "dog",
        "cat",
        "bird",
        "boat",
        "train",
        "airplane",
        "traffic-light",
        "stop-sign",
        "fire-hydrant",
        "bench",
        "chair",
        "table",
        "bottle",
        "cup",
        "book",
        "phone",
        "laptop",
        "backpack",
        "umbrella",
        "plant",
        "tree",
        "building",
        "road",
        "sky",
        "unknown",
    }
)

# ---------------------------------------------------------------------------
# Canonical signal / quality flag labels
# ---------------------------------------------------------------------------
SIGNAL_LABELS: frozenset[str] = frozenset(
    {
        "low-light",
        "overexposed",
        "blurry",
        "noisy",
        "occluded",
        "truncated",
        "motion-blur",
        "rain",
        "fog",
        "snow",
        "haze",
        "glare",
        "shadow",
        "crowded",
        "cluttered",
    }
)

# ---------------------------------------------------------------------------
# Canonical attribute labels
# ---------------------------------------------------------------------------
ATTRIBUTE_LABELS: frozenset[str] = frozenset(
    {
        "day",
        "night",
        "dusk",
        "dawn",
        "sunny",
        "cloudy",
        "overcast",
        "rainy",
        "snowy",
        "foggy",
        "streetlight",
        "artificial-light",
        "natural-light",
        "indoor-light",
        "wide-angle",
        "telephoto",
        "fisheye",
        "top-down",
        "eye-level",
        "low-angle",
        "high-angle",
    }
)

# ---------------------------------------------------------------------------
# Canonical relation predicates
# ---------------------------------------------------------------------------
RELATION_PREDICATES: frozenset[str] = frozenset(
    {
        "near",
        "behind",
        "in-front-of",
        "on",
        "under",
        "inside",
        "holding",
        "next-to",
        "above",
        "below",
    }
)

# ---------------------------------------------------------------------------
# Alias maps: non-canonical → canonical
# ---------------------------------------------------------------------------
SCENE_ALIASES: dict[str, str] = {
    "street": "urban-street",
    "city": "urban-street",
    "road": "urban-street",
    "highway-road": "highway",
    "freeway": "highway",
    "parking": "parking-lot",
    "office": "indoor-office",
    "home": "indoor-home",
    "house": "indoor-home",
    "factory": "indoor-industrial",
    "warehouse": "indoor-industrial",
    "park": "outdoor-park",
    "field": "outdoor-field",
    "drone": "aerial",
    "satellite": "aerial",
}

SUBJECT_ALIASES: dict[str, str] = {
    "automobile": "car",
    "vehicle": "car",
    "pedestrian": "person",
    "human": "person",
    "man": "person",
    "woman": "person",
    "child": "person",
    "cyclist": "person",
    "motorbike": "motorcycle",
    "bike": "bicycle",
    "aeroplane": "airplane",
    "plane": "airplane",
    "vessel": "boat",
    "ship": "boat",
    "lorry": "truck",
    "van": "truck",
    "fire hydrant": "fire-hydrant",
    "stop sign": "stop-sign",
    "traffic light": "traffic-light",
    "cell phone": "phone",
    "mobile phone": "phone",
}


def canonicalize_scene(label: str) -> str:
    """Return the canonical scene label for *label*.

    Unknown labels are returned unchanged so that novel scenes are not silently
    dropped.

    Args:
        label: Raw scene label string.

    Returns:
        Canonical label string.
    """
    normalized = label.lower().strip().replace(" ", "-")
    if normalized in SCENE_LABELS:
        return normalized
    return SCENE_ALIASES.get(normalized, normalized)


def canonicalize_subject(label: str) -> str:
    """Return the canonical subject label for *label*.

    Args:
        label: Raw subject label string.

    Returns:
        Canonical label string.
    """
    normalized = label.lower().strip()
    if normalized in SUBJECT_LABELS:
        return normalized
    return SUBJECT_ALIASES.get(normalized, normalized.replace(" ", "-"))


def canonicalize_signal(label: str) -> str:
    """Return the canonical signal/quality-flag label for *label*.

    Args:
        label: Raw signal label string.

    Returns:
        Canonical label string (lowercase, hyphenated).
    """
    return label.lower().strip().replace(" ", "-")


def canonicalize_predicate(predicate: str) -> str:
    """Return the canonical relation predicate for *predicate*.

    Args:
        predicate: Raw predicate string.

    Returns:
        Canonical predicate (lowercase, hyphenated).
    """
    return predicate.lower().strip().replace(" ", "-")
