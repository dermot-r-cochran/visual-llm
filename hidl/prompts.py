"""Prompt builder for HIDL annotations.

Builds the system prompt and per-image user message that are sent to a vision
LLM to obtain a structured HIDL JSON annotation.
"""

from __future__ import annotations

from hidl.modes import AnnotationMode


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an image annotation assistant that generates structured annotations \
using HIDL (Hierarchical Image Description Language).

## Rules
- Return ONLY valid JSON that follows the HIDL schema below.
- Do NOT generate long descriptive prose.
- Do NOT invent details not visible in the image.
- Use controlled vocabulary wherever possible.
- Prefer short tokens over sentences.
- Keep field ordering consistent.
- If uncertain about a field, omit it rather than guessing.

## HIDL Schema

### Always include
- id (string) – image identifier
- l0:
    scene (string)         – scene label, e.g. "urban-street"
    subjects (list)        – generic subject labels, e.g. ["person", "car"]
    signals (list)         – quality/condition signals, e.g. ["low-light"]
- l1:
    scene (string)
    subjects (list)        – [{{label, count}}, ...]
    attributes (list)      – lighting/weather/context tokens
    signals (list)
    quality_flags (list)

{l2_section}\
{l3_section}\
## Vocabulary
- Scene: urban-street, highway, rural-road, parking-lot, indoor, \
indoor-office, indoor-home, indoor-industrial, outdoor, outdoor-park, \
outdoor-field, aerial, underwater, unknown
- Subjects (singular): person, car, truck, bus, motorcycle, bicycle, dog, \
cat, bird, boat, train, airplane, traffic-light, stop-sign, fire-hydrant, \
bench, chair, table, bottle, cup, book, phone, laptop, backpack, umbrella, \
plant, tree, building, road, sky, unknown
- Signals / quality_flags: low-light, overexposed, blurry, noisy, occluded, \
truncated, motion-blur, rain, fog, snow, haze, glare, shadow, crowded, cluttered
- Attributes: day, night, dusk, dawn, sunny, cloudy, overcast, rainy, snowy, \
foggy, streetlight, artificial-light, natural-light, indoor-light, wide-angle, \
telephoto, fisheye, top-down, eye-level, low-angle, high-angle
- Relation predicates: near, behind, in-front-of, on, under, inside, \
holding, next-to, above, below

## Quality Rules
- Avoid subjective language (bad: "beautiful"; good: "low-light").
- Avoid storytelling and speculation.

## Example (fast-scan mode)
{{
  "id": "IMG_0001",
  "l0": {{
    "scene": "urban-street",
    "subjects": ["person", "car"],
    "signals": ["low-light"]
  }},
  "l1": {{
    "scene": "urban-street",
    "subjects": [
      {{"label": "person", "count": 1}},
      {{"label": "car", "count": 2}}
    ],
    "attributes": ["night", "streetlight"],
    "signals": ["low-light"],
    "quality_flags": []
  }}
}}
"""

_L2_SECTION = """\
### Include in detailed / refinement mode
- l2:
    objects (list) – [{{id, label, attributes: [...]}}, ...]
      id format: label + index, e.g. "p1", "car1"
    relations (list) – [{{subject, predicate, object}}, ...]
      predicate must be from: near, behind, in-front-of, on, under, \
inside, holding, next-to, above, below

"""

_L3_SECTION = """\
### Include in refinement mode only
- l3:
    ambiguity (list)    – uncertain label tokens
    composition (list)  – composition hints, e.g. ["rule-of-thirds"]
    uniqueness (list)   – unusual aspect tokens

"""


def build_system_prompt(mode: AnnotationMode) -> str:
    """Build the system prompt for the given annotation mode.

    Args:
        mode: The :class:`~hidl.modes.AnnotationMode` to use.

    Returns:
        A string to use as the ``system`` message for the vision LLM.
    """
    l2_section = _L2_SECTION if mode in (
        AnnotationMode.DETAILED,
        AnnotationMode.REFINEMENT,
    ) else ""
    l3_section = _L3_SECTION if mode == AnnotationMode.REFINEMENT else ""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        l2_section=l2_section,
        l3_section=l3_section,
    )


def build_user_message(image_id: str, mode: AnnotationMode) -> str:
    """Build the user message for a single image annotation request.

    Args:
        image_id: Identifier for the image (e.g. ``"IMG_0042"``).
        mode: Annotation mode — determines which levels to generate.

    Returns:
        A string to use as the ``user`` message content (text portion).
    """
    level_str = {
        AnnotationMode.FAST_SCAN: "L0 and L1",
        AnnotationMode.DETAILED: "L0, L1, and L2",
        AnnotationMode.REFINEMENT: "L0, L1, L2, and L3",
    }[mode]

    return (
        f'Annotate this image. Image ID: "{image_id}". '
        f"Generate {level_str}. "
        "Return ONLY valid JSON following the HIDL schema. "
        "Do not include any explanatory text outside the JSON."
    )
