"""Main HIDLAnnotator class.

Orchestrates mode inference, prompt building, LLM calls, JSON parsing, and
optional vocabulary canonicalisation to produce :class:`~hidl.schema.HIDLAnnotation`
objects from images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from hidl.backends import VisionBackend, ImageSource
from hidl.modes import AnnotationMode, infer_mode
from hidl.parser import parse_hidl_response
from hidl.prompts import build_system_prompt, build_user_message
from hidl.schema import HIDLAnnotation
from hidl.vocabulary import (
    canonicalize_scene,
    canonicalize_signal,
    canonicalize_subject,
    canonicalize_predicate,
)


class HIDLAnnotator:
    """Annotate images using HIDL via a vision LLM backend.

    Example usage::

        from hidl import HIDLAnnotator, AnnotationMode
        from hidl.backends import OpenAIVisionBackend

        backend = OpenAIVisionBackend(api_key="sk-…")
        annotator = HIDLAnnotator(backend)

        annotation = annotator.annotate("image.jpg", image_id="IMG_0001")
        print(annotation.to_json())

    Args:
        backend: A :class:`~hidl.backends.VisionBackend` implementation.
        canonicalize: If ``True`` (default), apply controlled vocabulary
            normalisation to L0/L1 fields after parsing.
    """

    def __init__(
        self,
        backend: VisionBackend,
        *,
        canonicalize: bool = True,
    ) -> None:
        self._backend = backend
        self._canonicalize = canonicalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        image: ImageSource,
        *,
        image_id: str = "unknown",
        mode: Optional[Union[AnnotationMode, str]] = None,
        instruction: Optional[str] = None,
    ) -> HIDLAnnotation:
        """Generate a HIDL annotation for a single image.

        Args:
            image: Image source — a file path (``str`` or :class:`~pathlib.Path`),
                raw image bytes, or an HTTPS URL string.
            image_id: Identifier for the image used in the ``id`` field of the
                output annotation.
            mode: Annotation mode.  Can be an :class:`~hidl.modes.AnnotationMode`
                value or its string equivalent (``"fast-scan"``, ``"detailed"``,
                ``"refinement"``).  If ``None``, *instruction* is used to infer
                the mode; if both are ``None``, defaults to
                :attr:`~hidl.modes.AnnotationMode.FAST_SCAN`.
            instruction: Free-text instruction used to infer the mode when
                *mode* is not explicitly supplied (e.g.
                ``"annotate in detailed mode"``).

        Returns:
            A fully populated :class:`~hidl.schema.HIDLAnnotation`.

        Raises:
            ValueError: If the LLM response cannot be parsed into a valid
                HIDL annotation.
        """
        resolved_mode = self._resolve_mode(mode, instruction)

        system_prompt = build_system_prompt(resolved_mode)
        user_message = build_user_message(image_id, resolved_mode)

        raw_response = self._backend.call(system_prompt, user_message, image)

        annotation = parse_hidl_response(raw_response, image_id=image_id)

        if self._canonicalize:
            annotation = _canonicalize_annotation(annotation)

        return annotation

    def annotate_batch(
        self,
        images: list[tuple[str, ImageSource]],
        *,
        mode: Optional[Union[AnnotationMode, str]] = None,
        instruction: Optional[str] = None,
    ) -> list[HIDLAnnotation]:
        """Annotate a batch of images sequentially.

        Args:
            images: A list of ``(image_id, image_source)`` pairs.
            mode: Annotation mode (same semantics as :meth:`annotate`).
            instruction: Free-text instruction for mode inference.

        Returns:
            A list of :class:`~hidl.schema.HIDLAnnotation` objects in the same
            order as *images*.
        """
        return [
            self.annotate(src, image_id=img_id, mode=mode, instruction=instruction)
            for img_id, src in images
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_mode(
        mode: Optional[Union[AnnotationMode, str]],
        instruction: Optional[str],
    ) -> AnnotationMode:
        if mode is not None:
            if isinstance(mode, AnnotationMode):
                return mode
            return AnnotationMode(mode)
        if instruction:
            return infer_mode(instruction)
        return AnnotationMode.FAST_SCAN


# ---------------------------------------------------------------------------
# Vocabulary canonicalisation
# ---------------------------------------------------------------------------


def _canonicalize_annotation(ann: HIDLAnnotation) -> HIDLAnnotation:
    """Apply controlled vocabulary normalisation to *ann* in-place."""
    # L0
    ann.l0.scene = canonicalize_scene(ann.l0.scene)
    ann.l0.subjects = [canonicalize_subject(s) for s in ann.l0.subjects]
    ann.l0.signals = [canonicalize_signal(s) for s in ann.l0.signals]

    # L1
    ann.l1.scene = canonicalize_scene(ann.l1.scene)
    for subj in ann.l1.subjects:
        subj.label = canonicalize_subject(subj.label)
    ann.l1.signals = [canonicalize_signal(s) for s in ann.l1.signals]
    ann.l1.quality_flags = [canonicalize_signal(s) for s in ann.l1.quality_flags]

    # L2 (if present)
    if ann.l2 is not None:
        for obj in ann.l2.objects:
            obj.label = canonicalize_subject(obj.label)
        for rel in ann.l2.relations:
            rel.predicate = canonicalize_predicate(rel.predicate)

    return ann
