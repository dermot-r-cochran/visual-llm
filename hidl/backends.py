"""Vision LLM backend abstraction.

Provides an abstract base class :class:`VisionBackend` and a concrete
:class:`OpenAIVisionBackend` implementation.  Additional backends (Anthropic,
Google Gemini, etc.) can be added by subclassing :class:`VisionBackend`.
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


# Type alias for an image source: file path, bytes, or a URL string.
ImageSource = Union[str, Path, bytes]


class VisionBackend(ABC):
    """Abstract base class for vision LLM backends.

    Subclasses must implement :meth:`call` which sends a system prompt,
    a user text message, and an image to the underlying model and returns
    the raw string response.
    """

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_message: str,
        image: ImageSource,
    ) -> str:
        """Send a vision request to the LLM and return the raw response.

        Args:
            system_prompt: The HIDL system prompt string.
            user_message: The per-image user message string.
            image: Image as a file path, raw bytes, or a URL string starting
                with ``http://`` or ``https://``.

        Returns:
            Raw response string from the model (may include markdown fences).
        """


def _encode_image(image: ImageSource) -> tuple[str, str]:
    """Encode an image as a base64 data URI or return its URL.

    Args:
        image: A file path, raw bytes, or a URL string.

    Returns:
        A ``(type, data)`` tuple where *type* is ``"base64"`` or ``"url"``
        and *data* is the encoded string or URL.
    """
    if isinstance(image, (str, Path)) and str(image).startswith(("http://", "https://")):
        return "url", str(image)

    if isinstance(image, (str, Path)):
        image_bytes = Path(image).read_bytes()
    else:
        image_bytes = image

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return "base64", encoded


class OpenAIVisionBackend(VisionBackend):
    """Vision backend that uses the OpenAI API (GPT-4o or compatible model).

    Requires the ``openai`` package to be installed::

        pip install openai

    The API key is read from the ``OPENAI_API_KEY`` environment variable or
    can be passed directly.

    Args:
        model: OpenAI model name (default: ``"gpt-4o"``).
        api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable if not provided.
        max_tokens: Maximum number of tokens for the completion.
        temperature: Sampling temperature.  Use 0 for deterministic output.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        try:
            import openai  # noqa: PLC0415  (lazy import for optional dep)
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIVisionBackend. "
                "Install it with: pip install openai"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An OpenAI API key must be provided via the 'api_key' argument "
                "or the OPENAI_API_KEY environment variable."
            )

        self._client = openai.OpenAI(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def call(
        self,
        system_prompt: str,
        user_message: str,
        image: ImageSource,
    ) -> str:
        img_type, img_data = _encode_image(image)

        if img_type == "url":
            image_content: dict = {
                "type": "image_url",
                "image_url": {"url": img_data},
            }
        else:
            # Detect MIME type from magic bytes.
            raw_bytes: bytes
            if isinstance(image, bytes):
                raw_bytes = image
            else:
                raw_bytes = Path(str(image)).read_bytes()
            mime = _detect_mime(raw_bytes)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{img_data}",
                },
            }

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        image_content,
                    ],
                },
            ],
        )
        return response.choices[0].message.content or ""


def _detect_mime(data: bytes) -> str:
    """Detect image MIME type from magic bytes.

    Args:
        data: Raw image bytes.

    Returns:
        A MIME type string such as ``"image/jpeg"`` or ``"image/png"``.
    """
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] in (b"GIF8", b"GIF9"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    # Fall back to octet-stream; most APIs handle unknown types gracefully.
    return "application/octet-stream"
