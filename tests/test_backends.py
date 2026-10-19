"""Tests for hidl.backends – image encoding, MIME detection and the OpenAI backend's wiring.

Nothing here calls a model. The OpenAI backend is driven through a stub
client that records the request it would have sent, so what is pinned is
the shape of that request (a URL passed through, bytes and files as a
data URI with the right MIME type), never a vendor's reply.
"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

import pytest

from hidl.backends import OpenAIVisionBackend, _detect_mime, _encode_image

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
GIF = b"GIF89a" + b"\x00" * 8
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4


# ---------------------------------------------------------------------------
# _encode_image
# ---------------------------------------------------------------------------


class TestEncodeImage:
    def test_https_url_passes_through(self):
        assert _encode_image("https://example.org/a.jpg") == ("url", "https://example.org/a.jpg")

    def test_http_url_passes_through(self):
        assert _encode_image("http://example.org/a.jpg") == ("url", "http://example.org/a.jpg")

    def test_bytes_are_base64(self):
        kind, data = _encode_image(JPEG)
        assert kind == "base64"
        assert base64.b64decode(data) == JPEG

    def test_str_path_is_read_and_encoded(self, tmp_path: Path):
        file = tmp_path / "a.png"
        file.write_bytes(PNG)
        kind, data = _encode_image(str(file))
        assert kind == "base64"
        assert base64.b64decode(data) == PNG

    def test_path_object_is_read_and_encoded(self, tmp_path: Path):
        file = tmp_path / "a.png"
        file.write_bytes(PNG)
        assert _encode_image(file) == _encode_image(str(file))

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _encode_image(tmp_path / "absent.jpg")


# ---------------------------------------------------------------------------
# _detect_mime
# ---------------------------------------------------------------------------


class TestDetectMime:
    @pytest.mark.parametrize(
        ("data", "mime"),
        [
            (JPEG, "image/jpeg"),
            (PNG, "image/png"),
            (GIF, "image/gif"),
            (b"GIF87a" + b"\x00" * 8, "image/gif"),
            (WEBP, "image/webp"),
        ],
    )
    def test_known_magic_bytes(self, data: bytes, mime: str):
        assert _detect_mime(data) == mime

    def test_unknown_bytes_fall_back(self):
        assert _detect_mime(b"not an image") == "application/octet-stream"

    def test_riff_without_webp_is_not_webp(self):
        assert _detect_mime(b"RIFF\x00\x00\x00\x00WAVE") == "application/octet-stream"


# ---------------------------------------------------------------------------
# OpenAIVisionBackend – constructed against a stub `openai` module
# ---------------------------------------------------------------------------


class _StubCompletions:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = types.SimpleNamespace(content=self.reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class _StubClient:
    def __init__(self, api_key: str, reply: str = "{}") -> None:
        self.api_key = api_key
        self.chat = types.SimpleNamespace(completions=_StubCompletions(reply))


@pytest.fixture
def stub_openai(monkeypatch: pytest.MonkeyPatch):
    """Install a stand-in `openai` module whose client records each request."""
    clients: list[_StubClient] = []

    def make(api_key: str) -> _StubClient:
        client = _StubClient(api_key)
        clients.append(client)
        return client

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=make))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return clients


class TestOpenAIVisionBackendConstruction:
    def test_missing_package_raises_import_error(self, monkeypatch: pytest.MonkeyPatch):
        # A None entry in sys.modules makes `import openai` raise ImportError.
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="pip install openai"):
            OpenAIVisionBackend(api_key="sk-test")

    def test_missing_key_raises_value_error(self, stub_openai):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIVisionBackend()

    def test_key_from_environment(self, stub_openai, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        OpenAIVisionBackend()
        assert stub_openai[0].api_key == "sk-from-env"

    def test_explicit_key_wins_over_environment(
        self, stub_openai, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        OpenAIVisionBackend(api_key="sk-explicit")
        assert stub_openai[0].api_key == "sk-explicit"


class TestOpenAIVisionBackendCall:
    def _request(self, stub_openai, image, **kwargs) -> dict:
        backend = OpenAIVisionBackend(api_key="sk-test", **kwargs)
        reply = backend.call("SYSTEM", "USER", image)
        assert reply == "{}"
        requests = stub_openai[0].chat.completions.requests
        assert len(requests) == 1
        return requests[0]

    def test_url_image_is_sent_as_url(self, stub_openai):
        request = self._request(stub_openai, "https://example.org/a.jpg")
        image_part = request["messages"][1]["content"][1]
        assert image_part == {
            "type": "image_url",
            "image_url": {"url": "https://example.org/a.jpg"},
        }

    def test_bytes_image_is_sent_as_data_uri_with_mime(self, stub_openai):
        request = self._request(stub_openai, PNG)
        url = request["messages"][1]["content"][1]["image_url"]["url"]
        prefix = "data:image/png;base64,"
        assert url.startswith(prefix)
        assert base64.b64decode(url[len(prefix):]) == PNG

    def test_file_image_is_sent_as_data_uri_with_mime(self, stub_openai, tmp_path: Path):
        file = tmp_path / "a.jpg"
        file.write_bytes(JPEG)
        request = self._request(stub_openai, file)
        url = request["messages"][1]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_prompts_model_and_sampling_are_forwarded(self, stub_openai):
        request = self._request(
            stub_openai, JPEG, model="gpt-4o-mini", max_tokens=99, temperature=0.5
        )
        assert request["model"] == "gpt-4o-mini"
        assert request["max_tokens"] == 99
        assert request["temperature"] == 0.5
        assert request["messages"][0] == {"role": "system", "content": "SYSTEM"}
        assert request["messages"][1]["content"][0] == {"type": "text", "text": "USER"}

    def test_empty_reply_becomes_empty_string(self, stub_openai):
        backend = OpenAIVisionBackend(api_key="sk-test")
        stub_openai[0].chat.completions.reply = None
        assert backend.call("SYSTEM", "USER", JPEG) == ""
