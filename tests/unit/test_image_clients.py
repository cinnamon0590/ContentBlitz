import base64
import io

import pytest
from PIL import Image

from src.integrations.image_clients import (
    ImageChain,
    ImageGenerationError,
    OpenAIImageClient,
    render_placeholder,
    validate_image,
)
from src.utils.observability import CostTracker
from tests.conftest import FakeImageClient

SIZES = {"gpt-image-1": "1536x1024", "gpt-image-1-mini": "1536x1024"}
CHAIN = ["gpt-image-1", "gpt-image-1-mini", "placeholder"]


def test_primary_success():
    tracker = CostTracker()
    result = ImageChain(FakeImageClient(), CHAIN, SIZES, tracker).generate("a cafe")
    assert result.provider == "gpt-image-1" and result.attempts == []
    assert tracker.images == {"gpt-image-1": 1}


def test_falls_back_to_mini():
    client = FakeImageClient(fail_models={"gpt-image-1"})
    result = ImageChain(client, CHAIN, SIZES).generate("x")
    assert result.provider == "gpt-image-1-mini" and len(result.attempts) == 1


def test_placeholder_when_everything_fails():
    client = FakeImageClient(fail_models={"gpt-image-1", "gpt-image-1-mini"})
    result = ImageChain(client, CHAIN, SIZES).generate("x", title="Coffee Trends 2026", brand="Bean Co")
    assert result.provider == "placeholder" and len(result.attempts) == 2
    assert Image.open(io.BytesIO(result.png_bytes)).format == "PNG"


def test_chain_without_placeholder_raises():
    with pytest.raises(ImageGenerationError):
        ImageChain(FakeImageClient({"gpt-image-1"}), ["gpt-image-1"], SIZES).generate("x")


def test_validate_rejects_small_and_garbage():
    buf = io.BytesIO()
    Image.new("RGB", (50, 50)).save(buf, format="PNG")
    with pytest.raises(ImageGenerationError, match="too small"):
        validate_image(buf.getvalue())
    with pytest.raises(ImageGenerationError, match="invalid"):
        validate_image(b"not an image")


def test_placeholder_bad_hex_falls_back():
    png = render_placeholder("t", palette=["zzz"], size=(400, 300))
    assert Image.open(io.BytesIO(png)).size == (400, 300)


class _Item:
    def __init__(self, b64):
        self.b64_json = b64
        self.url = None


class _Images:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        buf = io.BytesIO()
        Image.new("RGB", (300, 300)).save(buf, format="PNG")

        class R:
            data = [_Item(base64.b64encode(buf.getvalue()).decode())]

        return R()


class _SDK:
    def __init__(self):
        self.images = _Images()


def test_openai_image_client_params():
    sdk = _SDK()
    client = OpenAIImageClient("k", client=sdk)
    assert client.generate("gpt-image-1", "p" * 40000, "1536x1024")
    kwargs = sdk.images.kwargs
    assert kwargs["quality"] == "medium" and len(kwargs["prompt"]) == 32000 and "response_format" not in kwargs
    OpenAIImageClient("k", client=sdk, quality=None).generate("gpt-image-1-mini", "p", "1536x1024")
    assert "quality" not in sdk.images.kwargs
    with pytest.raises(ImageGenerationError):
        OpenAIImageClient(None).generate("gpt-image-1", "p", "1536x1024")


# ---- Google Gemini (cross-provider fallback) -------------------------------------------
def _png_b64(size=(600, 400)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 120, 40)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_gemini_client_request_and_parsing():
    from src.integrations.image_clients import GeminiImageClient
    from tests.unit.test_search_clients import fake_session

    payload = {"candidates": [{"content": {"parts": [{"text": "here you go"},
                                                     {"inlineData": {"mimeType": "image/png", "data": _png_b64()}}]},
                               "finishReason": "STOP"}]}
    session = fake_session(payload)
    raw = GeminiImageClient("k", session=session).generate("gemini-2.5-flash-image", "p" * 40000, "1536x1024")
    assert Image.open(io.BytesIO(raw)).size == (600, 400)
    call = session.request.call_args
    assert call.args[1].endswith("/models/gemini-2.5-flash-image:generateContent")
    body = call.kwargs["json"]
    assert body["generationConfig"] == {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "3:2"}}
    assert len(body["contents"][0]["parts"][0]["text"]) == 32000 and call.kwargs["headers"]["x-goog-api-key"] == "k"


def test_gemini_client_errors():
    from src.integrations.image_clients import GeminiImageClient
    from tests.unit.test_search_clients import fake_session

    blocked = {"candidates": [{"content": {"parts": [{"text": "I can't draw that"}]}, "finishReason": "IMAGE_SAFETY"}]}
    with pytest.raises(ImageGenerationError, match="IMAGE_SAFETY"):
        GeminiImageClient("k", session=fake_session(blocked)).generate("gemini-x", "p", "1024x1024")
    with pytest.raises(ImageGenerationError, match="PROHIBITED"):
        GeminiImageClient("k", session=fake_session({"promptFeedback": {"blockReason": "PROHIBITED"}})).generate(
            "gemini-x", "p", "1024x1024")
    with pytest.raises(ImageGenerationError, match="GOOGLE_API_KEY"):
        GeminiImageClient(None).generate("gemini-x", "p", "1024x1024")


def test_aspect_ratio_mapping():
    from src.integrations.image_clients import aspect_ratio

    assert [aspect_ratio(s) for s in ("1536x1024", "1024x1024", "1024x1536", "1920x1080")] == [
        "3:2", "1:1", "2:3", "16:9"]
    assert aspect_ratio("4:3") == "4:3" and aspect_ratio("weird") == "1:1"


def test_image_clients_route_by_model_and_chain_reaches_gemini():
    from src.integrations.image_clients import ImageClients

    openai, google = FakeImageClient(fail_models={"gpt-image-1", "gpt-image-1-mini"}), FakeImageClient()
    clients = ImageClients(openai, google)
    chain = ["gpt-image-1", "gpt-image-1-mini", "gemini-2.5-flash-image", "placeholder"]
    result = ImageChain(clients, chain, {**SIZES, "gemini-2.5-flash-image": "1536x1024"}).generate("x")
    assert result.provider == "gemini-2.5-flash-image" and len(result.attempts) == 2
    assert google.calls == ["gemini-2.5-flash-image"] and "gemini-2.5-flash-image" not in openai.calls
    assert clients.available
    with pytest.raises(ImageGenerationError, match="no Google image client"):
        ImageClients(openai).generate("gemini-2.5-flash-image", "x", "1024x1024")
