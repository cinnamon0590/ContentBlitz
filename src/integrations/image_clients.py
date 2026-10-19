"""Image generation clients and the image fallback chain.

Chain (``config/services.yaml``):
``gpt-image-1 → gpt-image-1-mini → gemini-2.5-flash-image → placeholder``.

* ``gpt-image-1`` needs a verified OpenAI organization. If the key isn't
  verified, the call fails and the chain moves on to ``gpt-image-1-mini``.
* ``gemini-2.5-flash-image`` (Google) is the cross-provider fallback: an OpenAI
  outage or billing problem no longer leaves only the placeholder. It needs a
  ``GOOGLE_API_KEY`` on a billing-enabled project (the free tier allows 0 images).
* DALL-E 2/3 are retired by OpenAI and no longer part of the chain.
* ``placeholder`` is a branded, Pillow-rendered card (never fails), so the
  workflow always returns an image and records why the real providers failed.

Every image is validated with Pillow (decodable, big enough) and normalized to
PNG before it is saved.
"""

from __future__ import annotations

import base64
import io
import textwrap
from dataclasses import dataclass, field
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont

from src.integrations import request_json
from src.utils.observability import CostTracker, get_logger

log = get_logger("images")

PROMPT_LIMIT = 32000  # gpt-image family
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_ASPECT_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")


class ImageGenerationError(RuntimeError):
    pass


@dataclass
class ImageResult:
    png_bytes: bytes
    provider: str
    width: int
    height: int
    attempts: list[str] = field(default_factory=list)


def validate_image(data: bytes, min_dimension: int = 256) -> tuple[bytes, int, int]:
    """Decode and sanity-check image bytes; return ``(png_bytes, width, height)``."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = img.size
            if min(width, height) < min_dimension:
                raise ImageGenerationError(f"image too small: {width}x{height}")
            out = io.BytesIO()
            img.convert("RGB").save(out, format="PNG", optimize=True)
            return out.getvalue(), width, height
    except ImageGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001 - PIL raises many types
        raise ImageGenerationError(f"invalid image data: {exc}") from exc


class OpenAIImageClient:
    """Wraps ``client.images.generate`` for the gpt-image family."""

    def __init__(self, api_key: str | None, client: Any = None, quality: str | None = "medium"):
        self.api_key = api_key
        self.quality = quality
        self._client = client

    @property
    def available(self) -> bool:
        return bool(self.api_key or self._client)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, model: str, prompt: str, size: str) -> bytes:
        if not self.available:
            raise ImageGenerationError("OPENAI_API_KEY is not set")
        kwargs: dict[str, Any] = {"model": model, "prompt": prompt[:PROMPT_LIMIT], "size": size, "n": 1}
        if self.quality:
            kwargs["quality"] = self.quality
        response = self._get_client().images.generate(**kwargs)
        item = response.data[0]
        if getattr(item, "b64_json", None):
            return base64.b64decode(item.b64_json)
        url = getattr(item, "url", None)
        if url:  # pragma: no cover - gpt-image models return base64
            import requests

            return requests.get(url, timeout=60).content
        raise ImageGenerationError("empty image response")


def aspect_ratio(size: str) -> str:
    """'1536x1024' → '3:2': the closest aspect ratio Gemini supports (it takes ratios, not pixel sizes)."""
    try:
        width, height = (int(x) for x in size.lower().split("x"))
    except ValueError:
        return size if size in GEMINI_ASPECT_RATIOS else "1:1"
    target = width / height

    def distance(ratio: str) -> float:
        w, h = (int(x) for x in ratio.split(":"))
        return abs(w / h - target)

    return min(GEMINI_ASPECT_RATIOS, key=distance)


class GeminiImageClient:
    """Google Gemini image models over the REST API (no SDK needed)."""

    def __init__(self, api_key: str | None, timeout: float = 120, session: requests.Session | None = None):
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, model: str, prompt: str, size: str) -> bytes:
        if not self.available:
            raise ImageGenerationError("GOOGLE_API_KEY is not set")
        data = request_json(
            "gemini", "POST", GEMINI_URL.format(model=model), session=self.session, timeout=self.timeout,
            headers={"x-goog-api-key": self.api_key},
            json={"contents": [{"parts": [{"text": prompt[:PROMPT_LIMIT]}]}],
                  "generationConfig": {"responseModalities": ["IMAGE"],
                                       "imageConfig": {"aspectRatio": aspect_ratio(size)}}},
        )
        candidate = (data.get("candidates") or [{}])[0]
        for part in (candidate.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
        reason = candidate.get("finishReason") or (data.get("promptFeedback") or {}).get("blockReason") or "no image"
        raise ImageGenerationError(f"Gemini returned no image ({reason})")


class ImageClients:
    """Routes each model in the chain to its provider's client (``gemini-*`` → Google, else OpenAI)."""

    def __init__(self, openai: Any, google: Any = None):
        self.openai = openai
        self.google = google

    @property
    def available(self) -> bool:
        return bool(self.openai.available or (self.google and self.google.available))

    def for_model(self, model: str) -> Any:
        if model.startswith("gemini"):
            if self.google is None:
                raise ImageGenerationError("no Google image client configured")
            return self.google
        return self.openai

    def generate(self, model: str, prompt: str, size: str) -> bytes:
        return self.for_model(model).generate(model, prompt, size)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        return (31, 78, 121)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def render_placeholder(title: str, palette: list[str] | None = None, size: tuple[int, int] = (1536, 1024),
                       brand: str = "") -> bytes:
    """Branded gradient card with the topic title. This is the last link in the image fallback chain."""
    palette = palette or ["#1F4E79", "#F2A900", "#FFFFFF"]
    start, end = _hex_to_rgb(palette[0]), _hex_to_rgb(palette[1] if len(palette) > 1 else palette[0])
    width, height = size
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(height - 1, 1)
        color = tuple(int(start[i] + (end[i] - start[i]) * t) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color)
    text_color = _hex_to_rgb(palette[2]) if len(palette) > 2 else (255, 255, 255)
    try:
        font = ImageFont.load_default(size=max(28, width // 22))
        small = ImageFont.load_default(size=max(16, width // 60))
    except TypeError:  # pragma: no cover - very old Pillow
        font = small = ImageFont.load_default()
    lines = textwrap.wrap(title.strip() or "ContentBlitz", width=28)[:4]
    y = height // 2 - len(lines) * (width // 22) // 2
    for line in lines:
        w = draw.textlength(line, font=font)
        draw.text(((width - w) / 2, y), line, fill=text_color, font=font)
        y += int(width // 22 * 1.3)
    footer = f"{brand} · placeholder image (image service unavailable)" if brand else "placeholder image"
    draw.text((24, height - 24 - width // 60), footer, fill=text_color, font=small)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


class ImageChain:
    """Ordered image-provider fallback."""

    def __init__(self, client: OpenAIImageClient, chain: list[str], sizes: dict[str, str] | None = None,
                 tracker: CostTracker | None = None, min_dimension: int = 256):
        self.client = client
        self.chain = chain
        self.sizes = sizes or {}
        self.tracker = tracker
        self.min_dimension = min_dimension

    def generate(self, prompt: str, title: str = "", palette: list[str] | None = None,
                 brand: str = "") -> ImageResult:
        attempts: list[str] = []
        for model in self.chain:
            if model == "placeholder":
                png, w, h = validate_image(render_placeholder(title or prompt[:80], palette, brand=brand),
                                           self.min_dimension)
                if self.tracker:
                    self.tracker.record_image("placeholder")
                return ImageResult(png, "placeholder", w, h, attempts)
            try:
                raw = self.client.generate(model, prompt, self.sizes.get(model, "1024x1024"))
                png, w, h = validate_image(raw, self.min_dimension)
                if self.tracker:
                    self.tracker.record_image(model)
                return ImageResult(png, model, w, h, attempts)
            except Exception as exc:  # noqa: BLE001 - any failure → next provider
                attempts.append(f"{model}: {str(exc)[:160]}")
                log.warning("image provider %s failed: %s", model, exc)
        raise ImageGenerationError("all image providers failed: " + "; ".join(attempts))
