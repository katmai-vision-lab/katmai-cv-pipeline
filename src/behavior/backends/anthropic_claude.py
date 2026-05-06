"""
Anthropic backend (Claude with vision — Sonnet/Opus 4.x).

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Pros: very strong reasoning quality, good at structured output.
Cons: per-frame cost, internet required, footage uploaded to Anthropic.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Optional

from PIL import Image

from .base import BaseBehaviorBackend


def _pil_to_b64(image_pil: Image.Image, max_side: int = 1568) -> tuple[str, str]:
    img = image_pil.copy()
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


class AnthropicBackend(BaseBehaviorBackend):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6",
                 api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Run `pip install anthropic`."
            ) from e
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY env var or pass api_key=..."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        print(f"[backend anthropic] Using model {model}")

    def _chat(self, image_pil: Image.Image, prompt: str,
              max_tokens: int = 400) -> str:
        b64, media_type = _pil_to_b64(image_pil)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return resp.content[0].text.strip()

    def analyze_frame(self, image_pil: Image.Image, prompt: str) -> str:
        return self._chat(image_pil, prompt, max_tokens=300)

    def summarize_video(self, timeline_text: str,
                         reference_image_pil: Optional[Image.Image] = None) -> str:
        prompt = (
            "You are a wildlife biologist. Below is a timeline of feeding behavior "
            "for brown bears at Brooks Falls, Alaska:\n\n"
            f"{timeline_text}\n\n"
            "Write a 2–4 sentence narrative summary."
        )
        if reference_image_pil is not None:
            return self._chat(reference_image_pil, prompt, max_tokens=300)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
