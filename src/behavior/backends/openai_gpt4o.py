"""
OpenAI backend (GPT-4o or any other vision-capable OpenAI model).

Setup:
    pip install openai
    export OPENAI_API_KEY=sk-...

Pros: highest accuracy, no GPU needed locally.
Cons: per-frame API cost (~$0.01–0.03 per frame at 1024×1024); needs internet;
      sponsor must accept that footage is uploaded to OpenAI.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Optional

from PIL import Image

from .base import BaseBehaviorBackend


def _pil_to_data_url(image_pil: Image.Image, max_side: int = 1024) -> str:
    """Encode a PIL image as a base64 data URL, downscaled to max_side."""
    img = image_pil.copy()
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class OpenAIBackend(BaseBehaviorBackend):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Run `pip install openai`."
            ) from e
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key=..."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"[backend openai] Using model {model}")

    def _chat(self, image_pil: Image.Image, prompt: str,
              max_tokens: int = 400) -> str:
        data_url = _pil_to_data_url(image_pil)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

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
        # Text-only fallback
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
