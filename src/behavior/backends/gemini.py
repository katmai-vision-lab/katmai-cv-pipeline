"""
Google Gemini backend (Gemini 1.5 Pro / Flash, with vision).

Setup:
    pip install google-generativeai
    export GOOGLE_API_KEY=...        # or GEMINI_API_KEY

Pros: 1M-token context (can ingest very long video natively); fast Flash variant
      is cheap; students get free quota.
Cons: footage uploaded to Google; rate-limited; reproducibility weaker than
      local models (sampling not fully deterministic).
"""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image

from .base import BaseBehaviorBackend


class GeminiBackend(BaseBehaviorBackend):
    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-pro",
                 api_key: Optional[str] = None):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai not installed. Run `pip install google-generativeai`."
            ) from e
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY env var."
            )
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = genai.GenerativeModel(model)
        print(f"[backend gemini] Using model {model}")

    def analyze_frame(self, image_pil: Image.Image, prompt: str) -> str:
        resp = self.model.generate_content(
            [prompt, image_pil],
            generation_config={"temperature": 0.0, "max_output_tokens": 300},
        )
        return resp.text.strip() if resp.text else ""

    def summarize_video(self, timeline_text: str,
                         reference_image_pil: Optional[Image.Image] = None) -> str:
        prompt = (
            "You are a wildlife biologist. Below is a timeline of feeding behavior "
            "for brown bears at Brooks Falls, Alaska:\n\n"
            f"{timeline_text}\n\n"
            "Write a 2–4 sentence narrative summary."
        )
        parts = [prompt]
        if reference_image_pil is not None:
            parts.append(reference_image_pil)
        resp = self.model.generate_content(
            parts,
            generation_config={"temperature": 0.0, "max_output_tokens": 300},
        )
        return resp.text.strip() if resp.text else ""
