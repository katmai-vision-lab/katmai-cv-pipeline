"""
Abstract interface that every behavior-classification backend must implement.

A "backend" is whatever runs the actual vision-language reasoning over a frame
crop. The default is a local Molmo2-8B; alternatives include cloud APIs
(OpenAI GPT-4o, Anthropic Claude, Google Gemini) and other open-weights VLMs.

The pipeline only ever talks to the backend through these two methods:

    backend.analyze_frame(pil_image, prompt) -> raw_text_response
    backend.summarize_video(timeline_str, reference_pil) -> summary_text

The pipeline does NOT know whether the backend is local or remote, fast or slow,
free or metered — the abstraction is intentionally minimal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class BaseBehaviorBackend(ABC):
    """Subclass this to wire a new VLM into the bear-feeding pipeline."""

    #: Human-readable name shown in console / JSON output
    name: str = "base"

    @abstractmethod
    def analyze_frame(self, image_pil: Image.Image, prompt: str) -> str:
        """Run the model on one frame and return the raw text response.

        The pipeline expects the response to follow this format
        (one line per detected bear):

            Bear 1: [STAGE] short description of what the bear is doing
            Bear 2: [STAGE] ...

        where STAGE ∈ {WAITING, LUNGING, CATCHING, EATING, MISSED}.

        If your model returns something different, normalize it inside
        `analyze_frame` so the rest of the pipeline keeps working.
        """
        raise NotImplementedError

    @abstractmethod
    def summarize_video(
        self,
        timeline_text: str,
        reference_image_pil: Optional[Image.Image] = None,
    ) -> str:
        """Produce a 2–4 sentence narrative summary of the whole video.

        Inputs:
          - `timeline_text`: a multi-line string of "t=Xs Bear N: [STAGE] ..." entries
          - `reference_image_pil`: an optional middle-of-video frame the model can
            visually anchor against. Pass `None` for text-only models.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Optional teardown (e.g. release GPU memory, close API clients)."""
        pass
