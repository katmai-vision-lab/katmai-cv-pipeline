"""
Pluggable behavior-classification backends.

The pipeline (`analyze_feeding.py`) calls a backend through this interface:

    backend = get_backend("molmo2")           # or "openai", "anthropic", "gemini"
    text    = backend.analyze_frame(pil_img, prompt)
    summary = backend.summarize_video(timeline_str, reference_pil)

To add a new backend:
  1. subclass `BaseBehaviorBackend` (see `base.py`)
  2. register it in the `BACKEND_REGISTRY` dict below
  3. document required env vars / dependencies in the docstring
"""

from .base import BaseBehaviorBackend


def get_backend(name: str, **kwargs) -> BaseBehaviorBackend:
    """Factory that returns an instantiated backend by name."""
    name = name.lower().strip()
    if name == "molmo2":
        from .molmo2 import Molmo2Backend
        return Molmo2Backend(**kwargs)
    if name == "openai":
        from .openai_gpt4o import OpenAIBackend
        return OpenAIBackend(**kwargs)
    if name in ("anthropic", "claude"):
        from .anthropic_claude import AnthropicBackend
        return AnthropicBackend(**kwargs)
    if name == "gemini":
        from .gemini import GeminiBackend
        return GeminiBackend(**kwargs)
    raise ValueError(
        f"Unknown backend '{name}'. Supported: molmo2, openai, anthropic, gemini. "
        f"To add a new one, see src/behavior/backends/__init__.py."
    )


SUPPORTED_BACKENDS = ["molmo2", "openai", "anthropic", "gemini"]
