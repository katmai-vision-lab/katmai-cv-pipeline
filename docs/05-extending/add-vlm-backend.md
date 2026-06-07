# Add a New VLM Backend

The feeding behavior module uses a pluggable backend system. Any vision-language model — local or API-based — can be added by implementing one abstract class.

---

## How backends work

`src/behavior/backends/base.py` defines `BaseBehaviorBackend`:

```python
class BaseBehaviorBackend:
    def analyze_frame(
        self,
        image_pil: Image.Image,
        prompt: str,
        reference_image_pil: Optional[Image.Image] = None,
    ) -> str:
        """Classify behavior in a single frame. Returns a string description."""
        raise NotImplementedError

    def summarize_video(
        self,
        timeline_str: str,
        reference_image_pil: Optional[Image.Image] = None,
    ) -> str:
        """Summarize a full video given a text timeline. Returns summary string."""
        raise NotImplementedError
```

That's the entire interface. Your backend receives a PIL image and a text prompt; it returns a string.

---

## Step 1 — Create the backend file

Create `src/behavior/backends/my_model.py`:

```python
from __future__ import annotations
from typing import Optional
from PIL import Image
from .base import BaseBehaviorBackend


class MyModelBackend(BaseBehaviorBackend):
    def __init__(self, model_path: str = "path/to/weights"):
        # load your model here, once
        self.model = load_my_model(model_path)

    def analyze_frame(
        self,
        image_pil: Image.Image,
        prompt: str,
        reference_image_pil: Optional[Image.Image] = None,
    ) -> str:
        # convert PIL → your model's input format
        result = self.model.classify(image_pil, prompt)
        # return a string — ideally in the pipeline's 5-stage format:
        # [WAITING] / [LUNGING] / [CATCHING] / [EATING] / [MISSED]
        return result

    def summarize_video(
        self,
        timeline_str: str,
        reference_image_pil: Optional[Image.Image] = None,
    ) -> str:
        return self.model.summarize(timeline_str)
```

---

## Step 2 — Register the backend

Open `src/behavior/analyze_feeding.py` and find the backend selection block (search for `"molmo2"`):

```python
if args.backend == "molmo2":
    from .backends.molmo2 import Molmo2Backend
    backend = Molmo2Backend()
elif args.backend == "anthropic":
    from .backends.anthropic_claude import AnthropicBackend
    backend = AnthropicBackend()
# ADD YOUR BACKEND HERE:
elif args.backend == "mymodel":
    from .backends.my_model import MyModelBackend
    backend = MyModelBackend()
```

---

## Step 3 — Use it

```bash
python -m src.behavior.analyze_feeding \
    --video path/to/clip.mp4 \
    --backend mymodel
```

---

## Output format convention

The pipeline parses behavior strings to extract the stage label. Follow this format for the labeled stage to appear in the viewer:

```
[STAGE] Description of what the bear is doing.
```

Valid stages: `WAITING`, `LUNGING`, `CATCHING`, `EATING`, `MISSED`.

If your model doesn't produce this format naturally, add a normalization step in `analyze_frame` that maps its output to one of these labels.

---

## Existing backends for reference

| Backend | File | Notes |
|---|---|---|
| Molmo2-8B (local) | `molmo2.py` | Default; ~16 GB RAM; free |
| Claude Sonnet | `anthropic_claude.py` | API; exponential backoff built in |
| GPT-4o | `openai_gpt4o.py` | API; fastest response time |
| Gemini 2.0 Flash | `gemini.py` | API; cheapest per frame |

`anthropic_claude.py` is the cleanest reference implementation — it handles image resizing, base64 encoding, retry logic, and structured output parsing.
