"""
Default backend: Molmo2-8B (Allen Institute, open weights, runs locally).

Hardware requirements: ~22 GB VRAM in bf16 (fits on dual RTX 2080 Ti or one A100).
Latency: ~5–8 s per frame on dual 2080 Ti.
License: Apache 2.0 (model weights), CC BY 4.0 (data).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .base import BaseBehaviorBackend


class Molmo2Backend(BaseBehaviorBackend):
    name = "molmo2"

    def __init__(self, model_name: str = "allenai/Molmo2-8B",
                 device_map: str = "auto"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[backend molmo2] Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        devices = sorted(set(str(p.device) for p in self.model.parameters()))
        print(f"[backend molmo2] Loaded on {devices}")

    def analyze_frame(self, image_pil: Image.Image, prompt: str) -> str:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=text, images=[image_pil], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        generated = output[0, input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def summarize_video(self, timeline_text: str,
                         reference_image_pil: Optional[Image.Image] = None) -> str:
        prompt = (
            "You are a wildlife biologist. Below is a timeline of feeding behavior "
            "for brown bears at Brooks Falls, Alaska, observed over a short video.\n\n"
            f"{timeline_text}\n\n"
            "Write a 2–4 sentence narrative summary of the whole video. Mention how "
            "many salmon were caught, which bears were active, and any notable events."
        )
        if reference_image_pil is None:
            # Some chat templates require an image; pass a tiny blank if none given.
            reference_image_pil = Image.new("RGB", (224, 224), color=(0, 0, 0))
        return self.analyze_frame(reference_image_pil, prompt)

    def close(self) -> None:
        del self.model, self.processor
        torch.cuda.empty_cache()
