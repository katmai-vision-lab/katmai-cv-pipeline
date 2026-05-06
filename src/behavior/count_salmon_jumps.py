"""
Count salmon jumps in a video using Molmo2-8B's native video understanding.

Passes the entire video (or a trimmed clip) directly to the model,
rather than analyzing frame by frame.

Usage:
    venv/bin/python3 -m src.behavior.count_salmon_jumps \
        --video feed/data_video/salmon_jump_2_from4s.mov
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR


def resolve_video(path_str):
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    for candidate in [PROJECT_ROOT / p, RAW_DATA_DIR / p]:
        if candidate.exists():
            return candidate
    return p


def main():
    parser = argparse.ArgumentParser(description="Count salmon jumps via Molmo2-8B video input")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--vision-model", default="allenai/Molmo2-8B",
                        help="Vision model HuggingFace name")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt (default: salmon jump counting)")
    args = parser.parse_args()

    video_path = resolve_video(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    prompt = args.prompt or (
        "Watch this video of a river with salmon. "
        "Count how many distinct salmon jumps you see (a fish leaping above the water surface). "
        "Answer in this exact format:\n\n"
        "Total jumps: <number>\n"
        "Details:\n"
        "- Jump 1: <brief description, approximate time>\n"
        "- Jump 2: <brief description, approximate time>\n"
        "...\n\n"
        "Do NOT output point coordinates. Only output text."
    )

    print(f"Video : {video_path}")
    print(f"Model : {args.vision_model}")
    print(f"Prompt: {prompt[:80]}...")
    print()

    # Load model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print()

    print("Loading vision model...")
    processor = AutoProcessor.from_pretrained(args.vision_model, trust_remote_code=True)
    # Use high frame sampling so short clips get nearly every frame
    # Default max_fps=2.0 — ~10 frames fits in 22GB VRAM at typical resolution
    # Higher fps needs either lower resolution or more VRAM
    model = AutoModelForImageTextToText.from_pretrained(
        args.vision_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    devices = sorted(set(str(p.device) for p in model.parameters()))
    print(f"  Loaded on: {devices}\n")

    # Build messages with video input
    messages = [
        {"role": "user", "content": [
            {"type": "video", "path": str(video_path)},
            {"type": "text", "text": prompt},
        ]}
    ]

    # Process with chat template — handles video loading internally
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Running inference on full video...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    generated = output[0, input_len:]
    response = processor.decode(generated, skip_special_tokens=True).strip()

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()
