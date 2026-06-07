# Installation

This guide sets up the Katmai CV Pipeline from scratch on a fresh machine.

---

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.10 (other versions untested) |
| GPU | None (CPU fallback) | NVIDIA 8 GB+ VRAM (CUDA 12.x) |
| RAM | 8 GB | 16 GB+ |
| Disk | 5 GB | 20 GB (models + data) |
| OS | macOS / Windows / Linux | Ubuntu 22.04 or macOS 14+ |

See [Hardware Guide](hardware-guide.md) for GPU/CPU/cloud tradeoffs before proceeding.

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/katmai-vision-lab/katmai-cv-pipeline.git
cd katmai-cv-pipeline
```

---

## Step 2 — Create the Python environment

**Conda (recommended):**
```bash
conda create -n katmai python=3.10 -y
conda activate katmai
```

**venv:**
```bash
python3.10 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

---

## Step 3 — Install PyTorch

Install the version that matches your hardware. **Do this before `requirements.txt`.**

**NVIDIA GPU (Linux / Windows):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon (M1/M2/M3/M4):**
```bash
pip install torch torchvision
```

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available(), getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
```

---

## Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Critical:** `transformers==4.47.1` is pinned. v5.x has breaking changes that will break Grounding DINO and OWL-ViT. Do not upgrade.

If you see errors about `huggingface_hub`, also pin it:
```bash
pip install huggingface-hub==0.36.2
```

---

## Step 5 — Install system dependencies

**FFmpeg** is required for rendering annotated output videos:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH
```

---

## Step 6 — Download model weights

The fine-tuned bear detector weights are stored in `models/trained/bear_detector3/weights/best.pt`. They should already be present in the repo (tracked via Git LFS or included directly).

Check:
```bash
ls models/trained/bear_detector3/weights/
# should show: best.pt  last.pt
```

If missing, ask a team member for the SharePoint link and place them at that path.

---

## Step 7 — (Optional) Set up VLM API keys

The feeding behavior module defaults to Molmo2-8B running locally (requires ~16 GB RAM or GPU). To use a cloud backend instead:

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # Claude (Sonnet)
export OPENAI_API_KEY=sk-...           # GPT-4o
export GOOGLE_API_KEY=...              # Gemini 2.0 Flash
```

Add these to your shell profile (`~/.zshrc`, `~/.bashrc`) to persist across sessions.

---

## Step 8 — Verify the install

```bash
python -m src.cli
```

You should see the interactive menu. Press `q` to quit.

For a quick non-interactive smoke test:
```bash
python -c "from src.detection.detector import BearDetector; d = BearDetector(); print('OK')"
```

---

## Common install issues

**`ModuleNotFoundError: No module named 'groundingdino'`**  
Run: `pip install groundingdino-py` or check if the package installed as `groundingdino` vs `groundingdino_py`.

**`transformers` version conflict**  
Force: `pip install transformers==4.47.1 --force-reinstall`

**CUDA not detected on Windows**  
Ensure you have the matching CUDA toolkit from https://developer.nvidia.com/cuda-downloads and that `nvcc --version` matches your PyTorch build.

**`MPS` device errors on macOS**  
Some models fall back to CPU on MPS. This is expected — the pipeline handles it automatically.

---

Next: [Your First Run →](first-run.md)
