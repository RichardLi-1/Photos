"""Shared bits for the SigLIP2-B/16-384 -> ONNX (Android) scripts. Offline only, guarded (see search/photosearch/guard.py).

Run with the search venv (it has open_clip + timm + transformers) and private ONNX libs in ml/android/.pylib
(kept out of the shared venvs):
    search/.venv/bin/python -m pip install --target ml/android/.pylib onnx onnxscript onnxruntime
    rm -rf ml/android/.pylib/{numpy,sympy,mpmath,packaging,typing_extensions}*   # use the venv's copies
    PYTHONPATH=ml/android/.pylib search/.venv/bin/python ml/android/<script>.py
"""
from __future__ import annotations

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("GUARD_MIN_DISK_GB", "10")  # repo rule: stop if free disk < 10 GB

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "search"))
from photosearch import guard  # noqa: E402  (read-only import; owned by search/)

ARCH, PRETRAINED, HF_REPO = "ViT-B-16-SigLIP2-384", "webli", "timm/ViT-B-16-SigLIP2-384"
IMAGE_SIZE, CONTEXT_LENGTH, DIM, VOCAB = 384, 64, 768, 256000
ASSETS = os.path.join(ROOT, "android", "app", "src", "main", "assets", "models")  # small files, always in the APK
BIG_ASSETS = os.path.join(ROOT, "android", "app", "bigassets", "models")  # ~390 MB; in the APK unless -PbundleModels=false
TEST_RES = os.path.join(ROOT, "android", "app", "src", "test", "resources")
WORK = os.path.join(HERE, "work")  # fp32 intermediates (gitignored, deleted at the end)
DEMO = os.path.join(ROOT, "search", "data", "demo", "unsplash")

PARITY_TEXTS = [
    "a photo of a dog", "Sunset over the BEACH!", "café menu with prices: $12.50", "東京タワー at night",
    "  multiple   spaces\tand\nnewlines ", "emoji 🐶🌅 party", "receipt_from_costco",
    "the quick brown fox jumps over the lazy dog " * 8, "naïve résumé", "x",
    "my passport", "kids playing in the snow", "wifi password on a whiteboard", "red vintage car parked by the sea",
    "birthday cake with candles", "hotel room with two beds", "screenshot of a boarding pass", "Eiffel Tower",
    "a plate of sushi", "foggy mountains at sunrise",
]
TOKENIZER_EXTRA = [  # more edge cases for the Kotlin tokenizer test (curly quotes, entities, fullwidth, ligatures, NBSP)
    "it’s Mike’s “dog”", "Tom &amp; Jerry &lt;3", "ＡＢＣ　ｆｕｌｌｗｉｄｔｈ", "ﬁsh and ﬂowers", "non breaking space",
    "ÜBER straße ΣΊΣΥΦΟΣ", "Ελληνικά και русский текст", "😀", "a-b_c.d/e", "",
]


def load_model():
    """open_clip SigLIP2 on CPU, fp32, eval. ~1.5 GB RAM."""
    import open_clip
    import torch
    torch.set_grad_enabled(False)
    torch.backends.mha.set_fastpath_enabled(False)
    model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED, device="cpu")
    return model.eval(), preprocess


def tokenizer():
    import open_clip
    return open_clip.get_tokenizer(ARCH)  # ground truth: exactly what the Python prototype uses


def demo_images(n: int = 20) -> list[str]:
    names = sorted(f for f in os.listdir(DEMO) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return [os.path.join(DEMO, f) for f in names[:n]]
