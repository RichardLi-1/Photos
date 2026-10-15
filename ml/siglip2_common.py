"""Shared bits for the SigLIP2-B/16-384 -> Core ML scripts (convert, tokenizer export, concepts, parity).

Everything runs offline from the Hugging Face cache (HF_HUB_OFFLINE=1). Heavy steps hold the machine-wide ML budget
from search/photosearch/guard.py (stdlib only), so they never run alongside other big model jobs.
"""
from __future__ import annotations

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("GUARD_MIN_DISK_GB", "10")  # repo rule: stop if free disk < 10 GB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "search"))
from photosearch import guard  # noqa: E402  (read-only import; owned by search/)

ARCH, PRETRAINED, HF_REPO = "ViT-B-16-SigLIP2-384", "webli", "timm/ViT-B-16-SigLIP2-384"
IMAGE_SIZE, CONTEXT_LENGTH, DIM = 384, 64, 768
OUT = os.path.join(ROOT, "ios", "Core", "Models")
IMAGE_PKG = os.path.join(OUT, "SigLIP2ImageEncoder.mlpackage")
TEXT_PKG = os.path.join(OUT, "SigLIP2TextEncoder.mlpackage")
FIXTURES = os.path.join(ROOT, "ios", "PhotosTests", "siglip2_token_fixtures.json")

# Strings used for tokenizer fixtures and text parity (punctuation, case, unicode, emoji, long input -> truncation).
SAMPLE_TEXTS = [
    "a photo of a dog", "Sunset over the BEACH!", "café menu with prices: $12.50", "東京タワー at night",
    "  multiple   spaces\tand\nnewlines ", "emoji 🐶🌅 party", "receipt_from_costco",
    "the quick brown fox jumps over the lazy dog " * 8, "naïve résumé", "x",
    "my passport", "kids playing in the snow", "wifi password on a whiteboard", "red vintage car parked by the sea",
    "birthday cake with candles", "hotel room with two beds", "screenshot of a boarding pass", "Eiffel Tower",
    "a plate of sushi", "foggy mountains at sunrise",
]


def tokenizer_json_path() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, "tokenizer.json")


def clean(text: str) -> str:
    """open_clip's 'canonicalize' clean fn for SigLIP2 (ftfy + html unescape, drop punctuation, lowercase)."""
    from open_clip.tokenizer import _clean_canonicalize
    return _clean_canonicalize(text)


def hf_tokenizer():
    """Reference tokenizer: HF `tokenizers` on the cached tokenizer.json, exactly as open_clip's HFTokenizer pads it."""
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(tokenizer_json_path())
    t.enable_truncation(CONTEXT_LENGTH)
    t.enable_padding(length=CONTEXT_LENGTH, pad_id=0)
    return t


def tokenize(texts: list[str]) -> list[list[int]]:
    return [e.ids for e in hf_tokenizer().encode_batch([clean(s) for s in texts])]


def load_model():
    """open_clip SigLIP2 on CPU, fp32, eval. ~1.5 GB RAM."""
    import open_clip
    import torch
    torch.set_grad_enabled(False)
    torch.backends.mha.set_fastpath_enabled(False)  # plain ops instead of aten._native_multi_head_attention
    model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED, device="cpu")
    return model.eval(), preprocess


def towers(model=None):
    """(image_tower, text_tower) as plain nn.Modules, ready to trace/export for any runtime (Core ML here; Android reuses it).

    image_tower: float (N,3,384,384) already normalized to [-1, 1] (pixel/127.5 - 1, RGB, squash-resized) -> (N,768) unit vectors
    text_tower:  int (N,64) token ids (see tokenize()) -> (N,768) unit vectors
    Similarity -> calibrated probability: sigmoid(logit_scale * cos + logit_bias), see calibration().
    """
    import torch
    import torch.nn.functional as F
    model = model or load_model()[0]

    class ImageTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = model.visual

        def forward(self, x):
            return F.normalize(self.visual(x), dim=-1)

    class TextTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text = model.text

        def forward(self, ids):
            return F.normalize(self.text(ids.long()), dim=-1)

    return ImageTower().eval(), TextTower().eval()


def calibration(model) -> dict:
    """SigLIP's sigmoid head: P(match) = sigmoid(logit_scale * cos + logit_bias)."""
    return {"logit_scale": float(model.logit_scale.exp()), "logit_bias": float(model.logit_bias)}
