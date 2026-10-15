"""Qwen3-VL-Reranker (MLX, fully local): cross-encoder re-scoring of a short list of candidate photos.

Format follows the official Qwen3VLReranker: the system turn asks for a yes/no judgement, the user turn is
"<Instruct>: ...<Query>:...\\n<Document>:...", and the score is sigmoid(logit("yes") - logit("no")) at the
last token. Query may be text or an image (PIL image / pathlib.Path); documents are image paths, optionally
with a text note each (e.g. place/date metadata the pixels can't show).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import guard
from .encoders import load_image

IMG = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
INSTRUCT = "Given a search query, retrieve relevant candidates that answer the query."
SAME_PLACE = "Given a photo, retrieve photos taken in the same room or place."


class Reranker:
    def __init__(self, repo="mlx-community/Qwen3-VL-Reranker-2B-4bit", max_tokens=320):
        guard.acquire(2)
        from huggingface_hub import snapshot_download
        from mlx_vlm.encoder_loader import load_encoder_model
        from transformers import AutoProcessor
        path = Path(snapshot_download(repo, local_files_only=os.environ.get("PHOTOSEARCH_ALLOW_DOWNLOAD") != "1"))
        import mlx.core as mx
        mx.set_cache_limit(512 << 20)  # see QwenVLEmbed
        # the embedding variant = Qwen3-VL minus the LM head, with a padding-masked forward; the head is tied
        self.model = load_encoder_model(path, model_remapping={"qwen3_vl": "qwen3_vl_embedding"})
        self.proc = AutoProcessor.from_pretrained(path)
        self.proc.tokenizer.padding_side = "right"
        v = self.proc.tokenizer.get_vocab()
        self.yes_no = [v["yes"], v["no"]]
        self.max_px = max_tokens * 32 * 32

    def _resize(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        s = min(1.0, (self.max_px / (w * h)) ** 0.5)
        return im.resize((max(32, int(w * s) // 32 * 32), max(32, int(h * s) // 32 * 32)), Image.BICUBIC)

    @staticmethod
    def prompt(query: str, doc: str, instruction: str = INSTRUCT) -> str:
        return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}"
                f"<Query>:{query}\n<Document>:{doc}<|im_end|>\n<|im_start|>assistant\n")

    def _forward(self, texts, images) -> np.ndarray:
        import mlx.core as mx
        from mlx_vlm.models.qwen3_vl_embedding.qwen3_vl_embedding import last_non_padding_token
        enc = self.proc(text=texts, images=images, padding=True, return_tensors="np")
        lm = self.model.language_model
        lm._position_ids = lm._rope_deltas = None  # see QwenVLEmbed._run
        a = {k: mx.array(enc[k]) for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
        h = last_non_padding_token(self.model._last_hidden_state(**a), a["attention_mask"])
        logits = lm.model.embed_tokens.as_linear(h)[:, self.yes_no].astype(mx.float32)
        return np.array(mx.sigmoid(logits[:, 0] - logits[:, 1]))

    def score(self, query, candidates, notes=None, instruction=None, bs=6) -> np.ndarray:
        """query: text, or an image (PIL.Image / pathlib.Path). candidates: image paths. notes: optional text per candidate."""
        q_img = None
        if isinstance(query, (Image.Image, Path)):
            q_img = self._resize(query if isinstance(query, Image.Image) else load_image(str(query)))
        instruction = instruction or (SAME_PLACE if q_img is not None else INSTRUCT)
        notes = notes or [""] * len(candidates)
        texts = [self.prompt(IMG if q_img is not None else query, IMG + n, instruction) for n in notes]
        out = []
        with ThreadPoolExecutor(4) as pool:
            ims = list(pool.map(lambda p: self._resize(load_image(p)), candidates))
        for i in range(0, len(candidates), bs):
            guard.check()
            docs = ims[i:i + bs]
            images = [x for d in docs for x in ([q_img, d] if q_img is not None else [d])]
            out.append(self._forward(texts[i:i + bs], images))
        return np.concatenate(out) if out else np.zeros(0, np.float32)
