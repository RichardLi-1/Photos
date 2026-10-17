#!/usr/bin/env python3
"""SigLIP2-B/16-384 (open_clip) -> ONNX for ONNX Runtime Android, int8 weights, with parity checks.

Outputs (android/app/src/main/assets/models/, gitignored; regenerate with this script):
  siglip2_image_q8.onnx      pixels float[N,3,384,384] in [-1,1] (RGB, squash-resized) -> embedding float[N,768], unit length
  siglip2_text_q8.onnx       token embeddings float[N,64,768] -> embedding float[N,768], unit length
  siglip2_token_embed_q8.bin the 256k x 768 token-embedding table, int8 per row + fp32 row scales (looked up in Kotlin:
                             keeping it out of the graph saves ~600 MB of RAM, ORT would dequantize all of it at load)
  concepts.json + concepts_f16.bin  concept vocabulary + its text embeddings, for "why it matched" (src/main/assets)
  calibration.json           SigLIP sigmoid head (logit_scale, logit_bias) + model facts
ml/android/parity.json       cosine vs PyTorch fp32 on 20 demo images + 20 texts

usage:  PYTHONPATH=ml/android/.pylib search/.venv/bin/python ml/android/convert_siglip2_onnx.py [export|quant|concepts|parity|all]
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

from common import (ASSETS, BIG_ASSETS, CONTEXT_LENGTH, DIM, IMAGE_SIZE, PARITY_TEXTS, VOCAB, WORK, HERE, demo_images, guard,
                    load_model, tokenizer)

IMG_FP32, TXT_FP32 = os.path.join(WORK, "image_fp32.onnx"), os.path.join(WORK, "text_fp32.onnx")
IMG_Q8, TXT_Q8 = os.path.join(BIG_ASSETS, "siglip2_image_q8.onnx"), os.path.join(BIG_ASSETS, "siglip2_text_q8.onnx")
EMB_Q8 = os.path.join(BIG_ASSETS, "siglip2_token_embed_q8.bin")
IR_VERSION = 9  # ONNX Runtime Android 1.22 reads IR <= 10; newer onnx writes 11+ by default
OPSET = 17


def towers(model):
    import torch
    import torch.nn.functional as F
    t = model.text
    assert t.cls_emb is None and not getattr(t, "use_pad_mask", False) and t.pool_type == "last", "unexpected text tower"
    assert t.attn_mask is None, "SigLIP2 text tower is bidirectional"

    class Image(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = model.visual

        def forward(self, pixels):
            return F.normalize(self.visual(pixels), dim=-1)

    class TextFromEmbeds(torch.nn.Module):
        """open_clip TextTransformer.forward minus the token-embedding gather (done on device from the q8 table)."""

        def __init__(self):
            super().__init__()
            self.t = t

        def forward(self, emb):
            x = emb + self.t.positional_embedding[:CONTEXT_LENGTH]
            x = self.t.transformer(x, attn_mask=None)
            x = self.t.ln_final(x)
            pooled = x[:, -1]  # pool_type 'last': SigLIP pads to 64 and reads the final position
            proj = self.t.text_projection
            pooled = proj(pooled) if isinstance(proj, torch.nn.Linear) else pooled @ proj
            return F.normalize(pooled, dim=-1)

    return Image().eval(), TextFromEmbeds().eval()


def export(model):
    import torch
    os.makedirs(WORK, exist_ok=True)
    img, txt = towers(model)
    # sanity: the split text tower equals open_clip's own encode_text
    ids = tokenizer()(["a photo of a dog", "sunset"])
    ref = torch.nn.functional.normalize(model.encode_text(ids), dim=-1)
    mine = txt(model.text.token_embedding(ids))
    assert torch.allclose(ref, mine, atol=1e-5), (ref - mine).abs().max()
    kw = dict(dynamo=False, opset_version=OPSET, do_constant_folding=True)
    torch.onnx.export(img, (torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),), IMG_FP32, input_names=["pixels"],
                      output_names=["embedding"], dynamic_axes={"pixels": {0: "n"}, "embedding": {0: "n"}}, **kw)
    guard.check(force=True)
    torch.onnx.export(txt, (torch.zeros(1, CONTEXT_LENGTH, DIM),), TXT_FP32, input_names=["token_embeds"],
                      output_names=["embedding"], dynamic_axes={"token_embeds": {0: "n"}, "embedding": {0: "n"}}, **kw)
    guard.check(force=True)
    write_embed_table(model.text.token_embedding.weight.detach().float().numpy())
    os.makedirs(ASSETS, exist_ok=True)
    cal = {"model": "ViT-B-16-SigLIP2-384 (webli)", "image_size": IMAGE_SIZE, "context_length": CONTEXT_LENGTH,
           "dim": DIM, "mean": [0.5] * 3, "std": [0.5] * 3, "resize": "squash",
           "logit_scale": float(model.logit_scale.exp()), "logit_bias": float(model.logit_bias)}
    json.dump(cal, open(os.path.join(ASSETS, "calibration.json"), "w"), indent=1)
    print("exported", cal)


def write_embed_table(W: np.ndarray):
    """'SGE8' | u32 rows | u32 dim | f32 scale[rows] | i8 q[rows*dim]  (little endian). value = q * scale."""
    assert W.shape == (VOCAB, DIM), W.shape
    scale = np.abs(W).max(1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(W / scale[:, None]), -127, 127).astype(np.int8)
    os.makedirs(BIG_ASSETS, exist_ok=True)
    with open(EMB_Q8, "wb") as f:
        f.write(b"SGE8" + struct.pack("<II", VOCAB, DIM))
        f.write(scale.astype("<f4").tobytes())
        f.write(q.tobytes())


def read_embed_table() -> tuple[np.ndarray, np.ndarray]:
    raw = np.memmap(EMB_Q8, dtype=np.uint8, mode="r")
    assert bytes(raw[:4]) == b"SGE8"
    rows, dim = struct.unpack("<II", bytes(raw[4:12]))
    scale = np.frombuffer(raw[12:12 + 4 * rows].tobytes(), "<f4")
    q = np.asarray(raw[12 + 4 * rows:]).view(np.int8).reshape(rows, dim)
    return scale, q


def lookup(ids: np.ndarray) -> np.ndarray:
    scale, q = read_embed_table()
    return (q[ids].astype(np.float32) * scale[ids][..., None]).astype(np.float32)


def quantize():
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for src, dst in ((IMG_FP32, IMG_Q8), (TXT_FP32, TXT_Q8)):
        os.makedirs(BIG_ASSETS, exist_ok=True)
        tmp = dst + ".tmp.onnx"
        quantize_dynamic(src, tmp, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"])
        m = onnx.load(tmp)
        m.ir_version = IR_VERSION
        onnx.save(m, dst)
        os.remove(tmp)
        guard.check(force=True)
        print("quantized", dst, f"{os.path.getsize(dst) / 1e6:.0f} MB")


def cos(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(-1)


def parity(model, preprocess):
    import onnxruntime as ort
    import torch
    from PIL import Image
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = lambda p: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    paths = demo_images(20)
    load = lambda p: Image.open(p).convert("RGB")
    x = torch.stack([preprocess(load(p)) for p in paths])
    ref_i = torch.nn.functional.normalize(model.encode_image(x), dim=-1).numpy()
    pj = os.path.join(HERE, "parity.json")
    out = json.load(open(pj)) if os.path.exists(pj) else {}  # keep fp32 results once work/ has been cleaned up
    out.update({"images": len(paths), "texts": len(PARITY_TEXTS)})
    for name, path in (("image_fp32_onnx", IMG_FP32), ("image_q8_onnx", IMG_Q8)):
        if os.path.exists(path):
            s = sess(path)
            e = np.concatenate([s.run(None, {"pixels": x[i:i + 4].numpy()})[0] for i in range(0, len(x), 4)])
            c = cos(ref_i, e)
            out[name] = {"min": float(c.min()), "mean": float(c.mean())}
            guard.check()
    # Android-style preprocessing (bilinear squash from a decoder-downsampled bitmap) vs open_clip's bicubic, for info
    def android_like(p, resample):
        im = load(p)
        while min(im.size) >= 2 * IMAGE_SIZE:  # ImageDecoder setTargetSampleSize-style power-of-two decode
            im = im.reduce(2)
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), resample)
        a = np.asarray(im, np.float32) / 127.5 - 1.0
        return a.transpose(2, 0, 1)
    s = sess(IMG_Q8)
    for name, rs in (("bilinear", Image.BILINEAR), ("bicubic", Image.BICUBIC)):
        xa = np.stack([android_like(p, rs) for p in paths]).astype(np.float32)
        ea = np.concatenate([s.run(None, {"pixels": xa[i:i + 4]})[0] for i in range(0, len(xa), 4)])
        c = cos(ref_i, ea)
        out[f"image_q8_android_preprocess_{name}_vs_ref"] = {"min": float(c.min()), "mean": float(c.mean())}

    ids = tokenizer()(PARITY_TEXTS)
    ref_t = torch.nn.functional.normalize(model.encode_text(ids), dim=-1).numpy()
    emb_fp32 = model.text.token_embedding(ids).numpy()
    emb_q8 = lookup(ids.numpy())
    for name, path, emb in (("text_fp32_onnx", TXT_FP32, emb_fp32), ("text_q8_onnx+q8_table", TXT_Q8, emb_q8),
                            ("text_q8_onnx+fp32_table", TXT_Q8, emb_fp32)):
        if os.path.exists(path):
            s = sess(path)
            e = s.run(None, {"token_embeds": emb})[0]
            c = cos(ref_t, e)
            out[name] = {"min": float(c.min()), "mean": float(c.mean())}
    # retrieval sanity: does the q8 pipeline rank the same image first for each text?
    s_img = sess(IMG_Q8)
    e_img = np.concatenate([s_img.run(None, {"pixels": x[i:i + 4].numpy()})[0] for i in range(0, len(x), 4)])
    e_txt = sess(TXT_Q8).run(None, {"token_embeds": emb_q8})[0]
    out["top1_agreement_text_to_image"] = float(np.mean((ref_t @ ref_i.T).argmax(1) == (e_txt @ e_img.T).argmax(1)))
    out["sizes_mb"] = {os.path.basename(p): round(os.path.getsize(p) / 1e6, 1) for p in (IMG_Q8, TXT_Q8, EMB_Q8)}
    json.dump(out, open(pj, "w"), indent=1)
    print(json.dumps(out, indent=1))
    worst = min(v["min"] for k, v in out.items() if isinstance(v, dict) and "min" in v and "android" not in k)
    assert worst >= 0.99, f"parity below 0.99: {worst}"


def concepts(model):
    """Concept vocabulary for the "why it matched" chips (same list as search/photosearch/explain.py), embedded once."""
    import torch
    from photosearch.explain import CONCEPTS  # numpy-only module, read-only import
    prompts = [f"a photo of {c}" for c in CONCEPTS]
    T = torch.nn.functional.normalize(model.encode_text(tokenizer()(prompts)), dim=-1).numpy().astype("<f2")
    os.makedirs(ASSETS, exist_ok=True)
    json.dump({"prompt": "a photo of {}", "dim": DIM, "concepts": CONCEPTS},
              open(os.path.join(ASSETS, "concepts.json"), "w"), ensure_ascii=False)
    with open(os.path.join(ASSETS, "concepts_f16.bin"), "wb") as f:
        f.write(T.tobytes())  # [len(concepts), 768] little-endian fp16, row-major
    print("concepts", T.shape)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    guard.acquire(2)
    model, preprocess = load_model()
    if stage in ("export", "all"):
        export(model)
    if stage in ("quant", "all"):
        quantize()
    if stage in ("concepts", "all"):
        concepts(model)
    if stage in ("parity", "all"):
        parity(model, preprocess)


if __name__ == "__main__":
    main()
