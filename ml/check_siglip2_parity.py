#!/usr/bin/env python3
"""Parity: Core ML (coremltools predict, macOS) vs PyTorch fp32 embeddings, for 20 images and 20 texts. Target cos >= 0.99.

Both sides get identical inputs: the photo squash-resized to 384x384 with PIL bicubic (what open_clip's preprocessing
does), then PyTorch normalizes to [-1, 1] itself while Core ML does it in-model. Also checks the token ids.

usage:  ml/.venv/bin/python ml/check_siglip2_parity.py [--images DIR]   (default: 20 demo photos in search/data/demo)
"""
import argparse
import glob
import os

import numpy as np

from siglip2_common import (CONTEXT_LENGTH, IMAGE_PKG, IMAGE_SIZE, ROOT, SAMPLE_TEXTS, TEXT_PKG, guard, load_model,
                            tokenize, towers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default=os.path.join(ROOT, "search", "data", "demo", "unsplash"))
    p.add_argument("--n", type=int, default=20)
    a = p.parse_args()

    import coremltools as ct
    import torch
    from PIL import Image, ImageOps

    guard.acquire(2)
    model, _ = load_model()
    image_tower, text_tower = towers(model)
    paths = sorted(glob.glob(os.path.join(a.images, "*.jpg")))[:: max(1, len(glob.glob(os.path.join(a.images, "*.jpg"))) // a.n)][: a.n]
    assert len(paths) == a.n, f"need {a.n} images in {a.images}"

    def cos(x, y):
        return float(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y))

    ml = ct.models.MLModel(IMAGE_PKG, compute_units=ct.ComputeUnit.ALL)
    img_cos = []
    for path in paths:
        guard.check()
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        ref = image_tower(x)[0].numpy()
        got = ml.predict({"image": im})["embedding"].reshape(-1)
        img_cos.append(cos(ref, got))
    del ml

    ml = ct.models.MLModel(TEXT_PKG, compute_units=ct.ComputeUnit.ALL)
    ids = tokenize(SAMPLE_TEXTS)
    assert all(len(r) == CONTEXT_LENGTH for r in ids)
    txt_cos = []
    for row in ids:
        guard.check()
        t = torch.tensor([row])
        ref = text_tower(t)[0].numpy()
        got = ml.predict({"input_ids": np.array([row], dtype=np.int32)})["embedding"].reshape(-1)
        txt_cos.append(cos(ref, got))

    for name, c in (("images", img_cos), ("texts", txt_cos)):
        print(f"{name}: n={len(c)} min cos={min(c):.5f} mean={np.mean(c):.5f}")
    ok = min(img_cos) >= 0.99 and min(txt_cos) >= 0.99
    print("PARITY", "OK" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
