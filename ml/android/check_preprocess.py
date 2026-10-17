#!/usr/bin/env python3
"""How much does the Android preprocessing (power-of-two decoder subsampling, then bicubic or bilinear squash to 384)
move the int8 image embedding vs open_clip's (bicubic squash from full resolution), on phone-sized (12 MP-like) inputs?

Demo photos are 512 px wide, so they are upscaled 6x first (smooth content: a lower bound on aliasing effects).
ONNX Runtime only (no torch); light, guard weight 1. Adds results to ml/android/parity.json.

usage:  PYTHONPATH=ml/android/.pylib search/.venv/bin/python ml/android/check_preprocess.py
"""
import json
import os

import numpy as np
from PIL import Image

from common import BIG_ASSETS, HERE, IMAGE_SIZE, demo_images, guard


def tensor(im):
    return (np.asarray(im, np.float32) / 127.5 - 1.0).transpose(2, 0, 1)


def main():
    import onnxruntime as ort
    guard.acquire(1)
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    s = ort.InferenceSession(os.path.join(BIG_ASSETS, "siglip2_image_q8.onnx"), so, providers=["CPUExecutionProvider"])
    emb = lambda x: s.run(None, {"pixels": x[None]})[0][0]
    res = {}
    for p in demo_images(20):
        big = Image.open(p).convert("RGB")
        big = big.resize((big.width * 6, big.height * 6), Image.BICUBIC)
        ref = emb(tensor(big.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)))  # open_clip: bicubic squash, full res
        for min_side in (IMAGE_SIZE, 2 * IMAGE_SIZE):  # decoder keeps the short side >= min_side
            im = big
            while min(im.size) >= 2 * min_side:
                im = im.reduce(2)  # ImageDecoder.setTargetSampleSize
            for name, rs in (("bilinear", Image.BILINEAR), ("bicubic", Image.BICUBIC)):
                res.setdefault(f"{name}_decode{min_side}", []).append(float(np.dot(ref, emb(tensor(im.resize((IMAGE_SIZE, IMAGE_SIZE), rs))))))
        guard.check()
    pj = os.path.join(HERE, "parity.json")
    out = json.load(open(pj)) if os.path.exists(pj) else {}
    for k, v in res.items():
        out[f"android_preprocess_{k}_vs_openclip_phone_size"] = {"min": min(v), "mean": float(np.mean(v))}
    json.dump(out, open(pj, "w"), indent=1)
    print({k: (round(min(v), 4), round(float(np.mean(v)), 4)) for k, v in res.items()})


if __name__ == "__main__":
    main()
