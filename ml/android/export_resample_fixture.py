#!/usr/bin/env python3
"""Fixture for android Resample.kt: PIL bicubic resizes of small random RGB images (PIL only; no model, no guard).

usage:  search/.venv/bin/python ml/android/export_resample_fixture.py
"""
import json
import os

import numpy as np
from PIL import Image

from common import TEST_RES

rng = np.random.default_rng(0)
cases = []
for (w, h), (ow, oh) in [((37, 23), (16, 16)), ((50, 50), (13, 29)), ((12, 9), (24, 18)), ((200, 133), (96, 96))]:
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    if w > 100:  # smooth-ish photo-like content for the realistic size
        a = np.asarray(Image.fromarray(a).resize((w // 16, h // 16)).resize((w, h), Image.BILINEAR))
    out = np.asarray(Image.fromarray(a).resize((ow, oh), Image.BICUBIC))
    pack = lambda x: ((0xff << 24) | (x[..., 0].astype(np.int64) << 16) | (x[..., 1].astype(np.int64) << 8) | x[..., 2]).ravel()
    cases.append({"w": w, "h": h, "ow": ow, "oh": oh, "in": [int(v) - (1 << 32) if v >= 1 << 31 else int(v) for v in pack(a)],
                  "out": [int(v) - (1 << 32) if v >= 1 << 31 else int(v) for v in pack(out)]})
os.makedirs(TEST_RES, exist_ok=True)
json.dump(cases, open(os.path.join(TEST_RES, "resample_fixture.json"), "w"))
print("cases", len(cases))
