"""Export the trained "same place" head (search/data/models/hotel_head.pt) as plain Float32 weights for the desktop app.

The head is a residual MLP on the 768-d SigLIP2 vector (search/photosearch/heads.py):
    z = normalize(x @ P^T + W2 · GELU(W1 · x + b1) + b2)          P: 512×768, W1: 1024×768, W2: 512×1024
Swift applies it with vDSP (desktop/macos/Sources/Engine/PlaceHead.swift), so no Core ML model is needed.

Output (desktop/macos/Models/, gitignored next to the Core ML models):
    place_head.f32   little-endian float32: P^T (768×512) | W1^T (768×1024) | b1 (1024) | W2^T (1024×512) | b2 (512)
    place_head.json  shapes, offsets, fuse_w, and a parity fixture (4 library vectors → torch outputs)

usage: search/.venv/bin/python ml/export_place_head.py [--head search/data/models/hotel_head.pt] [--out desktop/macos/Models]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "search"))
from photosearch import guard  # noqa: E402  (read-only import; owned by search/)


def fixture_inputs(n: int = 4) -> np.ndarray:
    """A few real library vectors if an index exists (more realistic than noise), else seeded random unit vectors."""
    for path in sorted(glob.glob(os.path.join(ROOT, "search", "data", "index*", "vectors__siglip2-b16-384.npz"))):
        try:
            with np.load(path) as z:
                key = "X" if "X" in z.files else z.files[0]
                X = np.asarray(z[key][:n], np.float32)
            if X.ndim == 2 and X.shape[1] == 768 and len(X) == n:
                return X / np.linalg.norm(X, axis=1, keepdims=True)
        except Exception:
            continue
    X = np.random.default_rng(0).standard_normal((n, 768)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=os.path.join(ROOT, "search", "data", "models", "hotel_head.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "desktop", "macos", "Models"))
    args = ap.parse_args()

    guard.acquire(1)  # tiny model, but torch still counts against the machine-wide budget
    import torch
    from photosearch import heads

    h = heads.load(args.head)
    cfg = h.cfg
    assert cfg.get("inputs", ["pooled"]) == ["pooled"], f"only the pooled-input head is supported, got {cfg}"
    assert cfg["depth"] == 2 and h.mlp is not None, f"expected a 2-layer residual MLP, got {cfg}"
    lin1, act, lin2 = h.mlp[0], h.mlp[1], h.mlp[3]
    assert isinstance(act, torch.nn.GELU) and act.approximate == "none", "Swift side implements exact (erf) GELU"

    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in h.state_dict().items()}
    parts = [
        ("proj_t", sd["proj.weight"].T),     # 768×512
        ("w1_t", sd["mlp.0.weight"].T),      # 768×1024
        ("b1", sd["mlp.0.bias"]),            # 1024
        ("w2_t", sd["mlp.3.weight"].T),      # 1024×512
        ("b2", sd["mlp.3.bias"]),            # 512
    ]
    os.makedirs(args.out, exist_ok=True)
    offsets, blob, off = {}, [], 0
    for name, a in parts:
        a = np.ascontiguousarray(a, dtype="<f4")
        offsets[name] = {"offset": off, "shape": list(a.shape)}
        blob.append(a.tobytes())
        off += a.size
    with open(os.path.join(args.out, "place_head.f32"), "wb") as f:
        f.write(b"".join(blob))

    X = fixture_inputs()
    with torch.inference_mode():
        Z = h(torch.from_numpy(X)).numpy()
    meta = {
        "source": os.path.relpath(args.head, ROOT),
        "in_dim": cfg["in_dim"], "hidden": cfg["hidden"], "dim": cfg["dim"], "fuse_w": float(h.fuse_w),
        "activation": "gelu_erf", "floats": off, "tensors": offsets,
        "fixture": {"x": X.round(7).tolist(), "z": Z.round(7).tolist()},
    }
    with open(os.path.join(args.out, "place_head.json"), "w") as f:
        json.dump(meta, f)

    # Self-check with numpy, exactly the math the Swift side does.
    from math import erf, sqrt
    gelu = np.vectorize(lambda v: 0.5 * v * (1 + erf(v / sqrt(2))))
    zz = X @ parts[0][1] + gelu(X @ parts[1][1] + parts[2][1]) @ parts[3][1] + parts[4][1]
    zz /= np.linalg.norm(zz, axis=1, keepdims=True)
    cos = (zz * Z).sum(1)
    print(f"wrote {args.out}/place_head.f32 ({off * 4 / 1e6:.1f} MB) + place_head.json; numpy-vs-torch min cos {cos.min():.6f}")
    assert cos.min() > 0.9999


if __name__ == "__main__":
    main()
