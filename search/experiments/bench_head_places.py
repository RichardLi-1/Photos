"""Regression check: does a hotel-trained head hurt generic scene similarity? Places365 val, 10 photos x 365
scene classes, leave-one-out retrieval of same-class photos (raw SigLIP2 vs head vs fused).

usage: python experiments/bench_head_places.py data/models/hotel_head.pt [...]
"""
import os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import cached, D  # noqa: E402
from bench_head import metrics, fmt  # noqa: E402
from photosearch import heads  # noqa: E402

def enc():
    from photosearch import encoders
    return encoders.get("siglip2-b16-384")

if __name__ == "__main__":  # DataLoader workers re-import this module on macOS
    rows = [l.split() for l in open(f"{D}/places365/places365_val.txt")]
    rows = [r for r in rows if int(r[0].split("_")[-1].split(".")[0]) <= 50 * 365]  # first ~50/class, then 10 per class
    per = {}
    for f, c in rows:
        if len(per.setdefault(c, [])) < 10:
            per[c].append(f)
    files = [f for c in sorted(per, key=int) for f in per[c]]
    labels = np.array([c for c in sorted(per, key=int) for _ in per[c]])

    X = cached("siglip2-b16-384", "places3650", lambda: enc().images([f"{D}/places365/val_256/{f}" for f in files])).astype(np.float32)
    print(fmt("places raw siglip2", metrics(X @ X.T, labels)))
    for p in sys.argv[1:]:
        h = heads.load(p); Z = h.embed(X)
        print(fmt(f"places {os.path.basename(p)} head", metrics(Z @ Z.T, labels)))
        R, S = X @ X.T, Z @ Z.T
        np.fill_diagonal(R, -np.inf); np.fill_diagonal(S, -np.inf)
        for w in sorted({1.0, h.fuse_w}):
            print(fmt(f"places {os.path.basename(p)} fuse w={w}", metrics(heads.fuse(R, S, w), labels)))
