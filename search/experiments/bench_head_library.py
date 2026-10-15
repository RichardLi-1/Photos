"""Whole-library "same room" eval: every hotel photo in data/index (5,390 Hotels-50K test photos, 1,577 hotels;
none used in training) queries the full 21k-photo library (Unsplash/Flickr distractors included), using the stored
SigLIP2 vectors -> head. Reports R@1/R@5/mAP and how many top-10 results are non-hotel distractors.

usage: python experiments/bench_head_library.py data/models/hotel_head.pt [...]
"""
import json, os, sqlite3, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from photosearch import heads  # noqa: E402


def run(scores_fn, labels, is_hotel, name):
    q = np.where(is_hotel)[0]
    r1, r5, aps, junk = [], [], [], []
    for c in range(0, len(q), 500):
        qi = q[c:c + 500]
        S = scores_fn(qi)
        S[np.arange(len(qi)), qi] = -np.inf
        order = np.argsort(-S, 1)[:, :-1]
        rel = labels[order] == labels[qi][:, None]
        npos = rel.sum(1)
        prec = np.cumsum(rel, 1) / np.arange(1, rel.shape[1] + 1)
        ok = npos > 0
        aps += list(((prec * rel).sum(1) / np.maximum(npos, 1))[ok]); r1 += list(rel[ok, 0]); r5 += list(rel[ok, :5].any(1))
        junk += list((~is_hotel[order[:, :10]]).mean(1))
    print(f"{name:40s} R@1={np.mean(r1):.3f} R@5={np.mean(r5):.3f} mAP={np.mean(aps):.3f}  non-hotel in top10={np.mean(junk):.3f}", flush=True)


if __name__ == "__main__":
    z = np.load(f"{ROOT}/data/index/vectors__siglip2-b16-384.npz")
    ids, X = z["ids"], z["X"].astype(np.float32)
    db = sqlite3.connect(f"file:{ROOT}/data/index/library.sqlite?mode=ro", uri=True)
    path = dict(db.execute("SELECT id, path FROM photos"))
    gt = json.load(open(f"{ROOT}/data/hotels/gt.json"))
    labels = np.array([gt.get(os.path.basename(path[i]), "") if "/hotels/images/" in path[i] else f"x{i}" for i in ids])
    is_hotel = np.array([not l.startswith("x") for l in labels])
    print(f"{len(ids)} photos, {is_hotel.sum()} hotel queries")
    run(lambda qi: X[qi] @ X.T, labels, is_hotel, "raw siglip2")
    for p in sys.argv[1:]:
        h = heads.load(p); Z = h.embed(X)
        run(lambda qi: Z[qi] @ Z.T, labels, is_hotel, f"{os.path.basename(p)} head")
        for w in sorted({0.5, 1.0, 2.0, h.fuse_w}):
            run(lambda qi: heads.fuse(X[qi] @ X.T, Z[qi] @ Z.T, w), labels, is_hotel, f"{os.path.basename(p)} fuse w={w}")
