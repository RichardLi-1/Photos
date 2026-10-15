"""Hotels "same room" benchmark for learned heads (hotels2400, same metrics as bench.py --tasks hotels).

usage: python experiments/bench_head.py [HEAD.pt ...]     (no args: raw feature/pooling baselines)
"""
import os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photosearch import heads  # noqa: E402

FEATS = f"{ROOT}/data/hotels_train/feats"


def metrics(S: np.ndarray, labels: np.ndarray) -> dict:
    """Leave-one-out retrieval: R@1, R@5, mAP over queries that have >= 1 positive (identical to bench.hotels_task)."""
    S = S.astype(np.float64).copy()
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, 1)[:, :-1]
    rel = labels[order] == labels[:, None]
    npos = rel.sum(1)
    ok = npos > 0
    rel, npos = rel[ok], npos[ok]
    prec = np.cumsum(rel, 1) / np.arange(1, rel.shape[1] + 1)
    return {"R@1": rel[:, 0].mean(), "R@5": rel[:, :5].any(1).mean(), "mAP": ((prec * rel).sum(1) / npos).mean()}


def fmt(name, m):
    return f"{name:44s} R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} mAP={m['mAP']:.3f}"


def test_data():
    from bench import hotels_subset
    _, labels = hotels_subset()
    return dict(np.load(f"{FEATS}/test2400.npz")), labels


def eval_head(h, feats, labels, ws=(0.5, 1.0, 2.0)) -> dict:
    X = heads.inputs_from(feats, ("pooled",)); S0 = X @ X.T
    Z = h.embed(heads.inputs_from(feats, h.inputs)); S1 = Z @ Z.T
    out = {"head": metrics(S1, labels)}
    for w in ws:
        S = S1.copy(); np.fill_diagonal(S, -np.inf); R = S0.copy(); np.fill_diagonal(R, -np.inf)
        out[f"fuse w={w}"] = metrics(heads.fuse(R, S, w), labels)
    return out


if __name__ == "__main__":
    feats, labels = test_data()
    if len(sys.argv) == 1:
        for names in [("pooled",), ("mean",), ("gem",), ("pooled", "gem"), ("pooled", "mean")]:
            X = heads.inputs_from(feats, names)
            print(fmt("raw " + "+".join(names), metrics(X @ X.T, labels)))
        g = feats["grid"].astype(np.float32); g /= np.linalg.norm(g, axis=-1, keepdims=True)
        X = g.reshape(len(g), -1) / 4  # sum over 16 aligned regions
        print(fmt("raw grid4x4 (aligned)", metrics(X @ X.T, labels)))
        # region max-match (Chamfer): mean over my regions of best match among yours
        Sc = np.concatenate([np.einsum("ird,jsd->ijrs", g[i:i + 100], g).max(3).mean(2) for i in range(0, len(g), 100)])
        print(fmt("raw grid chamfer", metrics(Sc + Sc.T, labels)))
        R = X0 = heads.inputs_from(feats, ("pooled",))
        for w in (0.25, 0.5):
            print(fmt(f"pooled + {w}*chamfer (z-fused)", metrics(heads.fuse(X0 @ X0.T, Sc + Sc.T, w), labels)))
    for p in sys.argv[1:]:
        for k, m in eval_head(heads.load(p), feats, labels).items():
            print(fmt(f"{os.path.basename(p)} {k}", m))
