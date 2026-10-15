"""Hotels "same room": DINO pooling variants, score fusion with SigLIP2, and re-ranking tricks — all on cached embeddings."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench import CACHE, D, hotels_subset, hotels_task  # noqa: E402
from photosearch import encoders as E  # noqa: E402

files, labels = hotels_subset()
VARIANTS = ["cls", "mean", "gem", "cls+gem", "cls+mean"]


def dino_variants(timm_name, tag, res=336):
    if all(os.path.exists(f"{CACHE}/hotels2400__{tag}:{v}.npy") for v in VARIANTS):
        return
    d = E.Dino(timm_name, res, "cls")
    out = {v: [] for v in VARIANTS}
    with torch.inference_mode():
        for x in d._batches([f"{D}/hotels/images/{f}" for f in files], d.tf, 32):
            for v, t in d.pools(*d.tokens(x.to(E.DEVICE, E.DTYPE))).items():
                out[v].append(E._norm(t))
    for v in VARIANTS:
        np.save(f"{CACHE}/hotels2400__{tag}:{v}.npy", np.concatenate(out[v]))


def run(name, hook=None):
    r = hotels_task(name if hook is None else "siglip2-b16-384", None, hook)
    print(f"{name:42s} R@1={r['hotels_R@1']:.3f} R@5={r['hotels_R@5']:.3f} mAP={r['hotels_mAP']:.3f}", flush=True)


def load(name):
    return np.load(f"{CACHE}/hotels2400__{name}.npy").astype(np.float32)


def zs(S):  # per-query z-score so different models' similarities are comparable
    m = np.where(np.isfinite(S), S, np.nan)
    return (S - np.nanmean(m, 1, keepdims=True)) / np.nanstd(m, 1, keepdims=True)


def fuse(names, weights):
    def hook(S, _):
        out = 0
        for n, w in zip(names, weights):
            X = load(n); s = X @ X.T; np.fill_diagonal(s, -np.inf)
            out = out + w * zs(s)
        return out
    return hook


def alpha_qe(name, k=3, alpha=3.0):
    """Alpha query expansion: re-query with the weighted mean of the query and its top-k neighbours."""
    def hook(S, _):
        X = load(name)
        top = np.argsort(-S, 1)[:, :k]
        w = np.maximum(np.take_along_axis(S, top, 1), 0) ** alpha
        Xq = X + (w[..., None] * X[top]).sum(1)
        Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
        S2 = Xq @ X.T; np.fill_diagonal(S2, -np.inf)
        return S2
    return hook


if __name__ == "__main__":
    dino_variants("vit_base_patch16_dinov3.lvd1689m", "dinov3-b@336")
    for v in VARIANTS:
        run(f"dinov3-b@336:{v}")
    run("siglip2-b16-384")
    best_dino = "dinov3-b@336:cls"
    for w in (0.25, 0.5):
        run(f"fuse siglip2 + {w}*{best_dino}", fuse(["siglip2-b16-384", best_dino], [1, w]))
    run("fuse siglip2 + clip-b32", fuse(["siglip2-b16-384", "clip-b32"], [1, 0.5]))
    for k in (2, 3, 5):
        run(f"siglip2 + alphaQE k={k}", alpha_qe("siglip2-b16-384", k))
