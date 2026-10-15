"""Retrieval benchmarks.

  hotels   image->image, "other photos of this room" (Hotels-50K/TraffickCam, 5.4k imgs, leave-one-out)
  roxford  image->image, landmark instance retrieval (rOxford medium, 70 queries / 5k)
  flickr   text->image (Flickr30k 1k test, 5k captions)

usage: python experiments/bench.py ENCODER [ENCODER...] [--tasks hotels,roxford,flickr]
"""
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from photosearch import encoders  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = f"{D}/cache"
os.makedirs(CACHE, exist_ok=True)


def cached(enc_name, key, fn):
    p = f"{CACHE}/{key}__{enc_name.replace('/', '_')}.npy"
    if os.path.exists(p):
        return np.load(p)
    x = fn()
    np.save(p, x)
    return x


def ap(ranked_rel: np.ndarray, n_pos: int) -> float:
    hits = np.cumsum(ranked_rel)
    return float((hits / np.arange(1, len(ranked_rel) + 1) * ranked_rel).sum() / max(n_pos, 1))


def hotels_subset(n=2400):
    """All photos of a random set of hotels (seed 0) until ~n images; keeps same-chain confusers."""
    import collections, random
    gt = json.load(open(f"{D}/hotels/gt.json"))
    by = collections.defaultdict(list)
    for f, h in gt.items():
        by[h].append(f)
    hs = sorted(by); random.Random(0).shuffle(hs)
    files = []
    for h in hs:
        if len(files) >= n:
            break
        files += sorted(by[h])
    return files, np.array([gt[f] for f in files])


def hotels_task(name, enc_fn, sims_hook=None):
    files, labels = hotels_subset()
    X = cached(name, "hotels2400", lambda: enc_fn().images([f"{D}/hotels/images/{f}" for f in files]))
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    if sims_hook:
        S = sims_hook(S, files)
    aps, r1, r5 = [], [], []
    for i in range(len(files)):
        rel = (labels == labels[i])
        rel[i] = False
        if rel.sum() == 0:
            continue
        order = np.argsort(-S[i])[:-1]
        rr = rel[order]
        aps.append(ap(rr, rel.sum())); r1.append(rr[0]); r5.append(rr[:5].any())
    return {"hotels_mAP": np.mean(aps), "hotels_R@1": np.mean(r1), "hotels_R@5": np.mean(r5)}


def roxford_task(name, enc_fn):
    gt = json.load(open(f"{D}/roxford/gt.json"))
    corpus = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{D}/roxford/images/corpus/*.jpg"))
    queries = sorted(gt)
    enc = None

    def e():
        nonlocal enc
        enc = enc or enc_fn()
        return enc
    C = cached(name, "rox_corpus", lambda: e().images([f"{D}/roxford/images/corpus/{c}.jpg" for c in corpus]))
    Q = cached(name, "rox_query", lambda: e().images([f"{D}/roxford/images/query/{q}.jpg" for q in queries]))
    idx = {c: i for i, c in enumerate(corpus)}
    aps = []
    for qi, q in enumerate(queries):
        ev = np.array([idx[c] for c in gt[q]["eval"] if c in idx])  # junk excluded
        pos = set(idx[c] for c in gt[q]["pos"] if c in idx)
        s = C[ev] @ Q[qi]
        order = ev[np.argsort(-s)]
        aps.append(ap(np.array([o in pos for o in order]), len(pos)))
    return {"roxford_mAP": np.mean(aps)}


def flickr_data():
    import csv
    rows = list(csv.DictReader(open(f"{D}/flickr1k/captions.csv")))
    files = [r["filename"] for r in rows]
    caps, cap_img = [], []
    for i, r in enumerate(rows):
        for c in json.loads(r["raw"]):
            caps.append(c); cap_img.append(i)
    return files, caps, np.array(cap_img)


def flickr_task(name, enc_fn):
    files, caps, cap_img = flickr_data()
    enc = None

    def e():
        nonlocal enc
        enc = enc or enc_fn()
        return enc
    I = cached(name, "flickr_img", lambda: e().images([f"{D}/flickr1k/images_flickr_1k_test/{f}" for f in files]))
    T = cached(name, "flickr_txt", lambda: e().texts(caps))
    S = T @ I.T
    rank = (S > S[np.arange(len(caps)), cap_img][:, None]).sum(1)
    return {f"flickr_R@{k}": float((rank < k).mean()) for k in (1, 5, 10)}


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("encoders", nargs="+")
    ap_.add_argument("--tasks", default="hotels,flickr")
    a = ap_.parse_args()
    for name in a.encoders:
        _enc = None

        def enc_fn():
            global _enc
            if _enc is None:
                t = time.time()
                _enc = encoders.get(name)
                print(f"  loaded {name} in {time.time() - t:.0f}s", flush=True)
            return _enc
        res = {}
        t0 = time.time()
        for task in a.tasks.split(","):
            if task == "flickr" and not name.startswith(("clip", "mobileclip", "siglip", "pe")):
                continue
            res.update({"hotels": hotels_task, "roxford": roxford_task, "flickr": flickr_task}[task](name, enc_fn))
        print(f"{name:28s} " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
        _enc = None
