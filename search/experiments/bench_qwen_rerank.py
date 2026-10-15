"""Qwen3-VL-Reranker on top of SigLIP2's top-K.  usage: python bench_qwen_rerank.py [--flickr N] [--hotels N] [--eval] [--k 30]

Raw scores go to results/rerank_*.json so blends can be tried without re-running the model.
"""
import argparse, json, os, sys, time
from datetime import datetime
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench import CACHE, D, ap, flickr_data, hotels_subset  # noqa: E402
from photosearch.rerank import Reranker  # noqa: E402

SIG = "siglip2-b16-384"
HERE = os.path.dirname(__file__)


def load(key):
    return np.load(f"{CACHE}/{key}__{SIG}.npy").astype(np.float32)


def rerank_order(top, scores):
    return [top[i] for i in np.argsort(-np.asarray(scores), kind="stable")]


def sanity(rr):
    msgs = [{"role": "system", "content": [{"type": "text", "text": "S"}]},
            {"role": "user", "content": [{"type": "text", "text": "<Instruct>: I"}, {"type": "text", "text": "<Query>:"},
                                         {"type": "text", "text": "q"}, {"type": "text", "text": "\n<Document>:"},
                                         {"type": "image", "image": "x"}]}]
    ref = rr.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    from photosearch import rerank as R
    print("prompt == chat template:", ref == rr.prompt("q", R.IMG, "I").replace(R.SYSTEM, "S"))


def flickr(rr, n, k, out):
    files, caps, cap_img = flickr_data()
    T, I = load("flickr_txt"), load("flickr_img")
    qs = np.random.default_rng(0).choice(len(caps), n, replace=False)
    base, new, secs, raw = [], [], [], []
    for q in qs:
        order = list(np.argsort(-(I @ T[q])))
        t = time.time()
        s = rr.score(caps[q], [f"{D}/flickr1k/images_flickr_1k_test/{files[i]}" for i in order[:k]])
        secs.append(time.time() - t)
        new_order = rerank_order(order[:k], s) + order[k:]
        base.append(order.index(cap_img[q])); new.append(new_order.index(cap_img[q]))
        raw.append({"q": int(q), "top": [int(i) for i in order[:k]], "rr": s.tolist(), "sig": (I[order[:k]] @ T[q]).tolist()})
    f = lambda r: " ".join(f"R@{x}={np.mean(np.array(r) < x):.3f}" for x in (1, 5, 10))
    print(f"flickr n={n} top-{k}  siglip2 {f(base)}  | rerank {f(new)}  | gt in top-{k}: {np.mean(np.array(base) < k):.3f}"
          f"  | {np.mean(secs):.1f}s/query", flush=True)
    out["flickr"] = raw


def hotels(rr, n, k, out):
    files, labels = hotels_subset()
    X = load("hotels2400")
    S = X @ X.T; np.fill_diagonal(S, -np.inf)
    qs = np.random.default_rng(0).choice(len(files), n, replace=False)
    from pathlib import Path
    m = {"base": [], "new": []}; secs, raw = [], []
    for q in qs:
        rel = labels == labels[q]; rel[q] = False
        if not rel.any():
            continue
        order = list(np.argsort(-S[q])[:-1])
        t = time.time()
        s = rr.score(Path(f"{D}/hotels/images/{files[q]}"), [f"{D}/hotels/images/{files[i]}" for i in order[:k]])
        secs.append(time.time() - t)
        for key, o in (("base", order), ("new", rerank_order(order[:k], s) + order[k:])):
            rr_ = rel[o]; m[key].append((rr_[0], rr_[:5].any(), ap(rr_, rel.sum())))
        raw.append({"q": int(q), "top": [int(i) for i in order[:k]], "rr": s.tolist(), "sig": S[q, order[:k]].tolist()})
    f = lambda v: "R@1={:.3f} R@5={:.3f} mAP={:.3f}".format(*np.mean(v, 0))
    print(f"hotels n={len(raw)} top-{k}  siglip2 {f(m['base'])}  | rerank {f(m['new'])}  | {np.mean(secs):.1f}s/query", flush=True)
    out["hotels"] = raw


def note(p):
    bits = []
    if p.get("place"):
        bits.append(f"Place: {p['place']}.")
    if p.get("taken"):
        bits.append(f"Taken: {datetime.fromtimestamp(p['taken']):%B %Y}.")
    return " " + " ".join(bits) if bits else ""


def eval_set(rr, k, out, types=("object/scene", "caption", "place+object")):
    from eval_search import INDEX, score, table
    from photosearch.library import Library
    from photosearch.search import Query, Searcher
    lib = Library(INDEX, readonly=True)
    s = Searcher(lib)
    E = json.load(open(f"{HERE}/eval_set.json"))
    emb = set(s.ids.tolist())
    rows = {"siglip2": [], "rerank": [], "rerank+meta": []}; secs, raw = [], []
    for q in (q for q in E["queries"] if q["type"] in types):
        rel = set(q["relevant"]) & emb
        if not rel:
            continue
        res = s.search(Query(text=q["query"], k=60))
        ids = [r["id"] for r in res]
        paths = [r["path"] for r in res[:k]]
        t = time.time(); a = rr.score(q["query"], paths); secs.append(time.time() - t)
        b = rr.score(q["query"], paths, notes=[note(r) for r in res[:k]])
        for name, o in (("siglip2", ids), ("rerank", rerank_order(ids[:k], a) + ids[k:]),
                        ("rerank+meta", rerank_order(ids[:k], b) + ids[k:])):
            rows[name].append({"type": q["type"], "n_rel": len(rel), "metrics": score(o, rel)})
        raw.append({"id": q["id"], "query": q["query"], "ids": ids, "rr": a.tolist(), "rr_meta": b.tolist(),
                    "rel": sorted(rel)})
        print(f"  {q['id']:18s} {secs[-1]:.1f}s", flush=True)
    for name, r in rows.items():
        print(f"\n== {name}"); table(r)
    print(f"\neval-set rerank: {np.mean(secs):.1f}s/query (top-{k}, image-only docs)", flush=True)
    out["eval"] = raw


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--flickr", type=int, default=0)
    p.add_argument("--hotels", type=int, default=0)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=320)
    a = p.parse_args()
    t = time.time(); rr = Reranker(max_tokens=a.max_tokens); print(f"loaded in {time.time() - t:.0f}s", flush=True)
    sanity(rr)
    out = {"k": a.k, "max_tokens": a.max_tokens}
    path = f"{HERE}/results/rerank_{a.max_tokens}_{datetime.now():%Y%m%d-%H%M%S}.json"
    for name, fn, arg in (("flickr", flickr, a.flickr), ("hotels", hotels, a.hotels), ("eval", eval_set, a.eval)):
        if arg:
            fn(rr, arg, a.k, out) if name != "eval" else fn(rr, a.k, out)
            json.dump(out, open(path, "w"))  # after every task: the watchdog may stop us
    import mlx.core as mx
    print(f"mlx peak memory {mx.get_peak_memory() / 1e9:.2f}GB")
