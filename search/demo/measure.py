#!/usr/bin/env python3
"""Measure what each demo-bundle compression step costs in size, RAM, latency and quality.  Mac only (needs the index).

usage (from search/):
  PYTHONPATH=<dir with onnxruntime> .venv/bin/python demo/measure.py --models DIR [--index data/index-xl]
      [--q4 DIR_WITH_Q4_TABLE] [--skip-place] [--out demo/results]

--models: a model dir as build_bundle.py writes it (tokenizer.json, siglip2_text_q8.onnx, token_embed.bin, calibration.json).
--q4: a second model dir whose token_embed.bin is the int4 table (build_bundle.py --table q4).

Quality, all on the 72,887-photo XL index:
  * eval_set_xl.json through the real search box path (Query.parse=True; same scoring as experiments/eval_search.py)
  * Flickr30k-1k captions -> their photo among ALL 72,887 photos (5,000 captions, torch caption vectors from
    data/cache), i.e. text->image retrieval with 72k distractors: sensitive to vector compression.
  * same_room (the trained head's space) with Z in fp32 vs int8, and top-10 overlap on 500 query photos.
  * text tower: cosine of ONNX int8 (+ q8 / q4 token table) vs the torch embeddings cached by eval_search.py.
Guarded: guard.acquire(1) (ONNX text tower + the small torch head; never the SigLIP image tower).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import resource
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH = os.path.dirname(HERE)
sys.path[:0] = [SEARCH, HERE]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("GUARD_MIN_DISK_GB", "6.5")

import numpy as np  # noqa: E402

from photosearch import guard  # noqa: E402
from photosearch.search import Query  # noqa: E402
from engine import DemoEncoder, DemoLibrary, DemoSearcher, QMatrix, pca_basis, quantize_cols  # noqa: E402

METRICS = ("hit@1", "hit@10", "P@10", "R@60", "MRR")
K = 60


def score(ranked, rel):  # = experiments/eval_search.py:score
    hits = [i in rel for i in ranked[:K]]
    first = hits.index(True) if any(hits) else None
    return {"hit@1": float(any(hits[:1])), "hit@10": float(any(hits[:10])), "P@10": sum(hits[:10]) / 10,
            "R@60": sum(hits) / min(len(rel), K), "MRR": 0.0 if first is None else 1 / (first + 1)}


def summarize(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["type"]].append(r["m"])
    by["ALL"] = [r["m"] for r in rows]
    return {t: {"n": len(ms), **{m: round(float(np.mean([x[m] for x in ms])), 3) for m in METRICS}} for t, ms in by.items()}


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def median_ms(fn, n=30):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(SEARCH, "data", "index-xl"))
    ap.add_argument("--models", required=True)
    ap.add_argument("--q4")
    ap.add_argument("--head", default=os.path.join(SEARCH, "data", "models", "hotel_head.pt"))
    ap.add_argument("--skip-place", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    guard.acquire(1)
    out = {"index": a.index, "when": datetime.now().isoformat(timespec="seconds")}

    # ------------------------------------------------------------ text tower
    cache = np.load(os.path.join(SEARCH, "data", "cache", "eval_text__siglip2-b16-384.npz"))
    ctexts, cX = list(cache["t"]), cache["X"].astype(np.float32)
    r0 = rss_mb()
    enc = DemoEncoder(a.models)
    out["text"] = {"rss_after_load_mb": round(rss_mb() - r0, 1)}
    E = enc.text._run(ctexts[:1])  # warm
    t_single = median_ms(lambda: enc.text._run(["a red vintage car parked by the sea"]))
    E = np.concatenate([enc.text._run(ctexts[i:i + 16]) for i in range(0, len(ctexts), 16)])
    c = (E * cX).sum(1)
    out["text"]["onnx_q8_table"] = {"cos_vs_torch_min": round(float(c.min()), 4), "cos_vs_torch_mean": round(float(c.mean()), 4),
                                    "ms_per_query_1thread_batch1": t_single, "rss_mb_peak": round(rss_mb(), 1)}
    onnx_cache = dict(zip(ctexts, E))
    print("text q8:", out["text"])
    if a.q4:
        enc4 = DemoEncoder(a.q4)
        E4 = np.concatenate([enc4.text._run(ctexts[i:i + 16]) for i in range(0, len(ctexts), 16)])
        c4 = (E4 * cX).sum(1)
        out["text"]["onnx_q4_table"] = {"cos_vs_torch_min": round(float(c4.min()), 4), "cos_vs_torch_mean": round(float(c4.mean()), 4),
                                        "cos_vs_q8_min": round(float((E4 * E).sum(1).min()), 4)}
        print("text q4:", out["text"]["onnx_q4_table"])
    guard.check()

    # ------------------------------------------------------------ vectors
    with np.load(os.path.join(a.index, "vectors__siglip2-b16-384.npz")) as z:
        ids, X16 = z["ids"], z["X"]
    X = X16.astype(np.float32)
    N, D = X.shape
    variants = {"fp32 (in RAM today)": QMatrix(X, np.ones(D, np.float32))}
    q, s = quantize_cols(X)
    variants["int8 per-dim"] = QMatrix(q, s)
    variants["int8 per-dim, dense f32 in RAM"] = QMatrix(q, s, dense=True)
    for d in (384, 256):
        B, mu = pca_basis(X, d)
        P = (X - mu) @ B.T
        qp, sp = quantize_cols(P)
        variants[f"PCA{d} + int8"] = QMatrix(qp, sp, B, mu)
    disk = {"fp32 (in RAM today)": X16.nbytes + ids.nbytes}  # what's on disk today is fp16
    for k, m in variants.items():
        disk.setdefault(k, m.q.nbytes + m.scale.nbytes + (m.basis.nbytes + m.mean.nbytes if m.basis is not None else 0) + ids.nbytes)

    # Flickr captions vs all photos
    rows = list(csv.DictReader(open(os.path.join(SEARCH, "data", "flickr1k", "captions.csv"))))
    T = np.load(os.path.join(SEARCH, "data", "cache", "flickr_txt__siglip2-b16-384.npy")).astype(np.float32)
    import sqlite3
    db = sqlite3.connect(f"file:{a.index}/library.sqlite?mode=ro", uri=True)
    by_file = {os.path.basename(p): i for i, p in db.execute("SELECT id, path FROM photos WHERE album='flickr'")}
    row_of = {int(i): n for n, i in enumerate(ids)}
    cap_row = np.array([row_of[by_file[r["filename"]]] for r in rows for _ in json.loads(r["raw"])])

    def flickr(M):
        ranks = []
        for b in range(0, len(T), 500):
            S = M @ T[b:b + 500].T  # [N, 500]
            tgt = S[cap_row[b:b + 500], np.arange(S.shape[1])]
            ranks.append((S > tgt).sum(0))
            guard.check()
        r = np.concatenate(ranks)
        return {f"R@{k}": round(float((r < k).mean()), 4) for k in (1, 10)}

    # eval_set_xl through the search box, with torch query vectors (cache) where available, else ONNX
    ev = json.load(open(os.path.join(SEARCH, "experiments", "eval_set_xl.json")))
    lib = DemoLibrary(a.index, os.path.join(a.index, "library.sqlite"), enc)
    Z32 = None
    if not a.skip_place and os.path.exists(a.head):
        from photosearch import heads
        import torch
        with torch.inference_mode():
            Z32 = heads.load(a.head).embed(X).astype(np.float32)
    qz, sz = quantize_cols(Z32) if Z32 is not None else (None, None)

    def run_eval(M, Zm, text_src):
        S = DemoSearcher(lib, ids, M, Zm)
        miss = [0]
        if text_src == "torch":
            tc = dict(zip(ctexts, cX))

            def et(t, _o=S.embed_text):
                if t in tc:
                    return tc[t]
                miss[0] += 1
                return _o(t)
            S.embed_text = et
        emb = set(ids.tolist())
        res, lat = [], []
        for qq in ev["queries"]:
            rel, img = set(qq["relevant"]) & emb, qq.get("image_id")
            if not rel or (img is not None and img not in emb):
                continue
            t = time.perf_counter()
            if img is not None:
                r = S.search(Query(image=img, k=K, mode="place" if Zm is not None else "similar"))
            else:
                r = S.search(Query(text=qq["query"], k=K, parse=True))
            lat.append((time.perf_counter() - t) * 1000)
            res.append({"type": qq["type"], "m": score([x["id"] for x in r], rel)})
            guard.check()
        sm = summarize(res)
        return sm, miss[0], round(float(np.median(lat)), 1)

    v = X[0] / np.linalg.norm(X[0])
    out["vectors"] = {}
    for name, M in variants.items():
        sm, miss, lat = run_eval(M, None, "torch")
        out["vectors"][name] = {
            "disk_mb": round(disk[name] / 1e6, 1), "ram_mb": round((M._dense.nbytes if M._dense is not None else M.q.nbytes) / 1e6, 1),
            "matmul_ms": median_ms(lambda: M @ v), "flickr_in_72k": flickr(M),
            "eval_xl_ALL": sm["ALL"], "eval_xl_by_type": {k: v_ for k, v_ in sm.items() if k != "ALL"},
            "eval_median_ms": lat, "texts_not_in_torch_cache": miss}
        print(name, {k: v_ for k, v_ in out["vectors"][name].items() if k != "eval_xl_by_type"})

    # the demo configuration end to end: ONNX query vectors + int8 vectors (+ head space, int8)
    for name in ("int8 per-dim", "PCA384 + int8"):
        sm, _, lat = run_eval(variants[name], None, "onnx")
        out["vectors"][name]["eval_xl_ALL_onnx_text"] = sm["ALL"]
        out["vectors"][name]["eval_xl_by_type_onnx_text"] = {k: v_ for k, v_ in sm.items() if k != "ALL"}
        print(name, "onnx text:", sm["ALL"], f"{lat} ms")
    out["vectors"]["fp32 (in RAM today)"]["eval_xl_ALL_onnx_text"] = run_eval(variants["fp32 (in RAM today)"], None, "onnx")[0]["ALL"]

    # ------------------------------------------------------------ same-place head space
    if Z32 is not None:
        out["place"] = {}
        for name, Zm in (("Z fp32 (torch head at runtime today)", QMatrix(Z32, np.ones(Z32.shape[1], np.float32))),
                         ("Z int8 precomputed", QMatrix(qz, sz))):
            sm, _, lat = run_eval(variants["int8 per-dim"], Zm, "torch")
            out["place"][name] = {"same_room": sm.get("same_room"), "disk_mb": round(Zm.q.nbytes / 1e6, 1),
                                  "ram_mb": round(Zm.q.nbytes / 1e6, 1), "median_ms": lat}
        rng = np.random.default_rng(0)
        Zq = QMatrix(qz, sz)
        ov = []
        for r in rng.choice(N, 500, replace=False):
            a32 = np.argsort(-(Z32 @ Z32[r]))[1:11]
            a8 = np.argsort(-(Zq @ Zq[r]))[1:11]
            ov.append(len(set(a32) & set(a8)) / 10)
        out["place"]["top10_overlap_int8_vs_fp32"] = round(float(np.mean(ov)), 3)
        print("place:", out["place"])

    out["rss_peak_mb"] = round(rss_mb(), 1)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"measure_{datetime.now():%Y%m%d-%H%M%S}.json")
    json.dump(out, open(p, "w"), indent=1)
    print("->", p)


if __name__ == "__main__":
    main()
