#!/usr/bin/env python3
"""Export a compact, read-only demo bundle from an index (Mac side; the server never needs torch).

usage (from search/):
  .venv/bin/python demo/build_bundle.py --index data/index-pass --credits data/pass/credits.csv --out ../demo-bundle
      [--limit 20000] [--thumb avif:384:50] [--head data/models/hotel_head.pt | --head none] [--allow-uncredited]

What it writes (see engine.py for the layout) and how each part was shrunk (measure.py has the numbers):
  vectors.npz    SigLIP2 768-d vectors, fp16 -> int8 with one scale per dimension (no PCA: it cost recall)
  place.npz      Z = same-place head(X), precomputed and int8 (no torch at runtime; top-10 overlap 0.989 vs fp32)
  library.sqlite metadata + OCR + FTS5 + credits; `path` is "demo/<id>" (absolute paths never leave the Mac);
                 ocr_fast and any albums / trash / audit rows of the source index are NOT copied
  models/        int8 ONNX text tower (Android's), int8 token table pruned to ids reachable after open_clip's
                 clean() (lossless, 198 -> 139 MB), the compact BPE file (9 MB), calibration.json (from the weights)
  concepts.npz   "why it matched" prompt vectors
  thumbs.pack    every thumbnail in one file (thumbs.npz: ids / offsets / lengths / mime)
  CREDITS.csv + manifest.json

Only photos with a credits row are exported (a public demo must show licensed photos only); --allow-uncredited is for
local smoke tests with unpublishable data and marks the bundle publishable=false (demo_server.py then refuses to start
unless DEMO_ALLOW_UNPUBLISHABLE=1).
Guarded: guard.acquire(1) (the small torch head and the ONNX tower for concept prompts; no image model).
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SEARCH)
sys.path[:0] = [SEARCH, HERE]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402

from photosearch import guard  # noqa: E402
from photosearch.library import SCHEMA  # noqa: E402

HF_SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--timm--ViT-B-16-SigLIP2-384/snapshots/*/")
ONNX_TEXT = os.path.join(ROOT, "android", "app", "bigassets", "models", "siglip2_text_q8.onnx")
ANDROID_ASSETS = os.path.join(ROOT, "android", "app", "src", "main", "assets", "models")
CREDIT_COLS = ("creator", "license", "license_url", "source", "source_url", "flickr_url", "title")


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def hf_file(name: str) -> str:
    hits = glob.glob(HF_SNAP + name)
    if not hits:
        sys.exit(f"{name} not in the HF cache ({HF_SNAP}); run `make setup` once (online)")
    return hits[0]


# ---------------------------------------------------------------- models

def build_models(out: str, onnx_path: str):
    from safetensors import safe_open
    from textenc import reachable_ids, write_pruned_q8
    d = os.path.join(out, "models")
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(onnx_path):
        sys.exit(f"{onnx_path} missing: run ml/android/convert_siglip2_onnx.py export quant (see android/README.md)")
    shutil.copyfile(onnx_path, os.path.join(d, "siglip2_text_q8.onnx"))
    tj_path = hf_file("tokenizer.json")
    tj = json.load(open(tj_path, encoding="utf-8"))
    sgtk = os.path.join(ANDROID_ASSETS, "siglip2_tokenizer.bin")
    if os.path.exists(sgtk):  # the compact BPE (textenc.GemmaBPE, ~10 MB RSS; ids identical to open_clip on 7.5k texts)
        shutil.copyfile(sgtk, os.path.join(d, "siglip2_tokenizer.bin"))
    else:
        log(f"WARNING: {sgtk} missing (run ml/android/export_tokenizer.py); shipping tokenizer.json (+~470 MB RSS)")
        shutil.copyfile(tj_path, os.path.join(d, "tokenizer.json"))
    with safe_open(hf_file("open_clip_model.safetensors"), "numpy") as f:
        W = f.get_tensor("text.token_embedding.weight").astype(np.float32)
        scale, bias = float(np.exp(f.get_tensor("logit_scale")).reshape(-1)[0]), float(f.get_tensor("logit_bias").reshape(-1)[0])
    keep = reachable_ids(tj)
    write_pruned_q8(os.path.join(d, "token_embed.bin"), W, keep, fallback=tj["model"]["vocab"]["<unk>"])
    json.dump({"model": "ViT-B-16-SigLIP2-384 (webli), text tower int8 ONNX", "context_length": 64, "dim": 768,
               "logit_scale": scale, "logit_bias": bias, "token_rows_kept": int(len(keep)), "token_rows": int(len(W))},
              open(os.path.join(d, "calibration.json"), "w"), indent=1)
    log(f"models: token table {len(keep)}/{len(W)} rows, logit_scale {scale:.2f} bias {bias:.2f}")


def build_concepts(out: str):
    """Android's concept vectors (torch fp32) when they match explain.CONCEPTS, else embed with the ONNX tower."""
    from photosearch.explain import CONCEPTS
    cj, cb = os.path.join(ANDROID_ASSETS, "concepts.json"), os.path.join(ANDROID_ASSETS, "concepts_f16.bin")
    T = None
    if os.path.exists(cj) and os.path.exists(cb) and json.load(open(cj)).get("concepts") == CONCEPTS:
        T = np.fromfile(cb, "<f2").reshape(len(CONCEPTS), -1)
        src = "android assets (torch fp32)"
    if T is None:
        from textenc import TextEncoder
        T = TextEncoder(os.path.join(out, "models")).texts([f"a photo of {c}" for c in CONCEPTS]).astype(np.float16)
        src = "onnx int8"
    np.savez(os.path.join(out, "concepts.npz"), names=np.array(CONCEPTS), T=T.astype(np.float16))
    log(f"concepts: {len(CONCEPTS)} from {src}")


# ---------------------------------------------------------------- thumbnails

def encode_thumb(path: str, spec: str) -> tuple[bytes, str]:
    from PIL import Image, ImageOps
    fmt, side, q = spec.split(":")
    side, q = int(side), int(q)
    im = Image.open(path)
    im.draft("RGB", (side * 2, side * 2))
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    b = io.BytesIO()
    if fmt == "avif":
        im.save(b, "AVIF", quality=q, speed=6)
        return b.getvalue(), "image/avif"
    if fmt == "webp":
        im.save(b, "WEBP", quality=q, method=6)
        return b.getvalue(), "image/webp"
    im.save(b, "JPEG", quality=q, optimize=True, progressive=True)
    return b.getvalue(), "image/jpeg"


def build_thumbs(out: str, rows: list[tuple[int, str]], spec: str):
    offs, lens, ids, mime = [], [], [], None
    t0, pos = time.time(), 0
    with open(os.path.join(out, "thumbs.pack"), "wb") as f:
        for n, (pid, path) in enumerate(rows):
            if n % 200 == 0:
                guard.check()
                if n:
                    log(f"thumbs {n}/{len(rows)} ({pos / n / 1024:.1f} KB avg, {n / (time.time() - t0):.0f}/s)")
            try:
                b, mime = encode_thumb(path, spec)
            except Exception as e:
                log(f"  skip {pid}: {e}")
                continue
            f.write(b)
            ids.append(pid); offs.append(pos); lens.append(len(b)); pos += len(b)
    np.savez(os.path.join(out, "thumbs.npz"), ids=np.array(ids, np.int64), offsets=np.array(offs, np.int64),
             lengths=np.array(lens, np.int32), mime=np.array(mime or "image/jpeg"))
    log(f"thumbs: {len(ids)} in {pos / 1e6:.1f} MB ({pos / max(1, len(ids)) / 1024:.1f} KB avg)")
    return set(ids)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--credits", help="credits.csv from experiments/fetch_pass.py (column `file` = photo file name)")
    ap.add_argument("--allow-uncredited", action="store_true", help="local smoke tests only: bundle is marked unpublishable")
    ap.add_argument("--limit", type=int, help="random subset of N photos (seeded)")
    ap.add_argument("--thumb", default="avif:384:50", help="format:long_side:quality (avif|webp|jpeg)")
    ap.add_argument("--no-thumbs", action="store_true", help="serve thumbnails from DEMO_THUMB_BASE_URL instead")
    ap.add_argument("--head", default=os.path.join(SEARCH, "data", "models", "hotel_head.pt"), help="'none' = no Same place")
    ap.add_argument("--onnx", default=ONNX_TEXT)
    ap.add_argument("--model", default="siglip2-b16-384")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"{a.out} exists and is not empty")
    os.makedirs(a.out, exist_ok=True)
    guard.acquire(1)

    src = sqlite3.connect(f"file:{a.index}/library.sqlite?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    with np.load(os.path.join(a.index, f"vectors__{a.model}.npz"), allow_pickle=False) as z:
        vids, X16 = z["ids"], z["X"]
    have_vec = {int(i): n for n, i in enumerate(vids)}
    cols = [r[1] for r in src.execute("PRAGMA table_info(photos)")]
    where = "WHERE deleted_at IS NULL" if "deleted_at" in cols else ""
    photos = [dict(r) for r in src.execute(f"SELECT * FROM photos {where} ORDER BY id") if r["id"] in have_vec]

    credits = {}
    if a.credits:
        for r in csv.DictReader(open(a.credits, newline="", encoding="utf-8")):
            credits[r["file"]] = r
    keep = [p for p in photos if os.path.basename(p["path"]) in credits]
    publishable = True
    if len(keep) < len(photos):
        log(f"{len(photos) - len(keep)} of {len(photos)} photos have no credits row")
        if a.allow_uncredited:
            keep, publishable = photos, False
            log("--allow-uncredited: keeping them; bundle marked publishable=false (NOT for a public demo)")
    if a.limit and len(keep) > a.limit:
        rng = np.random.default_rng(0)
        keep = [keep[i] for i in sorted(rng.choice(len(keep), a.limit, replace=False))]
    if not keep:
        sys.exit("nothing to export")
    log(f"exporting {len(keep)} photos")

    # thumbnails first: a photo whose file can't be read is dropped everywhere
    if not a.no_thumbs:
        ok = build_thumbs(a.out, [(p["id"], p["path"]) for p in keep], a.thumb)
        keep = [p for p in keep if p["id"] in ok]

    # vectors (int8, per-dim scale) and the same-place space
    from engine import quantize_cols
    ids = np.array([p["id"] for p in keep], np.int64)
    X = X16[[have_vec[int(i)] for i in ids]].astype(np.float32)
    q, s = quantize_cols(X)
    np.savez(os.path.join(a.out, "vectors.npz"), ids=ids, q=q, scale=s)
    head_note = None
    if a.head != "none" and os.path.exists(a.head):
        from photosearch import heads
        import torch
        with torch.inference_mode():
            Z = heads.load(a.head).embed(X).astype(np.float32)
        qz, sz = quantize_cols(Z)
        np.savez(os.path.join(a.out, "place.npz"), q=qz, scale=sz)
        head_note = f"{os.path.basename(a.head)} applied at build time; only its outputs (Z) ship, not the weights"
        log("NOTE: the same-place head was trained on Hotels-50K, whose license is unverified "
            "(docs/data-and-licenses.md); use --head none for a public demo until that is cleared")
        log(f"place space: {Z.shape}")
    guard.check()

    # sqlite: fresh schema, no absolute paths, no albums/trash/audit/ocr_fast from the source
    db_path = os.path.join(a.out, "library.sqlite")
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    base = ("id", "album", "taken", "lat", "lon", "place", "camera", "width", "height", "ocr", "ocr_done", "session")
    db.executemany(f"INSERT INTO photos(path, {', '.join(base)}) VALUES (?{', ?' * len(base)})",
                   [(f"demo/{p['id']}", *(p.get(c) for c in base)) for p in keep])
    db.execute(f"CREATE TABLE credits (id INTEGER PRIMARY KEY, file TEXT, {', '.join(c + ' TEXT' for c in CREDIT_COLS)})")
    rows = []
    for p in keep:
        c = credits.get(os.path.basename(p["path"]))
        if c:
            rows.append((p["id"], os.path.basename(p["path"]), *(c.get(k) or None for k in CREDIT_COLS)))
    db.executemany(f"INSERT INTO credits VALUES (?, ?{', ?' * len(CREDIT_COLS)})", rows)
    db.commit()
    from photosearch import tools  # numpy/sqlite only: its tables (albums, audit_log, photo_stats, soft-delete columns)
    tools.ensure_schema(db)
    db.execute("INSERT INTO fts(fts) VALUES('optimize')")
    db.commit()
    db.execute("VACUUM")
    db.close()
    with open(os.path.join(a.out, "CREDITS.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("id", "file") + CREDIT_COLS)
        w.writerows(rows)

    build_models(a.out, a.onnx)
    build_concepts(a.out)

    sizes = {}
    for dp, _, fs in os.walk(a.out):
        for fn in fs:
            sizes[os.path.relpath(os.path.join(dp, fn), a.out)] = os.path.getsize(os.path.join(dp, fn))
    man = {"built": datetime.now().isoformat(timespec="seconds"), "photos": len(keep), "publishable": publishable,
           "source_index": os.path.basename(os.path.normpath(a.index)), "thumb": None if a.no_thumbs else a.thumb,
           "vectors": "int8 per-dim scale, 768-d", "place": head_note, "credited": len(rows),
           "albums": dict(sorted(Counter(p["album"] for p in keep).items())),
           "sizes_mb": {k: round(v / 1e6, 2) for k, v in sorted(sizes.items())},
           "total_mb": round(sum(sizes.values()) / 1e6, 1),
           "licenses": {"model": "SigLIP2 (Apache-2.0, timm/ViT-B-16-SigLIP2-384)",
                        "photos": "per CREDITS.csv", "places": "GeoNames (CC BY 4.0) when fetch_pass.py --geonames was used"}}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    log(f"bundle {a.out}: {man['total_mb']} MB, {len(keep)} photos, publishable={publishable}")


if __name__ == "__main__":
    main()
