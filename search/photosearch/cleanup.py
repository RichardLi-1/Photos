"""Photos you probably don't need: local, explainable cleanup candidates.

Every candidate carries `reasons` (plain words + the numbers behind them), a `confidence` in [0, 1] and, for
groups, a suggested photo to `keep`. Nothing here deletes anything; the agent trashes candidates via tools.py.

Signals (all on-device, no new models):
  photo_stats  cheap per-image stats, precomputed once (`python -m photosearch.cleanup stats`, resumable):
               sha1, Laplacian sharpness (global + best 4x4 tile), luminance histogram, colourfulness, "finger" blob.
  SigLIP       the existing image vectors scored against text prompts with SigLIP's sigmoid head (calibrated
               probabilities, as in explain.py), competing against explain.CONCEPTS as "ordinary photo" negatives.
  OCR          text density from the photos.ocr column.

Categories: exact_duplicate, near_duplicate, blurry, dark, overexposed, screenshot, receipt, document,
whiteboard, meme, accidental.
"""
from __future__ import annotations

import hashlib
import io
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps

from . import guard

STATS_VERSION = 2  # bump when image_stats changes: rows with an older version get recomputed
SIDE = 384  # normalise every photo to this long side so sharpness numbers are comparable across sources

STATS_SCHEMA = """
CREATE TABLE IF NOT EXISTS photo_stats (
  id INTEGER PRIMARY KEY, sha1 TEXT, bytes INT, w INT, h INT,
  lap_var REAL, lap_tile_max REAL, lap_tile_med REAL,
  luma_mean REAL, luma_std REAL, p01 REAL, p05 REAL, p50 REAL, p95 REAL, p99 REAL,
  frac_dark REAL, frac_bright REAL, entropy REAL, colorful REAL, finger REAL,
  p999 REAL, frac_lit REAL, edge_n INT, edge_ratio REAL, edge_ratio_x REAL, edge_ratio_y REAL,
  err TEXT, version INT, computed_at REAL);
CREATE INDEX IF NOT EXISTS photo_stats_sha1 ON photo_stats(sha1);
"""
STAT_COLS = ["sha1", "bytes", "w", "h", "lap_var", "lap_tile_max", "lap_tile_med", "luma_mean", "luma_std", "p01", "p05",
             "p50", "p95", "p99", "frac_dark", "frac_bright", "entropy", "colorful", "finger",
             "p999", "frac_lit", "edge_n", "edge_ratio", "edge_ratio_x", "edge_ratio_y"]
NEW_COLS = {"p999": "REAL", "frac_lit": "REAL", "edge_n": "INT", "edge_ratio": "REAL", "edge_ratio_x": "REAL",
            "edge_ratio_y": "REAL"}


def connect(db_path: str) -> sqlite3.Connection:
    db = sqlite3.connect(db_path, timeout=60, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=60000")
    return db


def ensure_schema(db: sqlite3.Connection):
    db.executescript(STATS_SCHEMA)
    have = {r[1] for r in db.execute("PRAGMA table_info(photo_stats)")}
    for c, t in NEW_COLS.items():  # upgrade a v1 table in place
        if c not in have:
            try:
                db.execute(f"ALTER TABLE photo_stats ADD COLUMN {c} {t}")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e):
                    raise
    db.commit()


# ---------------------------------------------------------------- per-image stats (CPU only)

def _lap(g: np.ndarray) -> np.ndarray:
    return 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]


def _finger(rgb: np.ndarray) -> float:
    """Share of an edge cell (3x3 grid) covered by a smooth, warm, skin-like blob, minus the same share in the centre.

    A finger over the lens shows up as a soft orange/red/brown blob creeping in from a border.
    """
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    warm = (r > 60) & (r > g + 12) & (g >= b - 5) & (r - b > 35)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(y)
    smooth = np.hypot(gx, gy) < 6
    m = warm & smooth
    H, W = m.shape
    cells = [m[i * H // 3:(i + 1) * H // 3, j * W // 3:(j + 1) * W // 3].mean() for i in range(3) for j in range(3)]
    centre = cells.pop(4)
    return float(max(cells) - centre)


def _edge_ratio(d1: np.ndarray, d2: np.ndarray) -> tuple[float, int]:
    """Edge sharpness independent of contrast: |2nd derivative| (max over +-1 px) / |1st derivative| at the strongest
    edges. A step edge blurred with width s scores ~1/s, so in-focus photos land around 1.7-3 and blurred ones below 0.8."""
    a = np.abs(d1)
    thr = max(float(np.percentile(a, 99)), 8.0)
    e = a >= thr
    n = int(e.sum())
    if n < 20:
        return float("nan"), n
    return float(d2[e].mean() / (a[e].mean() + 1e-6)), n


def _max3(x: np.ndarray, axis=None) -> np.ndarray:
    m = x.copy()
    for ax in ((0, 1) if axis is None else (axis,)):
        m = np.maximum(m, np.maximum(np.roll(m, 1, ax), np.roll(m, -1, ax)))
    return m


def image_stats(path: str) -> dict:
    data = open(path, "rb").read()
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    im.draft("RGB", (SIDE * 2, SIDE * 2))  # fast JPEG downscale on decode
    out = pixel_stats(ImageOps.exif_transpose(im).convert("RGB"))
    out.update(sha1=hashlib.sha1(data).hexdigest(), bytes=len(data), w=w, h=h)
    return out


def pixel_stats(im: Image.Image) -> dict:
    out = {}
    s = SIDE / max(im.size)
    if s < 1:
        im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.Resampling.BOX)
    g = np.asarray(im.convert("L"), dtype=np.float32)
    lap = _lap(g)
    out["lap_var"] = float(lap.var())
    H, W = lap.shape
    tiles = [lap[i * H // 4:(i + 1) * H // 4, j * W // 4:(j + 1) * W // 4].var() for i in range(4) for j in range(4)]
    out["lap_tile_max"], out["lap_tile_med"] = float(np.max(tiles)), float(np.median(tiles))
    out["luma_mean"], out["luma_std"] = float(g.mean()), float(g.std())
    for k, q in zip(("p01", "p05", "p50", "p95", "p99"), np.percentile(g, [1, 5, 50, 95, 99])):
        out[k] = float(q)
    out["frac_dark"], out["frac_bright"] = float((g < 20).mean()), float((g > 245).mean())
    h = np.bincount((g.astype(np.uint8) >> 2).ravel(), minlength=64) / g.size
    out["entropy"] = float(-(h[h > 0] * np.log2(h[h > 0])).sum())
    rgb = np.asarray(im, dtype=np.float32)
    rg, yb = rgb[..., 0] - rgb[..., 1], 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    out["colorful"] = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    small = np.asarray(im.resize((96, max(8, round(96 * im.height / im.width))) if im.width >= im.height
                                 else (max(8, round(96 * im.width / im.height)), 96), Image.Resampling.BOX))
    out["finger"] = _finger(small)
    out["p999"] = float(np.percentile(g, 99.9))
    out["frac_lit"] = float((g > 128).mean())
    gx, gy = np.zeros_like(g), np.zeros_like(g)
    gx[:, 1:-1], gy[1:-1] = (g[:, 2:] - g[:, :-2]) / 2, (g[2:] - g[:-2]) / 2
    alap = np.zeros_like(g); alap[1:-1, 1:-1] = np.abs(lap)
    out["edge_ratio"], out["edge_n"] = _edge_ratio(np.hypot(gx, gy), _max3(alap))
    dxx, dyy = np.zeros_like(g), np.zeros_like(g)
    dxx[:, 1:-1], dyy[1:-1] = np.abs(g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]), np.abs(g[2:] - 2 * g[1:-1] + g[:-2])
    out["edge_ratio_x"] = _edge_ratio(gx, _max3(dxx, 1))[0]  # motion blur smears one direction only
    out["edge_ratio_y"] = _edge_ratio(gy, _max3(dyy, 0))[0]
    for k, v in out.items():
        if isinstance(v, float) and v != v:
            out[k] = None
    return out


def _one(row):
    pid, path = row
    try:
        return pid, image_stats(path), None
    except Exception as e:  # unreadable file: record it so we don't retry forever
        return pid, {}, f"{type(e).__name__}: {e}"[:200]


def precompute_stats(db_path: str, batch: int = 100, workers: int = 2, limit: int | None = None, log=print) -> int:
    """Fill photo_stats for photos that lack a current row. Resumable; small WAL transactions; <=2 threads."""
    workers = max(1, min(2, workers))  # machine safety: this Mac panicked under parallel load
    db = connect(db_path)
    ensure_schema(db)
    todo = db.execute("""SELECT p.id, p.path FROM photos p LEFT JOIN photo_stats s ON s.id = p.id
                         WHERE s.id IS NULL OR s.version < ? ORDER BY p.id""", (STATS_VERSION,)).fetchall()
    todo = [(r[0], r[1]) for r in todo][:limit]
    cols = ["id"] + STAT_COLS + ["err", "version", "computed_at"]
    sql = f"INSERT OR REPLACE INTO photo_stats({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    t0, done = time.time(), 0
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(todo), batch):
            guard.check()
            rows = []
            for pid, st, err in pool.map(_one, todo[i:i + batch]):
                rows.append([pid] + [st.get(c) for c in STAT_COLS] + [err, STATS_VERSION, time.time()])
            with db:  # one short transaction per batch: the server keeps reading meanwhile
                db.executemany(sql, rows)
            done += len(rows)
            if (i // batch) % 10 == 0 or done == len(todo):
                log(f"stats {done}/{len(todo)}  {done / (time.time() - t0 + 1e-9):.0f} img/s")
    db.close()
    return done




# ---------------------------------------------------------------- zero-shot prompts (SigLIP, calibrated)

PROMPTS = {  # "is it this kind of photo?" -- must beat every ordinary subject in explain.CONCEPTS
    "screenshot": ["a screenshot of a phone screen", "a screenshot of a computer screen", "a screenshot of a website",
                   "a screenshot of a text message conversation", "a screenshot of an app user interface"],
    "receipt": ["a photo of a receipt", "a printed shopping receipt with prices", "a restaurant bill", "an invoice"],
    "document": ["a photo of a document", "a scanned page of text", "a photo of a printed page of text",
                 "a paper form filled in", "a photo of a typed letter"],
    "whiteboard": ["a photo of a whiteboard", "a whiteboard with handwritten notes", "a classroom blackboard covered in notes",
                   "a photo of a presentation slide on a projector screen"],
    "meme": ["a meme", "an internet meme with a caption in bold white letters", "a funny image macro with text",
             "a reaction meme"],
    "accidental": ["an accidental photo taken inside a pocket", "an accidental blurry photo of the floor",
                   "an accidental photo of the ceiling", "a finger covering the camera lens",
                   "a photo taken by mistake showing nothing", "a completely black photo", "a motion blurred accidental photo"],
}
CONTRAST = {  # quality defects: the defect wording must beat the "done on purpose" wording (a veto on the pixel stats)
    "blurry": (["a blurry out of focus photo", "a photo ruined by camera shake", "a shaky blurry snapshot"],
               ["a sharp photo", "a photo of fog", "a minimalist photo of the sky", "a long exposure photo",
                "a photo with a sharp subject and a softly blurred background", "an artistic soft focus photo"]),
    "dark": (["an underexposed photo that is almost completely black", "a photo taken in the dark by mistake"],
             ["a night sky with stars", "the moon at night", "a low-key photo on a black background",
              "city lights at night", "a silhouette against the sky", "a dark moody artistic photo"]),
    "overexposed": (["an overexposed washed out photo", "a photo blown out to white by too much light"],
                    ["a minimalist photo of an object on a white background", "a snowy landscape",
                     "a bright and airy photo", "a white wall", "a photo of the bright sky"]),
}
# ordinary-subject competitors = explain.CONCEPTS minus concepts that overlap a category on purpose
OVERLAP = {"document", "receipt", "newspaper", "notebook", "book", "menu", "sign with text", "poster", "map", "laptop",
           "phone", "black and white photo", "dark moody photo"}
DESC = {"screenshot": "looks like a screenshot", "receipt": "looks like a receipt", "document": "looks like a document",
        "whiteboard": "looks like a whiteboard / slide", "meme": "looks like a meme",
        "accidental": "looks like an accidental shot"}


class PromptBank:
    """Text embeddings for PROMPTS, CONTRAST and the negatives; cached next to the index so later runs need no model."""

    def __init__(self, cache_path: str, encoder_fn):
        from .explain import CONCEPTS
        self.negs = [c for c in CONCEPTS if c not in OVERLAP]
        self.labels = [(k, p) for k, ps in PROMPTS.items() for p in ps]
        self.contrast = [(k, side, p) for k, (d, i) in CONTRAST.items() for side, ps in (("defect", d), ("ok", i)) for p in ps]
        texts = [p for _, p in self.labels] + [p for *_, p in self.contrast] + [f"a photo of {c}" for c in self.negs]
        key = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:16]
        if os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=False)
            if str(z["key"]) == key:
                self.T, self.scale, self.bias = z["T"].astype(np.float32), float(z["scale"]), float(z["bias"])
                return
        enc = encoder_fn()  # goes through encoders.get -> guard.acquire(1)
        self.T = enc.texts(texts).astype(np.float32)
        m = enc.model
        self.scale = float(m.logit_scale.exp()) if hasattr(m, "logit_scale") else 100.0
        self.bias = float(m.logit_bias) if getattr(m, "logit_bias", None) is not None else -10.0
        tmp = cache_path + ".tmp.npz"
        np.savez(tmp, T=self.T, scale=self.scale, bias=self.bias, key=key)
        os.replace(tmp, cache_path)

    def score(self, X: np.ndarray, chunk: int = 4096) -> dict:
        """Per category: best prompt logit / index, calibrated p, and margin over the competitors."""
        n_pos, n_con = len(self.labels), len(self.contrast)
        idx = {c: [i for i, (k, _) in enumerate(self.labels) if k == c] for c in PROMPTS}
        cidx = {(c, side): [n_pos + i for i, (k, sd, _) in enumerate(self.contrast) if k == c and sd == side]
                for c in CONTRAST for side in ("defect", "ok")}
        n = len(X)
        out = {c: {f: np.empty(n, np.float32) for f in ("logit", "best", "margin")} for c in list(PROMPTS) + list(CONTRAST)}
        neg_best = np.empty(n, np.int32)
        for i in range(0, n, chunk):
            L = self.scale * (X[i:i + chunk].astype(np.float32) @ self.T.T) + self.bias
            neg = L[:, n_pos + n_con:]
            nl = neg.max(1)
            neg_best[i:i + chunk] = neg.argmax(1)
            for c, ix in idx.items():
                sub = L[:, ix]
                out[c]["logit"][i:i + chunk] = sub.max(1)
                out[c]["best"][i:i + chunk] = np.array(ix)[sub.argmax(1)]
                out[c]["margin"][i:i + chunk] = sub.max(1) - nl
            for c in CONTRAST:
                d, ok = L[:, cidx[(c, "defect")]], L[:, cidx[(c, "ok")]]
                out[c]["logit"][i:i + chunk] = d.max(1)
                out[c]["best"][i:i + chunk] = np.array(cidx[(c, "ok")])[ok.argmax(1)]  # the strongest "on purpose" reading
                out[c]["margin"][i:i + chunk] = d.max(1) - ok.max(1)
        for c in out:
            out[c]["p"] = 1 / (1 + np.exp(-out[c]["logit"]))
        out["_neg_best"] = neg_best
        return out

    def text(self, i: int) -> str:
        n_pos, n_con = len(self.labels), len(self.contrast)
        if i < n_pos:
            return self.labels[i][1]
        if i < n_pos + n_con:
            return self.contrast[i - n_pos][2]
        return self.negs[i - n_pos - n_con]


# ---------------------------------------------------------------- candidate detection

# Thresholds, calibrated on synthetic degradations of 150 library photos and checked on contact sheets (NOTES.md).
T = {
    "burst_cos": 0.95,         # same session, same camera (when known), shot within burst_window_s
    "burst_window_s": 120,
    "global_cos": 0.978,       # no timestamps / different sessions: stricter (0.97-0.978 was mostly look-alike
                               # textures, skies and moons from different shoots on the demo library)
    "link_slack": 0.02,        # every pair inside a group must be within this of the group's threshold
    "edge_n_min": 100,         # need real edges before judging focus
    "blur_edge_ratio": 1.0,    # Gaussian/defocus: in-focus photos >= 1.3, 1.5px blur <= 0.7
    "blur_dir_ratio": 0.7,     # motion blur along one axis
    "dark_mean": 30.0,         # average luma
    "dark_p999": 50.0,         # brightest 0.1% still dim: no highlight anywhere (moon/city-light shots have one)
    "over_frac": 0.40,         # share of clipped (>245) pixels
    "over_p05": 120.0,         # and no real shadows left
    "veto": {"blurry": 1.0, "dark": 1.0, "overexposed": 2.0},  # min logits by which the defect wording must win
    "black_p999": 25.0,        # pitch black (and not one pixel above 128, e.g. no moon): no SigLIP veto needed
    "acc_margin": 1.0,         # accidental prompt must beat every ordinary subject by this much
    "zs_margin": {"meme": 2.0},  # category prompt must beat every ordinary subject by this (default 0)...
    "zs_p": {"meme": 0.02},      # ...with at least this calibrated probability (default 0.05)
    "ocr_dense": 80,           # characters of OCR text that count as "text-heavy"
    "finger": 0.5,
}
TEXT_KINDS = ("screenshot", "receipt", "document", "whiteboard", "meme")
KINDS = ["exact_duplicate", "near_duplicate", "blurry", "dark", "overexposed", "screenshot", "receipt", "document",
         "whiteboard", "meme", "accidental"]


def _sig(x):
    return float(1 / (1 + np.exp(-x)))


def quality(st: dict) -> float:
    """Which photo of a group to keep: in focus, well exposed, large."""
    if not st:
        return -1e9
    q = np.log1p(st["lap_tile_max"] or 0) + 0.5 * np.log1p(st["lap_var"] or 0) + 1.0 * min(st.get("edge_ratio") or 0, 3)
    q -= 3 * max(0.0, (st["frac_dark"] or 0) - 0.25) + 3 * max(0.0, (st["frac_bright"] or 0) - 0.10)
    q += 0.15 * np.log((st["w"] or 1) * (st["h"] or 1))
    return float(q)


class Detector:
    """Computes all candidates in one pass (~10-30 s for 21k photos), cached until vectors/stats change."""

    def __init__(self, db: sqlite3.Connection, searcher, prompt_cache: str):
        self.db, self.s, self.prompt_cache = db, searcher, prompt_cache
        self._bank = None
        self._cache_key, self._cache = None, None
        self._mu = threading.Lock()

    @property
    def bank(self) -> PromptBank:
        if self._bank is None:
            self._bank = PromptBank(self.prompt_cache, lambda: self.s.lib.encoder)
        return self._bank

    def candidates(self, ids=None, X=None) -> dict[str, list[dict]]:
        ids = np.asarray(self.s.ids if ids is None else ids)
        X = self.s.X if X is None else X
        r = self.db.execute("SELECT count(*), max(computed_at) FROM photo_stats").fetchone()
        k = (len(ids), r[0], r[1])
        with self._mu:
            if self._cache is None or k != self._cache_key:
                self._cache, self._cache_key = self._compute(ids, X), k
            return self._cache

    # -------------------------------------------------------------- the pass
    def _compute(self, ids, X) -> dict[str, list[dict]]:
        stats = {r["id"]: dict(r) for r in self.db.execute("SELECT * FROM photo_stats WHERE err IS NULL")}
        meta = {r["id"]: dict(r) for r in self.db.execute("SELECT id, taken, session, camera, ocr FROM photos")}
        out = {k: [] for k in KINDS}

        # exact duplicates: identical bytes
        by_hash = {}
        for pid, st in stats.items():
            by_hash.setdefault(st["sha1"], []).append(pid)
        dup_of = {}
        for h, g in by_hash.items():
            if len(g) > 1:
                g = sorted(g)
                dup_of.update({p: g[0] for p in g[1:]})
                out["exact_duplicate"].append({
                    "kind": "exact_duplicate", "photo_ids": g, "keep": g[0], "remove": g[1:], "confidence": 0.99,
                    "reasons": [f"{len(g)} files with identical bytes (sha1 {h[:10]})", f"keep #{g[0]}: first imported"]})

        out["near_duplicate"] = self._near_dups(ids, X, stats, meta, dup_of)

        zs = self.bank.score(X)
        for r, pid in enumerate(ids.tolist()):
            st = stats.get(pid)
            if st is not None:
                for c in self.judge(st, zs, r, len((meta.get(pid) or {}).get("ocr") or "")):
                    out[c[0]].append(self._one(c[0], pid, c[1], c[2]))
        for k in out:
            out[k].sort(key=lambda c: -c["confidence"])
        return out

    def judge(self, st: dict, zs: dict, r: int, ocr_n: int = 0) -> list[tuple[str, float, list[str]]]:
        """All single-photo verdicts for row r of a PromptBank.score() result: [(kind, confidence, reasons)]."""
        B, out = self.bank, []
        for c in ("blurry", "dark", "overexposed"):
            hit = getattr(self, "_" + c)(st)
            if not hit:
                continue
            conf, reasons = hit
            m = float(zs[c]["margin"][r])
            pitch_black = c == "dark" and (st.get("p999") or 255) < T["black_p999"] and (st.get("frac_lit") or 0) == 0
            if m < T["veto"][c] and not pitch_black:  # SigLIP isn't convinced it's a mistake (soft-focus art,
                continue                                # low-key, high-key): veto
            if m >= T["veto"][c]:
                reasons.append(f"looks unintentional: \"{CONTRAST[c][0][0]}\" fits better than "
                               f"\"{B.text(int(zs[c]['best'][r]))}\" ({m:+.1f} logits)")
            m = max(m, 2.0) if pitch_black else m
            out.append((c, min(0.97, conf * (0.85 + 0.15 * min(1.0, m / 4))), reasons))
        texty = [c for c in TEXT_KINDS if float(zs[c]["p"][r]) > 0]
        best_text = max(texty, key=lambda c: float(zs[c]["p"][r])) if texty else None
        for c in PROMPTS:
            if c in TEXT_KINDS and c != best_text:  # one label per photo: a receipt is not also a document
                continue
            margin, p = float(zs[c]["margin"][r]), float(zs[c]["p"][r])
            mmin, pmin = T["zs_margin"].get(c, 0.0), T["zs_p"].get(c, 0.05)
            prompt = B.labels[int(zs[c]["best"][r])][1]
            text_heavy = c != "accidental" and ocr_n >= T["ocr_dense"]
            if c == "accidental":
                ok = self._accidental_ok(st, margin, p, prompt)
            else:
                ok = margin > mmin and (p >= pmin or (text_heavy and p >= pmin / 3))
            if not ok:
                continue
            rival = B.negs[int(zs["_neg_best"][r])]
            conf = 0.9 * _sig(1.0 * (margin - mmin) + 0.8 * np.log(p / pmin)) + (0.08 if text_heavy else 0)
            reasons = [f"{DESC[c]}: best match \"{prompt}\" (p={p:.2f}), beats the closest ordinary subject "
                       f"\"{rival}\" by {margin:+.1f} logits"]
            if text_heavy:
                reasons.append(f"text-heavy: {ocr_n} characters of OCR text")
            if c == "accidental":
                reasons += self._accidental_reasons(st)
            out.append((c, min(0.97, conf), reasons))
        return out

    @staticmethod
    def _one(kind, pid, conf, reasons):
        return {"kind": kind, "photo_ids": [pid], "confidence": round(float(conf), 3), "reasons": reasons}

    # pixel-stat rules: return (confidence, reasons) or None
    @staticmethod
    def _blurry(st):
        er, n = st.get("edge_ratio"), st.get("edge_n") or 0
        dirs = [v for v in (st.get("edge_ratio_x"), st.get("edge_ratio_y")) if v is not None]
        if er is None or n < T["edge_n_min"]:
            return None
        md = min(dirs) if dirs else 9.0
        if er < T["blur_edge_ratio"]:
            return (0.6 + 0.35 * min(1.0, (T["blur_edge_ratio"] - er) / 0.5),
                    [f"out of focus: even the strongest edges are soft (edge sharpness {er:.2f}; in-focus photos score "
                     f"above 1.3)"])
        if md < T["blur_dir_ratio"]:
            ax = "horizontal" if st.get("edge_ratio_x") == md else "vertical"
            return (0.55 + 0.35 * min(1.0, (T["blur_dir_ratio"] - md) / 0.4),
                    [f"motion blur: edges smeared in the {ax} direction (directional sharpness {md:.2f} vs > 0.8 when steady)"])
        return None

    @staticmethod
    def _dark(st):
        if st["luma_mean"] < T["dark_mean"] and (st.get("p999") or 255) < T["dark_p999"]:
            return (0.6 + 0.35 * min(1.0, (T["dark_p999"] - st["p999"]) / T["dark_p999"]),
                    [f"almost black: average brightness {st['luma_mean']:.0f}/255 and even the brightest spot only "
                     f"reaches {st['p999']:.0f}"])
        return None

    @staticmethod
    def _overexposed(st):
        if st["frac_bright"] > T["over_frac"] and st["p05"] > T["over_p05"]:
            return (0.45 + 0.15 * min(1.0, (st["frac_bright"] - T["over_frac"]) / 0.4),  # capped: high-key art looks alike
                    [f"washed out: {st['frac_bright']:.0%} of pixels blown to white, darkest tones only reach "
                     f"{st['p05']:.0f}/255"])
        return None

    @staticmethod
    def _low_detail(st) -> list[str]:
        """Pixel evidence that nothing was really photographed (low entropy alone is not enough: low-key art has it)."""
        r = []
        if (st.get("edge_n") or 0) < T["edge_n_min"]:
            r.append("almost no edges anywhere")
        if Detector._blurry(st):
            r.append(Detector._blurry(st)[1][0])
        if Detector._dark(st):
            r.append("nearly black")
        return r

    def _accidental_ok(self, st, margin, p, prompt) -> bool:
        if margin < T["acc_margin"] or p < 0.025:
            return False
        if "finger" in prompt:
            return st["finger"] >= T["finger"]
        return bool(self._low_detail(st))

    def _accidental_reasons(self, st):
        r = self._low_detail(st)
        if st["finger"] >= T["finger"]:
            r.append(f"soft skin-coloured blob covers {st['finger']:.0%} more of an edge than the centre (finger on lens?)")
        return r

    def _near_dups(self, ids, X, stats, meta, dup_of, chunk=1024) -> list[dict]:
        rows = np.flatnonzero([int(i) not in dup_of for i in ids])  # exact copies are reported separately
        Xr = X[rows].astype(np.float32)
        pid = ids[rows].astype(int)
        sess = np.array([(meta.get(int(p)) or {}).get("session") if (meta.get(int(p)) or {}).get("session") is not None
                         else -1 for p in pid])
        taken = np.array([(meta.get(int(p)) or {}).get("taken") or np.nan for p in pid], dtype=np.float64)
        cams = {}
        cam = np.array([cams.setdefault((meta.get(int(p)) or {}).get("camera"), len(cams)) if (meta.get(int(p)) or {}).get("camera")
                        else -1 for p in pid])
        lo = min(T["burst_cos"], T["global_cos"])
        pairs = []
        for i0 in range(0, len(rows), chunk):
            guard.check()
            S = Xr[i0:i0 + chunk] @ Xr.T
            ii, jj = np.nonzero(S >= lo)
            keep = jj > ii + i0
            for a, b in zip(ii[keep].tolist(), jj[keep].tolist()):
                a2, s = a + i0, float(S[a, b])
                burst = (sess[a2] >= 0 and sess[a2] == sess[b] and abs(taken[a2] - taken[b]) <= T["burst_window_s"]
                         and (cam[a2] < 0 or cam[b] < 0 or cam[a2] == cam[b]))
                if s >= (T["burst_cos"] if burst else T["global_cos"]):
                    pairs.append((s, a2, b, burst))
        # complete-ish linkage: strongest pairs first; merge two groups only if every cross pair is similar enough,
        # so A~B, B~C never drags in an unrelated C (union-find chained 17 textures into one group)
        pairs.sort(reverse=True)
        group, members, burst_of = {}, {}, {}
        for s, a, b, burst in pairs:
            ga, gb = group.get(a, ("s", a)), group.get(b, ("s", b))
            if ga == gb:
                continue
            A, Bm = members.get(ga, [a]), members.get(gb, [b])
            need = (T["burst_cos"] if burst else T["global_cos"]) - T["link_slack"]
            if len(A) * len(Bm) > 1 and (Xr[A] @ Xr[Bm].T).min() < need:
                continue
            g = ga if ga in members else gb if gb in members else ("g", a)
            for m in A + Bm:
                group[m] = g
            members.pop(ga, None); members.pop(gb, None)
            members[g] = A + Bm
            burst_of[g] = burst_of.get(ga, True) and burst_of.get(gb, True) and burst
        out = []
        for g, mem in members.items():
            ps = sorted(int(pid[m]) for m in mem)
            sub = Xr[mem] @ Xr[mem].T
            smin = float(sub[np.triu_indices(len(mem), 1)].min())
            burst = burst_of[g]
            q = {p: quality(stats.get(p, {})) for p in ps}
            keep = max(ps, key=q.get)
            ks = stats.get(keep, {})
            others = [stats[p]["edge_ratio"] or 0 for p in ps if p != keep and p in stats]
            runner_up = max((q[p] for p in ps if p != keep), default=-1e9)
            if q[keep] - runner_up < 0.05:
                why_keep = f"keep #{keep}: the photos are equally sharp and exposed; kept the earliest import"
            else:
                why_keep = (f"keep #{keep}: sharpest and best exposed (edge sharpness {ks.get('edge_ratio') or 0:.2f}"
                            + (f" vs {np.median(others):.2f} for the others)" if others else ")"))
            when = ""
            if burst:
                ts = taken[mem]
                span = float(np.nanmax(ts) - np.nanmin(ts))
                when = f", same camera, shot {span:.0f} s apart (burst)"
            base = T["burst_cos"] if burst else T["global_cos"]
            conf = 0.6 + 0.35 * min(1.0, max(0.0, smin - base + T["link_slack"]) / (1 - base + T["link_slack"]))
            out.append({"kind": "near_duplicate", "photo_ids": ps, "keep": keep, "remove": [p for p in ps if p != keep],
                        "confidence": round(conf, 3), "burst": bool(burst),
                        "reasons": [f"{len(ps)} near-identical photos (visual similarity >= {smin:.3f}){when}", why_keep]})
        return out


def flatten(cands: dict[str, list[dict]], kind: str | None, exclude: set[int]) -> list[dict]:
    """Drop trashed photos (groups shrink; groups of one vanish)."""
    kinds = [kind] if kind else KINDS
    out = []
    for k in kinds:
        for c in cands.get(k, []):
            ids = [p for p in c["photo_ids"] if p not in exclude]
            if not ids or (("keep" in c) and len(ids) < 2):
                continue
            if len(ids) != len(c["photo_ids"]):
                c = dict(c, photo_ids=ids)
                if "keep" in c:
                    c["keep"] = c["keep"] if c["keep"] in ids else ids[0]
                    c["remove"] = [p for p in ids if p != c["keep"]]
            out.append(c)
    return out


def contact_sheet(paths_labels: list[tuple[str, str]], out_path: str, cell: int = 200, cols: int = 8):
    """Numbered thumbnails for eyeballing precision."""
    from PIL import ImageDraw
    n = len(paths_labels)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (p, lab) in enumerate(paths_labels):
        try:
            im = Image.open(p); im.draft("RGB", (cell * 2, cell * 2))
            im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((cell - 4, cell - 4))
        except Exception:
            im = Image.new("RGB", (cell - 4, cell - 4), "red")
        x, y = (i % cols) * cell, (i // cols) * (cell + 16)
        sheet.paste(im, (x + 2, y + 2))
        d.rectangle([x, y + cell - 2, x + cell, y + cell + 14], fill="white")
        d.text((x + 3, y + cell), f"{i + 1}. {lab}"[:34], fill="black")
    sheet.save(out_path, quality=85)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "stats":
    INDEX = os.environ.get("PHOTOSEARCH_INDEX", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "index"))
    precompute_stats(f"{INDEX}/library.sqlite", limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
