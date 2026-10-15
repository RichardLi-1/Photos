"""Hybrid retrieval over a Library: embeddings (text/image/both) + OCR full-text + hard metadata filters.

Search-box text goes through query.parse (Query.parse=True): dates and places become filters that must ALL hold,
the rest is ranked visually. Soft-deleted photos (photos.deleted_at, when that column exists) never come back.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
from PIL import Image

from .library import Library, _haversine_km


@dataclass
class Filters:
    after: str | None = None    # ISO date
    before: str | None = None
    dates: list | None = None   # [(after, before), ...] ISO ranges, any may match ("every christmas")
    place: str | None = None    # place name(s), "A|B"; aliases + sub-places via places.py ("Germany" ~ "Deutschland")
    album: str | None = None
    near: tuple[float, float, float] | None = None  # lat, lon, km
    has_text: bool = False      # only photos with OCR'd text

    def active(self):
        return any([self.after, self.before, self.dates is not None, self.place, self.album, self.near, self.has_text])


@dataclass
class Query:
    text: str | None = None
    image: Image.Image | int | None = None  # PIL image or a library photo id
    text_weight: float = 0.5                # when both image + text: how much the text steers
    filters: Filters = field(default_factory=Filters)
    k: int = 60
    parse: bool = False                     # search box: pull dates/places out of `text` (photosearch.query)
    parsed: dict | None = None              # out: what the parser understood
    mode: str = "similar"                   # photo queries: "similar" (general look) | "place" (same room/place, trained head)


def _ts(s):
    return datetime.fromisoformat(s).timestamp()


def _toks(s):
    return re.findall(r"\w+", s.lower())


def _stem(w):
    for suf in ("ies", "es", "s", "ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


MAX_QUERY_WORDS, MAX_PHRASE = 32, 8
LITERAL = re.compile(r"[\"“”]([^\"“”]{2,})[\"“”]|\b(?:says?|saying|reads?|reading|written|labell?ed|titled|with the words?)\s+(.+)$", re.I)


class Searcher:
    RRF_K = 60

    def __init__(self, lib: Library):
        self.lib = lib
        self.refresh()

    def refresh(self):
        """Build everything into locals first, then swap it all in: a failure leaves the old state consistent."""
        from .places import Places
        ids, X16 = self.lib.read_vectors()
        meta = {r[0]: r[1:] for r in self.lib.db.execute("SELECT id, session, taken, place, album, ocr IS NOT NULL, lat, lon FROM photos")}
        sess = lambda v: int.from_bytes(v, "little", signed=True) if isinstance(v, bytes) else v  # old BLOB sessions
        m = [meta.get(int(i), (None,) * 7) for i in ids]
        session = np.array([-1 if r[0] is None else sess(r[0]) for r in m])
        taken = np.array([np.nan if r[1] is None else r[1] for r in m], dtype=np.float64)
        place_rows = defaultdict(list)
        for n, r in enumerate(m):
            if r[2]:
                place_rows[r[2]].append(n)
        album = np.array([r[3] or "" for r in m], dtype=object)  # per-row columns so filters are vectorised (no SQL per search)
        has_text = np.array([bool(r[4]) for r in m], dtype=bool)
        lat = np.array([np.nan if r[5] is None else r[5] for r in m], dtype=np.float64)
        lon = np.array([np.nan if r[6] is None else r[6] for r in m], dtype=np.float64)
        places, X, row_of = Places(place_rows), X16.astype(np.float32), {int(i): n for n, i in enumerate(ids)}
        self.lib.ids, self.lib.X, self.lib.row_of = ids, X16, row_of
        self.ids, self.X, self.session, self.taken, self.place_rows, self.places = ids, X, session, taken, place_rows, places
        self.album, self.has_text, self.lat, self.lon = album, has_text, lat, lon
        self.Z = self._place_space(X)
        self._has_deleted = False

    def _place_space(self, X):
        """Our own trained "same place" head (train/): SigLIP2 vectors -> a space where photos of one room/place cluster.

        hotels: R@1 0.196 -> 0.347, mAP 0.113 -> 0.242 vs raw SigLIP2 (NOTES.md). Optional: no file, no place mode.
        """
        path = os.environ.get("PHOTOSEARCH_HEAD", os.path.join(self.lib.dir, "..", "models", "hotel_head.pt"))
        if not os.path.exists(path) or not len(X):
            return None
        try:
            if getattr(self, "_head", None) is None:
                from . import heads
                self._head = heads.load(path)
            import torch
            with torch.inference_mode():
                return self._head.embed(X).astype(np.float32)
        except Exception as e:
            print(f"[search] place head unavailable: {e}")
            return None

    # ---------- pieces ----------
    def embed_text(self, text: str) -> np.ndarray:
        return self.lib.encoder.texts([text])[0]

    def embed_prompted(self, text: str) -> np.ndarray:
        """Prompt ensemble: mean of the text and "a photo of <text>" (eval: +0.03 hit@1, +0.015 MRR on the search box)."""
        v = self.embed_text(text) + self.embed_text(f"a photo of {text}")
        return v / np.linalg.norm(v)

    def embed_image(self, image) -> np.ndarray:
        if isinstance(image, (int, np.integer)):
            return self.X[self.lib.row_of[int(image)]]
        return self.lib.encoder.images([image])[0]

    def live(self) -> np.ndarray:
        """Rows not soft-deleted. Works before and after another module adds photos.deleted_at."""
        if not self._has_deleted:
            self._has_deleted = any(r[1] == "deleted_at" for r in self.lib.db.execute("PRAGMA table_info(photos)"))
        if not self._has_deleted:
            return np.ones(len(self.ids), bool)
        gone = [r[0] for r in self.lib.db.execute("SELECT id FROM photos WHERE deleted_at IS NOT NULL")]
        return ~np.isin(self.ids, gone) if gone else np.ones(len(self.ids), bool)

    def place_mask(self, place: str) -> np.ndarray:
        rows = []
        for name in place.split("|"):
            ok = self.places.match(name.strip())
            rows += [r for p, rs in self.place_rows.items() if ok(p) for r in rs]
        m = np.zeros(len(self.ids), bool)
        m[rows] = True
        return m

    def mask(self, f: Filters) -> np.ndarray:
        ok = self.live()
        if not f.active():
            return ok
        t = self.taken
        with np.errstate(invalid="ignore"):
            if f.after:
                ok &= t >= _ts(f.after)
            if f.before:
                ok &= t < _ts(f.before)
            if f.dates is not None:
                ok &= np.any([(t >= _ts(a)) & (t < _ts(b)) for a, b in f.dates], axis=0) if f.dates else False
        if f.place:
            ok &= self.place_mask(f.place)
        if f.album:
            ok &= self.album == f.album
        if f.has_text:
            ok &= self.has_text
        if f.near:
            lat, lon, km = f.near
            with np.errstate(invalid="ignore"):
                ok &= _haversine_km(lat, lon, self.lat, self.lon) <= km  # NaN (no GPS) compares False
        return ok

    def fulltext(self, text: str, limit: int = 300) -> tuple[list[int], set[int]]:
        """Photos whose OCR'd text matches the query, best first, and the subset matching a phrase exactly.

        Order: exact phrase, then more distinct query words, then BM25. Only the `ocr` column is searched: place
        names are handled as hard filters by the parser, and the album name never votes.
        """
        from .query import STOP
        toks = _toks(text)[:MAX_QUERY_WORDS]  # phrase matching below is O(words^2 * MAX_PHRASE) per hit
        words = [w for w in dict.fromkeys(toks) if len(w) > 2 and w not in STOP]
        if not words:
            return [], set()
        try:
            ids = [r[0] for r in self.lib.db.execute(
                "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
                ("ocr : (" + " OR ".join(f'"{w}"' for w in words) + ")", limit))]
        except Exception:
            return [], set()
        ocr = dict(self.lib.db.execute(f"SELECT id, ocr FROM photos WHERE id IN ({','.join('?' * len(ids))})", ids)) if ids else {}
        m = LITERAL.search(text)
        literal = _toks(m.group(1) or m.group(2)) if m else []
        stems, qs = {_stem(w) for w in words}, [_stem(w) for w in toks]
        exact, key = set(), {}
        for n, i in enumerate(ids):
            o = [_stem(w) for w in _toks(ocr.get(i) or "")]
            have, line = stems & set(o), " " + " ".join(o) + " "
            run = 0  # longest run of consecutive query words found as a phrase, if it carries meaning
            for a in range(len(qs)):
                for b in range(min(len(qs), a + MAX_PHRASE), a + 1, -1):
                    content = sum(w not in STOP for w in toks[a:b])
                    if b - a > run and (content >= 2 or (b - a >= 3 and content)) and " " + " ".join(qs[a:b]) + " " in line:
                        run = b - a
                        break
            lit = len(literal) >= 1 and " " + " ".join(map(_stem, literal)) + " " in line
            if lit or run:
                exact.add(i)
            key[i] = (-lit, -run, -len(have), n)
        return sorted(ids, key=key.get), exact

    # ---------- main entry ----------
    def _same_place(self, q: Query, f: Filters) -> list[dict]:
        """Other photos of the same room/place: cosine in the trained head's space (+ shooting-session propagation)."""
        if isinstance(q.image, (int, np.integer)):
            z = self.Z[self.lib.row_of[int(q.image)]]
        else:
            import torch
            with torch.inference_mode():
                z = self._head.embed(self.embed_image(q.image)[None].astype(np.float32))[0]
        sim = self._session_propagate(self.Z @ z, q.image)
        sim[~self.mask(f)] = -np.inf
        if isinstance(q.image, (int, np.integer)):
            sim[self.lib.row_of[int(q.image)]] = -np.inf
        rows = [r for r in np.argsort(-sim)[: q.k] if np.isfinite(sim[r])]
        out = self.lib.get([int(self.ids[r]) for r in rows])
        for o in out:
            o["score"] = round(float(sim[self.lib.row_of[o["id"]]]), 4)
        return out

    def search(self, q: Query) -> list[dict]:
        text, f = q.text, q.filters
        if q.parse and text:
            from .query import parse
            vis, pf = parse(text, self.places)
            f = Filters(**{k: v if v not in (None, False) else getattr(pf, k) for k, v in asdict(f).items()})  # explicit wins
            q.parsed = {"text": vis, **{k: v for k, v in asdict(pf).items() if v not in (None, False)}}
            if pf.active():
                text = vis or None
        if text is None and q.image is None:
            return self._by_date(f, q.k) if q.parse and f.active() else []
        if q.mode == "place" and q.image is not None and text is None and self.Z is not None:
            return self._same_place(q, f)
        vecs = []
        if q.image is not None:
            vecs.append(self.embed_image(q.image))
        if text:
            t = self.embed_prompted(text) if q.parse else self.embed_text(text)
            tw = min(max(q.text_weight, 0.01), 0.99)  # 1.0 would divide by zero
            vecs.append(t * (tw / (1 - tw)) if q.image is not None else t)
        v = np.sum(vecs, 0)
        v /= np.linalg.norm(v)
        sim = self.X @ v

        if q.image is not None and text is None:
            sim = self._session_propagate(sim, q.image)
        sim[~self.mask(f)] = -np.inf
        if isinstance(q.image, (int, np.integer)):
            sim[self.lib.row_of[int(q.image)]] = -np.inf

        order = np.argsort(-sim)[: max(q.k * 4, 200)]
        fused = {int(self.ids[r]): 1 / (self.RRF_K + n) for n, r in enumerate(order) if np.isfinite(sim[r])}
        if text:  # literal text in the photo gets a vote; an exact phrase match goes to the top
            ft, exact = self.fulltext(text)
            ok = {i for i in ft if (r := self.lib.row_of.get(i)) is not None and np.isfinite(sim[r])}
            for n, pid in enumerate(i for i in ft if i in ok):
                fused[pid] = fused.get(pid, 0) + 0.8 / (self.RRF_K + n) + (2 / self.RRF_K if pid in exact else 0)
        if text and not q.parse:  # raw text from other callers: places it names get a vote (the search box filters instead)
            for name in {n for *_, n in self.places.find(text)}:
                rows = np.flatnonzero(self.place_mask(name) & np.isfinite(sim))
                for n, r in enumerate(rows[np.argsort(-sim[rows])][:300]):
                    fused[int(self.ids[r])] = fused.get(int(self.ids[r]), 0) + 0.8 / (self.RRF_K + n)
        top = sorted(fused, key=fused.get, reverse=True)[: q.k]
        out = self.lib.get(top)
        for o in out:
            r = self.lib.row_of.get(o["id"])
            o["score"] = round(float(sim[r]), 4) if r is not None else None
        return out

    def _by_date(self, f: Filters, k: int) -> list[dict]:
        """Nothing visual to rank by ("sept 29 2025", "photos from Iceland"): the filter's matches, newest first."""
        rows = np.flatnonzero(self.mask(f))
        rows = rows[np.argsort(-np.nan_to_num(self.taken[rows], nan=-np.inf), kind="stable")][:k]
        out = self.lib.get(self.ids[rows])
        for o in out:
            o["score"] = None
        return out

    def _session_propagate(self, sim, image, strong=0.6, gain=0.9):
        """Photos shot in the same session as a strong visual match inherit (most of) its score.

        Bridges views that look nothing alike (bed vs bathroom of the same hotel room) but were taken minutes apart.
        """
        if (self.session < 0).all():
            return sim
        sim = sim.copy()
        best = {}
        for r in np.flatnonzero(sim >= strong):
            s = self.session[r]
            if s >= 0:
                best[s] = max(best.get(s, -1), sim[r])
        for s, b in best.items():
            rows = np.flatnonzero(self.session == s)
            sim[rows] = np.maximum(sim[rows], gain * b)
        return sim
