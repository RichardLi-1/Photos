"""Photo library index: SQLite (metadata, OCR, full-text) + an embedding matrix. Everything stays local."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

import numpy as np
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")
SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
  id INTEGER PRIMARY KEY, path TEXT UNIQUE, album TEXT, taken REAL, lat REAL, lon REAL,
  place TEXT, camera TEXT, width INT, height INT, ocr TEXT, ocr_done INT DEFAULT 0, session INT);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(ocr, place, album, content='photos', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS photos_au AFTER UPDATE OF ocr, place, album ON photos BEGIN
  INSERT INTO fts(fts, rowid, ocr, place, album) VALUES('delete', old.id, old.ocr, old.place, old.album);
  INSERT INTO fts(rowid, ocr, place, album) VALUES (new.id, new.ocr, new.place, new.album);
END;
CREATE TRIGGER IF NOT EXISTS photos_ai AFTER INSERT ON photos BEGIN
  INSERT INTO fts(rowid, ocr, place, album) VALUES (new.id, new.ocr, new.place, new.album);
END;
-- fast-mode OCR text kept when a photo is re-OCR'd in accurate mode (a row = upgraded). Search only ranks fts.ocr;
-- fts.album stays indexed for compatibility but never votes.
CREATE TABLE IF NOT EXISTS ocr_fast (id INTEGER PRIMARY KEY, ocr TEXT);
"""


def _exif(path: str) -> dict:
    """Capture time, GPS and camera from EXIF (what Apple Photos / a phone would record)."""
    out = {}
    try:
        im = Image.open(path)
        out["width"], out["height"] = im.size
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        dt = sub.get(36867) or ex.get(306)
        if dt:
            out["taken"] = datetime.strptime(str(dt)[:19], "%Y:%m:%d %H:%M:%S").timestamp()
        cam = " ".join(str(x).strip() for x in (ex.get(271), ex.get(272)) if x)
        if cam:
            out["camera"] = cam
        gps = ex.get_ifd(0x8825)
        if 2 in gps and 4 in gps:
            dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
            out["lat"] = dms(gps[2]) * (-1 if gps.get(1) == "S" else 1)
            out["lon"] = dms(gps[4]) * (-1 if gps.get(3) == "W" else 1)
    except Exception:
        pass
    return out


class Library:
    def __init__(self, index_dir: str, model: str = "siglip2-b16-384", readonly: bool = False):
        self.dir, self.model_name = index_dir, model
        if readonly:  # servers: never write, so indexers can keep running
            self.db = sqlite3.connect(f"file:{index_dir}/library.sqlite?mode=ro", uri=True, check_same_thread=False, timeout=60)
            self.db.row_factory = sqlite3.Row
        else:
            os.makedirs(index_dir, exist_ok=True)
            self.db = sqlite3.connect(f"{index_dir}/library.sqlite", check_same_thread=False, timeout=120)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")  # indexer jobs + server read/write concurrently
            old = self.db.execute("SELECT sql FROM sqlite_master WHERE name='photos_au'").fetchone()
            if old and "album" not in old[0].split("ON photos")[0]:  # older trigger missed album changes
                self.db.execute("DROP TRIGGER photos_au")
            self.db.executescript(SCHEMA)
        self._vec_path = f"{index_dir}/vectors__{model}.npz"
        self._load_vectors()
        self._enc = None

    # ---------- ingest ----------
    def add_folder(self, folder: str, album: str | None = None) -> int:
        """Register photos (metadata only). Optional `_meta.json` sidecar: {filename: {date, lat, lon, place, camera}}."""
        side = {}
        if os.path.exists(f"{folder}/_meta.json"):
            side = json.load(open(f"{folder}/_meta.json"))
        known = {r[0] for r in self.db.execute("SELECT path FROM photos")}
        rows = []
        for dirpath, _, names in os.walk(folder):
            for n in sorted(names):
                p = os.path.abspath(os.path.join(dirpath, n))
                if not n.lower().endswith(EXTS) or p in known:
                    continue
                m = _exif(p)
                s = side.get(n) or {}
                if s.get("date") and "taken" not in m:
                    m["taken"] = datetime.fromisoformat(s["date"][:19]).timestamp()
                for k in ("lat", "lon", "place", "camera"):
                    if s.get(k) is not None:
                        m.setdefault(k, s[k])
                rows.append((p, album or os.path.basename(folder.rstrip("/")), m.get("taken"), m.get("lat"), m.get("lon"),
                             m.get("place"), m.get("camera"), m.get("width"), m.get("height")))
        self.db.executemany("INSERT INTO photos(path, album, taken, lat, lon, place, camera, width, height) VALUES (?,?,?,?,?,?,?,?,?)", rows)
        self.db.commit()
        return len(rows)

    def assign_sessions(self, gap_s: float = 45 * 60, km: float = 1.0):
        """Group photos into shooting sessions: consecutive in time (< gap) and, when GPS exists, nearby."""
        rows = self.db.execute("SELECT id, taken, lat, lon FROM photos WHERE taken IS NOT NULL ORDER BY taken").fetchall()
        sess, prev, upd = -1, None, []
        for r in rows:
            new = prev is None or r["taken"] - prev["taken"] > gap_s
            if not new and None not in (r["lat"], prev["lat"]):
                new = _haversine_km(r["lat"], r["lon"], prev["lat"], prev["lon"]) > km
            sess += int(new)  # plain int: numpy ints are stored as BLOBs by sqlite3
            upd.append((sess, r["id"])); prev = r
        self.db.executemany("UPDATE photos SET session=? WHERE id=?", upd)
        self.db.commit()

    @property
    def encoder(self):
        if self._enc is None:
            from . import encoders
            self._enc = encoders.get(self.model_name)
        return self._enc

    def embed_pending(self, chunk: int = 256, log=print):
        have = set(self.ids.tolist())
        todo = [(r[0], r[1]) for r in self.db.execute("SELECT id, path FROM photos ORDER BY id") if r[0] not in have]
        buf, done = [], 0
        for X in self.encoder.iter_images([p for _, p in todo]):
            buf.append(X.astype(np.float16))
            n = sum(len(b) for b in buf)
            if n >= chunk or done + n == len(todo):
                X = np.concatenate(buf)
                self.ids = np.concatenate([self.ids, np.array([j for j, _ in todo[done:done + n]], dtype=np.int64)])
                self.X = np.concatenate([self.X, X]) if len(self.X) else X
                tmp = self._vec_path + ".tmp.npz"
                np.savez(tmp, ids=self.ids, X=self.X)
                os.replace(tmp, self._vec_path)  # atomic: readers never see a half-written file
                done += n; buf = []
                log(f"embedded {done}/{len(todo)}")

    def ocr_pending(self, log=print):
        from .ocr import ocr
        todo = self.db.execute("SELECT id, path FROM photos WHERE ocr_done=0").fetchall()
        for n, r in enumerate(todo):
            self.db.execute("UPDATE photos SET ocr=?, ocr_done=1 WHERE id=?", (ocr(r["path"]) or None, r["id"]))
            if n % 200 == 0:
                self.db.commit(); log(f"ocr {n}/{len(todo)}")
        self.db.commit()

    def ocr_accurate(self, ids, batch: int = 25, log=print):
        """Re-OCR `ids` with Vision's accurate mode + language correction. Resumable; old text kept in ocr_fast."""
        import objc
        from . import guard
        from .ocr import ocr
        guard.acquire(1)
        done = {r[0] for r in self.db.execute("SELECT id FROM ocr_fast")}
        todo = [int(i) for i in ids if int(i) not in done]
        for n in range(0, len(todo), batch):
            guard.check()
            rows = self.db.execute(f"SELECT id, path, ocr FROM photos WHERE id IN ({','.join('?' * len(todo[n:n + batch]))})",
                                   todo[n:n + batch]).fetchall()
            res = []
            for r in rows:  # OCR outside the transaction: the server keeps reading meanwhile
                with objc.autorelease_pool():
                    res.append((r["id"], r["ocr"], ocr(r["path"], fast=False) or None))
            with self.db:  # small WAL transaction per batch
                self.db.executemany("INSERT OR IGNORE INTO ocr_fast(id, ocr) VALUES (?,?)", [(i, old) for i, old, _ in res])
                self.db.executemany("UPDATE photos SET ocr=? WHERE id=?", [(new, i) for i, _, new in res])
            log(f"ocr accurate {n + len(rows)}/{len(todo)}")

    # ---------- read ----------
    def read_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """(ids, X) from disk without touching self; closes the file."""
        if not os.path.exists(self._vec_path):
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float16)
        with np.load(self._vec_path, allow_pickle=False) as z:
            return z["ids"], z["X"]

    def _load_vectors(self):
        self.ids, self.X = self.read_vectors()
        self.row_of = {int(i): n for n, i in enumerate(self.ids)}

    def get(self, ids) -> list[dict]:
        if not len(ids):
            return []
        q = f"SELECT * FROM photos WHERE id IN ({','.join('?' * len(ids))})"
        by = {r["id"]: dict(r) for r in self.db.execute(q, [int(i) for i in ids])}
        return [by[int(i)] for i in ids if int(i) in by]


def _haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))
