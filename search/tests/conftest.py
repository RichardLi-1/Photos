"""Shared fixtures: tiny generated photo library, a fake encoder (no models), and a sandboxed ML guard.

Every fast test runs with:
  - `Library.encoder` replaced by FakeEncoder (colour -> vector for images, colour words -> vector for text);
  - `photosearch.encoders.get` booby-trapped, so an accidental real model load fails loudly instead of loading;
  - guard slot files redirected to a temp dir, so tests never touch the real machine-wide ML budget in /tmp.
Tests marked `slow` (opt-in: `-m slow`) get none of this and go through the real guard.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import zlib
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

SEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SEARCH_DIR not in sys.path:
    sys.path.insert(0, SEARCH_DIR)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Modules compute their default index path at import time (tools.INDEX, server.INDEX). Point them at a path that
# does not exist, so nothing imported during collection can ever open (or migrate) the user's real data/index.
os.environ["PHOTOSEARCH_INDEX"] = os.path.join(tempfile.mkdtemp(prefix="photosearch-tests-"), "no-index")

from photosearch import guard  # noqa: E402
from photosearch.library import Library  # noqa: E402

REAL_CHECK = guard.check

# ---------------------------------------------------------------- fake encoder

DIM = 16
COLOR_AXIS = {"red": 0, "green": 1, "blue": 2}


def _word_axis(w: str) -> int:
    return 3 + zlib.crc32(w.encode()) % (DIM - 3)


class FakeEncoder:
    """Deterministic stand-in for SigLIP. Images: mean RGB on axes 0-2. Text: colour words on axes 0-2, others hashed."""

    name = "fake"
    has_text = True
    model = types.SimpleNamespace(logit_bias=None)  # Explainer falls back to scale=100, bias=-10

    def __init__(self):
        self.text_calls = 0
        self.image_calls = 0

    @staticmethod
    def _norm(v):
        return (v / np.linalg.norm(v)).astype(np.float32)

    def image_vec(self, im) -> np.ndarray:
        if not isinstance(im, Image.Image):
            im = Image.open(im)
        rgb = np.asarray(im.convert("RGB").resize((4, 4)), dtype=np.float32).reshape(-1, 3).mean(0) / 255
        v = np.zeros(DIM, np.float32)
        v[:3] = rgb
        v[3:] = 0.02
        return self._norm(v)

    def text_vec(self, t: str) -> np.ndarray:
        v = np.zeros(DIM, np.float32)
        for w in t.lower().split():
            if w in COLOR_AXIS:
                v[COLOR_AXIS[w]] += 1.0
            else:
                v[_word_axis(w)] += 0.3
        if not v.any():
            v[3] = 1.0
        return self._norm(v)

    def texts(self, texts, bs=256):
        self.text_calls += 1
        return np.stack([self.text_vec(t) for t in texts])

    def images(self, items, bs=32):
        self.image_calls += 1
        return np.stack([self.image_vec(x) for x in items])

    def iter_images(self, items, bs=32):
        for i in range(0, len(items), bs):
            yield self.images(items[i:i + bs])


FAKE = FakeEncoder()


@pytest.fixture
def fake_encoder():
    return FAKE


# ---------------------------------------------------------------- safety: no models, private guard slots

def _no_model(name, *a, **k):
    raise AssertionError(f"fast test tried to load a real model ({name!r}); mark it @pytest.mark.slow")


@pytest.fixture(autouse=True)
def _sandbox(request, monkeypatch, tmp_path_factory):
    if request.node.get_closest_marker("slow"):
        yield
        return
    monkeypatch.setattr(Library, "encoder", property(lambda self: FAKE))
    enc = sys.modules.get("photosearch.encoders")
    if enc is not None:
        monkeypatch.setattr(enc, "get", _no_model)
    slots = tmp_path_factory.mktemp("slots")
    monkeypatch.setattr(guard, "SLOT_PATH", str(slots / "slot{}"))
    monkeypatch.setattr(guard, "SERVER_SLOT", str(slots / "server"))  # never see the real UI server's reserved slot
    monkeypatch.setattr(guard, "_held", [])
    monkeypatch.setattr(guard, "check", lambda *a, **k: None)  # no sysctl / memory_pressure polling in fast tests
    yield
    for f in guard._held:
        f.close()


# ---------------------------------------------------------------- tiny photo library

SYDNEY, PARIS, LONDON = (-33.8688, 151.2093), (48.8566, 2.3522), (51.5072, -0.1276)

# name: (folder/album, rgb, taken, (lat, lon) or None)
PHOTOS = {
    "trip_1": ("trip", (230, 20, 20), "2023:06:01 10:00:00", SYDNEY),
    "trip_2": ("trip", (210, 40, 30), "2023:06:01 10:05:00", SYDNEY),
    "trip_3": ("trip", (250, 60, 40), "2023:06:01 10:10:00", SYDNEY),
    "trip_4": ("trip", (190, 30, 60), "2023:06:01 10:15:00", SYDNEY),
    "home_1": ("home", (20, 30, 230), "2024:01:10 09:00:00", PARIS),
    "home_2": ("home", (40, 40, 210), "2024:01:10 09:20:00", PARIS),
    "home_3": ("home", (30, 60, 250), "2024:01:10 09:30:00", LONDON),  # 10 min later but 340 km away: new session
    "misc_1": ("misc", (20, 220, 30), None, None),
    "misc_2": ("misc", (40, 200, 60), None, None),
    "misc_3": ("misc", (120, 120, 120), "2022:03:03 12:00:00", None),
}
PLACES = {"trip": "Sydney, Australia", "home": "Paris, France"}
OCR = {"misc_1": "Lunch MENU pasta 12.50", "home_2": "Louvre ticket admission"}


def _dms(x):
    x = abs(x)
    d = int(x); m = int((x - d) * 60); s = round(((x - d) * 60 - m) * 60, 4)
    return (float(d), float(m), float(s))


def write_jpeg(path, rgb, taken=None, gps=None, camera=("Apple", "iPhone 15"), size=(24, 16)):
    ex = Image.Exif()
    if camera:
        ex[271], ex[272] = camera
    if taken:
        ex.get_ifd(0x8769)[36867] = taken
    if gps:
        g = ex.get_ifd(0x8825)
        g[1], g[2] = ("N" if gps[0] >= 0 else "S"), _dms(gps[0])
        g[3], g[4] = ("E" if gps[1] >= 0 else "W"), _dms(gps[1])
    Image.new("RGB", size, rgb).save(path, "JPEG", quality=90, exif=ex)


def make_photos(root) -> dict[str, str]:
    """Write the 10 test photos (+ a non-image file) under root/<album>/; returns name -> path."""
    out = {}
    for name, (album, rgb, taken, gps) in PHOTOS.items():
        d = os.path.join(root, album)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{name}.jpg")
        write_jpeg(p, rgb, taken, gps)
        out[name] = os.path.abspath(p)
    open(os.path.join(root, "misc", "notes.txt"), "w").write("not a photo")
    return out


def build_index(index_dir, photo_root, embed=True) -> dict:
    """Ingest + sessions + places/OCR (via SQL, no Vision) + fake embeddings. Returns name -> id."""
    paths = make_photos(photo_root)
    lib = Library(index_dir)
    for album in ("trip", "home", "misc"):
        lib.add_folder(os.path.join(photo_root, album))
    lib.assign_sessions()
    for album, place in PLACES.items():
        lib.db.execute("UPDATE photos SET place=? WHERE album=?", (place, album))
    by_path = {r["path"]: r["id"] for r in lib.db.execute("SELECT id, path FROM photos")}
    ids = {n: by_path[p] for n, p in paths.items()}
    for n, text in OCR.items():
        lib.db.execute("UPDATE photos SET ocr=?, ocr_done=1 WHERE id=?", (text, ids[n]))
    lib.db.commit()
    if embed:
        lib._enc = FAKE
        lib.embed_pending(log=lambda *_: None)
    lib.db.close()
    return {"ids": ids, "paths": paths, "dir": str(index_dir), "photos": str(photo_root)}


@pytest.fixture(scope="session")
def shared_index(tmp_path_factory):
    """Built once per session. Treat as READ-ONLY (use `fresh_index` to mutate)."""
    root = tmp_path_factory.mktemp("lib")
    return build_index(root / "index", root / "photos")


@pytest.fixture
def fresh_index(tmp_path):
    return build_index(tmp_path / "index", tmp_path / "photos")


@pytest.fixture
def ro_lib(shared_index):
    lib = Library(shared_index["dir"], readonly=True)
    yield lib
    lib.db.close()


def ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()
