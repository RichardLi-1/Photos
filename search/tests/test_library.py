"""photosearch.library: ingest (EXIF dates/GPS/camera, sidecars), sessions, SQLite modes, FTS triggers, vectors."""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pytest

from conftest import LONDON, PARIS, PHOTOS, SYDNEY, build_index, make_photos, ts, write_jpeg
from photosearch.library import Library, _exif, _haversine_km


def _rows(lib):
    return {os.path.basename(r["path"])[:-4]: dict(r) for r in lib.db.execute("SELECT * FROM photos")}


@pytest.fixture
def lib(tmp_path):
    make_photos(tmp_path / "photos")
    lib = Library(str(tmp_path / "index"))
    yield lib
    lib.db.close()


# ---------------------------------------------------------------- EXIF

def test_exif_date_gps_camera(tmp_path):
    p = tmp_path / "a.jpg"
    write_jpeg(p, (1, 2, 3), "2023:06:01 10:00:00", SYDNEY, size=(40, 30))
    m = _exif(str(p))
    assert m["taken"] == ts("2023-06-01T10:00:00")
    assert m["lat"] == pytest.approx(SYDNEY[0], abs=1e-4) and m["lat"] < 0      # S -> negative
    assert m["lon"] == pytest.approx(SYDNEY[1], abs=1e-4)
    assert m["camera"] == "Apple iPhone 15"
    assert (m["width"], m["height"]) == (40, 30)


def test_exif_west_is_negative(tmp_path):
    p = tmp_path / "b.jpg"
    write_jpeg(p, (1, 2, 3), None, LONDON)
    lon = _exif(str(p))["lon"]
    assert lon < 0 and lon == pytest.approx(LONDON[1], abs=1e-4)


def test_exif_missing_and_garbage(tmp_path):
    p = tmp_path / "c.jpg"
    write_jpeg(p, (1, 2, 3), camera=None)
    assert set(_exif(str(p))) == {"width", "height"}
    bad = tmp_path / "d.jpg"
    bad.write_bytes(b"\xff\xd8 not really a jpeg")
    assert _exif(str(bad)) == {}


# ---------------------------------------------------------------- ingest

def test_add_folder_counts_and_ignores_non_images(lib, tmp_path):
    root = tmp_path / "photos"
    n = sum(lib.add_folder(str(root / a)) for a in ("trip", "home", "misc"))
    assert n == len(PHOTOS)                              # notes.txt skipped
    assert lib.db.execute("SELECT count(*) FROM photos").fetchone()[0] == len(PHOTOS)
    assert lib.add_folder(str(root / "trip")) == 0       # idempotent


def test_add_folder_metadata(lib, tmp_path):
    for a in ("trip", "home", "misc"):
        lib.add_folder(str(tmp_path / "photos" / a))
    r = _rows(lib)
    assert r["trip_1"]["album"] == "trip" and r["misc_1"]["album"] == "misc"
    assert r["trip_1"]["taken"] == ts("2023-06-01T10:00:00")
    assert r["home_1"]["lat"] == pytest.approx(PARIS[0], abs=1e-4)
    assert r["misc_1"]["taken"] is None and r["misc_1"]["lat"] is None
    assert os.path.isabs(r["trip_1"]["path"])
    assert r["trip_1"]["width"] == 24 and r["trip_1"]["camera"] == "Apple iPhone 15"


def test_album_override_and_recursion(lib, tmp_path):
    assert lib.add_folder(str(tmp_path / "photos"), album="everything") == len(PHOTOS)
    assert {r[0] for r in lib.db.execute("SELECT DISTINCT album FROM photos")} == {"everything"}


def test_sidecar_fills_gaps_but_exif_wins(tmp_path):
    d = tmp_path / "side"
    d.mkdir()
    write_jpeg(d / "x.jpg", (9, 9, 9), camera=None)                                # no EXIF date/GPS
    write_jpeg(d / "y.jpg", (9, 9, 9), "2020:01:01 00:00:00", PARIS)              # EXIF present
    (d / "_meta.json").write_text(json.dumps({
        "x.jpg": {"date": "2019-05-05T08:00:00Z", "lat": 1.5, "lon": 2.5, "place": "Rome", "camera": "Leica"},
        "y.jpg": {"date": "1999-01-01T00:00:00", "lat": 0.0, "lon": 0.0, "place": "Paris"},
    }))
    lib = Library(str(tmp_path / "index"))
    lib.add_folder(str(d))
    r = _rows(lib)
    assert r["x"]["taken"] == ts("2019-05-05T08:00:00") and r["x"]["place"] == "Rome" and r["x"]["camera"] == "Leica"
    assert (r["x"]["lat"], r["x"]["lon"]) == (1.5, 2.5)
    assert r["y"]["taken"] == ts("2020-01-01T00:00:00")                            # EXIF beats sidecar
    assert r["y"]["lat"] == pytest.approx(PARIS[0], abs=1e-4) and r["y"]["place"] == "Paris"


# ---------------------------------------------------------------- sessions

def test_sessions_are_integers_not_blobs(fresh_index):
    db = sqlite3.connect(f"{fresh_index['dir']}/library.sqlite")
    types_ = {t for (t,) in db.execute("SELECT DISTINCT typeof(session) FROM photos")}
    assert types_ <= {"integer", "null"} and "integer" in types_


def test_session_grouping(fresh_index):
    db = sqlite3.connect(f"{fresh_index['dir']}/library.sqlite")
    ids = fresh_index["ids"]
    s = {n: db.execute("SELECT session FROM photos WHERE id=?", (i,)).fetchone()[0] for n, i in ids.items()}
    assert len({s[f"trip_{i}"] for i in range(1, 5)}) == 1                 # 5-min gaps, same place
    assert s["home_1"] == s["home_2"]                                       # 20 min apart
    assert s["home_3"] != s["home_2"]                                       # 10 min later but in London
    assert s["trip_1"] != s["home_1"] != s["misc_3"]
    assert s["misc_1"] is None and s["misc_2"] is None                      # undated: no session
    assert sorted(set(v for v in s.values() if v is not None)) == list(range(4))  # dense, from 0, in time order
    assert s["misc_3"] == 0                                                 # 2022 is the earliest


def test_session_time_gap(tmp_path):
    d = tmp_path / "p"; d.mkdir()
    write_jpeg(d / "a.jpg", (1, 1, 1), "2023:01:01 10:00:00")
    write_jpeg(d / "b.jpg", (1, 1, 1), "2023:01:01 10:44:00")   # < 45 min
    write_jpeg(d / "c.jpg", (1, 1, 1), "2023:01:01 11:30:00")   # 46 min later
    lib = Library(str(tmp_path / "i")); lib.add_folder(str(d)); lib.assign_sessions()
    s = {os.path.basename(p): v for p, v in lib.db.execute("SELECT path, session FROM photos")}
    assert s["a.jpg"] == s["b.jpg"] != s["c.jpg"]


# ---------------------------------------------------------------- SQLite modes

def test_wal_mode(fresh_index):
    db = sqlite3.connect(f"{fresh_index['dir']}/library.sqlite")
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_readonly_refuses_writes(shared_index):
    lib = Library(shared_index["dir"], readonly=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        lib.db.execute("UPDATE photos SET place='x'")
    with pytest.raises(sqlite3.OperationalError):
        lib.db.execute("CREATE TABLE evil(x)")
    assert lib.db.execute("SELECT count(*) FROM photos").fetchone()[0] == len(PHOTOS)


def test_readonly_does_not_create_index(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        Library(str(tmp_path / "nope"), readonly=True)
    assert not (tmp_path / "nope").exists()


def test_readonly_sees_concurrent_writer(fresh_index, tmp_path):
    ro = Library(fresh_index["dir"], readonly=True)
    rw = Library(fresh_index["dir"])
    (tmp_path / "new").mkdir()
    write_jpeg(tmp_path / "new" / "z.jpg", (5, 5, 5))
    rw.add_folder(str(tmp_path / "new"))
    assert ro.db.execute("SELECT count(*) FROM photos").fetchone()[0] == len(PHOTOS) + 1


# ---------------------------------------------------------------- full text (triggers)

def _fts(db, q):
    return [r[0] for r in db.execute("SELECT rowid FROM fts WHERE fts MATCH ?", (q,))]


def test_fts_follows_ocr_and_place_updates(fresh_index):
    lib = Library(fresh_index["dir"])
    ids = fresh_index["ids"]
    assert _fts(lib.db, "menu") == [ids["misc_1"]]
    assert set(_fts(lib.db, "sydney")) == {ids[f"trip_{i}"] for i in range(1, 5)}
    assert set(_fts(lib.db, "trip")) == {ids[f"trip_{i}"] for i in range(1, 5)}       # album indexed on insert
    lib.db.execute("UPDATE photos SET ocr='dinner receipt' WHERE id=?", (ids["misc_1"],))
    assert _fts(lib.db, "menu") == [] and _fts(lib.db, "receipt") == [ids["misc_1"]]
    assert _fts(lib.db, "menus") == []
    assert _fts(lib.db, "receipts") == [ids["misc_1"]]                                  # porter stemming


# ---------------------------------------------------------------- vectors

def test_embed_pending_writes_vectors(fresh_index):
    lib = Library(fresh_index["dir"])
    assert sorted(lib.ids.tolist()) == sorted(fresh_index["ids"].values())
    assert lib.X.dtype == np.float16 and lib.X.shape == (len(PHOTOS), 16)
    assert np.allclose(np.linalg.norm(lib.X.astype(np.float32), axis=1), 1, atol=1e-2)
    assert lib.row_of == {int(i): n for n, i in enumerate(lib.ids)}
    assert not [f for f in os.listdir(fresh_index["dir"]) if ".tmp" in f]


def test_embed_pending_only_new(fresh_index, tmp_path, fake_encoder):
    lib = Library(fresh_index["dir"])
    lib.embed_pending(log=lambda *_: None)
    assert len(lib.ids) == len(PHOTOS)
    d = tmp_path / "more"; d.mkdir()
    write_jpeg(d / "m.jpg", (0, 0, 255))
    lib.add_folder(str(d))
    logs = []
    lib.embed_pending(log=logs.append)
    assert len(lib.ids) == len(PHOTOS) + 1 and logs == ["embedded 1/1"]
    assert Library(fresh_index["dir"], readonly=True).ids.tolist() == lib.ids.tolist()   # persisted


def test_empty_index(tmp_path):
    lib = Library(str(tmp_path / "i"))
    assert lib.ids.shape == (0,) and lib.get([]) == []
    lib.embed_pending(log=lambda *_: None)   # no photos: no-op
    assert not os.path.exists(lib._vec_path)


def test_get_preserves_order_and_skips_missing(ro_lib, shared_index):
    ids = shared_index["ids"]
    want = [ids["home_2"], 999999, ids["trip_1"], np.int64(ids["misc_1"])]
    got = ro_lib.get(want)
    assert [g["id"] for g in got] == [ids["home_2"], ids["trip_1"], ids["misc_1"]]
    assert got[0]["ocr"] == "Louvre ticket admission"


def test_haversine():
    assert _haversine_km(*SYDNEY, *SYDNEY) == 0
    assert _haversine_km(*PARIS, *LONDON) == pytest.approx(344, abs=5)
    assert _haversine_km(*SYDNEY, *PARIS) == pytest.approx(16960, rel=0.01)


def test_build_twice_is_stable(tmp_path):
    a = build_index(tmp_path / "i", tmp_path / "p")
    lib = Library(a["dir"])
    assert lib.add_folder(str(tmp_path / "p" / "trip")) == 0
