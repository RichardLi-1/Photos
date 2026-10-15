"""server.py over FastAPI's TestClient: temp index, fake encoder, no live server, no model loads.

Security regression tests (Host/CSRF/embedding/upload/query limits) cover fixes from the security review.
"""
from __future__ import annotations

import importlib
import io
import os
import struct
import sys
import zlib

import pytest
from PIL import Image

from conftest import PHOTOS

ORIGIN = "http://127.0.0.1:8000"


@pytest.fixture(scope="session")
def server(shared_index):
    old = os.environ.get("PHOTOSEARCH_INDEX")
    os.environ["PHOTOSEARCH_INDEX"] = shared_index["dir"]
    try:
        assert "server" not in sys.modules, "server imported before the temp index was set"
        srv = importlib.import_module("server")
    finally:
        if old is None:
            os.environ.pop("PHOTOSEARCH_INDEX", None)
        else:
            os.environ["PHOTOSEARCH_INDEX"] = old
    assert srv.INDEX == shared_index["dir"]
    return srv


@pytest.fixture(scope="session")
def client(server):
    from fastapi.testclient import TestClient
    with TestClient(server.app, base_url=ORIGIN, raise_server_exceptions=False) as c:  # Host: 127.0.0.1:8000
        yield c


@pytest.fixture
def ids(shared_index):
    return shared_index["ids"]


@pytest.fixture
def names(shared_index):
    by_id = {v: k for k, v in shared_index["ids"].items()}
    return lambda body: [by_id[r["id"]] for r in body["results"]]


def png_bytes(rgb=(255, 0, 0), size=(8, 8), fmt="PNG"):
    b = io.BytesIO()
    Image.new("RGB", size, rgb).save(b, fmt)
    return b.getvalue()


def png_header_only(w, h):
    """A PNG that *claims* w x h pixels (decompression-bomb probe) without any pixel data."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IEND", b"")


# ---------------------------------------------------------------- stats / refresh

def test_stats(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    j = r.json()
    lo, hi = j.pop("date_range")
    assert isinstance(j.pop("index"), str) and isinstance(j.pop("place_mode"), bool)
    assert j == {"photos": len(PHOTOS), "embedded": len(PHOTOS), "ocr_done": 2, "with_text": 2, "places": 2,
                 "albums": {"home": 3, "misc": 3, "trip": 4}}
    assert (lo is None and hi is None) or lo <= hi  # ISO dates of the oldest/newest photo


def test_refresh(client):
    assert client.post("/api/refresh").json() == {"embedded": len(PHOTOS)}


def test_no_docs_or_openapi(client):
    for p in ("/docs", "/redoc", "/openapi.json"):
        assert "openapi" not in client.get(p).text.lower()


# ---------------------------------------------------------------- /api/search

def test_search(client, names):
    r = client.get("/api/search", params={"q": "red", "k": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["embedded"] == len(PHOTOS) and set(names(body)) == {f"trip_{i}" for i in range(1, 5)}
    row = body["results"][0]
    assert {"id", "album", "date", "place", "ocr", "score", "path"} <= set(row)
    assert row["date"].startswith("2023-06-01 10:")


def test_search_literal_text(client, names):
    assert names(client.get("/api/search", params={"q": "menu", "k": 1}).json()) == ["misc_1"]


@pytest.mark.parametrize("params,expect", [
    ({"album": "home"}, {"home_1", "home_2", "home_3"}),
    ({"after": "2024-01-01"}, {"home_1", "home_2", "home_3"}),
    ({"before": "2023-01-01"}, {"misc_3"}),
    ({"place": "sydney"}, {f"trip_{i}" for i in range(1, 5)}),
    ({"has_text": "true"}, {"misc_1", "home_2"}),
    ({"has_text": "1", "album": "misc"}, {"misc_1"}),
    ({"album": "  "}, set(PHOTOS)),                           # blank filters are ignored
])
def test_search_filters(client, names, params, expect):
    r = client.get("/api/search", params={"q": "red", "k": 50, **params})
    assert r.status_code == 200 and set(names(r.json())) == expect


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "   "}])
def test_search_requires_q(client, params):
    r = client.get("/api/search", params=params)
    assert r.status_code == 400 and r.json()["detail"] == "q required"


@pytest.mark.parametrize("bad", ["yesterday", "2024-13-01", "2024-01-01'; --"])
def test_search_bad_date(client, bad):
    r = client.get("/api/search", params={"q": "red", "after": bad})
    assert r.status_code == 400 and "bad date" in r.json()["detail"]


@pytest.mark.parametrize("k,want", [("3", 3), ("0", 1), ("-7", 1), ("abc", 60), ("", 60), ("1e9", 60), (str(10**12), 500)])
def test_k_is_clamped(client, server, monkeypatch, k, want):
    seen = []
    real = server.S.search
    monkeypatch.setattr(server.S, "search", lambda q: seen.append(q.k) or real(q))
    assert client.get("/api/search", params={"q": "red", "k": k}).status_code == 200
    assert seen == [want]


def test_search_long_query_is_rejected_or_capped(client):
    import time
    t0 = time.perf_counter()
    r = client.get("/api/search", params={"q": "lunch menu " * 400})   # 800 words, ~9 KB: a GET any web page can send
    assert r.status_code in (200, 400, 414, 422) and time.perf_counter() - t0 < 0.5


# ---------------------------------------------------------------- /api/similar

def test_similar(client, ids, names):
    r = client.get(f"/api/similar/{ids['home_1']}", params={"k": 2})
    assert r.status_code == 200 and set(names(r.json())) == {"home_2", "home_3"}


def test_similar_with_text_and_filters(client, ids, names):
    r = client.get(f"/api/similar/{ids['home_1']}", params={"q": "red", "album": "trip", "k": 50})
    assert r.status_code == 200 and set(names(r.json())) == {f"trip_{i}" for i in range(1, 5)}


@pytest.mark.parametrize("pid,code", [("999999", 404), ("-1", 404), ("abc", 422), ("1.5", 422)])
def test_similar_unknown(client, pid, code):
    assert client.get(f"/api/similar/{pid}").status_code == code


# ---------------------------------------------------------------- /api/browse

def test_browse(client, ids):
    b = client.get("/api/browse").json()
    assert b["total"] == len(PHOTOS) and sorted(b["ids"]) == sorted(ids.values())
    assert client.get("/api/browse").json() == b                     # stable order
    page = client.get("/api/browse", params={"offset": 3, "limit": 4}).json()
    assert page["ids"] == b["ids"][3:7]
    assert client.get("/api/browse", params={"offset": 100}).json()["ids"] == []
    assert len(client.get("/api/browse", params={"limit": 0}).json()["ids"]) == 1      # clamped to >= 1
    assert client.get("/api/browse", params={"limit": 10**9}).status_code == 200
    assert client.get("/api/browse", params={"limit": "x"}).status_code == 422


def test_browse_negative_offset(client):
    r = client.get("/api/browse", params={"offset": -3, "limit": 2})
    assert r.status_code == 422 or r.json()["ids"] == client.get("/api/browse", params={"limit": 2}).json()["ids"]


# ---------------------------------------------------------------- /photo/{id}

def test_photo_original(client, ids, shared_index):
    r = client.get(f"/photo/{ids['trip_1']}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content == open(shared_index["paths"]["trip_1"], "rb").read()


@pytest.mark.parametrize("w", [64, 1, 10**6])
def test_photo_thumbnail(client, ids, w):
    r = client.get(f"/photo/{ids['home_1']}", params={"w": w})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert "private" in r.headers["cache-control"]
    im = Image.open(io.BytesIO(r.content))
    assert max(im.size) <= max(32, min(1024, w))


@pytest.mark.parametrize("path,codes", [
    ("/photo/999999", {404}),
    ("/photo/0", {404}),
    ("/photo/-1", {404}),
    ("/photo/abc", {422}),
    ("/photo/1%00", {404, 422}),
    ("/photo/..%2F..%2F..%2Fetc%2Fpasswd", {404, 422}),
    ("/photo/%2e%2e/%2e%2e/%2e%2e/etc/passwd", {404, 422}),
    ("/photo/1/../../../../etc/passwd", {404, 422}),
    ("/photo/1?w=abc", {422}),
])
def test_photo_bad_ids_and_traversal(client, path, codes):
    r = client.get(path)
    assert r.status_code in codes
    assert b"root:" not in r.content and b"FastAPI" not in r.content


@pytest.mark.parametrize("path", ["/../search/server.py", "/..%2fsearch%2fserver.py", "/%2e%2e/search/server.py",
                                  "/assets/..%2f..%2f..%2fsearch%2fserver.py", "/....//search/server.py"])
def test_static_mount_traversal(client, path):
    r = client.get(path)
    assert b"from fastapi import" not in r.content


def test_photo_missing_file_is_404(client, ids, shared_index):
    p = shared_index["paths"]["misc_2"]
    os.rename(p, p + ".moved")
    try:
        assert client.get(f"/photo/{ids['misc_2']}").status_code == 404
        assert client.get(f"/photo/{ids['misc_2']}", params={"w": 64}).status_code in (200, 404)  # may be cached
    finally:
        os.rename(p + ".moved", p)


# ---------------------------------------------------------------- /api/explain

def test_explain_modes(client, ids):
    b = client.get(f"/api/explain/{ids['trip_1']}").json()
    assert b["mode"] == "browse" and b["place"] == "Sydney, Australia" and b["date"].startswith("2023-06-01")
    assert isinstance(b["shared"], list) and "red tones" in b["shared"]

    t = client.get(f"/api/explain/{ids['misc_1']}", params={"q": "lunch menu"}).json()
    assert t["mode"].startswith("text") and t["text_hits"] == ["lunch", "menu"] and t["place_hits"] == []

    p = client.get(f"/api/explain/{ids['trip_1']}", params={"q": "sydney"}).json()
    assert p["place_hits"] == ["sydney"]

    s = client.get(f"/api/explain/{ids['trip_1']}", params={"ref": ids["trip_2"]}).json()
    assert s["mode"].startswith("similar") and "red tones" in s["shared"]

    g = client.get(f"/api/explain/{ids['trip_1']}", params={"tags": "red tones, blue tones,nonsense"}).json()
    assert g["mode"] == "photo" and g["shared"][0] == "red tones" and set(g["shared"]) <= {"red tones", "blue tones"}


def test_explain_unknown(client, ids):
    assert client.get("/api/explain/999999").status_code == 404
    assert client.get("/api/explain/abc").status_code == 422
    r = client.get(f"/api/explain/{ids['trip_1']}", params={"ref": 999999})   # unknown ref: falls back
    assert r.status_code == 200 and r.json()["mode"] == "browse"


# ---------------------------------------------------------------- /api/search_image (uploads)

def test_search_image(client, names):
    r = client.post("/api/search_image", files={"file": ("q.png", png_bytes((0, 0, 255)), "image/png")}, data={"k": "3"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(names(body)) == {"home_1", "home_2", "home_3"}
    assert isinstance(body["qtags"], list) and "blue tones" in body["qtags"]


def test_search_image_with_text(client, names):
    r = client.post("/api/search_image", files={"file": ("q.jpg", png_bytes((0, 0, 255), fmt="JPEG"), "image/jpeg")},
                    data={"text": "red", "text_weight": "0.95", "k": "4"})
    assert r.status_code == 200 and set(names(r.json())) == {f"trip_{i}" for i in range(1, 5)}


@pytest.mark.parametrize("tw", ["-5", "5", "0", "1", "nan"])
def test_search_image_text_weight_clamped(client, tw):
    r = client.post("/api/search_image", files={"file": ("q.png", png_bytes(), "image/png")}, data={"text": "red", "text_weight": tw})
    assert r.status_code in (200, 400)
    if r.status_code == 200:
        assert all(x["score"] is None or x["score"] == x["score"] for x in r.json()["results"])  # no NaN scores


@pytest.mark.parametrize("files,data,detail", [
    (None, {"k": "3"}, "file required"),
    ({"file": ("", b"", "image/png")}, None, "file required"),
    ({"file": ("q.png", b"", "image/png")}, None, "file required"),
    ({"file": ("q.png", b"definitely not an image", "image/png")}, None, "not an image"),
    ({"file": ("q.png", png_bytes()[:40], "image/png")}, None, "not an image"),               # truncated
    ({"file": ("bomb.png", png_header_only(20000, 20000), "image/png")}, None, "not an image"),  # > 2x MAX_IMAGE_PIXELS
    ({"file": ("q.png", png_bytes(), "image/png")}, {"text_weight": "heavy"}, "bad text_weight"),
])
def test_search_image_rejects(client, files, data, detail):
    r = client.post("/api/search_image", files=files, data=data)
    assert r.status_code == 400 and r.json()["detail"] == detail


def test_search_image_bad_date(client):
    r = client.post("/api/search_image", files={"file": ("q.png", png_bytes(), "image/png")}, data={"after": "soon"})
    assert r.status_code == 400


def test_search_image_get_not_allowed(client):
    assert client.get("/api/search_image").status_code in (404, 405)   # falls through to the static mount


def test_search_image_size_cap(client):
    r = client.post("/api/search_image", files={"file": ("big.png", b"\0" * (21 * 2**20), "image/png")})
    assert r.status_code == 413


@pytest.mark.parametrize("fmt", ["BMP", "TIFF", "PPM"])
def test_search_image_format_allowlist(client, fmt):
    r = client.post("/api/search_image", files={"file": (f"q.{fmt.lower()}", png_bytes(fmt=fmt), "application/octet-stream")})
    assert r.status_code in (400, 415)


def test_search_image_pixel_cap(client):
    r = client.post("/api/search_image", files={"file": ("big.png", png_header_only(10000, 10000), "image/png")})  # 100 MP
    assert r.status_code == 413


# ---------------------------------------------------------------- /api/ask (validation only: never loads the VLM)

@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "   "}, {"question": None}])
def test_ask_requires_question(client, body):
    r = client.post("/api/ask", json=body)
    assert r.status_code == 400 and r.json()["detail"] == "question required"


def test_ask_status(client):
    r = client.get("/api/ask/status")
    assert r.status_code == 200 and r.json() == {"loaded": False, "queued": False}


# ---------------------------------------------------------------- security regressions (local-only middleware, upload limits)

def test_rejects_foreign_host(client):
    assert client.get("/api/stats", headers={"Host": "attacker.example:8000"}).status_code in (400, 403, 421)


def test_rejects_cross_site_post(client):
    r = client.post("/api/ask", content=b'{"question": ""}', headers={"Origin": "https://evil.example", "Content-Type": "text/plain"})
    assert r.status_code == 403


def test_photo_not_embeddable_cross_site(client, ids):
    r = client.get(f"/photo/{ids['trip_1']}", headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "image"})
    assert r.status_code == 403 or r.headers.get("cross-origin-resource-policy") == "same-origin"


def test_same_origin_requests_still_work(client):
    """Guard-rail for the fixes above: the app's own requests must keep working."""
    h = {"Origin": ORIGIN, "Sec-Fetch-Site": "same-origin"}
    assert client.get("/api/stats", headers=h).status_code == 200
    assert client.post("/api/refresh", headers=h).status_code == 200
    dev = {"Origin": "http://127.0.0.1:5173", "Sec-Fetch-Site": "same-origin"}   # vite dev proxy
    assert client.get("/api/stats", headers=dev).status_code == 200
