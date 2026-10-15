"""Local photosearch API + web app.  usage: python server.py  ->  http://127.0.0.1:8000

Private: binds to 127.0.0.1, no external requests, no access log, read-only on the index.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import importlib, io, math, random, threading, time  # noqa: E402
from collections import OrderedDict  # noqa: E402
from datetime import datetime  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from photosearch.encoders import load_image  # noqa: E402  (also registers HEIC)
from photosearch.explain import Explainer, word_hits  # noqa: E402
from photosearch.guard import ResourceError  # noqa: E402
from photosearch.library import Library  # noqa: E402
from photosearch.search import Filters, Query, Searcher  # noqa: E402

INDEX = os.environ.get("PHOTOSEARCH_INDEX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "index"))
MODEL = "siglip2-b16-384"


for _try in range(10):  # the indexer may be mid-write on the vector file
    try:
        S = Searcher(Library(INDEX, MODEL, readonly=True)); break
    except Exception:
        if _try == 9:
            raise
        time.sleep(2)
lock, tlock, ask_lock = threading.RLock(), threading.Lock(), threading.Lock()
_mtime = [os.path.getmtime(S.lib._vec_path) if os.path.exists(S.lib._vec_path) else 0]
_asker = [None]
_explainer = [None]
_order: list = [None]
_thumbs: OrderedDict = OrderedDict()
_thumb_bytes = [0]
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# Local-only: other websites in the user's browser can reach 127.0.0.1, so refuse DNS-rebinding (Host), cross-site
# requests (Sec-Fetch-Site / Origin) and oversized bodies before any handler runs.
PORT = int(os.environ.get("PORT", 8000))
_HOSTS = {"127.0.0.1", "localhost"}
ALLOWED_HOSTS = _HOSTS | {f"{h}:{p}" for h in _HOSTS for p in (PORT, 5173)}  # 5173: the vite dev proxy keeps Host
ALLOWED_ORIGINS = {f"http://{h}" for h in ALLOWED_HOSTS}
MAX_BODY, MAX_Q, MAX_TAGS = 25 * 2**20, 300, 32
UPLOAD_MAX_BYTES, UPLOAD_MAX_PIXELS = 20 * 2**20, 50_000_000
UPLOAD_FORMATS = ("JPEG", "MPO", "PNG", "WEBP", "HEIF", "GIF")
THUMB_SIZES, THUMB_CACHE_BYTES = (128, 256, 384, 512, 768, 1024), 96 * 2**20


@app.middleware("http")
async def local_only(request: Request, call_next):
    h = request.headers
    if h.get("host", "") not in ALLOWED_HOSTS:
        return JSONResponse({"detail": "unknown host"}, status_code=421)
    if h.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
        return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
    if request.method not in ("GET", "HEAD") and h.get("origin") not in (None, *ALLOWED_ORIGINS):
        return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
    try:
        if int(h.get("content-length") or 0) > MAX_BODY:
            return JSONResponse({"detail": "request too large"}, status_code=413)
    except ValueError:
        return JSONResponse({"detail": "bad content-length"}, status_code=400)
    r = await call_next(request)
    r.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    r.headers["X-Content-Type-Options"] = "nosniff"
    return r


@app.exception_handler(ResourceError)
def _busy(_, e):  # guard refused (memory/swap/disk pressure)
    return JSONResponse({"detail": str(e)}, status_code=503)


def refresh(force=False):
    """Pick up new embeddings from the background indexer (cheap mtime check)."""
    try:
        m = os.path.getmtime(S.lib._vec_path)
    except OSError:
        return
    if force or m != _mtime[0]:
        try:
            S.refresh(); _mtime[0] = m
        except Exception:
            pass  # file mid-write; retry next request


def _s(v):
    return v.strip() if isinstance(v, str) and v.strip() else None


def filters(p) -> Filters:
    for k in ("after", "before"):
        if _s(p.get(k)):
            try:
                datetime.fromisoformat(p[k].strip())
            except ValueError:
                raise HTTPException(400, f"bad date: {k}")
    return Filters(after=_s(p.get("after")), before=_s(p.get("before")), place=_s(p.get("place")),
                   album=_s(p.get("album")), has_text=str(p.get("has_text", "")).lower() in ("1", "true", "on", "yes"))


def k_of(p) -> int:
    try:
        return max(1, min(500, int(p.get("k") or 60)))
    except ValueError:
        return 60


def rows(rs):
    for r in rs:
        r["date"] = datetime.fromtimestamp(r["taken"]).isoformat(" ", "minutes") if r.get("taken") else None
    return rs


def run(q: Query):
    with lock:
        refresh()
        return {"results": rows(S.search(q)), "embedded": len(S.ids)}


async def upload_image(f) -> Image.Image | None:
    if not f or not getattr(f, "filename", None):
        return None
    data = await f.read(UPLOAD_MAX_BYTES + 1)
    if len(data) > UPLOAD_MAX_BYTES:
        raise HTTPException(413, "image too large")
    if not data:
        return None

    def decode():  # off the event loop; only common photo formats; refuse huge pixel counts before decoding
        try:
            Image.init()  # unknown/unloaded names in `formats` raise KeyError, so keep only registered ones
            im = Image.open(io.BytesIO(data), formats=[f for f in UPLOAD_FORMATS if f in Image.ID])
        except Exception:
            raise HTTPException(400, "not an image")
        if im.width * im.height > UPLOAD_MAX_PIXELS:
            raise HTTPException(413, "image has too many pixels")
        try:
            return ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            raise HTTPException(400, "not an image")
    return await run_in_threadpool(decode)


@app.get("/api/browse")
def browse(offset: int = 0, limit: int = 400):
    """The default wall: every photo in a stable shuffled order (nicest album first)."""
    if _order[0] is None:
        rows_ = S.lib.db.execute("SELECT id, album FROM photos").fetchall()
        rng = random.Random(7)
        first = [r[0] for r in rows_ if r[1] == "unsplash"]; rest = [r[0] for r in rows_ if r[1] != "unsplash"]
        rng.shuffle(first); rng.shuffle(rest)
        _order[0] = first + rest
    limit, offset = max(1, min(1000, limit)), max(0, offset)
    try:  # hide Recently Deleted (soft-deleted) photos from the wall; the trash is small, so this is cheap
        gone = {r[0] for r in S.lib.db.execute("SELECT id FROM photos WHERE deleted_at IS NOT NULL")}
    except Exception:  # older index without the column
        gone = set()
    order = [i for i in _order[0] if i not in gone] if gone else _order[0]
    return {"ids": order[offset:offset + limit], "total": len(order)}


@app.get("/api/search")
def search(request: Request):
    p = request.query_params
    if not _s(p.get("q")):
        raise HTTPException(400, "q required")
    if len(p["q"]) > MAX_Q:
        raise HTTPException(414, f"query longer than {MAX_Q} characters")
    q = Query(text=p["q"].strip(), filters=filters(p), k=k_of(p), parse=True)  # dates/places in the text -> hard filters
    return {**run(q), "parsed": q.parsed}


@app.post("/api/search_image")
async def search_image(request: Request):
    form = await request.form(max_files=1, max_fields=20)
    im = await upload_image(form.get("file"))
    if im is None:
        raise HTTPException(400, "file required")
    try:
        tw = float(form.get("text_weight") or 0.5)
    except ValueError:
        raise HTTPException(400, "bad text_weight")
    if not math.isfinite(tw):
        raise HTTPException(400, "bad text_weight")
    q = Query(text=_s(form.get("text")), image=im, text_weight=min(max(tw, 0.05), 0.95), filters=filters(form), k=k_of(form),
              mode="place" if form.get("mode") == "place" else "similar")

    def go():
        out = run(q)
        with lock:  # the query photo's own concepts, so result cards can say what they share with it
            out["qtags"] = explainer().tags(S.embed_image(im), k=12, floor=0.01)
        return out
    return await run_in_threadpool(go)


def explainer() -> Explainer:
    if _explainer[0] is None:
        _explainer[0] = Explainer(S)
    return _explainer[0]


@app.get("/api/explain/{pid}")
def explain(pid: int, q: str | None = None, ref: int | None = None, tags: str | None = None):
    """Why a photo matched the current query, in a few words."""
    q = q[:MAX_Q] if q else q
    with lock:
        refresh()
        if pid not in S.lib.row_of:
            raise HTTPException(404, "photo not embedded")
        E, v = explainer(), S.X[S.lib.row_of[pid]]
        row = rows(S.lib.get([pid]))[0]
        out = {"place": row["place"], "date": row["date"], "text_hits": [], "place_hits": []}
        if ref is not None and ref in S.lib.row_of:
            out["mode"], out["shared"] = "similar", E.shared_with_image(S.X[S.lib.row_of[ref]], v)
        elif tags:
            out["mode"], out["shared"] = "photo", E.shared_with_tags([t.strip() for t in tags.split(",")][:MAX_TAGS], v)
        elif _s(q):
            out["mode"], out["shared"] = "text", E.matched_text(q, v)
            out["text_hits"], out["place_hits"] = word_hits(q, row["ocr"]), word_hits(q, row["place"])
        else:
            out["mode"], out["shared"] = "browse", E.tags(v, k=4)
        if not out["shared"] and out["mode"] != "browse":
            out["shared"] = E.tags(v, k=3)
            out["mode"] += "-overall"
        return out


@app.get("/api/similar/{pid}")
def similar(pid: int, request: Request):
    p = request.query_params
    with lock:
        refresh()
        if pid not in S.lib.row_of:
            raise HTTPException(404, "photo not embedded yet")
        mode = "place" if p.get("mode") == "place" else "similar"  # place = our trained same-room/place head
        return run(Query(text=_s(p.get("q")), image=pid, filters=filters(p), k=k_of(p), mode=mode))


@app.post("/api/ask")
async def ask(request: Request):
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/"):
        form = await request.form(max_files=1, max_fields=20)
        question, im = _s(form.get("question")), await upload_image(form.get("file"))
    elif ctype.startswith("application/json"):  # not text/plain: that's a cross-site "simple request"
        body = await request.json()
        question, im = _s(body.get("question") if isinstance(body, dict) else None), None
    else:
        raise HTTPException(415, "send JSON or multipart")
    if not question:
        raise HTTPException(400, "question required")
    if len(question) > MAX_Q:
        raise HTTPException(414, f"question longer than {MAX_Q} characters")

    def go():
        with lock:  # short: just pick up new vectors / create the Asker
            refresh()
            if _asker[0] is None:
                from photosearch.ask import Asker
                _asker[0] = Asker(S)
        with ask_lock:  # one question at a time (one VLM), without blocking search/explain for 10-60 s
            out = _asker[0].ask(question, image=im)
        rows(out["results"])
        return out
    return await run_in_threadpool(go)


@app.post("/api/refresh")
def force_refresh():
    with lock:
        refresh(force=True)
        return {"embedded": len(S.ids)}


@app.get("/api/stats")
def stats():
    with lock:
        refresh()
        db = S.lib.db
        n, ocr, txt, places = db.execute("SELECT count(*), sum(ocr_done), count(ocr), count(DISTINCT place) FROM photos").fetchone()
        albums = {r[0]: r[1] for r in db.execute("SELECT album, count(*) FROM photos GROUP BY album ORDER BY album")}
        lo, hi = db.execute("SELECT min(taken), max(taken) FROM photos").fetchone()
        span = [datetime.fromtimestamp(t).date().isoformat() if t else None for t in (lo, hi)]
        return {"photos": n, "embedded": len(S.ids), "ocr_done": ocr or 0, "with_text": txt, "places": places, "albums": albums,
                "date_range": span, "index": os.path.basename(os.path.normpath(INDEX)),
                "place_mode": getattr(S, "Z", None) is not None}  # same-place head loaded (Query.mode="place")


@app.get("/photo/{pid}")
def photo(pid: int, w: int | None = None):
    r = S.lib.db.execute("SELECT path FROM photos WHERE id=?", (pid,)).fetchone()
    if not r or not os.path.isfile(r[0]):
        raise HTTPException(404)
    path = r[0]
    if w is None:
        if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return FileResponse(path)
        buf = io.BytesIO(); load_image(path).save(buf, "JPEG", quality=90)  # HEIC -> JPEG
        return Response(buf.getvalue(), media_type="image/jpeg")
    w = next((s_ for s_ in THUMB_SIZES if s_ >= w), THUMB_SIZES[-1])  # a few fixed sizes keep the cache small
    key = (pid, w)
    with tlock:
        b = _thumbs.get(key)
        if b:
            _thumbs.move_to_end(key)
    if not b:
        im = Image.open(path)
        im.draft("RGB", (w, w))  # fast JPEG downscale on decode
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((w, w))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
        b = buf.getvalue()
        with tlock:
            _thumbs[key] = b
            _thumb_bytes[0] += len(b)
            while _thumb_bytes[0] > THUMB_CACHE_BYTES and _thumbs:
                _thumb_bytes[0] -= len(_thumbs.popitem(last=False)[1])
    return Response(b, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


# Feature routers live in their own modules (api_*.py next to this file) and get the shared searcher via `S`.
# Each module exposes `router` (fastapi.APIRouter) and optionally `init(S, lock)`; must be registered before the "/" mount.
for _mod in ("api_tools", "api_ask", "api_agent"):
    try:
        _m = importlib.import_module(_mod)
    except ModuleNotFoundError as _e:
        if _e.name != _mod:
            raise
        continue
    if hasattr(_m, "init"):
        _m.init(S, lock)
    app.include_router(_m.router)


WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "dist")


class _Web(StaticFiles):
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        if not path.startswith("assets/"):  # index.html must revalidate so new builds show up; hashed assets can cache
            r.headers["Cache-Control"] = "no-cache"
        return r


if os.path.isdir(WEB):  # the React app (cd web && npm run build); for dev use `npm run dev` instead
    app.mount("/", _Web(directory=WEB, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    from photosearch import guard
    guard.SERVER_ROLE = PORT == 8000  # the user's UI server rides on a reserved ML slot (agents' test servers don't)
    S.lib.encoder  # load SigLIP now so the first search is fast
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), access_log=False)
