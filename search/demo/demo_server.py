"""Public, read-only demo server: FastAPI + onnxruntime + numpy over a bundle from build_bundle.py. No torch.

  DEMO_BUNDLE=/bundle DEMO_ALLOWED_HOSTS=photos-demo.example.org,*.a.run.app uvicorn demo_server:app --port 8080

Same API shapes as search/server.py, so the unmodified web/dist works. What differs from the local app, and why:
  * Ask (/api/ask) needs a local VLM on a Mac (MLX): it answers with plain search results and says so.
  * Search by uploaded photo (/api/search_image) is off: no image tower ships (smaller, cheaper) and the demo never
    receives visitors' photos. "More like this" / "Same place" on demo photos work (precomputed vectors).
  * The agent can't change shared state: album / trash / purge / restore tools are refused before they run, and
    cleanup suggestions are off (they need on-device image stats). Agent sessions live in memory only (30 min).
  * Nothing is written to disk at runtime except a private temp copy of the sqlite file; no access log, no query
    log. Rate limits keep per-IP counters in memory only.
  * Visitor analytics, off unless configured (visits.py; never in the local app): /api/track forwards the browser's
    visit card + IP + Fly edge region to Discord, and AI fetchers get one log line each. Search text, photos and
    photo IDs are never sent anywhere.

Env:
  DEMO_BUNDLE                bundle dir (default ./bundle)
  DEMO_WEB                   web/dist (default ../../web/dist relative to this file, or /app/web)
  DEMO_ALLOWED_HOSTS         comma list of Host values; "*.suffix" allowed; default 127.0.0.1/localhost only
  DEMO_CLIENT_IP_HEADER      header holding the client IP (fly-client-ip, x-forwarded-for, ...); default: socket peer
  DEMO_PROXY_HOPS            with x-forwarded-for: how many trusted proxies append to it (default 1: last entry)
  DEMO_RATE_API              API requests per IP per minute (default 60)
  DEMO_RATE_PHOTO            thumbnail requests per IP per minute (default 900)
  DEMO_THUMB_BASE_URL        serve thumbnails by redirect, e.g. https://thumbs.example.org (files <id>.<ext>)
  DEMO_FRAME_ANCESTORS       CSP frame-ancestors (default 'none'; Hugging Face Spaces: https://huggingface.co)
  DEMO_DENSE=1               keep a float32 copy of the vectors (4x RAM, ~5x faster matmul)
  DEMO_ALLOW_UNPUBLISHABLE=1 start with a bundle built with --allow-uncredited (local tests only)
  DEMO_IP_DEBUG=1            add X-Demo-Client-Key (the caller's own IP as the rate limiter sees it) to responses
  DISCORD_WEBHOOK_URL        visit cards to Discord (see visits.py); webhook URLs never reach the browser
"""
from __future__ import annotations

import html
import json
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):  # demo/ and search/ (photosearch, mock_agent)
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.middleware.gzip import GZipMiddleware  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

import visits  # noqa: E402
from engine import manifest, open_bundle  # noqa: E402
from photosearch.explain import Explainer, word_hits  # noqa: E402
from photosearch.search import Filters, Query  # noqa: E402

BUNDLE = os.path.abspath(os.environ.get("DEMO_BUNDLE", os.path.join(os.getcwd(), "bundle")))
MAN = manifest(BUNDLE)
if MAN.get("publishable") is False and os.environ.get("DEMO_ALLOW_UNPUBLISHABLE") != "1":
    sys.exit("this bundle was built with --allow-uncredited (photos without a license/credit): refusing to serve it")

# private, throwaway copy of the sqlite file: photosearch.tools opens a WAL connection and needs a writable dir
STATE = tempfile.mkdtemp(prefix="photos-demo-")
shutil.copyfile(os.path.join(BUNDLE, "library.sqlite"), os.path.join(STATE, "library.sqlite"))
S = open_bundle(BUNDLE, os.path.join(STATE, "library.sqlite"), dense=os.environ.get("DEMO_DENSE") == "1")
lock = threading.RLock()  # one search at a time: 1 vCPU, and it keeps peak RAM flat
_explainer: list = [None]
_order: list = [None]

# ---------------------------------------------------------------- thumbnails
THUMB_BASE = os.environ.get("DEMO_THUMB_BASE_URL", "").rstrip("/")
_pack = None
if os.path.exists(os.path.join(BUNDLE, "thumbs.npz")):
    with np.load(os.path.join(BUNDLE, "thumbs.npz"), allow_pickle=False) as z:
        _toff = {int(i): (int(o), int(n)) for i, o, n in zip(z["ids"], z["offsets"], z["lengths"])}
        THUMB_MIME = str(z["mime"])
    _pack = os.open(os.path.join(BUNDLE, "thumbs.pack"), os.O_RDONLY)
else:
    _toff, THUMB_MIME = {}, MAN.get("thumb_mime", "image/avif")
THUMB_EXT = {"image/avif": "avif", "image/webp": "webp", "image/jpeg": "jpg"}.get(THUMB_MIME, "jpg")

# ---------------------------------------------------------------- app + guards
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(GZipMiddleware, minimum_size=2048, compresslevel=5)  # search JSON (k=300 rows) ~6x smaller
_hosts = [h.strip().lower() for h in os.environ.get("DEMO_ALLOWED_HOSTS", "").split(",") if h.strip()]
LOCAL = {"127.0.0.1", "localhost"}
MAX_BODY, MAX_Q, MAX_TAGS, MAX_K = 64 * 1024, 300, 32, 300
IP_HEADER = os.environ.get("DEMO_CLIENT_IP_HEADER", "").lower()
HOPS = max(1, int(os.environ.get("DEMO_PROXY_HOPS", "1")))
RATE = {"api": int(os.environ.get("DEMO_RATE_API", "60")), "photo": int(os.environ.get("DEMO_RATE_PHOTO", "900"))}
FRAME = os.environ.get("DEMO_FRAME_ANCESTORS", "'none'")
IP_DEBUG = os.environ.get("DEMO_IP_DEBUG") == "1"  # echo the caller's own rate-limit key back to them (never logged)
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
       f"img-src 'self' data: blob:{' ' + THUMB_BASE if THUMB_BASE else ''}; connect-src 'self'; font-src 'self' data:; "
       f"object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors {FRAME}")
TRACK_PER_MIN, MAX_TRACK_BODY = 10, 8 * 1024  # /api/track: on top of the api bucket, so one IP can't flood Discord


def host_name(host: str) -> str:
    h = host.lower()
    return h.rsplit(":", 1)[0] if not h.startswith("[") else h


def host_ok(host: str) -> bool:
    h, name = host.lower(), host_name(host)
    if name in LOCAL:
        return True
    for a in _hosts:
        if a.startswith("*.") and (name.endswith(a[1:]) or h.endswith(a[1:])):
            return True
        if h == a or name == a:
            return True
    return False


def client_ip(request: Request) -> str:
    if IP_HEADER:
        v = request.headers.get(IP_HEADER, "")
        if v:
            parts = [p.strip() for p in v.split(",") if p.strip()]
            if parts:
                return parts[max(0, len(parts) - HOPS)] if IP_HEADER == "x-forwarded-for" else parts[0]
    return request.client.host if request.client else "?"


class Buckets:
    """Fixed-window counters per (ip, kind), memory only, bounded (oldest IPs dropped first). Never logged."""

    def __init__(self, cap: int = 50_000):
        self.d: OrderedDict = OrderedDict()
        self.cap, self.mu = cap, threading.Lock()

    def hit(self, key, limit: int) -> bool:
        now = int(time.time() // 60)
        with self.mu:
            w, n = self.d.get(key, (now, 0))
            if w != now:
                w, n = now, 0
            self.d[key] = (w, n + 1)
            self.d.move_to_end(key)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)
            return n + 1 <= limit


_buckets = Buckets()


@app.middleware("http")
async def public_demo_guard(request: Request, call_next):
    h = request.headers
    if not host_ok(h.get("host", "")) and request.url.path != "/healthz":  # platform checks use the machine IP as Host
        return JSONResponse({"detail": "unknown host"}, status_code=421)
    if request.method not in ("GET", "HEAD", "POST"):
        return JSONResponse({"detail": "method not allowed"}, status_code=405)
    if h.get("sec-fetch-site", "same-origin") not in ("same-origin", "none") and request.method != "GET":
        return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
    try:
        if int(h.get("content-length") or 0) > MAX_BODY:
            return JSONResponse({"detail": "request too large"}, status_code=413)
    except ValueError:
        return JSONResponse({"detail": "bad content-length"}, status_code=400)
    path = request.url.path
    # AI assistants fetching pages run no JavaScript: one log line each, Discord for "user" fetches (queued, not awaited)
    visits.report_ai_fetch(h.get("user-agent"), path, local=host_name(h.get("host", "")) in LOCAL)
    kind = "photo" if path.startswith("/photo/") else "api" if path.startswith("/api/") else None
    ip = client_ip(request)
    if kind and not _buckets.hit((ip, kind), RATE[kind]):
        return JSONResponse({"detail": "too many requests; try again in a minute"}, status_code=429,
                            headers={"Retry-After": "60"})
    r = await call_next(request)
    r.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["Referrer-Policy"] = "no-referrer"
    r.headers["Content-Security-Policy"] = CSP
    r.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), interest-cohort=()"
    if IP_DEBUG:  # deploy check: is DEMO_CLIENT_IP_HEADER / DEMO_PROXY_HOPS picking *your* address?
        r.headers["X-Demo-Client-Key"] = ip
    return r


# ---------------------------------------------------------------- helpers (as in server.py)
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
        return max(1, min(MAX_K, int(p.get("k") or 60)))
    except ValueError:
        return 60


_credit_cache: dict = {}


def credit(pid: int) -> str | None:
    if pid not in _credit_cache:
        r = S.lib.db.execute("SELECT creator, license FROM credits WHERE id=?", (pid,)).fetchone() if _has_credits else None
        _credit_cache[pid] = f"{r[0] or 'unknown'} · {r[1] or ''}".strip(" ·") if r else None
        if len(_credit_cache) > 100_000:
            _credit_cache.clear()
    return _credit_cache[pid]


_has_credits = bool(S.lib.db.execute("SELECT 1 FROM sqlite_master WHERE name='credits'").fetchone())


def rows(rs):
    for r in rs:
        r["date"] = datetime.fromtimestamp(r["taken"]).isoformat(" ", "minutes") if r.get("taken") else None
        r["credit"] = credit(r["id"])
    return rs


def run(q: Query):
    with lock:
        return {"results": rows(S.search(q)), "embedded": len(S.ids)}


def explainer() -> Explainer:
    if _explainer[0] is None:
        _explainer[0] = Explainer(S)
    return _explainer[0]


# ---------------------------------------------------------------- routes
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/stats")
def stats():
    db = S.lib.db
    n, ocr, txt, places = db.execute("SELECT count(*), sum(ocr_done), count(ocr), count(DISTINCT place) FROM photos").fetchone()
    albums = {r[0]: r[1] for r in db.execute("SELECT album, count(*) FROM photos GROUP BY album ORDER BY album")}
    lo, hi = db.execute("SELECT min(taken), max(taken) FROM photos").fetchone()
    span = [datetime.fromtimestamp(t).date().isoformat() if t else None for t in (lo, hi)]
    return {"photos": n, "embedded": len(S.ids), "ocr_done": ocr or 0, "with_text": txt, "places": places, "albums": albums,
            "date_range": span, "index": "demo", "place_mode": S.Z is not None,
            "demo": {"ask": False, "upload": False, "agent_writes": False, "credits": "/credits"}}


@app.post("/api/track", status_code=204)
async def track(request: Request):
    """Visit card from web/src/visits.ts -> Discord (visits.py). Always 204; never waits on Discord.
    Same-origin browsers only, never from localhost, at most TRACK_PER_MIN per IP."""
    h, ip = request.headers, client_ip(request)
    if (visits.enabled() and h.get("sec-fetch-site") == "same-origin" and host_name(h.get("host", "")) not in LOCAL
            and _buckets.hit((ip, "track"), TRACK_PER_MIN)):
        raw = await request.body()
        try:
            body = json.loads(raw) if len(raw) <= MAX_TRACK_BODY else None
        except ValueError:
            body = None
        if isinstance(body, dict):
            visits.report_visit(body, ip=ip if ip != "?" else None, region=h.get("fly-region"))
    return Response(status_code=204)


@app.get("/api/browse")
def browse(offset: int = 0, limit: int = 400):
    if _order[0] is None:
        ids = [r[0] for r in S.lib.db.execute("SELECT id FROM photos")]
        random.Random(7).shuffle(ids)
        _order[0] = ids
    limit, offset = max(1, min(1000, limit)), max(0, offset)
    return {"ids": _order[0][offset:offset + limit], "total": len(_order[0])}


@app.get("/api/search")
def search(request: Request):
    p = request.query_params
    if not _s(p.get("q")):
        raise HTTPException(400, "q required")
    if len(p["q"]) > MAX_Q:
        raise HTTPException(414, f"query longer than {MAX_Q} characters")
    q = Query(text=p["q"].strip(), filters=filters(p), k=k_of(p), parse=True)
    return {**run(q), "parsed": q.parsed}


@app.post("/api/search_image")
def search_image():
    raise HTTPException(403, "Search by your own photo is off in the public demo: your photo would have to be "
                             "uploaded. On your Mac or phone it never leaves the device. Try 'More like this' instead.")


@app.get("/api/explain/{pid}")
def explain(pid: int, q: str | None = None, ref: int | None = None, tags: str | None = None):
    q = q[:MAX_Q] if q else q
    with lock:
        if pid not in S.lib.row_of:
            raise HTTPException(404, "photo not found")
        E, v = explainer(), S.X[S.lib.row_of[pid]]
        row = rows(S.lib.get([pid]))[0]
        out = {"place": row["place"], "date": row["date"], "text_hits": [], "place_hits": [], "credit": row["credit"]}
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
    if pid not in S.lib.row_of:
        raise HTTPException(404, "photo not found")
    mode = "place" if p.get("mode") == "place" else "similar"
    q = _s(p.get("q"))
    return run(Query(text=q[:MAX_Q] if q else None, image=pid, filters=filters(p), k=k_of(p), mode=mode))


ASK_NOTE = ("Ask runs a vision-language model on your own Mac, so it's off in this hosted demo. "
            "Here are the closest search results instead.")


@app.post("/api/ask")
async def ask(request: Request):
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(403, "Ask with a photo is off in the public demo (it would upload your photo).")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "body is not valid JSON")
    question = _s(body.get("question")) if isinstance(body, dict) else None
    if not question:
        raise HTTPException(400, "question required")
    if len(question) > MAX_Q:
        raise HTTPException(414, f"question longer than {MAX_Q} characters")

    def go():
        t = time.time()
        q = Query(text=question, k=60, parse=True)
        res = run(q)["results"]
        return {"plan": {"queries": [q.parsed.get("text") if q.parsed else question], "answer_type": "photos"},
                "answer": ASK_NOTE, "results": [{**r, "verified": False} for r in res],
                "timing": {"plan+recall_s": round(time.time() - t, 3), "verify_s": 0}}
    return await run_in_threadpool(go)


@app.get("/api/ask/status")
def ask_status():
    return {"loaded": True, "queued": False, "disabled": True}


# ---------------------------------------------------------------- agent: read-only tools, sessions in memory
import mock_agent  # noqa: E402
from photosearch import cleanup, tools  # noqa: E402

READ_ONLY = {"search_photos", "get_photos", "explain_photo", "list_albums", "get_album", "list_trash"}
REFUSAL = ("{tool} is turned off in the public demo: the photos here are shared by everyone, so nothing can be "
           "changed. On your own device the agent previews it, asks you to confirm, and keeps an undo.")


CLEANUP_REFUSAL = ("cleanup suggestions are off in the public demo: they need image statistics computed on your own "
                   "device. Try searching instead.")


def demo_call(name: str, args: dict | None = None, actor: str = "agent") -> dict:
    if name == "find_cleanup_candidates":
        return {"ok": False, "error": CLEANUP_REFUSAL}
    if name not in READ_ONLY:
        return {"ok": False, "error": REFUSAL.format(tool=name.replace("_", " "))}
    return tools.call(name, args, actor=actor)


try:
    tools.init(S, lock, STATE)
    tools._candidates = lambda: {k: [] for k in cleanup.KINDS}  # cleanup needs on-device image stats: off in the demo
    EXECUTOR = mock_agent.Executor(call=demo_call)
except Exception as e:  # never take search down because the agent couldn't start
    print(f"[demo] agent disabled: {type(e).__name__}", file=sys.stderr, flush=True)
    EXECUTOR = None


async def _json_body(request: Request) -> dict:
    if not (request.headers.get("content-type") or "").startswith("application/json"):
        raise HTTPException(415, "send a JSON object (Content-Type: application/json)")
    try:
        b = await request.json()
    except Exception:
        raise HTTPException(400, "body is not valid JSON")
    if not isinstance(b, dict):
        raise HTTPException(400, "body must be a JSON object")
    return b


def _sid(b: dict, k: str, required=True):
    v = b.get(k)
    if not required and (v is None or v == ""):
        return None
    if not isinstance(v, str) or not v or len(v) > 64:
        raise HTTPException(400, f"{k} required (string)")
    return v


async def _agent(fn, *a):
    if EXECUTOR is None:
        raise HTTPException(503, "the photo agent is unavailable in this demo")
    try:
        return await run_in_threadpool(fn, *a)
    except mock_agent.AgentError as e:
        raise HTTPException(e.status, str(e))


@app.post("/api/agent/message")
async def agent_message(request: Request):
    b = await _json_body(request)
    return await _agent(EXECUTOR.message, b.get("text"), _sid(b, "session_id", required=False))


@app.post("/api/agent/confirm")
async def agent_confirm(request: Request):
    b = await _json_body(request)
    return await _agent(EXECUTOR.confirm, _sid(b, "session_id"), _sid(b, "pending_id"))


@app.post("/api/agent/cancel")
async def agent_cancel(request: Request):
    b = await _json_body(request)
    return await _agent(EXECUTOR.cancel, _sid(b, "session_id"), _sid(b, "pending_id"))


@app.post("/api/agent/undo")
async def agent_undo(request: Request):
    b = await _json_body(request)
    return await _agent(EXECUTOR.undo, _sid(b, "session_id"), _sid(b, "undo_id"))


@app.get("/api/albums")
def albums():
    return tools.call("list_albums", {})


@app.get("/api/albums/{album_id}")
def album(album_id: int, limit: int = 60, offset: int = 0):
    out = tools.call("get_album", {"album_id": album_id, "limit": max(1, min(200, limit)), "offset": max(0, offset)})
    if not out.get("ok"):
        raise HTTPException(404, out.get("error", "not found"))
    return out


# ---------------------------------------------------------------- photos + credits
@app.get("/photo/{pid}")
def photo(pid: int, w: int | None = None):
    loc = _toff.get(pid)
    if loc is not None and _pack is not None:
        b = os.pread(_pack, loc[1], loc[0])
        return Response(b, media_type=THUMB_MIME, headers={"Cache-Control": "public, max-age=86400"})
    if THUMB_BASE and pid in S.lib.row_of:
        return RedirectResponse(f"{THUMB_BASE}/{pid}.{THUMB_EXT}", status_code=302,
                                headers={"Cache-Control": "public, max-age=86400"})
    raise HTTPException(404)


@app.get("/api/credits/{pid}")
def credits_one(pid: int):
    if not _has_credits:
        raise HTTPException(404)
    r = S.lib.db.execute("SELECT * FROM credits WHERE id=?", (pid,)).fetchone()
    if not r:
        raise HTTPException(404)
    return dict(r)


@app.get("/credits", response_class=HTMLResponse)
def credits_page(page: int = 0, id: int | None = None):
    per = 500
    db = S.lib.db
    if not _has_credits:
        return HTMLResponse("<p>No credits in this bundle.</p>")
    if id is not None:
        rs = db.execute("SELECT * FROM credits WHERE id=?", (id,)).fetchall()
    else:
        rs = db.execute("SELECT * FROM credits ORDER BY id LIMIT ? OFFSET ?", (per, max(0, page) * per)).fetchall()
    total = db.execute("SELECT count(*) FROM credits").fetchone()[0]
    e = lambda v: html.escape(str(v)) if v is not None else ""
    link = lambda u, t: f'<a href="{e(u)}" rel="noopener noreferrer nofollow">{e(t)}</a>' if u else e(t)
    body = "".join(f"<tr><td><a href='/photo/{r['id']}'>{r['id']}</a></td><td>{e(r['creator'])}</td>"
                   f"<td>{link(r['license_url'], r['license'])}</td><td>{link(r['flickr_url'] or r['source_url'], r['source'])}</td></tr>"
                   for r in rs)
    pages = math.ceil(total / per)
    nav = " ".join(f"<a href='/credits?page={i}'>{i + 1}</a>" for i in range(pages)) if id is None and pages > 1 else ""
    return HTMLResponse(f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Photo credits</title><style>body{{font:14px system-ui;margin:16px;max-width:960px}}td,th{{padding:2px 8px;text-align:left}}
@media(prefers-color-scheme:dark){{body{{background:#111;color:#ddd}}a{{color:#8ab4ff}}}}</style>
<h1>Photo credits</h1><p>{total} photos from <a href="https://www.robots.ox.ac.uk/~vgg/research/pass/">PASS</a>
(Asano et al., NeurIPS 2021; CC BY 4.0), a subset of YFCC100M. Each photo is by the Flickr user named below, under the
license shown; thumbnails here are resized. Place names: GeoNames (CC BY 4.0). Search model: SigLIP2 (Apache-2.0).
<a href="/CREDITS.csv">CREDITS.csv</a>. To request removal of a photo, contact the demo's operator.</p>
<table><tr><th>photo</th><th>creator</th><th>license</th><th>source</th></tr>{body}</table><p>{nav}</p>""")


@app.get("/CREDITS.csv")
def credits_csv():
    p = os.path.join(BUNDLE, "CREDITS.csv")
    if not os.path.exists(p):
        raise HTTPException(404)
    return Response(open(p, "rb").read(), media_type="text/csv; charset=utf-8")


# ---------------------------------------------------------------- web app (unchanged web/dist + a credits/demo note)
WEB = os.environ.get("DEMO_WEB") or next((p for p in (os.path.join(HERE, "..", "..", "web", "dist"), "/app/web")
                                          if os.path.isdir(p)), "")
NOTE = ('<div style="position:fixed;left:8px;bottom:6px;z-index:2147483647;font:11px system-ui;opacity:.75;'
        'background:rgba(0,0,0,.45);color:#fff;padding:3px 8px;border-radius:9px">Public demo: searches run on this '
        'server · <a href="/credits" style="color:#fff">photo credits</a></div>')


def _index_html() -> bytes:
    raw = open(os.path.join(WEB, "index.html"), encoding="utf-8").read()
    return raw.replace("</body>", NOTE + "</body>").encode()


if WEB and os.path.isdir(WEB):
    _INDEX = _index_html()

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    def index():
        return Response(_INDEX, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})

    class _Web(StaticFiles):
        async def get_response(self, path, scope):
            r = await super().get_response(path, scope)
            r.headers["Cache-Control"] = "public, max-age=31536000, immutable" if path.startswith("assets/") else "no-cache"
            return r

    app.mount("/", _Web(directory=WEB, html=True), name="web")


def _warm():  # before the port opens: the first visitor shouldn't pay for ONNX graph init / the concept prompts
    with lock:
        S.embed_prompted("warm up")
        explainer()
    S.lib.encoder.text._cache.clear()


_warm()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8080)), access_log=False,
                proxy_headers=False, server_header=False)
