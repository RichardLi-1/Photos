"""Agent-facing photo tools: search, albums, soft delete (trash), cleanup suggestions, explanations.

Three ways in, same rails:
    tools.call("trash_photos", {"photo_ids": [1, 2], "reason": "blurry"})   -> dict (never raises)
    tools.trash_photos([1, 2], "blurry")                                    -> dict (plain function)
    tools.schemas()                                                         -> JSON-schema tool definitions

Safety model
  * Nothing here deletes or moves a file, ever. "Trash" sets photos.deleted_at (restorable, like Recently Deleted);
    "purge" only sets photos.purged_at. Freeing disk space is a separate, unwired hook: purge_files_hook().
  * Destructive tools (DESTRUCTIVE) take dry_run, default True: they return what *would* happen plus a
    confirm_token. A real run must pass dry_run=false AND that token, which binds it to exactly the previewed
    photo ids and expires after PREVIEW_TTL_S. The agent's own confirmation policy sits between the two calls.
  * Per-call batch caps (MAX_*). Every mutating tool call -- through call() or as a plain function, dry run or not,
    ok or failed -- writes audit_log; a real mutation writes its audit row in the SAME transaction, so a change
    without its audit row can't be committed.
  * Small WAL transactions on a private write connection; reads use a separate connection (committed data only),
    since the server reads the same SQLite file.
"""
from __future__ import annotations

import functools
import hashlib
import hmac
import inspect
import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime

import numpy as np

from . import cleanup

INDEX = os.environ.get("PHOTOSEARCH_INDEX", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "index"))
MODEL = "siglip2-b16-384"

MAX_TRASH = 200          # photos per trash/restore call
MAX_PURGE = 500          # trash rows marked purged per call
MAX_ALBUM_BATCH = 1000   # photos per add/remove-from-album call
MAX_GET = 100            # photos per get_photos call
MAX_LIMIT = 200          # results per search / listing call
PREVIEW_TTL_S = 15 * 60  # confirm_token lifetime
DESTRUCTIVE = {"trash_photos", "purge_trash", "delete_album", "remove_from_album"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS albums (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at REAL, updated_at REAL);
CREATE UNIQUE INDEX IF NOT EXISTS albums_name ON albums(name COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS album_photos (
  album_id INTEGER NOT NULL, photo_id INTEGER NOT NULL, added_at REAL, PRIMARY KEY (album_id, photo_id));
CREATE INDEX IF NOT EXISTS album_photos_photo ON album_photos(photo_id);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, actor TEXT, tool TEXT NOT NULL, args TEXT, dry_run INT, ok INT,
  n_affected INT, summary TEXT);
"""
PHOTO_COLUMNS = {"deleted_at": "REAL", "deleted_reason": "TEXT", "purged_at": "REAL"}


class ToolError(Exception):
    pass


# ---------------------------------------------------------------- context

class _Reader:
    """Read-only access on its own connection: sees committed data only, never a half-done write transaction."""

    def __init__(self, path):
        self.conn = cleanup.connect(path)
        self.lock = threading.Lock()

    def all(self, sql, args=()):
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def one(self, sql, args=()):
        with self.lock:
            return self.conn.execute(sql, args).fetchone()


class _Ctx:
    def __init__(self, searcher, lock, index_dir):
        path = f"{index_dir}/library.sqlite"
        self.s = searcher
        self.lock = lock or threading.RLock()  # guards the Searcher / encoder (shared with server.py)
        self.wlock = threading.RLock()          # guards the write connection; every write goes through it
        self.db = cleanup.connect(path)
        ensure_schema(self.db)
        self.r = _Reader(path)
        self.secret = secrets.token_bytes(16)
        self.tokens: dict[str, float] = {}
        self.detector = cleanup.Detector(cleanup.connect(path), searcher, f"{index_dir}/cleanup_prompts__{MODEL}.npz")
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            from .explain import Explainer
            self._explainer = Explainer(self.s)
        return self._explainer


_ctx: _Ctx | None = None
_init_lock = threading.Lock()


def ensure_schema(db: sqlite3.Connection):
    """Idempotent: our tables, photo_stats, and the soft-delete columns on photos."""
    with db:
        db.executescript(SCHEMA)
    cleanup.ensure_schema(db)
    have = {r[1] for r in db.execute("PRAGMA table_info(photos)")}
    for col, typ in PHOTO_COLUMNS.items():
        if col not in have:
            try:
                with db:
                    db.execute(f"ALTER TABLE photos ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError as e:  # another process added it first
                if "duplicate column" not in str(e):
                    raise
    with db:
        db.execute("CREATE INDEX IF NOT EXISTS photos_deleted ON photos(deleted_at)")


def init(searcher=None, lock=None, index_dir: str = INDEX) -> _Ctx:
    """Call once with the server's Searcher and lock. Without a searcher, opens the index read-only itself."""
    global _ctx
    with _init_lock:
        if searcher is None:
            from .library import Library
            from .search import Searcher
            searcher = Searcher(Library(index_dir, MODEL, readonly=True))
        _ctx = _Ctx(searcher, lock, index_dir)
        return _ctx


def ctx() -> _Ctx:
    return _ctx or init()


# ---------------------------------------------------------------- helpers

def _now():
    return time.time()


def _iso(ts):
    return datetime.fromtimestamp(ts).isoformat(" ", "minutes") if ts else None


def _ids(xs, cap, what="photo_ids") -> list[int]:
    if xs is None:
        raise ToolError(f"{what} required")
    if isinstance(xs, (int, str)):
        xs = [xs]
    try:
        out = list(dict.fromkeys(int(x) for x in xs))
    except (TypeError, ValueError):
        raise ToolError(f"{what} must be a list of integers")
    if not out:
        raise ToolError(f"{what} is empty")
    if len(out) > cap:
        raise ToolError(f"too many {what}: {len(out)} > cap {cap} per call; split into batches")
    return out


def _q(n):
    return ",".join("?" * n)


def _photo_rows(ids) -> dict[int, dict]:
    if not ids:
        return {}
    return {r["id"]: dict(r) for r in ctx().r.all(f"SELECT * FROM photos WHERE id IN ({_q(len(ids))})", ids)}


def _trashed() -> set[int]:
    return {r[0] for r in ctx().r.all("SELECT id FROM photos WHERE deleted_at IS NOT NULL")}


def _brief(r: dict, score=None) -> dict:
    o = {"id": r["id"], "date": _iso(r.get("taken")), "place": r.get("place"), "folder": r.get("album"),
         "size": f"{r.get('width')}x{r.get('height')}" if r.get("width") else None,
         "thumb": f"/photo/{r['id']}?w=256"}
    if r.get("ocr"):
        o["text"] = r["ocr"].replace("\n", " ")[:80]
    if score is not None:
        o["score"] = score
    if r.get("deleted_at"):
        o["trashed_at"] = _iso(r["deleted_at"])
    return o


def _album(album_id) -> dict:
    try:
        album_id = int(album_id)
    except (TypeError, ValueError):
        raise ToolError("album_id must be an integer")
    r = ctx().r.one("SELECT * FROM albums WHERE id=?", (album_id,))
    if not r:
        raise ToolError(f"no album with id {album_id}; call list_albums")
    return dict(r)


def _name(name) -> str:
    name = (name or "").strip() if isinstance(name, str) else ""
    if not name or len(name) > 120:
        raise ToolError("album name must be 1-120 characters")
    return name


def _token(tool: str, plan: dict) -> str:
    c = ctx()
    msg = f"{tool}|{json.dumps(plan, sort_keys=True)}".encode()
    t = hmac.new(c.secret, msg, hashlib.sha256).hexdigest()[:20]
    c.tokens = {k: v for k, v in c.tokens.items() if v > _now()}
    c.tokens[t] = _now() + PREVIEW_TTL_S
    return t


def _check_token(tool: str, plan: dict, token: str | None):
    """A real destructive run must replay the token from a dry run of exactly the same plan."""
    c = ctx()
    if not token:
        raise ToolError(f"{tool} with dry_run=false needs confirm_token: call it with dry_run=true first, show the user "
                        "what will happen, then repeat the call with the returned confirm_token")
    msg = f"{tool}|{json.dumps(plan, sort_keys=True)}".encode()
    want = hmac.new(c.secret, msg, hashlib.sha256).hexdigest()[:20]
    if not hmac.compare_digest(want, str(token)):
        raise ToolError("confirm_token does not match this exact request (photos or arguments changed since the "
                        "preview); run the dry run again")
    if c.tokens.get(want, 0) < _now():
        raise ToolError("confirm_token expired or already used; run the dry run again")
    c.tokens.pop(want, None)


_tl = threading.local()  # .actor (set by call()), .stack (one frame per running audited tool)


def _audit_row(db, frame, dry_run, ok, n, summary):
    db.execute("INSERT INTO audit_log(ts, actor, tool, args, dry_run, ok, n_affected, summary) VALUES (?,?,?,?,?,?,?,?)",
               (_now(), getattr(_tl, "actor", None) or "python", frame["tool"],
                json.dumps(frame["args"], default=str)[:4000], int(bool(dry_run)), int(ok), n, str(summary)[:500]))


def _audit_tx(db, n, summary):
    """Call INSIDE the `with c.wlock, c.db:` block of a real mutation: the audit row commits (or rolls back) with it."""
    frame = _tl.stack[-1]
    _audit_row(db, frame, False, True, n, summary)
    frame["logged"] = True


def _summary(out: dict) -> str:
    return json.dumps({k: out[k] for k in ("photo_ids", "album", "reason") if k in out}, default=str)


def _count(out: dict):
    return next((out[k] for k in ("trashed", "would_trash", "purged", "would_purge", "removed", "would_remove",
                                  "added", "restored") if isinstance(out.get(k), int)), None)


def _audited(fn):
    """Every call of a mutating tool is audited, whether it came through call() or as a plain function."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            b = sig.bind(*a, **kw)
            b.apply_defaults()
            args = dict(b.arguments)
        except TypeError:
            args = {"args": a, **kw}
        frame = {"tool": fn.__name__, "args": args, "logged": False}
        if not hasattr(_tl, "stack"):
            _tl.stack = []
        _tl.stack.append(frame)
        try:
            out = fn(*a, **kw)
        except Exception as e:
            _log_outside_tx(frame, args.get("dry_run", False), False, None, f"{type(e).__name__}: {e}")
            raise
        finally:
            _tl.stack.pop()
        if not frame["logged"]:  # dry runs and no-op calls: nothing was mutated, log on its own
            _log_outside_tx(frame, out.get("dry_run", False), True, _count(out), _summary(out))
        return out
    return wrapper


def _log_outside_tx(frame, dry_run, ok, n, summary):
    c = ctx()
    with c.wlock, c.db:
        _audit_row(c.db, frame, dry_run, ok, n, summary)


# ---------------------------------------------------------------- search & read

def search_photos(query: str | None = None, similar_to: int | None = None, after: str | None = None,
                  before: str | None = None, place: str | None = None, folder: str | None = None,
                  album_id: int | None = None, near: dict | None = None, has_text: bool = False,
                  limit: int = 24) -> dict:
    from .search import Filters, Query
    if not (query and str(query).strip()) and similar_to is None:
        raise ToolError("give query (text) and/or similar_to (a photo id)")
    for k, v in (("after", after), ("before", before)):
        if v:
            try:
                datetime.fromisoformat(v)
            except ValueError:
                raise ToolError(f"{k} must be an ISO date like 2024-06-01")
    limit = max(1, min(MAX_LIMIT, int(limit)))
    nr = None
    if near:
        try:
            nr = (float(near["lat"]), float(near["lon"]), float(near.get("km", 5)))
        except (KeyError, TypeError, ValueError):
            raise ToolError("near must be {lat, lon, km}")
    f = Filters(after=after, before=before, place=place, album=folder, near=nr, has_text=bool(has_text))
    c = ctx()
    trashed = _trashed()
    text = str(query).strip() if query else None
    with c.lock:
        S = c.s
        if similar_to is not None and int(similar_to) not in S.lib.row_of:
            raise ToolError(f"photo {similar_to} not found / not embedded")
        if album_id is None:
            # text-only queries go through the search-box parser: "Japan trip 2019" -> place + date filters + "trip"
            res = S.search(Query(text=text, image=int(similar_to) if similar_to is not None else None, filters=f,
                                 k=limit + min(len(trashed), 200), parse=similar_to is None))
            hits = [(r["id"], r.get("score")) for r in res if r["id"] not in trashed][:limit]
        else:  # rank inside a user album
            _album(album_id)
            members = [r[0] for r in c.r.all("SELECT photo_id FROM album_photos WHERE album_id=?", (int(album_id),))]
            m = S.mask(f)
            rows = [S.lib.row_of[p] for p in members if p in S.lib.row_of and p not in trashed and m[S.lib.row_of[p]]]
            vecs = []
            if text:
                vecs.append(S.embed_text(text))
            if similar_to is not None:
                vecs.append(S.X[S.lib.row_of[int(similar_to)]])
            v = np.sum(vecs, 0); v /= np.linalg.norm(v)
            sim = S.X[rows] @ v if rows else np.zeros(0)
            order = np.argsort(-sim)[:limit]
            hits = [(int(S.ids[rows[i]]), round(float(sim[i]), 4)) for i in order]
    meta = _photo_rows([h for h, _ in hits])
    return {"count": len(hits), "results": [_brief(meta[h], s) for h, s in hits if h in meta]}


def get_photos(photo_ids: list[int]) -> dict:
    ids = _ids(photo_ids, MAX_GET)
    meta = _photo_rows(ids)
    R = ctx().r
    alb = {}
    for r in R.all(f"""SELECT ap.photo_id, a.id, a.name FROM album_photos ap JOIN albums a ON a.id = ap.album_id
                       WHERE ap.photo_id IN ({_q(len(ids))})""", ids):
        alb.setdefault(r[0], []).append({"id": r[1], "name": r[2]})
    st = {r["id"]: dict(r) for r in R.all(f"SELECT * FROM photo_stats WHERE id IN ({_q(len(ids))})", ids)}
    out = []
    for i in ids:
        r = meta.get(i)
        if not r:
            out.append({"id": i, "error": "not found"}); continue
        o = {"id": i, "path": r["path"], "folder": r["album"], "date": _iso(r["taken"]), "place": r["place"],
             "lat": r["lat"], "lon": r["lon"], "camera": r["camera"], "width": r["width"], "height": r["height"],
             "text": r["ocr"], "albums": alb.get(i, []), "trashed": r.get("deleted_at") is not None,
             "thumb": f"/photo/{i}?w=256"}
        if r.get("deleted_at"):
            o.update(trashed_at=_iso(r["deleted_at"]), trash_reason=r.get("deleted_reason"), purged=r.get("purged_at") is not None)
        if i in st and st[i]["err"] is None:
            s = st[i]
            o["quality"] = {"sharpness": round(s["lap_tile_max"]), "brightness": round(s["luma_mean"]),
                            "clipped_white": round(s["frac_bright"], 3), "near_black": round(s["frac_dark"], 3)}
        out.append(o)
    return {"photos": out}


def explain_photo(photo_id: int) -> dict:
    """What's in a photo (zero-shot concepts) + metadata + any cleanup flags."""
    pid = int(photo_id)
    c = ctx()
    p = get_photos([pid])["photos"][0]
    if "error" in p:
        raise ToolError(f"photo {pid} not found")
    with c.lock:
        if pid not in c.s.lib.row_of:
            raise ToolError(f"photo {pid} is not embedded yet")
        v = c.s.X[c.s.lib.row_of[pid]]
        tags = c.explainer.tags(v, k=6)
    flags = [{"kind": k["kind"], "confidence": k["confidence"], "reasons": k["reasons"],
              **({"keep": k["keep"], "group": k["photo_ids"]} if "keep" in k else {})}
             for k in cleanup.flatten(_candidates(), None, _trashed()) if pid in k["photo_ids"]]
    words = f"Shows {', '.join(tags) or 'nothing clearly recognisable'}"
    if p["date"] or p["place"]:
        words += f"; taken {p['date'] or ''}{' in ' + p['place'] if p['place'] else ''}".rstrip()
    if flags:
        words += ". Possibly not needed: " + "; ".join(f"{f['kind']} ({f['reasons'][0]})" for f in flags)
    return {"id": pid, "summary": words, "concepts": tags, "photo": p, "cleanup_flags": flags}


# ---------------------------------------------------------------- albums

def list_albums() -> dict:
    rows = ctx().r.all("""
        SELECT a.id, a.name, a.created_at, a.updated_at, count(p.id) AS n, max(p.id) AS cover
        FROM albums a LEFT JOIN album_photos ap ON ap.album_id = a.id
        LEFT JOIN photos p ON p.id = ap.photo_id AND p.deleted_at IS NULL
        GROUP BY a.id ORDER BY a.name COLLATE NOCASE""")
    return {"albums": [{"id": r["id"], "name": r["name"], "count": r["n"], "cover": r["cover"],
                        "created": _iso(r["created_at"]), "updated": _iso(r["updated_at"])} for r in rows]}


def get_album(album_id: int, limit: int = 60, offset: int = 0) -> dict:
    a = _album(album_id)
    limit = max(1, min(MAX_LIMIT, int(limit)))
    R = ctx().r
    n = R.one("""SELECT count(*) FROM album_photos ap JOIN photos p ON p.id = ap.photo_id
                 WHERE ap.album_id=? AND p.deleted_at IS NULL""", (a["id"],))[0]
    rows = R.all("""SELECT p.* FROM album_photos ap JOIN photos p ON p.id = ap.photo_id
                    WHERE ap.album_id=? AND p.deleted_at IS NULL ORDER BY ap.added_at, p.id LIMIT ? OFFSET ?""",
                 (a["id"], limit, max(0, int(offset))))
    return {"album": {"id": a["id"], "name": a["name"], "count": n}, "photos": [_brief(dict(r)) for r in rows]}


@_audited
def create_album(name: str, photo_ids: list[int] | None = None) -> dict:
    name = _name(name)
    c = ctx()
    with c.wlock:
        if c.db.execute("SELECT 1 FROM albums WHERE name=? COLLATE NOCASE", (name,)).fetchone():
            raise ToolError(f"an album named {name!r} already exists")
        with c.db:
            aid = c.db.execute("INSERT INTO albums(name, created_at, updated_at) VALUES (?,?,?)", (name, _now(), _now())).lastrowid
            _audit_tx(c.db, None, json.dumps({"album": {"id": aid, "name": name}}))
    out = {"album": {"id": aid, "name": name}, "added": 0}
    if photo_ids:
        out["added"] = add_to_album(aid, photo_ids)["added"]
    return out


@_audited
def add_to_album(album_id: int, photo_ids: list[int]) -> dict:
    a = _album(album_id)
    ids = _ids(photo_ids, MAX_ALBUM_BATCH)
    meta = _photo_rows(ids)
    missing = [i for i in ids if i not in meta]
    trashed = [i for i in ids if i in meta and meta[i].get("deleted_at")]
    ok = [i for i in ids if i in meta and not meta[i].get("deleted_at")]
    c = ctx()
    with c.wlock, c.db:
        cur = c.db.executemany("INSERT OR IGNORE INTO album_photos(album_id, photo_id, added_at) VALUES (?,?,?)",
                               [(a["id"], i, _now()) for i in ok])
        added = cur.rowcount
        c.db.execute("UPDATE albums SET updated_at=? WHERE id=?", (_now(), a["id"]))
        _audit_tx(c.db, added, json.dumps({"album": {"id": a["id"], "name": a["name"]}, "photo_ids": ok}))
    return {"album": {"id": a["id"], "name": a["name"]}, "added": added, "already_in_album": len(ok) - added,
            "skipped_missing": missing, "skipped_trashed": trashed}


@_audited
def remove_from_album(album_id: int, photo_ids: list[int], dry_run: bool = True, confirm_token: str | None = None) -> dict:
    a = _album(album_id)
    ids = _ids(photo_ids, MAX_ALBUM_BATCH)
    c = ctx()
    present = sorted(r[0] for r in c.r.all(
        f"SELECT photo_id FROM album_photos WHERE album_id=? AND photo_id IN ({_q(len(ids))})", [a["id"], *ids]))
    plan = {"album_id": a["id"], "photo_ids": present}
    out = {"album": {"id": a["id"], "name": a["name"]}, "would_remove" if dry_run else "removed": len(present),
           "not_in_album": [i for i in ids if i not in set(present)], "photo_ids": present,
           "note": "photos stay in the library; only album membership changes"}
    if dry_run:
        return {**out, "dry_run": True, "confirm_token": _token("remove_from_album", plan) if present else None}
    if not present:
        return {**out, "dry_run": False}
    _check_token("remove_from_album", plan, confirm_token)
    with c.wlock, c.db:
        c.db.executemany("DELETE FROM album_photos WHERE album_id=? AND photo_id=?", [(a["id"], i) for i in present])
        c.db.execute("UPDATE albums SET updated_at=? WHERE id=?", (_now(), a["id"]))
        _audit_tx(c.db, len(present), _summary(out))
    return {**out, "dry_run": False}


@_audited
def rename_album(album_id: int, new_name: str) -> dict:
    a = _album(album_id)
    name = _name(new_name)
    c = ctx()
    with c.wlock:
        clash = c.db.execute("SELECT id FROM albums WHERE name=? COLLATE NOCASE AND id<>?", (name, a["id"])).fetchone()
        if clash:
            raise ToolError(f"another album is already named {name!r}")
        with c.db:
            c.db.execute("UPDATE albums SET name=?, updated_at=? WHERE id=?", (name, _now(), a["id"]))
            _audit_tx(c.db, None, json.dumps({"album": {"id": a["id"], "name": name, "old_name": a["name"]}}))
    return {"album": {"id": a["id"], "name": name, "old_name": a["name"]}}


@_audited
def delete_album(album_id: int, dry_run: bool = True, confirm_token: str | None = None) -> dict:
    a = _album(album_id)
    c = ctx()
    n = c.r.one("SELECT count(*) FROM album_photos WHERE album_id=?", (a["id"],))[0]
    plan = {"album_id": a["id"], "name": a["name"], "n": n}
    out = {"album": {"id": a["id"], "name": a["name"]}, "photos_in_album": n,
           "note": "deletes the album only; its photos stay in the library untouched"}
    if dry_run:
        return {**out, "dry_run": True, "confirm_token": _token("delete_album", plan)}
    _check_token("delete_album", plan, confirm_token)
    with c.wlock, c.db:
        c.db.execute("DELETE FROM album_photos WHERE album_id=?", (a["id"],))
        c.db.execute("DELETE FROM albums WHERE id=?", (a["id"],))
        _audit_tx(c.db, n, _summary(out))
    return {**out, "dry_run": False, "deleted": True}


# ---------------------------------------------------------------- trash (soft delete)

@_audited
def trash_photos(photo_ids: list[int], reason: str, dry_run: bool = True, confirm_token: str | None = None) -> dict:
    ids = _ids(photo_ids, MAX_TRASH)
    reason = (reason or "").strip()[:200] if isinstance(reason, str) else ""
    if not reason:
        raise ToolError("reason required (shown to the user in Recently Deleted)")
    meta = _photo_rows(ids)
    todo = sorted(i for i in ids if i in meta and not meta[i].get("deleted_at"))
    plan = {"photo_ids": todo, "reason": reason}
    in_albums = ctx().r.one(f"SELECT count(DISTINCT album_id) FROM album_photos WHERE photo_id IN ({_q(len(todo))})",
                            todo)[0] if todo else 0
    out = {"would_trash" if dry_run else "trashed": len(todo), "photo_ids": todo,
           "already_in_trash": [i for i in ids if i in meta and meta[i].get("deleted_at")],
           "not_found": [i for i in ids if i not in meta], "reason": reason,
           "photos": [_brief(meta[i]) for i in todo[:12]],
           "albums_affected": in_albums,
           "note": "soft delete: hidden from search and albums, restorable with restore_photos; files are not touched"}
    if dry_run:
        return {**out, "dry_run": True, "confirm_token": _token("trash_photos", plan) if todo else None}
    if not todo:
        return {**out, "dry_run": False}
    _check_token("trash_photos", plan, confirm_token)
    c = ctx()
    with c.wlock, c.db:
        c.db.executemany("UPDATE photos SET deleted_at=?, deleted_reason=? WHERE id=? AND deleted_at IS NULL",
                         [(_now(), reason, i) for i in todo])
        _audit_tx(c.db, len(todo), _summary(out))
    return {**out, "dry_run": False}


@_audited
def restore_photos(photo_ids: list[int]) -> dict:
    ids = _ids(photo_ids, MAX_TRASH)
    meta = _photo_rows(ids)
    ok = [i for i in ids if i in meta and meta[i].get("deleted_at") and not meta[i].get("purged_at")]
    purged = [i for i in ids if i in meta and meta[i].get("purged_at")]
    c = ctx()
    if ok:
        with c.wlock, c.db:
            c.db.executemany("UPDATE photos SET deleted_at=NULL, deleted_reason=NULL WHERE id=? AND purged_at IS NULL",
                             [(i,) for i in ok])
            _audit_tx(c.db, len(ok), json.dumps({"photo_ids": ok}))
    return {"restored": len(ok), "photo_ids": ok, "not_in_trash": [i for i in ids if i in meta and not meta[i].get("deleted_at")],
            "not_found": [i for i in ids if i not in meta], "already_purged": purged,
            "note": "albums the photos were in are kept, so they reappear there too"}


def list_trash(limit: int = 60, offset: int = 0, include_purged: bool = False) -> dict:
    limit = max(1, min(MAX_LIMIT, int(limit)))
    w = "deleted_at IS NOT NULL" + ("" if include_purged else " AND purged_at IS NULL")
    R = ctx().r
    n = R.one(f"SELECT count(*) FROM photos WHERE {w}")[0]
    rows = R.all(f"SELECT * FROM photos WHERE {w} ORDER BY deleted_at DESC LIMIT ? OFFSET ?", (limit, max(0, int(offset))))
    out = []
    for r in rows:
        o = _brief(dict(r))
        o.update(reason=r["deleted_reason"], days_in_trash=round((_now() - r["deleted_at"]) / 86400, 1),
                 purged=r["purged_at"] is not None)
        out.append(o)
    return {"total": n, "photos": out}


def purge_files_hook(photos: list[dict]) -> None:
    """HOOK, deliberately NOT wired: purge_trash only marks rows purged in the DB and never touches files.

    If you later want purge to free disk space, implement it here (e.g. move each photo["path"] to the macOS Trash
    with NSFileManager.trashItemAtURL), then call it from purge_trash. Until then this does nothing on purpose.
    """
    return None


@_audited
def purge_trash(older_than_days: float = 30, dry_run: bool = True, confirm_token: str | None = None) -> dict:
    try:
        days = float(older_than_days)
    except (TypeError, ValueError):
        raise ToolError("older_than_days must be a number")
    if days < 0:
        raise ToolError("older_than_days must be >= 0")
    cutoff = _now() - days * 86400
    R = ctx().r
    total = R.one("SELECT count(*) FROM photos WHERE deleted_at IS NOT NULL AND purged_at IS NULL AND deleted_at <= ?",
                  (cutoff,))[0]
    ids = [r[0] for r in R.all("""SELECT id FROM photos WHERE deleted_at IS NOT NULL AND purged_at IS NULL
                                  AND deleted_at <= ? ORDER BY id LIMIT ?""", (cutoff, MAX_PURGE))]
    plan = {"photo_ids": ids, "older_than_days": days}
    out = {"would_purge" if dry_run else "purged": len(ids), "photo_ids": ids, "eligible_total": total,
           "remaining_after_this_call": total - len(ids),
           "note": "marks photos purged in the database only (no longer restorable through tools); "
                   "files on disk are NOT deleted or moved"}
    if dry_run:
        return {**out, "dry_run": True, "confirm_token": _token("purge_trash", plan) if ids else None}
    if not ids:
        return {**out, "dry_run": False}
    _check_token("purge_trash", plan, confirm_token)
    c = ctx()
    with c.wlock, c.db:
        c.db.executemany("UPDATE photos SET purged_at=? WHERE id=? AND deleted_at IS NOT NULL", [(_now(), i) for i in ids])
        _audit_tx(c.db, len(ids), _summary(out))
    # purge_files_hook(...) intentionally not called; see its docstring.
    return {**out, "dry_run": False}


# ---------------------------------------------------------------- cleanup

def _candidates():
    c = ctx()
    with c.lock:  # snapshot vectors + (once) build the prompt bank with the shared encoder, under the server lock
        ids, X = c.s.ids, c.s.X
        c.detector.bank
    return c.detector.candidates(ids, X)  # the slow pass runs outside the lock, so search stays responsive


def find_cleanup_candidates(kind: str | None = None, limit: int = 20, offset: int = 0, min_confidence: float = 0.0) -> dict:
    if kind is not None and kind not in cleanup.KINDS:
        raise ToolError(f"kind must be one of {cleanup.KINDS}")
    limit = max(1, min(MAX_LIMIT, int(limit)))
    trashed = _trashed()
    allc = _candidates()
    summary = {k: len(cleanup.flatten(allc, k, trashed)) for k in cleanup.KINDS}
    items = [x for x in cleanup.flatten(allc, kind, trashed) if x["confidence"] >= float(min_confidence)]
    items.sort(key=lambda x: -x["confidence"])
    page = items[int(offset):int(offset) + limit]
    meta = _photo_rows(sorted({p for x in page for p in x["photo_ids"]}))
    for x in page:
        x["photos"] = [_brief(meta[p]) for p in x["photo_ids"] if p in meta]
    return {"summary": summary, "total": len(items), "offset": int(offset), "candidates": page,
            "note": "suggestions only; to act, pass a candidate's 'remove' ids (groups) or photo_ids to trash_photos"}


# ---------------------------------------------------------------- registry

def _obj(props: dict, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


IDS = {"type": "array", "items": {"type": "integer"}, "minItems": 1}
DRY = {"type": "boolean", "description": "true (default) = only preview what would happen and get a confirm_token"}
TOKEN = {"type": "string", "description": "confirm_token from the dry run of this exact request; required when dry_run=false"}

TOOLS = [
    {"name": "search_photos", "fn": search_photos,
     "description": "Find photos by meaning (\"dog on a beach at sunset\"), by text visible in the photo, and/or by visual "
                    "similarity to a photo id. Optional filters: date range, place name, source folder, user album, "
                    "near a location, has text. Trashed photos are never returned. Returns ids with date/place/score.",
     "input_schema": _obj({
         "query": {"type": "string", "description": "natural-language description of the photos wanted"},
         "similar_to": {"type": "integer", "description": "photo id: find photos that look like this one"},
         "after": {"type": "string", "description": "ISO date, inclusive lower bound on capture time"},
         "before": {"type": "string", "description": "ISO date, exclusive upper bound on capture time"},
         "place": {"type": "string", "description": "substring of the place name, e.g. 'Paris'"},
         "folder": {"type": "string", "description": "source folder the photo was imported from (not a user album)"},
         "album_id": {"type": "integer", "description": "only search inside this user album"},
         "near": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"},
                                                   "km": {"type": "number"}}, "required": ["lat", "lon"]},
         "has_text": {"type": "boolean", "description": "only photos with readable text in them"},
         "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "default": 24}})},
    {"name": "get_photos", "fn": get_photos,
     "description": f"Full metadata for up to {MAX_GET} photo ids: date, place, GPS, camera, size, OCR text, albums it is "
                    "in, trash status and simple quality numbers (sharpness, brightness).",
     "input_schema": _obj({"photo_ids": {**IDS, "maxItems": MAX_GET}}, ["photo_ids"])},
    {"name": "explain_photo", "fn": explain_photo,
     "description": "Describe one photo in words: what it shows (zero-shot concepts), when/where, and whether it was "
                    "flagged as possibly unneeded (duplicate, blurry, screenshot...) with the reasons. Use before "
                    "recommending that the user deletes a specific photo.",
     "input_schema": _obj({"photo_id": {"type": "integer"}}, ["photo_id"])},
    {"name": "list_albums", "fn": list_albums,
     "description": "List the user's albums with id, name and photo count.", "input_schema": _obj({})},
    {"name": "get_album", "fn": get_album,
     "description": "List the photos in one album (paged).",
     "input_schema": _obj({"album_id": {"type": "integer"}, "limit": {"type": "integer", "default": 60},
                           "offset": {"type": "integer", "default": 0}}, ["album_id"])},
    {"name": "create_album", "fn": create_album,
     "description": "Create a new album (names are unique, case-insensitive), optionally filling it with photo ids.",
     "input_schema": _obj({"name": {"type": "string"}, "photo_ids": {**IDS, "maxItems": MAX_ALBUM_BATCH}}, ["name"])},
    {"name": "add_to_album", "fn": add_to_album,
     "description": f"Add photos to an album (up to {MAX_ALBUM_BATCH} per call). Already-present photos are ignored; "
                    "trashed or unknown ids are skipped and reported.",
     "input_schema": _obj({"album_id": {"type": "integer"}, "photo_ids": {**IDS, "maxItems": MAX_ALBUM_BATCH}},
                          ["album_id", "photo_ids"])},
    {"name": "remove_from_album", "fn": remove_from_album,
     "description": "Take photos out of an album; the photos themselves stay in the library. Two steps like "
                    "trash_photos: dry_run=true (default) to preview and get a confirm_token, then dry_run=false with it.",
     "input_schema": _obj({"album_id": {"type": "integer"}, "photo_ids": {**IDS, "maxItems": MAX_ALBUM_BATCH},
                           "dry_run": {**DRY, "default": True}, "confirm_token": TOKEN},
                          ["album_id", "photo_ids"])},
    {"name": "rename_album", "fn": rename_album,
     "description": "Rename an album.",
     "input_schema": _obj({"album_id": {"type": "integer"}, "new_name": {"type": "string"}}, ["album_id", "new_name"])},
    {"name": "delete_album", "fn": delete_album,
     "description": "Delete an album. Only the album goes away; its photos stay in the library. Two steps: call with "
                    "dry_run=true (default) to preview and get a confirm_token, get the user's OK, then call again "
                    "with dry_run=false and the token.",
     "input_schema": _obj({"album_id": {"type": "integer"}, "dry_run": {**DRY, "default": True}, "confirm_token": TOKEN},
                          ["album_id"])},
    {"name": "trash_photos", "fn": trash_photos,
     "description": f"Move photos to Recently Deleted (soft delete, restorable; files are never touched). Max {MAX_TRASH} "
                    "per call. Two steps: call with dry_run=true (default) to see exactly what would be trashed and get "
                    "a confirm_token; show the user and get their OK; then call again with the same photo_ids and "
                    "reason, dry_run=false and the token.",
     "input_schema": _obj({"photo_ids": {**IDS, "maxItems": MAX_TRASH},
                           "reason": {"type": "string", "description": "short reason shown to the user, e.g. 'blurry duplicate of #123'"},
                           "dry_run": {**DRY, "default": True}, "confirm_token": TOKEN}, ["photo_ids", "reason"])},
    {"name": "restore_photos", "fn": restore_photos,
     "description": "Bring photos back from Recently Deleted (undo trash_photos). Safe; no confirmation needed.",
     "input_schema": _obj({"photo_ids": {**IDS, "maxItems": MAX_TRASH}}, ["photo_ids"])},
    {"name": "list_trash", "fn": list_trash,
     "description": "List photos in Recently Deleted, newest first, with the reason and days in trash.",
     "input_schema": _obj({"limit": {"type": "integer", "default": 60}, "offset": {"type": "integer", "default": 0},
                           "include_purged": {"type": "boolean", "default": False}})},
    {"name": "purge_trash", "fn": purge_trash,
     "description": f"Permanently empty Recently Deleted for photos trashed at least older_than_days ago (max {MAX_PURGE} "
                    "per call). After purge they can no longer be restored through these tools. Two steps like "
                    "trash_photos: dry_run=true (default) first, user OK, then dry_run=false with the confirm_token.",
     "input_schema": _obj({"older_than_days": {"type": "number", "minimum": 0, "default": 30},
                           "dry_run": {**DRY, "default": True}, "confirm_token": TOKEN})},
    {"name": "find_cleanup_candidates", "fn": find_cleanup_candidates,
     "description": "Suggest photos the user probably doesn't need, each with reasons and a confidence (0-1). Kinds: "
                    "exact_duplicate and near_duplicate (groups, with a suggested 'keep' photo and 'remove' list), "
                    "blurry, dark, overexposed, screenshot, receipt, document, whiteboard, meme, accidental (pocket/"
                    "floor/ceiling/finger over lens). Receipts/documents/whiteboards may still be important; ask the "
                    "user. Returns per-kind counts in 'summary'. Never deletes anything.",
     "input_schema": _obj({"kind": {"type": "string", "enum": cleanup.KINDS},
                           "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "default": 20},
                           "offset": {"type": "integer", "default": 0},
                           "min_confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0}})},
]
REGISTRY = {t["name"]: t for t in TOOLS}
MUTATING = DESTRUCTIVE | {"create_album", "add_to_album", "rename_album", "restore_photos"}


def schemas(annotations: bool = False) -> list[dict]:
    """Tool definitions in the {name, description, input_schema} shape most LLM tool-calling APIs accept.

    annotations=True adds MCP-style hints (destructiveHint / readOnlyHint) for your confirmation policy.
    """
    out = []
    for t in TOOLS:
        d = {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        if annotations:
            d["annotations"] = {"destructiveHint": t["name"] in DESTRUCTIVE, "readOnlyHint": t["name"] not in MUTATING}
        out.append(d)
    return out


_PY = {"integer": int, "number": (int, float), "string": str, "boolean": bool, "array": list, "object": dict}


def _validate(schema: dict, args: dict) -> dict:
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    props = schema["properties"]
    extra = set(args) - set(props)
    if extra:
        raise ToolError(f"unknown argument(s) {sorted(extra)}; allowed: {sorted(props)}")
    missing = [k for k in schema.get("required", []) if args.get(k) is None]
    if missing:
        raise ToolError(f"missing required argument(s) {missing}")
    out = {}
    for k, v in args.items():
        if v is None:
            continue
        typ = props[k].get("type")
        if typ == "integer" and isinstance(v, str) and v.strip().lstrip("-").isdigit():
            v = int(v)  # LLMs sometimes quote numbers
        if typ == "boolean" and isinstance(v, str) and v.lower() in ("true", "false"):
            v = v.lower() == "true"
        if typ and (not isinstance(v, _PY[typ]) or (typ in ("integer", "number") and isinstance(v, bool))):
            raise ToolError(f"argument {k!r} must be {typ}")
        if typ == "array" and props[k].get("maxItems") and len(v) > props[k]["maxItems"]:
            raise ToolError(f"{k}: at most {props[k]['maxItems']} items per call; split into batches")
        if "enum" in props[k] and v not in props[k]["enum"]:
            raise ToolError(f"{k} must be one of {props[k]['enum']}")
        out[k] = v
    return out


def call(name: str, args: dict | None = None, actor: str = "agent") -> dict:
    """Run a tool by name with JSON args. Never raises: errors come back as {"ok": false, "error": ...}.

    Mutating tools audit themselves (see _audited); call() only adds rows for requests rejected before the tool ran.
    """
    t = REGISTRY.get(name)
    if t is None:
        return {"ok": False, "error": f"unknown tool {name!r}; available: {sorted(REGISTRY)}"}
    args = args if args is not None else {}
    prev, _tl.actor = getattr(_tl, "actor", None), actor
    try:
        try:
            a = _validate(t["input_schema"], args)
        except ToolError as e:
            out = {"ok": False, "error": str(e)}
            if name in MUTATING:
                default = inspect.signature(t["fn"]).parameters.get("dry_run")
                dry = args.get("dry_run", default.default if default is not None else False) if isinstance(args, dict) else True
                try:
                    _log_outside_tx({"tool": name, "args": args}, dry, False, None, out["error"])
                except Exception:
                    pass
            return out
        try:
            return {"ok": True, **t["fn"](**a)}
        except ToolError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:  # tool bug or resource guard: report, don't crash the agent loop
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        _tl.actor = prev


def audit_log(limit: int = 50) -> list[dict]:
    """Recent audit entries (not exposed as an LLM tool; for the UI / you)."""
    return [dict(r) for r in ctx().r.all("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (int(limit),))]
