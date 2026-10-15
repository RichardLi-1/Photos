"""HTTP routes for the agent tools (photosearch/tools.py): albums, trash (soft delete), cleanup suggestions.

server.py auto-includes this module (`router`, then `init(S, lock)`); it binds nothing itself. Local only: the
server's middleware enforces the Host allow-list, same-origin checks and a body cap on every route here.

  GET  /api/tools/schema             tool definitions (JSON schema) + destructive flags, for your agent / the UI
  POST /api/tools/{name}             run any tool with a JSON object body -> {"ok": bool, ...}
  GET  /api/albums                   POST /api/albums {name, photo_ids?}
  GET  /api/albums/{id}              PATCH /api/albums/{id} {name}
  POST /api/albums/{id}/add {photo_ids}       POST /api/albums/{id}/remove {photo_ids, dry_run?, confirm_token?}
  POST /api/albums/{id}/delete {dry_run?, confirm_token?}
  GET  /api/trash                    POST /api/trash {photo_ids, reason, dry_run?, confirm_token?}
  POST /api/trash/restore {photo_ids}         POST /api/trash/purge {older_than_days?, dry_run?, confirm_token?}
  GET  /api/cleanup?kind=&limit=&offset=&min_confidence=
  GET  /api/photos/{id}/details      explain_photo
  GET  /api/audit?limit=

Destructive routes are dry runs unless the body says "dry_run": false AND carries the confirm_token from the dry run.
Tool-level errors: POST /api/tools/{name} answers 200 {"ok": false, "error"}; convenience routes answer 4xx.
"""
from __future__ import annotations

import re
import sys

from fastapi import APIRouter, HTTPException, Request

from photosearch import tools

router = APIRouter()
_state = {"ready": False, "error": None}


def init(S, lock):
    try:
        tools.init(S, lock, S.lib.dir)  # same index the server opened (tools.INDEX is fixed at import time)
        _state.update(ready=True, error=None)
    except Exception as e:  # never take the search server down because the tool tables couldn't be set up
        _state.update(ready=False, error=f"{type(e).__name__}: {e}")
        print(f"[api_tools] disabled: {_state['error']}", file=sys.stderr, flush=True)


def _ready():
    if not _state["ready"]:
        raise HTTPException(503, f"photo tools unavailable: {_state['error'] or 'not initialised'}")


def _actor(request: Request, default: str) -> str:
    a = request.headers.get("x-actor") or default
    return re.sub(r"[^\w.-]", "", a)[:40] or default


async def _body(request: Request) -> dict:
    if not (request.headers.get("content-type") or "").startswith("application/json"):
        if not await request.body():
            return {}
        raise HTTPException(415, "send a JSON object (Content-Type: application/json)")
    try:
        b = await request.json()
    except Exception:
        raise HTTPException(400, "body is not valid JSON")
    if b is None:
        return {}
    if not isinstance(b, dict):
        raise HTTPException(400, "body must be a JSON object")
    return b


def _run(name: str, args: dict, request: Request, default_actor: str = "ui") -> dict:
    """Convenience routes: same validation + audit as the agent path, but errors become HTTP errors."""
    _ready()
    out = tools.call(name, args, actor=_actor(request, default_actor))
    if not out.get("ok"):
        err = out.get("error", "error")
        code = 404 if re.search(r"\bno album\b|not found", err) else 409 if "already exists" in err or "already named" in err else 400
        raise HTTPException(code, err)
    return out


def _pick(d: dict, *keys) -> dict:
    return {k: d[k] for k in keys if k in d}


# ---------------------------------------------------------------- generic tool endpoint

@router.get("/api/tools/schema")
def tool_schema():
    return {"tools": tools.schemas(annotations=True), "destructive": sorted(tools.DESTRUCTIVE),
            "protocol": "Destructive tools default to dry_run=true and return confirm_token; repeat the identical call "
                        "with dry_run=false and that token (after the user confirms) to execute. Tokens are "
                        f"single-use and expire after {tools.PREVIEW_TTL_S // 60} minutes."}


@router.post("/api/tools/{name}")
async def tool_call(name: str, request: Request):
    _ready()
    if name not in tools.REGISTRY:
        raise HTTPException(404, f"unknown tool {name!r}")
    args = await _body(request)
    actor = _actor(request, "agent")
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(tools.call, name, args, actor)


# ---------------------------------------------------------------- albums

@router.get("/api/albums")
def albums(request: Request):
    return _run("list_albums", {}, request)


@router.post("/api/albums")
async def album_create(request: Request):
    b = await _body(request)
    return _run("create_album", _pick(b, "name", "photo_ids"), request)


@router.get("/api/albums/{album_id}")
def album_get(album_id: int, request: Request, limit: int = 60, offset: int = 0):
    return _run("get_album", {"album_id": album_id, "limit": limit, "offset": offset}, request)


@router.patch("/api/albums/{album_id}")
async def album_rename(album_id: int, request: Request):
    b = await _body(request)
    return _run("rename_album", {"album_id": album_id, "new_name": b.get("name") or b.get("new_name")}, request)


@router.post("/api/albums/{album_id}/add")
async def album_add(album_id: int, request: Request):
    b = await _body(request)
    return _run("add_to_album", {"album_id": album_id, **_pick(b, "photo_ids")}, request)


@router.post("/api/albums/{album_id}/remove")
async def album_remove(album_id: int, request: Request):
    b = await _body(request)
    return _run("remove_from_album", {"album_id": album_id, **_pick(b, "photo_ids", "dry_run", "confirm_token")}, request)


@router.post("/api/albums/{album_id}/delete")
async def album_delete(album_id: int, request: Request):
    b = await _body(request)
    return _run("delete_album", {"album_id": album_id, **_pick(b, "dry_run", "confirm_token")}, request)


# ---------------------------------------------------------------- trash

@router.get("/api/trash")
def trash_list(request: Request, limit: int = 60, offset: int = 0):
    return _run("list_trash", {"limit": limit, "offset": offset}, request)


@router.post("/api/trash")
async def trash(request: Request):
    b = await _body(request)
    return _run("trash_photos", _pick(b, "photo_ids", "reason", "dry_run", "confirm_token"), request)


@router.post("/api/trash/restore")
async def trash_restore(request: Request):
    b = await _body(request)
    return _run("restore_photos", _pick(b, "photo_ids"), request)


@router.post("/api/trash/purge")
async def trash_purge(request: Request):
    b = await _body(request)
    return _run("purge_trash", _pick(b, "older_than_days", "dry_run", "confirm_token"), request)


# ---------------------------------------------------------------- cleanup & details

@router.get("/api/cleanup")
def cleanup_candidates(request: Request, kind: str | None = None, limit: int = 20, offset: int = 0,
                       min_confidence: float = 0.0):
    args = {"limit": limit, "offset": offset, "min_confidence": min_confidence}
    if kind:
        args["kind"] = kind
    return _run("find_cleanup_candidates", args, request)


@router.get("/api/photos/{pid}/details")
def photo_details(pid: int, request: Request):
    return _run("explain_photo", {"photo_id": pid}, request)


@router.get("/api/audit")
def audit(limit: int = 50):
    _ready()
    return {"entries": tools.audit_log(max(1, min(500, limit)))}
