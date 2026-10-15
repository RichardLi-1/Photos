"""HTTP routes for the mock photo agent (mock_agent.py). JSON only, local only (server.py's middleware guards every route).

  POST /api/agent/message {text, session_id?}
       -> {session_id, reply, steps: [{tool, description, status}],
           results?: {photo_ids, view, groups?, albums?, album?, total?},
           pending?: {id, summary, count, photo_ids, groups?: [{kind, keep, remove, confidence, reason}]},
           undo?: {id, label}}
  POST /api/agent/confirm {session_id, pending_id}   runs the previewed destructive calls with their confirm tokens
  POST /api/agent/cancel  {session_id, pending_id}   drops the preview; nothing changes
  POST /api/agent/undo    {session_id, undo_id}      reverts what that response did (restore, delete album, ...)

step.status: done | pending (waiting for /confirm) | skipped | failed | cancelled | partial.
Showing an album's photos: GET /api/albums/{id} (api_tools.py).

Swap the planner (see mock_agent.py's docstring):  api_agent.EXECUTOR.planner = MyPlanner()
"""
from __future__ import annotations

import sys

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

import mock_agent
from photosearch import tools

router = APIRouter()
EXECUTOR = mock_agent.Executor()
_state = {"error": None}


def init(S, lock):
    """api_tools normally initialises photosearch.tools first (same server); do it here only if it hasn't."""
    if tools._ctx is not None:
        return
    try:
        tools.init(S, lock, S.lib.dir)
    except Exception as e:  # never take the search server down
        _state["error"] = f"{type(e).__name__}: {e}"
        print(f"[api_agent] disabled: {_state['error']}", file=sys.stderr, flush=True)


async def _body(request: Request) -> dict:
    if not (request.headers.get("content-type") or "").startswith("application/json"):
        raise HTTPException(415, "send a JSON object (Content-Type: application/json)")
    try:
        b = await request.json()
    except Exception:
        raise HTTPException(400, "body is not valid JSON")
    if not isinstance(b, dict):
        raise HTTPException(400, "body must be a JSON object")
    return b


async def _run(fn, *args):
    if tools._ctx is None:
        raise HTTPException(503, f"photo agent unavailable: {_state['error'] or 'tools not initialised'}")
    try:
        return await run_in_threadpool(fn, *args)
    except mock_agent.AgentError as e:
        raise HTTPException(e.status, str(e))


def _str(b: dict, k: str, required: bool = True) -> str | None:
    v = b.get(k)
    if not required and (v is None or v == ""):
        return None
    if not isinstance(v, str) or not v or len(v) > 64:
        raise HTTPException(400, f"{k} required (string)")
    return v


@router.post("/api/agent/message")
async def agent_message(request: Request):
    b = await _body(request)
    return await _run(EXECUTOR.message, b.get("text"), _str(b, "session_id", required=False))


@router.post("/api/agent/confirm")
async def agent_confirm(request: Request):
    b = await _body(request)
    return await _run(EXECUTOR.confirm, _str(b, "session_id"), _str(b, "pending_id"))


@router.post("/api/agent/cancel")
async def agent_cancel(request: Request):
    b = await _body(request)
    return await _run(EXECUTOR.cancel, _str(b, "session_id"), _str(b, "pending_id"))


@router.post("/api/agent/undo")
async def agent_undo(request: Request):
    b = await _body(request)
    return await _run(EXECUTOR.undo, _str(b, "session_id"), _str(b, "undo_id"))
