"""GET /api/ask/status: lets the web UI say what a pending /api/ask is waiting on (model load, or the ML budget queue).

Read-only and lock-free, so it answers while an ask holds the search lock.
"""
import os
import sys

from fastapi import APIRouter

from photosearch import guard

router = APIRouter()
ASK_WEIGHT = 3  # the units Asker._load() acquires (photosearch/ask.py)


def _asker():
    for name in ("__main__", "server"):  # `python server.py` or `uvicorn server:app`
        a = getattr(sys.modules.get(name), "_asker", None)
        if a and a[0] is not None:
            return a[0]
    return None


@router.get("/api/ask/status")
def ask_status():
    a = _asker()
    loaded = bool(a and getattr(a, "_m", None) is not None)
    me = f"pid {os.getpid()}:"
    others = [h for h in guard.holders() if not h.startswith(me)]
    need = min(ASK_WEIGHT, guard.SLOTS)  # job slots the VLM takes (the UI server's own model sits on its reserved slot)
    return {"loaded": loaded, "queued": not loaded and len(others) > guard.SLOTS - need}
