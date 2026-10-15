"""Keep ML jobs from taking the machine down (a 16GB Mac kernel-panicked when several model jobs ran at once).

- `acquire(w)` machine-wide budget of SLOTS units: a process holding models takes w units (search server = 1,
               big model / training = 2) and others wait, so two heavy jobs can never run at once.
- `check()`    called between batches: pause while swap is high / disk or memory is low; abort if it doesn't recover.
"""
from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import sys
import time

SLOTS = int(os.environ.get("GUARD_SLOTS", 2))  # batch-job budget: one heavy job (2) or two light ones; at 4, swap hit 15 GB
SERVER_SLOT = "/tmp/photosearch-ml.server"  # reserved for the user-facing server: it never queues behind batch jobs
SLOT_PATH = "/tmp/photosearch-ml.slot{}"
MAX_SWAP_GB = float(os.environ.get("GUARD_MAX_SWAP_GB", 16))  # swap stays allocated after pressure passes; disk + free memory are the real signals
MIN_DISK_GB = float(os.environ.get("GUARD_MIN_DISK_GB", 6.5))  # swap lives on this disk; below ~5 GB macOS can freeze
MIN_MEM_FREE_PCT = float(os.environ.get("GUARD_MIN_MEM_FREE_PCT", 12))

_held: list = []
_server = None
SERVER_ROLE = False  # set True only by `python server.py` on port 8000 (the user's UI server)
_last = 0.0


class ResourceError(RuntimeError):
    pass


def acquire(weight: int = 1, wait_s: float = 3 * 3600):
    """Hold `weight` units of the machine-wide ML budget (all-or-nothing; re-entrant, can grow; released at exit)."""
    global _server
    if SERVER_ROLE and _server is None:
        f = open(SERVER_SLOT, "a+")
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:  # another user-facing server is up
                time.sleep(5)
        f.seek(0); f.truncate(); f.write(f"pid {os.getpid()}: {' '.join(sys.argv)[:100]}"); f.flush()
        _server = f
        weight -= 1  # the server's own model (SigLIP) rides on the reserved slot
    need = min(weight, SLOTS) - len(_held)
    if need <= 0:
        return
    t0, said = time.time(), False
    # First come, first served: only the holder of the queue ticket may grab slots, so a waiting 2-unit job can't be
    # starved by 1-unit jobs that keep restarting and snatching whichever slot frees first.
    q = open(SLOT_PATH.format("-queue"), "a+")
    try:
        while True:
            try:
                fcntl.flock(q, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 > wait_s:
                    raise ResourceError(f"ML budget busy for {wait_s:.0f}s; refusing to overload the machine")
                if not said:
                    print(f"[guard] queued for {need} ML slot(s): busy with {holders()}", file=sys.stderr, flush=True)
                    said = True
                time.sleep(2)
        while True:
            got = []
            for i in range(SLOTS):
                if len(got) == need:
                    break
                f = open(SLOT_PATH.format(i), "a+")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    got.append(f)
                except BlockingIOError:
                    f.close()
            if len(got) == need:
                break
            for f in got:  # never sit on a partial grab: two jobs each holding half would deadlock
                fcntl.flock(f, fcntl.LOCK_UN); f.close()
            if time.time() - t0 > wait_s:
                raise ResourceError(f"ML budget busy for {wait_s:.0f}s; refusing to overload the machine")
            if not said:
                print(f"[guard] waiting for {need} ML slot(s): busy with {holders()}", file=sys.stderr, flush=True)
                said = True
            time.sleep(5)
    finally:
        fcntl.flock(q, fcntl.LOCK_UN); q.close()
    for f in got:
        f.seek(0); f.truncate(); f.write(f"pid {os.getpid()}: {' '.join(sys.argv)[:100]}"); f.flush()
    _held.extend(got)
    check(force=True)


def holders() -> list[str]:
    out = []
    for path in [SERVER_SLOT] + [SLOT_PATH.format(i) for i in range(SLOTS)]:
        try:
            with open(path) as fh:
                t = fh.read().strip()
            f = open(path, "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(f, fcntl.LOCK_UN)
            except BlockingIOError:
                out.append(t or "?")
            f.close()
        except FileNotFoundError:
            pass
    return out


def status() -> dict:
    swap = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
    mem = re.search(r"(\d+)%", subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout)
    return {"swap_gb": float(swap.group(1)) / 1024 if swap else 0.0,
            "disk_free_gb": shutil.disk_usage(os.path.expanduser("~")).free / 1e9,
            "mem_free_pct": float(mem.group(1)) if mem else 100.0}


def _problems(s):
    out = []
    if s["swap_gb"] > MAX_SWAP_GB:
        out.append(f"swap {s['swap_gb']:.1f}GB > {MAX_SWAP_GB}")
    if s["disk_free_gb"] < MIN_DISK_GB:
        out.append(f"disk free {s['disk_free_gb']:.1f}GB < {MIN_DISK_GB}")
    if s["mem_free_pct"] < MIN_MEM_FREE_PCT:
        out.append(f"memory free {s['mem_free_pct']:.0f}% < {MIN_MEM_FREE_PCT}")
    return out


def check(force: bool = False, every_s: float = 10, max_pause_s: float = 300):
    """Cheap to call often (rate-limited). Pauses up to max_pause_s for pressure to clear, then raises."""
    global _last
    if not force and time.time() - _last < every_s:
        return
    _last = time.time()
    t0 = time.time()
    while bad := _problems(status()):
        if time.time() - t0 > max_pause_s:
            raise ResourceError("aborting to protect the machine: " + "; ".join(bad))
        print("[guard] pausing: " + "; ".join(bad), file=sys.stderr, flush=True)
        time.sleep(15)
