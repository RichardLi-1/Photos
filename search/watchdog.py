"""Last-resort machine watchdog: if memory/swap/disk go critical, stop the newest guarded ML job (never the :8000 server).

usage: .venv/bin/python watchdog.py &   (pairs with photosearch/guard.py; see docs/development.md)
"""
import os, signal, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photosearch import guard  # noqa: E402

LOG = os.environ.get("WATCHDOG_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "watchdog.log"))
CRIT = {"swap_gb": 18.0, "mem_free_pct": 8.0, "disk_free_gb": 5.0}  # swap is sticky on macOS; watch memory + disk


def log(m):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + m + "\n")


def server_pid():
    out = subprocess.run(["lsof", "-nP", "-t", "-iTCP:8000", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def guarded_pids():
    pids = set()
    for h in guard.holders():
        if h.startswith("pid "):
            pids.add(int(h.split()[1].rstrip(":")))
    return pids


log("watchdog started")
while True:
    s = guard.status()
    bad = (s["swap_gb"] > CRIT["swap_gb"] or s["mem_free_pct"] < CRIT["mem_free_pct"] or s["disk_free_gb"] < CRIT["disk_free_gb"])
    if bad:
        keep = server_pid()
        victims = sorted(guarded_pids() - {keep, os.getpid()}, reverse=True)  # newest pid first
        if victims:
            v = victims[0]
            log(f"CRITICAL {s} -> SIGTERM pid {v}")
            try:
                os.kill(v, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(30)
            continue
        log(f"CRITICAL {s} but no guarded job to stop")
    time.sleep(20)
