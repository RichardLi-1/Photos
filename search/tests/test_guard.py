"""photosearch.guard: machine-wide ML slot budget (flock files) and the resource-pressure thresholds.

Slot tests run real separate processes against a private slot directory, never the real /tmp/photosearch-ml.slot*.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from conftest import REAL_CHECK, SEARCH_DIR
from photosearch import guard

HOLDER = textwrap.dedent("""
    import sys
    sys.path.insert(0, {search!r})
    from photosearch import guard
    guard.SLOT_PATH, guard.SLOTS = {slot!r}, {slots}
    guard.check = lambda *a, **k: None
    mode, w = sys.argv[1], int(sys.argv[2])
    try:
        guard.acquire(w, wait_s=0)
    except guard.ResourceError:
        print("busy", flush=True); sys.exit(3)
    print("got", len(guard._held), flush=True)
    if mode == "hold":
        sys.stdin.read()  # hold until the parent closes stdin
""")


class Slots:
    def __init__(self, tmp_path, n=3):
        self.path, self.n = str(tmp_path / "slot{}"), n
        self.script = HOLDER.format(search=SEARCH_DIR, slot=self.path, slots=n)
        self.procs = []

    def hold(self, w):
        p = subprocess.Popen([sys.executable, "-c", self.script, "hold", str(w)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
        self.procs.append(p)
        assert p.stdout.readline().strip() == f"got {w}"
        return p

    def try_once(self, w) -> str:
        r = subprocess.run([sys.executable, "-c", self.script, "once", str(w)], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()

    def release(self, p):
        p.stdin.close(); p.wait(timeout=10)

    def close(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill(); p.wait()


@pytest.fixture
def slots(tmp_path):
    s = Slots(tmp_path)
    yield s
    s.close()


def test_acquire_in_other_process_blocks_until_released(slots):
    a = slots.hold(2)
    assert slots.try_once(2) == "busy"       # only 1 of 3 free
    assert slots.try_once(1) == "got 1"
    slots.release(a)                          # lock dies with the process
    assert slots.try_once(3) == "got 3"


def test_all_or_nothing_never_keeps_a_partial_grab(slots):
    a = slots.hold(2)
    assert slots.try_once(3) == "busy"
    # the failed 3-unit attempt must not have kept the one free slot: a 1-unit job still fits
    b = slots.hold(1)
    assert slots.try_once(1) == "busy"
    slots.release(a); slots.release(b)


def test_crashed_holder_releases(slots):
    a = slots.hold(3)
    a.kill(); a.wait()
    assert slots.try_once(3) == "got 3"


def test_weight_is_clamped_to_slots(slots):
    assert slots.try_once(99) == f"got {slots.n}"


def test_in_process_reentrant_and_grows(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "SLOTS", 3)
    guard.acquire(1, wait_s=0)
    assert len(guard._held) == 1
    guard.acquire(1, wait_s=0)            # re-entrant: no extra slot
    assert len(guard._held) == 1
    guard.acquire(2, wait_s=0)            # grows to 2
    assert len(guard._held) == 2
    held = guard.holders()
    assert len(held) == 2 and all(h.startswith(f"pid {os.getpid()}:") for h in held)


def test_holders_empty_when_free(tmp_path, monkeypatch):
    assert guard.holders() == []


def test_holder_label_visible_to_other_processes(slots, monkeypatch):
    a = slots.hold(1)
    monkeypatch.setattr(guard, "SLOT_PATH", slots.path)
    monkeypatch.setattr(guard, "SLOTS", slots.n)
    held = guard.holders()
    assert held == [f"pid {a.pid}: -c hold 1"]
    slots.release(a)
    assert guard.holders() == []


# ---------------------------------------------------------------- thresholds

OK = {"swap_gb": 1.0, "disk_free_gb": 100.0, "mem_free_pct": 50.0}


def test_problems_none_when_healthy():
    assert guard._problems(OK) == []


@pytest.mark.parametrize("key,bad,word", [
    ("swap_gb", guard.MAX_SWAP_GB + 0.1, "swap"),
    ("disk_free_gb", guard.MIN_DISK_GB - 0.1, "disk"),
    ("mem_free_pct", guard.MIN_MEM_FREE_PCT - 1, "memory"),
])
def test_problems_each_threshold(key, bad, word):
    out = guard._problems({**OK, key: bad})
    assert len(out) == 1 and out[0].startswith(word)


@pytest.mark.parametrize("key,edge", [
    ("swap_gb", guard.MAX_SWAP_GB), ("disk_free_gb", guard.MIN_DISK_GB), ("mem_free_pct", guard.MIN_MEM_FREE_PCT)])
def test_problems_thresholds_are_strict(key, edge):
    assert guard._problems({**OK, key: edge}) == []


def test_problems_reports_all():
    assert len(guard._problems({"swap_gb": 99, "disk_free_gb": 0, "mem_free_pct": 0})) == 3


def test_thresholds_env_overridable():
    code = f"import sys; sys.path.insert(0, {SEARCH_DIR!r}); from photosearch import guard; print(guard.SLOTS, guard.MAX_SWAP_GB, guard.MIN_DISK_GB, guard.MIN_MEM_FREE_PCT)"
    env = {**os.environ, "GUARD_SLOTS": "7", "GUARD_MAX_SWAP_GB": "3", "GUARD_MIN_DISK_GB": "2", "GUARD_MIN_MEM_FREE_PCT": "5"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30).stdout.split()
    assert out == ["7", "3.0", "2.0", "5.0"]


# ---------------------------------------------------------------- check(): pause / abort (real function, fake status)

@pytest.fixture
def real_check(monkeypatch):
    """The real check() (the sandbox stubs it out); it reads guard.status / guard.time through module globals."""
    sleeps = []
    monkeypatch.setattr(guard.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(guard, "check", REAL_CHECK)
    return guard, sleeps


def test_check_raises_when_pressure_persists(real_check, monkeypatch):
    g, _ = real_check
    monkeypatch.setattr(g, "status", lambda: {**OK, "swap_gb": 999})
    with pytest.raises(g.ResourceError, match="swap"):
        g.check(force=True, max_pause_s=-1)


def test_check_pauses_then_recovers(real_check, monkeypatch):
    g, sleeps = real_check
    seq = iter([{**OK, "mem_free_pct": 1}, {**OK, "mem_free_pct": 1}, OK])
    monkeypatch.setattr(g, "status", lambda: next(seq))
    g.check(force=True, max_pause_s=3600)
    assert len(sleeps) == 2


def test_check_is_rate_limited(real_check, monkeypatch):
    g, _ = real_check
    calls = []
    monkeypatch.setattr(g, "status", lambda: calls.append(1) or OK)
    g.check(force=True)
    g.check()                 # within every_s: skipped
    g.check(every_s=0)        # not skipped
    assert len(calls) == 2
