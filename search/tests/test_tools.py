"""photosearch.tools: the agent's safety rails. Soft delete only, dry-run first, confirm tokens, caps, audit log."""
from __future__ import annotations

import os
import sqlite3

import pytest

tools = pytest.importorskip("photosearch.tools")

from photosearch.library import Library  # noqa: E402
from photosearch.search import Query, Searcher  # noqa: E402


@pytest.fixture
def T(fresh_index, monkeypatch):
    S = Searcher(Library(fresh_index["dir"], readonly=True))
    monkeypatch.setattr(tools, "_ctx", None)
    tools.init(S, None, fresh_index["dir"])
    yield tools, fresh_index, S
    tools._ctx.db.close()


def _db(idx):
    db = sqlite3.connect(f"{idx['dir']}/library.sqlite")
    db.row_factory = sqlite3.Row
    return db


def _audit(idx):
    return [dict(r) for r in _db(idx).execute("SELECT * FROM audit_log ORDER BY id")]


def test_trash_defaults_to_dry_run(T):
    t, idx, _ = T
    pid = idx["ids"]["trip_1"]
    out = t.trash_photos([pid], "blurry")
    assert out["dry_run"] is True and out["would_trash"] == 1 and out["confirm_token"]
    assert _db(idx).execute("SELECT deleted_at FROM photos WHERE id=?", (pid,)).fetchone()[0] is None


def test_real_trash_needs_matching_single_use_token(T):
    t, idx, _ = T
    a, b = idx["ids"]["trip_1"], idx["ids"]["trip_2"]
    with pytest.raises(t.ToolError, match="confirm_token"):
        t.trash_photos([a], "blurry", dry_run=False)                         # no token
    tok = t.trash_photos([a], "blurry")["confirm_token"]
    with pytest.raises(t.ToolError, match="does not match"):
        t.trash_photos([a, b], "blurry", dry_run=False, confirm_token=tok)  # different photos
    with pytest.raises(t.ToolError, match="does not match"):
        t.trash_photos([a], "dupes", dry_run=False, confirm_token=tok)      # different reason
    out = t.trash_photos([a], "blurry", dry_run=False, confirm_token=tok)
    assert out["trashed"] == 1 and out["dry_run"] is False
    tok2 = t.trash_photos([b], "blurry")["confirm_token"]
    t.trash_photos([b], "blurry", dry_run=False, confirm_token=tok2)
    with pytest.raises(t.ToolError):
        t.restore_photos([b]); t.trash_photos([b], "blurry", dry_run=False, confirm_token=tok2)  # replay refused


def test_token_expires(T, monkeypatch):
    t, idx, _ = T
    pid = idx["ids"]["home_1"]
    tok = t.trash_photos([pid], "old")["confirm_token"]
    now = t._now()
    monkeypatch.setattr(t, "_now", lambda: now + t.PREVIEW_TTL_S + 1)
    with pytest.raises(t.ToolError, match="expired"):
        t.trash_photos([pid], "old", dry_run=False, confirm_token=tok)


def test_trash_is_soft_and_restorable(T):
    t, idx, S = T
    pid = idx["ids"]["misc_1"]
    tok = t.trash_photos([pid], "receipt")["confirm_token"]
    t.trash_photos([pid], "receipt", dry_run=False, confirm_token=tok)
    assert os.path.isfile(idx["paths"]["misc_1"])                          # the file is never touched
    assert pid not in [r["id"] for r in S.search(Query(text="menu", k=20))]  # hidden from search
    assert t.list_trash()["total"] == 1
    assert t.restore_photos([pid])["restored"] == 1
    assert pid in [r["id"] for r in S.search(Query(text="menu", k=20))]


def test_purge_marks_only(T, monkeypatch):
    t, idx, _ = T
    pid = idx["ids"]["misc_2"]
    tok = t.trash_photos([pid], "dupe")["confirm_token"]
    t.trash_photos([pid], "dupe", dry_run=False, confirm_token=tok)
    p = t.purge_trash(older_than_days=0)
    assert p["dry_run"] is True and p["would_purge"] == 1
    t.purge_trash(older_than_days=0, dry_run=False, confirm_token=p["confirm_token"])
    assert os.path.isfile(idx["paths"]["misc_2"])
    assert t.restore_photos([pid])["restored"] == 0                          # purged: no longer restorable
    with pytest.raises(t.ToolError):
        t.purge_trash(older_than_days=-1)


def test_caps_and_validation(T):
    t, _, _ = T
    with pytest.raises(t.ToolError, match="cap"):
        t.trash_photos(list(range(1, t.MAX_TRASH + 2)), "x")
    with pytest.raises(t.ToolError, match="reason"):
        t.trash_photos([1], "   ")
    with pytest.raises(t.ToolError):
        t.trash_photos(["1; DROP TABLE photos"], "x")
    with pytest.raises(t.ToolError):
        t.trash_photos([], "x")


def test_delete_album_keeps_photos(T):
    t, idx, _ = T
    a = t.create_album("Best of", [idx["ids"]["trip_1"], idx["ids"]["trip_2"]])
    aid = a["album"]["id"] if "album" in a else a["id"]
    d = t.delete_album(aid)
    assert d["dry_run"] is True and d["photos_in_album"] == 2
    t.delete_album(aid, dry_run=False, confirm_token=d["confirm_token"])
    assert _db(idx).execute("SELECT count(*) FROM photos").fetchone()[0] == 10


def test_call_never_raises_and_audits_everything(T):
    t, idx, _ = T
    pid = idx["ids"]["trip_3"]
    assert t.call("nope")["ok"] is False
    assert t.call("trash_photos", {"photo_ids": [pid]})["ok"] is False                           # missing reason
    assert t.call("trash_photos", {"photo_ids": [pid], "reason": "x", "evil": 1})["ok"] is False  # unknown arg
    dry = t.call("trash_photos", {"photo_ids": [pid], "reason": "blurry"})
    assert dry["ok"] and dry["dry_run"]
    bad = t.call("trash_photos", {"photo_ids": [pid], "reason": "blurry", "dry_run": False})
    assert bad["ok"] is False
    ok = t.call("trash_photos", {"photo_ids": [pid], "reason": "blurry", "dry_run": False, "confirm_token": dry["confirm_token"]})
    assert ok["ok"] and ok["trashed"] == 1
    log = [r for r in _audit(idx) if r["tool"] == "trash_photos"]
    assert [r["ok"] for r in log] == [0, 0, 1, 0, 1]                  # failures are logged too
    assert [r["dry_run"] for r in log][2:] == [1, 0, 0]               # (rejected calls log dry_run=0: review L)
    assert all(r["actor"] == "agent" for r in log)


def test_destructive_tools_are_flagged(T):
    t, _, _ = T
    hints = {s["name"]: s["annotations"] for s in t.schemas(annotations=True)}
    for name in t.DESTRUCTIVE:
        assert hints[name]["destructiveHint"] is True and hints[name]["readOnlyHint"] is False


def test_every_destructive_tool_defaults_to_dry_run(T):
    import inspect
    t, _, _ = T
    for name in t.DESTRUCTIVE:
        assert inspect.signature(getattr(t, name)).parameters["dry_run"].default is True, name


def test_direct_calls_are_audited(T):
    t, idx, _ = T
    pid = idx["ids"]["home_3"]
    tok = t.trash_photos([pid], "x")["confirm_token"]
    t.trash_photos([pid], "x", dry_run=False, confirm_token=tok)
    assert any(r["tool"] == "trash_photos" and r["dry_run"] == 0 for r in _audit(idx))
