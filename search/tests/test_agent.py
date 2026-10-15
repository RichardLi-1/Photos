"""mock_agent: RulePlanner intents + the Executor's dry run -> confirm -> undo guarantees, on a temp index (fake encoder)."""
from __future__ import annotations

import sqlite3

import pytest

tools = pytest.importorskip("photosearch.tools")

import mock_agent as M  # noqa: E402
from photosearch import cleanup  # noqa: E402
from photosearch.library import Library  # noqa: E402
from photosearch.search import Searcher  # noqa: E402

ORIGIN = "http://127.0.0.1:8000"


def _fake_candidates(ids):
    c = {k: [] for k in cleanup.KINDS}
    t1, t2, t3, h1, h2, m3 = (ids[n] for n in ("trip_1", "trip_2", "trip_3", "home_1", "home_2", "misc_3"))
    c["exact_duplicate"] = [{"kind": "exact_duplicate", "photo_ids": [t1, t2, t3], "keep": t1, "remove": [t2, t3],
                             "confidence": 0.99, "reasons": ["3 files with identical bytes"]}]
    c["near_duplicate"] = [{"kind": "near_duplicate", "photo_ids": [h1, h2], "keep": h1, "remove": [h2],
                            "confidence": 0.8, "reasons": ["burst"]}]
    c["blurry"] = [{"kind": "blurry", "photo_ids": [m3], "confidence": 0.9, "reasons": ["soft edges"]}]
    return c


@pytest.fixture
def T(fresh_index, monkeypatch):
    S = Searcher(Library(fresh_index["dir"], readonly=True))
    monkeypatch.setattr(tools, "_ctx", None)
    tools.init(S, None, fresh_index["dir"])
    cands = _fake_candidates(fresh_index["ids"])
    monkeypatch.setattr(tools, "_candidates", lambda: cands)
    yield M.Executor(), fresh_index
    tools._ctx.db.close()


def _db(idx):
    db = sqlite3.connect(f"{idx['dir']}/library.sqlite")
    db.row_factory = sqlite3.Row
    return db


def _trashed(idx):
    return {r[0] for r in _db(idx).execute("SELECT id FROM photos WHERE deleted_at IS NOT NULL")}


def _albums(idx):
    return {r["name"]: r["id"] for r in _db(idx).execute("SELECT id, name FROM albums")}


# ---------------------------------------------------------------- planner intents (no index needed)

class FakeCtx(M.Context):
    def __init__(self, last=(), albums=()):
        al = [{"id": i + 1, "name": n, "count": 3, "cover": None} for i, n in enumerate(albums)]
        summary = {k: 0 for k in cleanup.KINDS} | {"exact_duplicate": 36, "near_duplicate": 124, "blurry": 5}
        canned = {"list_albums": {"ok": True, "albums": al},
                  "find_cleanup_candidates": {"ok": True, "summary": summary, "total": 36, "candidates": []}}
        super().__init__("s", [], list(last), lambda tool, args=None, d=None: canned[tool])


@pytest.mark.parametrize("text,intent,first_tool", [
    ("make an album of my beach photos", "create_album", "search_photos"),
    ("album: Kyoto 2024", "create_album", "search_photos"),
    ("album: Japan trip", "add_to_album", "search_photos"),  # exists: add instead of a clash
    ("Create a new album called Paris with photos of the eiffel tower", "create_album", "search_photos"),
    ("put my sunset photos in a new album", "create_album", "search_photos"),
    ("add these to Japan trip", "add_to_album", "add_to_album"),
    ("clean up duplicates", "cleanup", "find_cleanup_candidates"),
    ("clean up near-duplicates", "cleanup", "find_cleanup_candidates"),
    ("get rid of blurry photos", "cleanup", "find_cleanup_candidates"),
    ("delete my screenshots", "cleanup", "find_cleanup_candidates"),
    ("trash the blurry ones", "cleanup", "find_cleanup_candidates"),
    ("delete photos of receipts", "cleanup", "find_cleanup_candidates"),
    ("find photos I probably don't need", "cleanup", "find_cleanup_candidates"),
    ("tidy up the whiteboards", "cleanup", "find_cleanup_candidates"),
    ("delete photos of my old car", "trash", "search_photos"),
    ("delete these", "trash", "trash_photos"),
    ("move these to the trash", "trash", "trash_photos"),
    ("restore", "restore", "list_trash"),
    ("bring them back", "restore", "restore_photos"),
    ("undo", "undo", None),
    ("show my albums", "show_albums", "list_albums"),
    ("show trash", "show_trash", "list_trash"),
    ("Recently Deleted", "show_trash", "list_trash"),
    ("open album Japan trip", "show_album", "get_album"),
    ("find dogs in snow", "find", "search_photos"),
    ("show me blurry photos", "review", "find_cleanup_candidates"),
    ("delete the album Japan trip", "delete_album", "get_album"),
    ("remove these from Japan trip", "remove_from_album", "remove_from_album"),
    ("rename Japan trip to Tokyo 2024", "rename_album", "rename_album"),
    ("what can you do?", "help", None),
    ("empty the trash", "cant", None),
    ("share my beach photos with mom", "cant", None),
])
def test_rule_planner_intents(text, intent, first_tool):
    p = M.RulePlanner().plan(text, FakeCtx(last=[1, 2, 3], albums=["Japan trip"]))
    assert p.intent == intent, (text, p.intent)
    assert (p.steps[0].tool if p.steps else None) == first_tool
    for s in p.steps:
        assert s.tool in tools.REGISTRY and s.description


def test_planner_parses_names_and_queries():
    P, ctx = M.RulePlanner(), FakeCtx()
    p = P.plan("make an album of my beach photos", ctx)
    assert p.steps[0].args["query"] == "beach" and p.steps[1].description == "Made the album “Beach”"
    p = P.plan("Create an album called Paris with photos of the eiffel tower", ctx)
    assert p.steps[0].args["query"] == "eiffel tower" and "“Paris”" in p.steps[1].description
    p = P.plan("album: Japan trip", ctx)
    assert p.steps[0].args["query"] == "Japan trip" and "“Japan trip”" in p.steps[1].description
    assert P.plan("clean up near duplicates", ctx).steps[0].args["kind"] == "near_duplicate"
    assert P.plan("photos I probably don't need", ctx).steps[0].args["kind"] == "exact_duplicate"  # safest kind first
    assert P.plan("delete photos of my old car", ctx).steps[0].args["query"] == "old car"


def test_planner_without_context_asks_instead_of_guessing():
    p = M.RulePlanner().plan("delete these", FakeCtx(last=[]))
    assert not p.steps and "Which photos" in p.reply


# ---------------------------------------------------------------- executor: dry run -> confirm -> undo

def test_cleanup_duplicates_preview_confirm_undo(T):
    ex, idx = T
    ids = idx["ids"]
    r = ex.message("clean up duplicates")
    p = r["pending"]
    assert set(p["photo_ids"]) == {ids["trip_2"], ids["trip_3"]} and p["count"] == 2
    assert p["groups"] == [{"kind": "exact_duplicate", "keep": ids["trip_1"], "remove": [ids["trip_2"], ids["trip_3"]],
                            "confidence": 0.99, "reason": "3 files with identical bytes"}]
    assert "Recently Deleted" in p["summary"] and [s["status"] for s in r["steps"]] == ["done", "pending"]
    assert _trashed(idx) == set()                                 # a preview changes nothing
    assert "undo" not in r

    c = ex.confirm(r["session_id"], p["id"])
    assert _trashed(idx) == {ids["trip_2"], ids["trip_3"]}
    assert c["reply"] == "Done: 2 photos moved to Recently Deleted." and c["steps"][-1]["status"] == "done"
    assert c["undo"]["label"] == "Restore 2 photos"
    with pytest.raises(M.AgentError) as e:                         # single use
        ex.confirm(r["session_id"], p["id"])
    assert e.value.status == 404

    u = ex.undo(r["session_id"], c["undo"]["id"])
    assert _trashed(idx) == set() and u["reply"].startswith("Undone: 2 photos restored")
    with pytest.raises(M.AgentError):
        ex.undo(r["session_id"], c["undo"]["id"])


def test_cancel_changes_nothing(T):
    ex, idx = T
    r = ex.message("trash the blurry ones")
    assert r["pending"]["photo_ids"] == [idx["ids"]["misc_3"]]
    c = ex.cancel(r["session_id"], r["pending"]["id"])
    assert c["reply"].startswith("Cancelled") and c["steps"][-1]["status"] == "cancelled"
    assert _trashed(idx) == set()
    with pytest.raises(M.AgentError):
        ex.confirm(r["session_id"], r["pending"]["id"])


def test_none_found_suggests_near_duplicates(T):
    ex, idx = T
    r = ex.message("clean up screenshots")
    assert "pending" not in r and "didn’t find any screenshots" in r["reply"]
    assert "near-duplicates" in r["reply"] and _trashed(idx) == set()


def test_album_create_then_undo_deletes_it(T):
    ex, idx = T
    r = ex.message("make an album of my red photos")
    assert "Beach" not in r["reply"] and "Red" in _albums(idx)
    assert r["results"]["view"] == "album" and r["results"]["photo_ids"]
    assert all(s["status"] == "done" for s in r["steps"]) and "pending" not in r
    u = ex.undo(r["session_id"], r["undo"]["id"])                  # delete_album: dry run + auto-confirm inside
    assert "Red" not in _albums(idx) and "deleted" in u["reply"]


def test_add_to_existing_album_undo_removes_only_new(T):
    ex, idx = T
    ids = idx["ids"]
    tools.create_album("Trip", [ids["trip_1"]])
    s = ex.message("find red")["session_id"]
    r = ex.message("add these to trip", s)
    members = {x[0] for x in _db(idx).execute("SELECT photo_id FROM album_photos")}
    assert ids["trip_1"] in members and len(members) > 1
    ex.undo(s, r["undo"]["id"])
    assert {x[0] for x in _db(idx).execute("SELECT photo_id FROM album_photos")} == {ids["trip_1"]}


def test_delete_album_needs_confirm_and_undo_recreates(T):
    ex, idx = T
    ids = idx["ids"]
    tools.create_album("Japan trip", [ids["trip_1"], ids["trip_2"]])
    r = ex.message("delete the album Japan trip")
    assert "Japan trip" in _albums(idx) and "Delete the album “Japan trip”" in r["pending"]["summary"]
    c = ex.confirm(r["session_id"], r["pending"]["id"])
    assert "Japan trip" not in _albums(idx)
    ex.undo(r["session_id"], c["undo"]["id"])
    aid = _albums(idx)["Japan trip"]
    assert {x[0] for x in _db(idx).execute("SELECT photo_id FROM album_photos WHERE album_id=?", (aid,))} == \
        {ids["trip_1"], ids["trip_2"]}


def test_delete_these_uses_last_results_and_restore_undo_retrashes(T):
    ex, idx = T
    r = ex.message("find blue")
    shown = r["results"]["photo_ids"]
    r = ex.message("delete these", r["session_id"])
    assert r["pending"]["photo_ids"] == sorted(shown)
    ex.confirm(r["session_id"], r["pending"]["id"])
    assert _trashed(idx) == set(shown)
    r2 = ex.message("restore", r["session_id"])
    assert _trashed(idx) == set() and r2["reply"].startswith(f"Restored {len(shown)}")
    ex.message("undo", r["session_id"])                            # "undo" = latest undo: back to the trash
    assert _trashed(idx) == set(shown)
    reasons = {x[0] for x in _db(idx).execute("SELECT deleted_reason FROM photos WHERE deleted_at IS NOT NULL")}
    assert reasons == {"deleted by you via the agent"}             # original reason kept


def test_large_trash_is_chunked_under_tool_caps(T, monkeypatch):
    ex, idx = T
    monkeypatch.setitem(M.CHUNK, "trash_photos", 3)
    s = ex.message("find red")["session_id"]
    sess = ex._sessions[s]
    sess.last_photo_ids = list(idx["ids"].values())                # 10 photos -> 4 dry runs
    r = ex.message("delete these", s)
    assert r["pending"]["count"] == 10 and len(sess.pending[r["pending"]["id"]].calls) == 4
    c = ex.confirm(s, r["pending"]["id"])
    assert len(_trashed(idx)) == 10 and c["reply"] == "Done: 10 photos moved to Recently Deleted."
    ex.undo(s, c["undo"]["id"])
    assert _trashed(idx) == set()


def test_pending_expires(T):
    ex, idx = T
    now = [1000.0]
    ex.clock = lambda: now[0]
    r = ex.message("trash the blurry ones")
    now[0] += tools.PREVIEW_TTL_S + 1
    with pytest.raises(M.AgentError) as e:
        ex.confirm(r["session_id"], r["pending"]["id"])
    assert e.value.status == 410 and _trashed(idx) == set()


def test_sessions_expire(T):
    ex, _ = T
    now = [0.0]
    ex.clock = lambda: now[0]
    s = ex.message("help")["session_id"]
    now[0] += M.SESSION_TTL_S + 1
    with pytest.raises(M.AgentError):
        ex.undo(s, "u_x")
    assert ex.message("help", s)["session_id"] != s


class SneakyPlanner:
    """A planner (think: an LLM) that tries to skip the confirmation."""

    def plan(self, message, ctx):
        if message == "write via read":
            ctx.read("trash_photos", {"photo_ids": [1], "reason": "x", "dry_run": False})
        return M.Plan([M.Step("trash_photos", {"photo_ids": ctx.last_photo_ids or [1], "reason": "x", "dry_run": False,
                                               "confirm_token": "forged"}, "Trash it")], "trashing!")


def test_executor_enforces_dry_run_for_any_planner(T):
    ex, idx = T
    ex.planner = SneakyPlanner()
    r = ex.message("go")
    assert r["steps"][0]["status"] == "pending" and _trashed(idx) == set()
    with pytest.raises(M.AgentError) as e:
        ex.message("write via read")
    assert e.value.status == 400 and _trashed(idx) == set()
    audit = _db(idx).execute("SELECT count(*) FROM audit_log WHERE tool='trash_photos' AND dry_run=0").fetchone()[0]
    assert audit == 0


def test_message_validation(T):
    ex, _ = T
    for bad in ("", "   ", None, 5, "x" * (M.MAX_TEXT + 1)):
        with pytest.raises(M.AgentError):
            ex.message(bad)


# ---------------------------------------------------------------- HTTP

@pytest.fixture
def client(T):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api_agent
    app = FastAPI()
    app.include_router(api_agent.router)
    api_agent.EXECUTOR = T[0]
    with TestClient(app, base_url=ORIGIN) as c:
        yield c, T[1]


def test_http_flow(client):
    c, idx = client
    r = c.post("/api/agent/message", json={"text": "clean up duplicates"}).json()
    assert set(r) >= {"session_id", "reply", "steps", "pending", "results"}
    assert set(r["steps"][0]) == {"tool", "description", "status"}
    assert set(r["pending"]) == {"id", "summary", "count", "photo_ids", "groups"}
    sid = r["session_id"]
    d = c.post("/api/agent/confirm", json={"session_id": sid, "pending_id": r["pending"]["id"]}).json()
    assert len(_trashed(idx)) == 2 and d["undo"]["label"]
    assert c.post("/api/agent/confirm", json={"session_id": sid, "pending_id": r["pending"]["id"]}).status_code == 404
    u = c.post("/api/agent/undo", json={"session_id": sid, "undo_id": d["undo"]["id"]})
    assert u.status_code == 200 and _trashed(idx) == set()
    assert c.post("/api/agent/cancel", json={"session_id": "nope", "pending_id": "p"}).status_code == 404
    assert c.post("/api/agent/message", content=b"text=hi").status_code == 415
    assert c.post("/api/agent/message", json={"text": ""}).status_code == 400

