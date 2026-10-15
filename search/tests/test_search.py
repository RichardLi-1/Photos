"""photosearch.search: vector ranking, RRF fusion with full text, metadata filters, session propagation (fake encoder)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import PHOTOS, SYDNEY
from photosearch.search import Filters, Query, Searcher

TRIP = {f"trip_{i}" for i in range(1, 5)}
HOME = {"home_1", "home_2", "home_3"}


@pytest.fixture
def S(ro_lib):
    return Searcher(ro_lib)


@pytest.fixture
def name(shared_index):
    by_id = {v: k for k, v in shared_index["ids"].items()}
    return lambda rows: [by_id[r["id"]] for r in rows]


@pytest.fixture
def ids(shared_index):
    return shared_index["ids"]


# ---------------------------------------------------------------- basics

def test_refresh_shapes(S):
    assert S.X.dtype == np.float32 and S.X.shape == (len(PHOTOS), 16)
    assert len(S.session) == len(S.ids)
    assert (S.session >= -1).all() and (S.session == -1).sum() == 2      # misc_1/misc_2 undated


def test_empty_query(S):
    assert S.search(Query()) == []


def test_text_query_ranks_by_similarity(S, name):
    out = S.search(Query(text="red", k=4))
    assert set(name(out)) == TRIP
    scores = [o["score"] for o in out]
    assert scores == sorted(scores, reverse=True) and all(isinstance(s, float) for s in scores)


def test_k_limits_results(S):
    assert len(S.search(Query(text="blue", k=3))) == 3
    assert len(S.search(Query(text="blue", k=500))) == len(PHOTOS)


def test_pil_image_query(S, name):
    out = S.search(Query(image=Image.new("RGB", (8, 8), (0, 0, 255)), k=3))
    assert set(name(out)) == HOME


def test_image_by_id_excludes_itself(S, name, ids):
    out = S.search(Query(image=ids["home_1"], k=len(PHOTOS)))
    got = name(out)
    assert "home_1" not in got and len(got) == len(PHOTOS) - 1
    assert set(got[:2]) == {"home_2", "home_3"}


def test_unknown_image_id_raises(S):
    with pytest.raises(KeyError):
        S.search(Query(image=999999))


@pytest.mark.parametrize("tw,expect", [(0.95, TRIP), (0.05, {"home_2", "home_3"})])
def test_text_weight_steers_image_plus_text(S, name, ids, tw, expect):
    out = S.search(Query(text="red", image=ids["home_1"], text_weight=tw, k=len(expect)))
    assert set(name(out)) == expect


# ---------------------------------------------------------------- filters

@pytest.mark.parametrize("f,expect", [
    (Filters(), set(PHOTOS)),
    (Filters(album="home"), HOME),
    (Filters(album="nope"), set()),
    (Filters(after="2024-01-01"), HOME),
    (Filters(before="2024-01-01"), TRIP | {"misc_3"}),                   # undated photos never pass a date filter
    (Filters(after="2023-01-01", before="2023-12-31"), TRIP),
    (Filters(after="2024-01-10T09:20:00"), {"home_2", "home_3"}),          # after is inclusive
    (Filters(before="2024-01-10T09:20:00"), TRIP | {"misc_3", "home_1"}),  # before is exclusive
    (Filters(place="paris"), {"home_1", "home_2", "home_3"}),              # case-insensitive
    (Filters(place="Australia"), TRIP),
    (Filters(place="sydney"), TRIP),
    (Filters(place="Germany"), set()),
    (Filters(has_text=True), {"misc_1", "home_2"}),
    (Filters(near=(*SYDNEY, 5)), TRIP),
    (Filters(near=(48.85, 2.35, 50)), {"home_1", "home_2"}),
    (Filters(near=(0, 0, 100)), set()),
    (Filters(album="home", has_text=True), {"home_2"}),
])
def test_mask(S, shared_index, f, expect):
    by_id = {v: k for k, v in shared_index["ids"].items()}
    got = {by_id[int(i)] for i, ok in zip(S.ids, S.mask(f)) if ok}
    assert got == expect
    assert f.active() == (f != Filters())


def test_filters_apply_to_search(S, name):
    out = S.search(Query(text="red", filters=Filters(album="home"), k=10))
    assert set(name(out)) == HOME           # filtered rows never appear, however good the match
    assert S.search(Query(text="red", filters=Filters(album="nope"))) == []


def test_bad_date_raises(S):
    with pytest.raises(ValueError):
        S.mask(Filters(after="last tuesday"))


# ---------------------------------------------------------------- full text

def ft(S, text, **kw) -> list[int]:
    """fulltext() returns ids, or (ids, exact-phrase ids) in newer versions."""
    out = S.fulltext(text, **kw)
    return list(out[0] if isinstance(out, tuple) else out)


def test_fulltext_matches_ocr(S, ids):
    assert ft(S, "the lunch menu") == [ids["misc_1"]]
    assert set(ft(S, "louvre menu")) == {ids["misc_1"], ids["home_2"]}     # words are OR'ed


def test_fulltext_ignores_short_and_stop_words(S):
    assert ft(S, "a to of") == []
    assert ft(S, "") == []
    assert ft(S, "the photos") == []


@pytest.mark.parametrize("evil", [
    'menu" OR "x', "menu*", "NEAR(menu pasta)", "menu AND NOT pasta", "'); DROP TABLE photos;--",
    "ocr:menu", "{ocr place}: menu", "^menu", '"""', "\x00menu", "menu -pasta", "menu + pasta",
])
def test_fulltext_operators_are_inert(S, ids, evil):
    out = ft(S, evil)                                     # never raises, never treats input as FTS syntax
    assert set(out) <= {ids["misc_1"]}
    assert S.lib.db.execute("SELECT count(*) FROM photos").fetchone()[0] == len(PHOTOS)


def test_fulltext_limit(S, fresh_index):
    assert len(ft(S, "louvre menu", limit=1)) == 1


def _time_ft(S, n_words):
    import time
    t0 = time.perf_counter()
    ft(S, "lunch menu pasta " * (n_words // 3))
    return time.perf_counter() - t0


def test_fulltext_short_queries_are_fast(S):
    assert _time_ft(S, 60) < 0.5


def test_fulltext_long_query_is_bounded(S):
    assert _time_ft(S, 800) < 0.5


# ---------------------------------------------------------------- fusion

def test_k_is_a_prefix(S):
    for text in ("red", "menu", "louvre ticket", "blue"):
        full = [r["id"] for r in S.search(Query(text=text, k=len(PHOTOS)))]
        assert [r["id"] for r in S.search(Query(text=text, k=3))] == full[:3]
        assert len(set(full)) == len(full)


def test_literal_text_match_is_promoted(S, ids):
    # "menu" is visually meaningless to the fake encoder; only OCR can find it
    assert S.search(Query(text="menu", k=1))[0]["id"] == ids["misc_1"]
    assert S.search(Query(text="louvre", k=1))[0]["id"] == ids["home_2"]
    assert S.search(Query(text="louvre ticket", k=1))[0]["id"] == ids["home_2"]


def test_literal_match_beats_strong_visual_match(S, ids):
    # every trip photo is a near-perfect visual match for "red"; an OCR phrase hit still makes the top 5
    top = [r["id"] for r in S.search(Query(text="red lunch menu", k=5))]
    assert ids["misc_1"] in top


def test_literal_match_respects_filters(S, ids):
    out = S.search(Query(text="menu", filters=Filters(album="trip"), k=10))
    assert ids["misc_1"] not in [r["id"] for r in out]


# ---------------------------------------------------------------- session propagation

def test_session_propagate_unit(S):
    S.session = np.array([0, 0, 0, 1, 1, -1, -1, 2, 2, 2])[: len(S.ids)]
    sim = np.array([0.9, 0.1, 0.2, 0.5, 0.1, 0.95, 0.0, 0.61, 0.3, 0.9])[: len(S.ids)]
    out = S._session_propagate(sim, image=None)
    assert out[:3].tolist() == pytest.approx([0.9, 0.81, 0.81])     # inherit gain * best
    assert out[3:5].tolist() == [0.5, 0.1]                           # no strong match in session 1
    assert out[5:7].tolist() == [0.95, 0.0]                           # no session: untouched
    assert out[7:].tolist() == pytest.approx([0.81, 0.81, 0.9])      # best of the session, not the first
    assert sim[1] == 0.1                                              # input not mutated


def test_session_propagate_noop_without_sessions(S):
    S.session = np.full(len(S.ids), -1)
    sim = np.linspace(0, 1, len(S.ids))
    assert S._session_propagate(sim, None) is sim


def test_text_queries_skip_session_propagation(S, monkeypatch):
    called = []
    monkeypatch.setattr(S, "_session_propagate", lambda sim, img, **k: called.append(1) or sim)
    S.search(Query(text="red"))
    assert called == []
    S.search(Query(image=Image.new("RGB", (4, 4), "red")))
    assert called == [1]


def test_refresh_picks_up_new_vectors(fresh_index, tmp_path):
    from conftest import write_jpeg
    from photosearch.library import Library
    rw = Library(fresh_index["dir"])
    s = Searcher(Library(fresh_index["dir"], readonly=True))
    (tmp_path / "n").mkdir()
    write_jpeg(tmp_path / "n" / "new.jpg", (255, 0, 0))
    rw.add_folder(str(tmp_path / "n")); rw.embed_pending(log=lambda *_: None)
    assert len(s.ids) == len(PHOTOS)
    s.refresh()
    assert len(s.ids) == len(PHOTOS) + 1 and len(s.session) == len(s.ids) and s.X.shape[0] == len(s.ids)
