"""photosearch.query.parse: search-box text -> (visual text, Filters with dates / place). Today is pinned."""
from __future__ import annotations

from datetime import date

import pytest

query = pytest.importorskip("photosearch.query")
places_mod = pytest.importorskip("photosearch.places")

TODAY = date(2026, 10, 15)  # a Thursday


@pytest.fixture(scope="module")
def P():
    return places_mod.Places(["Sydney, Australia", "Paris, France", "Erlangen, Deutschland"])


def parse(text, P=None):
    return query.parse(text, P, today=TODAY)


@pytest.mark.parametrize("text", ["red car", "menu", "dog on a beach", "a sign that says hello", ""])
def test_nothing_parsed_returns_text_unchanged(text, P):
    vis, f = parse(text, P)
    assert vis == text and not f.active()


@pytest.mark.parametrize("text,vis,after,before", [
    ("flowers in Germany last summer", "flowers", "2026-06-01", "2026-09-01"),   # module docstring example
    ("winter 2017 snow", "snow", "2016-12-01", "2017-03-01"),                   # winter spans the new year
    ("photos from 2023", "", "2023-01-01", "2024-01-01"),
    ("dogs yesterday", "dogs", "2026-10-14", "2026-10-15"),
    ("beach in july 2019", "beach", "2019-07-01", "2019-08-01"),
])
def test_dates(text, vis, after, before, P):
    v, f = parse(text, P)
    assert (v, f.after, f.before) == (vis, after, before)


def test_bare_month_means_every_year(P):
    v, f = parse("beach in March", P)
    assert v == "beach" and f.after is None and f.before is None
    assert ("2025-03-01", "2025-04-01") in f.dates and ("2019-03-01", "2019-04-01") in f.dates
    assert all(a[5:7] == "03" for a, _ in f.dates)


@pytest.mark.parametrize("text,vis,place", [
    ("sunset in Paris", "sunset", "Paris"),
    ("flowers in Germany", "flowers", "Germany"),
    ("beach Sydney", "beach", "Sydney"),
])
def test_places(text, vis, place, P):
    v, f = parse(text, P)
    assert v == vis and f.place == place


def test_place_and_date_together(P):
    v, f = parse("sunset in Paris last summer", P)
    assert v == "sunset" and f.place == "Paris" and (f.after, f.before) == ("2026-06-01", "2026-09-01")


def test_works_without_library_places():
    v, f = parse("sunset in Paris")
    assert v == "sunset" and f.place == "Paris"


@pytest.mark.parametrize("text", ["2023-13-45", "in the year 99999", "'; DROP TABLE photos; --", "\x00\x01", "🙂" * 50,
                                  "from to between and", "last last last summer", "christmas " * 50])
def test_garbage_never_raises(text, P):
    v, f = parse(text, P)
    assert isinstance(v, str)
    for a in (f.after, f.before):
        assert a is None or date.fromisoformat(a)


def test_parse_is_fast_on_long_input(P):
    import time
    t0 = time.perf_counter()
    parse("flowers in Paris last summer " * 100, P)
    assert time.perf_counter() - t0 < 2.0


def test_search_uses_parser(ro_lib, shared_index):
    """Query(parse=True): place/date words become hard filters, the rest is ranked visually."""
    from photosearch.search import Query, Searcher
    S = Searcher(ro_lib)
    by_id = {v: k for k, v in shared_index["ids"].items()}
    q = Query(text="red in Paris", parse=True, k=10)
    got = {by_id[r["id"]] for r in S.search(q)}
    assert got == {"home_1", "home_2", "home_3"}
    assert q.parsed["place"] == "Paris" and q.parsed["text"] == "red"
    q = Query(text="photos from 2023", parse=True, k=10)                  # nothing visual left: by date
    assert {by_id[r["id"]] for r in S.search(q)} == {f"trip_{i}" for i in range(1, 5)}
