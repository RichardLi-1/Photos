"""Check the rule-based query parser (no models).  usage: python experiments/eval_parse.py

Cases are written by hand from how people type into a photo search box, not taken from eval_set.json.
Each: query -> (visual text, {filter: value}); today is pinned to 2026-10-15.
"""
import os, sqlite3, sys, time
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from index import INDEX  # noqa: E402
from photosearch.places import Places  # noqa: E402
from photosearch.query import parse  # noqa: E402

TODAY = date(2026, 10, 15)
CASES = [
    ("sept 29 2025", "", dict(after="2025-09-29", before="2025-09-30")),
    ("9/29/2025", "", dict(after="2025-09-29", before="2025-09-30")),
    ("29.09.2025", "", dict(after="2025-09-29", before="2025-09-30")),
    ("in 2019", "", dict(after="2019-01-01", before="2020-01-01")),
    ("summer 2018", "", dict(after="2018-06-01", before="2018-09-01")),
    ("last year", "", dict(after="2025-01-01", before="2026-01-01")),
    ("christmas 2020", "", dict(after="2020-12-24", before="2020-12-27")),
    ("from March to May 2021", "", dict(after="2021-03-01", before="2021-06-01")),
    ("from december to february 2018", "", dict(after="2017-12-01", before="2018-03-01")),
    ("between 2015 and 2017", "", dict(after="2015-01-01", before="2018-01-01")),
    ("hiking since 2020", "hiking", dict(after="2020-01-01")),
    ("last summer", "", dict(after="2026-06-01", before="2026-09-01")),
    ("3 weeks ago", "", dict(after="2026-09-21", before="2026-09-28")),
    ("last weekend", "", dict(after="2026-10-10", before="2026-10-12")),
    ("easter 2021", "", dict(after="2021-04-02", before="2021-04-06")),
    ("thanksgiving 2022", "", dict(after="2022-11-24", before="2022-11-28")),
    ("the 90s", "", dict(after="1990-01-01", before="2000-01-01")),
    ("summer photos from 2019", "", dict(after="2019-06-01", before="2019-09-01")),
    ("kids on halloween", "kids", dict(dates=True)),  # every Halloween
    ("flowers in Germany", "flowers", dict(place="Germany")),
    ("Deutschland", "", dict(place="Germany")),
    ("Schweiz", "", dict(place="Switzerland")),
    ("日本の写真", "", dict(place="Japan")),
    ("at the beach in Bali", "beach", dict(place="Bali")),
    ("coffee in Paris", "coffee", dict(place="Paris")),
    ("Paris, France", "", dict(place="Paris")),
    ("photos from the UK", "", dict(place="United Kingdom")),
    ("LA at night", "night", dict(place="Los Angeles")),
    ("in New Zealand last year", "", dict(place="New Zealand", after="2025-01-01", before="2026-01-01")),
    ("dog in the snow in Colorado in december 2020", "dog in the snow", dict(place="Colorado", after="2020-12-01", before="2021-01-01")),
    # must NOT become filters
    ("a cat sleeping", "a cat sleeping", {}),
    ("hotel", "hotel", {}),
    ("us at the beach", "us at the beach", {}),
    ("turkey dinner", "turkey dinner", {}),
    ("may flowers", "may flowers", {}),
    ("spring flowers", "spring flowers", {}),
    ("march of the penguins", "march of the penguins", {}),
    ("Finishing a summer time treat, a popsicle.", "Finishing a summer time treat, a popsicle.", {}),
    ("sign saying 9-11 was an inside job", "sign saying 9-11 was an inside job", {}),
    ("framed print of Lookout Point, Hot Springs", "framed print of Lookout Point, Hot Springs", {}),
    ("woman in her 20s", "woman in her 20s", {}),
]

if __name__ == "__main__":
    db = sqlite3.connect(f"file:{INDEX}/library.sqlite?mode=ro", uri=True)
    P = Places([r[0] for r in db.execute("SELECT DISTINCT place FROM photos WHERE place IS NOT NULL")])
    bad, worst = 0, 0.0
    for q, vis, want in CASES:
        t = time.perf_counter()
        v, f = parse(q, P, TODAY)
        worst = max(worst, time.perf_counter() - t)
        got = {k: x for k, x in vars(f).items() if x not in (None, False)}
        if "dates" in got and want.get("dates") is True:
            got["dates"] = True
        ok = v == vis and got == want
        bad += not ok
        print(("ok  " if ok else "FAIL") + f"  {q!r:50} -> {v!r:22} {got}" + ("" if ok else f"   want {vis!r} {want}"))
    print(f"\n{len(CASES) - bad}/{len(CASES)} ok, slowest parse {worst * 1000:.1f} ms")
    sys.exit(bad > 0)
