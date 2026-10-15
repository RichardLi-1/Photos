"""Fast vs accurate Vision OCR on the photos re-OCR'd by Library.ocr_accurate (old text in ocr_fast). No models.

usage: python experiments/eval_ocr.py   "legible" = at least 2 dictionary words of 4+ letters.
"""
import os, re, sqlite3, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from index import INDEX  # noqa: E402
from photosearch.places import is_common  # noqa: E402


def words(t):
    return [w for w in re.findall(r"[a-z]{4,}", (t or "").lower()) if is_common(w)]


if __name__ == "__main__":
    db = sqlite3.connect(f"file:{INDEX}/library.sqlite?mode=ro", uri=True)
    rows = db.execute("SELECT f.ocr, p.ocr FROM ocr_fast f JOIN photos p USING(id)").fetchall()
    leg = lambda t: len(words(t)) >= 2
    fast, acc = sum(leg(a) for a, _ in rows), sum(leg(b) for _, b in rows)
    gained, lost = sum(leg(b) and not leg(a) for a, b in rows), sum(leg(a) and not leg(b) for a, b in rows)
    print(f"{len(rows)} photos re-OCR'd: any text fast {sum(bool(a) for a, _ in rows)} -> accurate {sum(bool(b) for _, b in rows)}")
    print(f"legible: fast {fast} -> accurate {acc}  (gained {gained}, lost {lost})")
    print(f"dictionary words per photo: fast {sum(len(words(a)) for a, _ in rows) / max(len(rows), 1):.2f} -> "
          f"accurate {sum(len(words(b)) for _, b in rows) / max(len(rows), 1):.2f}")
