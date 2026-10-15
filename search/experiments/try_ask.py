"""Run the Ask pipeline on a fixed set of questions and print what came back.  usage: python experiments/try_ask.py [N]"""
import json, os, sys, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from index import INDEX  # noqa: E402
from photosearch.ask import Asker  # noqa: E402
from photosearch.library import Library  # noqa: E402
from photosearch.search import Searcher  # noqa: E402

lib = Library(INDEX)
s = Searcher(lib)
a = Asker(s)
gt = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "hotels", "gt.json")))
room = lib.db.execute("SELECT id, path FROM photos WHERE album='hotel-rooms' ORDER BY id LIMIT 1 OFFSET 40").fetchone()
room_hotel = gt[os.path.basename(room["path"])]

QUESTIONS = [
    ("show me all the other pictures of this room", room["id"]),
    ("waterfalls in Iceland", None),
    ("photos of a dog at the beach", None),
    ("what did the menu say coffee costs?", None),
    ("how many photos have a hot air balloon?", None),
    ("pictures from Japan at night", None),
    ("someone surfing a big wave", None),
    ("a red car parked on a street", None),
]

for q, img in QUESTIONS[: int(sys.argv[1]) if len(sys.argv) > 1 else None]:
    t = time.time()
    r = a.ask(q, image=img)
    top = r["results"][:8]
    print(f"\n### {q}  ({time.time() - t:.0f}s, {r['timing']})")
    print("plan:", json.dumps({k: v for k, v in r["plan"].items() if v}))
    print("answer:", r["answer"])
    for x in top:
        tag = ""
        if img is not None:
            tag = "SAME-HOTEL" if gt.get(os.path.basename(x["path"])) == room_hotel else ""
        print(f"  {'✓' if x['verified'] else ' '} {x['id']:6d} {x['album']:11s} {(x['place'] or '')[:28]:28s} "
              f"{(x['ocr'] or '').replace(chr(10), ' ')[:40]:40s} {tag}")
