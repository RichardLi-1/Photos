"""Build experiments/eval_set.json from metadata ground truth (never from embeddings).  usage: python experiments/make_eval.py"""
import collections, csv, json, os, random, re, sqlite3, time
from datetime import datetime

import numpy as np

D = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "eval_set.json")
csv.field_size_limit(10**9)
rng = random.Random(0)

db = sqlite3.connect(f"file:{D}/index/library.sqlite?mode=ro", uri=True, timeout=60)
db.row_factory = sqlite3.Row
photos = {r["id"]: dict(r) for r in db.execute("SELECT id, path, album, place, taken, ocr, ocr_done FROM photos")}
by_file = {(p["album"], os.path.basename(p["path"])): i for i, p in photos.items()}

# Free text per photo: Unsplash human + AI descriptions, Flickr's 5 captions. Hotels have none.
desc = {}
for r in csv.DictReader(open(f"{D}/demo/photos.tsv000"), delimiter="\t"):
    i = by_file.get(("unsplash", r["photo_id"] + ".jpg"))
    if i:
        desc[i] = f"{r['photo_description']} | {r['ai_description']}".lower()
flickr = list(csv.DictReader(open(f"{D}/flickr1k/captions.csv")))
for r in flickr:
    desc[by_file[("flickr", r["filename"])]] = " | ".join(json.loads(r["raw"])).lower()


def where(desc_pats=(), place=None, after=None, before=None):
    """Ids satisfying every condition. desc_pats: regexes that must ALL match the description."""
    out = []
    for i, p in photos.items():
        if desc_pats and not (i in desc and all(re.search(x, desc[i]) for x in desc_pats)):
            continue
        if place and not (p["place"] and re.search(place, p["place"].lower())):
            continue
        if after and not (p["taken"] and p["taken"] >= datetime.fromisoformat(after).timestamp()):
            continue
        if before and not (p["taken"] and p["taken"] < datetime.fromisoformat(before).timestamp()):
            continue
        out.append(i)
    return sorted(out)


OBJECTS = [  # (query, regexes that must all match the description)
    ("lighthouse", [r"\blighthouses?\b"]),
    ("hot air balloons", [r"hot.air.balloons?"]),
    ("giraffe", [r"\bgiraffes?\b"]),
    ("show me the elephants", [r"\belephants?\b"]),
    ("a rainbow in the sky", [r"\brainbows?\b"]),
    ("penguins", [r"\bpenguins?\b"]),
    ("close-up of a butterfly", [r"\bbutterfl(y|ies)\b"]),
    ("volcano", [r"\bvolcano(es)?\b|\blava\b|erupt"]),
    ("fireworks at night", [r"\bfireworks?\b"]),
    ("someone paddling a kayak or canoe", [r"\b(kayak|canoe)"]),
    ("a fox", [r"\bfox(es)?\b"]),
    ("a cat sleeping", [r"\b(cats?|kittens?)\b", r"\b(sleep|asleep|nap|napping|resting)"]),
    ("dog playing in the snow", [r"\b(dogs?|puppy|puppies)\b", r"\bsnow"]),
    ("surfer riding a wave", [r"\bsurf(er|ers|ing|board)\b", r"\bwaves?\b"]),
    ("tent under the stars", [r"\btents?\b", r"\bstar(s|ry)\b|milky way|galaxy|night"]),
]
PLACES = [  # (query, regex on the place name; aliases included so ground truth is complete)
    ("photos from Iceland", r"iceland|ísland"),
    ("my trip to Japan", r"japan|日本"),
    ("pictures I took in London", r"\blondon\b"),
    ("New York City", r"new york|\bnyc\b|manhattan|brooklyn"),
    ("San Francisco pictures", r"san francisco"),
    ("Yosemite", r"yosemite"),
    ("Maldives vacation photos", r"maldives"),
    ("Bali", r"\bbali\b"),
    ("photos from Antarctica", r"antarctica"),
    ("pictures from my Switzerland trip", r"switzerland|schweiz|suisse|svizzera"),
    ("Paris", r"\bparis\b"),
    ("Lofoten islands", r"lofoten"),
    ("photos I took in Germany", r"germany|deutschland"),
]
PLACE_OBJECTS = [  # (query, place regex, description regexes)
    ("waterfalls in Iceland", r"iceland|ísland", [r"waterfall"]),
    ("northern lights in Iceland", r"iceland|ísland", [r"aurora|northern lights"]),
    ("beach photos from Australia", r"australia", [r"\bbeach"]),
    ("snowy mountains in Switzerland", r"switzerland|schweiz|suisse|svizzera", [r"\bmountains?\b", r"\bsnow"]),
    ("penguins in Antarctica", r"antarctica", [r"\bpenguins?\b"]),
    ("buildings in New York", r"new york|\bnyc\b|manhattan|brooklyn", [r"\bbuildings?\b|skyscraper"]),
    ("Yosemite waterfall", r"yosemite", [r"waterfall"]),
    ("beaches in Bali", r"\bbali\b", [r"\bbeach"]),
    ("lakes in Canada", r"canada", [r"\blakes?\b"]),
    ("northern lights in Norway", r"norway|norge|lofoten", [r"aurora|northern lights"]),
    ("mountains in New Zealand", r"new zealand", [r"\bmountains?\b"]),
    ("flowers in Germany", r"germany|deutschland", [r"\bflowers?\b"]),
]
TIMES = [  # (query, after, before, description regexes). Unsplash upload dates stand in for capture dates.
    ("photos from summer 2017", "2017-06-01", "2017-09-01", []),
    ("pictures from December 2018", "2018-12-01", "2019-01-01", []),
    ("what did I photograph in March 2016", "2016-03-01", "2016-04-01", []),
    ("photos from 2014", "2014-01-01", "2015-01-01", []),
    ("pictures from the first week of May 2020", "2020-05-01", "2020-05-08", []),
    ("photos from 2021", "2021-01-01", "2022-01-01", []),
    ("snow photos from winter 2017", "2016-12-01", "2017-03-01", [r"\bsnow"]),
    ("beach photos from summer 2019", "2019-06-01", "2019-09-01", [r"\bbeach"]),
    ("mountain pictures from autumn 2018", "2018-09-01", "2018-12-01", [r"\bmountains?\b"]),
    ("sunsets I shot in 2020", "2020-01-01", "2021-01-01", [r"\bsunset"]),
    ("flowers from spring 2019", "2019-03-01", "2019-06-01", [r"\bflowers?\b"]),
    ("dog photos from 2020", "2020-01-01", "2021-01-01", [r"\b(dogs?|puppy|puppies)\b"]),
]
TEXTS = [  # (query, substring of the OCR'd text, lowercased). OCR is Vision "fast" mode: mostly garbled; legible words
          # almost never repeat across photos, so most of these have 1 relevant photo (not the ideal 2-20).
    ("the sign warning about rattlesnakes", "rattlesnake"),
    ("parking sign that says violators will be towed", "towed"),
    ("coin laundry sign", "coin laun"),
    ("caution sign about avalanches", "caution"),
    ("a page from the Gospel of Matthew", "gospel"),
    ("extinction rebellion protest sign", "rebellion"),
    ("climate protest sign saying our planet is getting hotter", "planet"),
    ("snack sticks package", "snack sticks"),
    ("Marlboro cigarette ad", "marlboro"),
    ("Tissot watch sign", "tissot"),
    ("sign saying no child left behind", "child left"),
    ("sign saying 9-11 was an inside job", "inside job"),
    ("gelati tabacchi shop sign", "gelati"),
    ("framed print of Lookout Point, Hot Springs, signed Beaman", "beaman"),
    ("do not disturb sign", "do not d"),
]

queries = []


def add(typ, query, relevant, notes, image_id=None, lo=1, hi=None):
    if not (lo <= len(relevant) and (hi is None or len(relevant) <= hi)):
        print(f"  skip {typ} {query!r}: {len(relevant)} relevant")
        return
    q = {"id": f"{typ}-{sum(x['type'] == typ for x in queries) + 1:02d}", "type": typ, "query": query}
    if image_id is not None:
        q["image_id"] = image_id
    queries.append({**q, "relevant": relevant, "notes": notes})


for query, pats in OBJECTS:
    add("object/scene", query, where(pats), f"description (Unsplash photo/ai_description, Flickr captions) ~ {pats}; incomplete ground truth", lo=5, hi=80)

for i in rng.sample(range(len(flickr)), 12):
    caps = json.loads(flickr[i]["raw"])
    add("caption", rng.choice(caps), [by_file[("flickr", flickr[i]["filename"])]], f"Flickr30k caption of {flickr[i]['filename']}")

for query, pl in PLACES:
    add("place", query, where(place=pl), f"place ~ /{pl}/", lo=5)

for query, pl, pats in PLACE_OBJECTS:
    add("place+object", query, where(pats, place=pl), f"place ~ /{pl}/ and description ~ {pats}", lo=3)

gt = json.load(open(f"{D}/hotels/gt.json"))
hotel_of = {by_file[("hotel-rooms", f)]: h for f, h in gt.items() if ("hotel-rooms", f) in by_file}
rooms = collections.defaultdict(list)
for i, h in hotel_of.items():
    rooms[h].append(i)
for h in rng.sample(sorted(h for h, v in rooms.items() if len(v) >= 3), 12):
    img = rng.choice(sorted(rooms[h]))
    add("same_room", "other photos of this hotel room", sorted(set(rooms[h]) - {img}), f"hotel {h}", image_id=img)

for query, a, b, pats in TIMES:
    add("time", query, where(pats, after=a, before=b), f"taken in [{a}, {b})" + (f" and description ~ {pats}" if pats else ""), lo=3)

for query, s in TEXTS:
    add("text_in_photo", query, sorted(i for i, p in photos.items() if p["ocr"] and s in p["ocr"].lower()), f"OCR contains {s!r}")

for _ in range(5):  # the indexer rewrites the npz while we read it
    try:
        emb = set(np.load(f"{D}/index/vectors__siglip2-b16-384.npz")["ids"].tolist()); break
    except Exception:
        time.sleep(2)
ocr_done = sum(p["ocr_done"] == 1 for p in photos.values())
meta = {"built": datetime.now().isoformat(timespec="seconds"), "seed": 0, "photos": len(photos),
        "embedded_at_build": len(emb), "ocr_done": ocr_done,
        "note": "relevant = metadata ground truth over all photos; eval_search.py reports embedding coverage at run time"}
json.dump({"meta": meta, "queries": queries}, open(OUT, "w"), indent=1)
print(f"{len(queries)} queries -> {OUT}  (embedded {len(emb)}/{len(photos)}, ocr {ocr_done}/{len(photos)})")
for t, n in collections.Counter(q["type"] for q in queries).items():
    rel = [len(q["relevant"]) for q in queries if q["type"] == t]
    print(f"  {t:14s} {n:3d} queries, relevant per query median {int(np.median(rel))} (min {min(rel)}, max {max(rel)})")
