"""Extra demo sources for the XL index (data/index-xl).  usage: python experiments/fetch_xl.py [places] [coco]

Resumable (curl -C -) with retries; refuses to start a download that would leave < MIN_FREE_GB on disk.
Archives go to data/downloads/<name>/, extracted photos to data/<name>/, archive deleted after a good extract.

- places : Places365-Standard validation, 256x256 (36,500 photos, 365 scene classes). MIT CSAIL.
           Labels: data/places365/places365_val.txt + categories_places365.txt.
- coco   : COCO val2017 (5,000 photos). images.cocodataset.org. Shrunk afterwards to 512px max side (saves ~0.5GB).
- eval   : build experiments/eval_set_xl.json (see build_eval) from the xl index; eval_set.json is not touched.
- coco_captions : only captions_val2017.json out of the 253MB annotations zip, via HTTP range requests (eval ground truth).
"""
import os, shutil, subprocess, sys, tarfile, time, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MIN_FREE_GB = 12

SOURCES = {
    "places": [("places365", "http://data.csail.mit.edu/places/places365/val_256.tar", 525158400),
               ("places365", "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar", 67498496)],
    "coco": [("coco_val2017", "http://images.cocodataset.org/zips/val2017.zip", 815585330)],
}


def free_gb():
    return shutil.disk_usage(os.path.expanduser("~")).free / 1e9


def download(url, dest, size):
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have == size:
        return
    need = (size - have) * 2 / 1e9  # archive + extracted copy
    if free_gb() - need < MIN_FREE_GB:
        sys.exit(f"refusing {url}: {free_gb():.1f}GB free, needs ~{need:.1f}GB, floor {MIN_FREE_GB}GB")
    for attempt in range(30):
        r = subprocess.run(["curl", "-L", "-sS", "-C", "-", "--retry", "5", "--retry-delay", "5",
                            "--connect-timeout", "30", "--speed-limit", "20000", "--speed-time", "60", "-o", dest, url])
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        print(f"{os.path.basename(dest)}: {have/1e6:.0f}/{size/1e6:.0f} MB (curl rc={r.returncode})", flush=True)
        if have == size:
            return
        if have > size:  # corrupt resume; start over
            os.remove(dest)
        time.sleep(min(60, 5 * (attempt + 1)))
    sys.exit(f"gave up on {url}")


def safe_members(names, out):
    for n in names:
        p = os.path.realpath(os.path.join(out, n))
        if not p.startswith(os.path.realpath(out) + os.sep) and p != os.path.realpath(out):
            raise RuntimeError(f"unsafe path in archive: {n}")


def extract(arc, out):
    os.makedirs(out, exist_ok=True)
    if arc.endswith(".zip"):
        with zipfile.ZipFile(arc) as z:
            safe_members(z.namelist(), out); z.extractall(out)
    else:
        with tarfile.open(arc) as t:
            ms = [m for m in t.getmembers() if m.isfile() or m.isdir()]  # no links/devices
            safe_members([m.name for m in ms], out); t.extractall(out, members=ms)


class HttpFile:
    """Seekable read-only file over HTTP range requests, so zipfile can pull one member out of a remote zip."""
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        import urllib.request
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{min(self.size, self.pos + n) - 1}"})
                data = urllib.request.urlopen(req, timeout=60).read(); break
            except Exception:
                if attempt == 7: raise
                time.sleep(5 * (attempt + 1))
        self.pos += len(data); return data


def shrink(folder, m=512):
    from PIL import Image
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if not f.lower().endswith(".jpg"): continue
        im = Image.open(p)
        if max(im.size) <= m: continue
        exif = im.info.get("exif")
        im = im.convert("RGB"); im.thumbnail((m, m), Image.LANCZOS)
        im.save(p + ".tmp", "JPEG", quality=85, **({"exif": exif} if exif else {})); os.replace(p + ".tmp", p)


SCENES = [  # (query, Places365 categories). ~100 val photos each; other albums can hold unlabelled matches.
    ("bowling alley", ["bowling_alley"]),
    ("laundromat with washing machines", ["laundromat"]),
    ("server room", ["server_room"]),
    ("hospital operating room", ["operating_room"]),
    ("subway station platform", ["subway_station/platform"]),
    ("indoor swimming pool", ["swimming_pool/indoor"]),
    ("kitchen", ["kitchen"]),
    ("living room", ["living_room"]),
    ("kid's bedroom", ["childs_room"]),
    ("classroom", ["classroom"]),
    ("inside a library", ["library/indoor"]),
    ("attic", ["attic"]),
    ("locker room", ["locker_room"]),
]


def build_eval(index_dir=f"{ROOT}/index-xl", seed=0):
    """eval_set_xl.json = eval_set.json's queries (ids are unchanged: index-xl started as a copy of data/index) with
    ground truth extended to the new photos by the same rules as make_eval.py, plus Places365 scene queries and
    Oxford same-building queries."""
    import ast, csv, json, random, re, sqlite3
    from datetime import datetime
    here = os.path.dirname(os.path.abspath(__file__))
    base = json.load(open(f"{here}/eval_set.json"))
    lists = {}  # the literal query lists in make_eval.py, read without executing it
    for node in ast.parse(open(f"{here}/make_eval.py").read()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("OBJECTS", "PLACES", "PLACE_OBJECTS", "TIMES", "TEXTS"):
            lists[node.targets[0].id] = {t[0]: t[1:] for t in ast.literal_eval(node.value)}
    db = sqlite3.connect(f"file:{index_dir}/library.sqlite?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    photos = {r["id"]: dict(r) for r in db.execute("SELECT id, path, album, place, taken, ocr FROM photos")}
    old_max = base["meta"]["photos"]  # ids 1..21389 came from data/index; ground truth for them is already final
    assert max(i for i in photos if photos[i]["album"] in ("flickr", "hotel-rooms")) <= old_max
    new = {i: p for i, p in photos.items() if i > old_max}
    by_file = {(p["album"], os.path.basename(p["path"])): i for i, p in photos.items()}

    desc = {}  # free text per new photo, like make_eval.py: Unsplash descriptions, COCO captions, Places labels
    csv.field_size_limit(10**9)
    for r in csv.DictReader(open(f"{ROOT}/demo/photos.tsv000"), delimiter="\t"):
        i = by_file.get(("unsplash", r["photo_id"] + ".jpg"))
        if i and i > old_max:  # data/demo/unsplash_more/, registered as album "unsplash"
            desc[i] = f"{r['photo_description']} | {r['ai_description']}".lower()
    cap = json.load(open(f"{ROOT}/coco_val2017/captions_val2017.json"))
    fn = {im["id"]: im["file_name"] for im in cap["images"]}
    for a in cap["annotations"]:
        i = by_file.get(("coco", fn[a["image_id"]]))
        if i:
            desc[i] = desc.get(i, "") + " | " + a["caption"].lower()
    cats = [l.split()[0][3:] for l in open(f"{ROOT}/places365/categories_places365.txt")]
    label = {}
    for l in open(f"{ROOT}/places365/places365_val.txt"):
        f, c = l.split()
        i = by_file.get(("places365", f))
        if i:
            label[i] = cats[int(c)]
            desc[i] = cats[int(c)].replace("_", " ").replace("/", " ")

    def match(i, pats=(), place=None, after=None, before=None):
        p = new[i]
        if pats and not (i in desc and all(re.search(x, desc[i]) for x in pats)):
            return False
        if place and not (p["place"] and re.search(place, p["place"].lower())):
            return False
        if after and not (p["taken"] and p["taken"] >= datetime.fromisoformat(after).timestamp()):
            return False
        if before and not (p["taken"] and p["taken"] < datetime.fromisoformat(before).timestamp()):
            return False
        return True

    queries, added = [], {}
    for q in base["queries"]:
        q = dict(q); t, extra = q["type"], []
        if t == "object/scene":
            (pats,) = lists["OBJECTS"][q["query"]]; extra = [i for i in new if match(i, pats)]
        elif t == "place":
            (pl,) = lists["PLACES"][q["query"]]; extra = [i for i in new if match(i, place=pl)]
        elif t == "place+object":
            pl, pats = lists["PLACE_OBJECTS"][q["query"]]; extra = [i for i in new if match(i, pats, place=pl)]
        elif t == "time":
            a, b, pats = lists["TIMES"][q["query"]]; extra = [i for i in new if match(i, pats, after=a, before=b)]
        elif t == "text_in_photo":
            (sub,) = lists["TEXTS"][q["query"]]; extra = [i for i in new if new[i]["ocr"] and sub in new[i]["ocr"].lower()]
        if extra:
            q["relevant"] = sorted(set(q["relevant"]) | set(extra)); added[q["id"]] = len(extra)
            q["notes"] += f"; +{len(extra)} from the xl sources"
        queries.append(q)

    for n, (query, cs) in enumerate(SCENES, 1):
        rel = sorted(i for i, c in label.items() if c in cs)
        queries.append({"id": f"scene-{n:02d}", "type": "scene", "query": query, "relevant": rel,
                        "notes": f"Places365 val label in {cs}; photos of this scene in other albums are unlabelled (incomplete ground truth)"})

    rox = f"{ROOT}/roxford/gt.json"  # query photo -> positives (easy+hard); data/roxford was deleted mid-run (disk)
    gt = json.load(open(rox)) if os.path.exists(rox) else {}
    rng, seen = random.Random(seed), set()
    for qname in rng.sample(sorted(gt), len(gt)):
        building = qname.rsplit("_", 1)[0]
        img = by_file.get(("oxford-buildings", qname + ".jpg"))
        rel = sorted(by_file[("oxford-buildings", x + ".jpg")] for x in gt[qname]["pos"] if ("oxford-buildings", x + ".jpg") in by_file)
        if building in seen or img is None or not rel:
            continue
        seen.add(building)
        queries.append({"id": f"same_building-{len(seen):02d}", "type": "same_building", "query": "other photos of this building",
                        "image_id": img, "relevant": rel, "notes": f"Revisited Oxford query {qname} ({building}); relevant = easy+hard positives"})
        if len(seen) == 8:
            break

    albums = dict(db.execute("SELECT album, count(*) FROM photos GROUP BY album").fetchall())
    meta = {"built": datetime.now().isoformat(timespec="seconds"), "seed": seed, "photos": len(photos), "albums": albums,
            "ocr_done": db.execute("SELECT count(*) FROM photos WHERE ocr_done=1").fetchone()[0], "base": base["meta"],
            "extended": added,
            "note": "index: PHOTOSEARCH_INDEX=data/index-xl. Base queries from eval_set.json (same ids), ground truth extended to "
                    "new photos via Unsplash descriptions / COCO captions / Places365 labels / OCR; still incomplete."}
    out = f"{here}/eval_set_xl.json"
    json.dump({"meta": meta, "queries": queries}, open(out, "w"), indent=1)
    import collections
    print(f"{len(queries)} queries -> {out}", dict(collections.Counter(q["type"] for q in queries)), "extended:", added)


if __name__ == "__main__":
    if "eval" in sys.argv[1:]:
        build_eval()
    if "coco_captions" in sys.argv[1:]:
        z = zipfile.ZipFile(HttpFile("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", 252907541))
        open(f"{ROOT}/coco_val2017/captions_val2017.json", "wb").write(z.read("annotations/captions_val2017.json"))
        print("got captions_val2017.json")
    for key in [k for k in sys.argv[1:] if k in SOURCES] or ([] if sys.argv[1:] else ["places"]):
        for name, url, size in SOURCES[key]:
            dl = f"{ROOT}/downloads/{name}"
            os.makedirs(dl, exist_ok=True)
            arc = f"{dl}/{os.path.basename(url)}"
            mark = f"{ROOT}/{name}/.done_{os.path.basename(url)}"
            if os.path.exists(mark):
                print("already have", url); continue
            print(f"{free_gb():.1f}GB free; fetching {url}", flush=True)
            download(url, arc, size)
            extract(arc, f"{ROOT}/{name}")
            if key == "coco":
                shrink(f"{ROOT}/{name}/val2017")
            open(mark, "w").close()
            os.remove(arc)
            print(f"extracted {name}; {free_gb():.1f}GB free", flush=True)
