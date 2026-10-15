"""Download N Unsplash Lite photos (512px) + metadata sidecar into data/demo/unsplash/."""
import csv, json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
D = os.path.join(os.path.dirname(__file__), "..", "data", "demo")
OUT = f"{D}/unsplash"
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10**9)
rows = list(csv.DictReader(open(f"{D}/photos.tsv000"), delimiter="\t"))
random.Random(0).shuffle(rows)
rows = rows[:N]

def meta(r):
    f = lambda k: (r.get(k) or "").strip() or None
    lat, lon = f("photo_location_latitude"), f("photo_location_longitude")
    return {"date": f("photo_submitted_at"), "description": f("photo_description") or f("ai_description"),
            "place": ", ".join(dict.fromkeys(t.strip() for x in [f("photo_location_name"), f("photo_location_city"), f("photo_location_country")] if x for t in x.split(",") if t.strip())) or None,
            "lat": float(lat) if lat else None, "lon": float(lon) if lon else None,
            "camera": " ".join(x for x in [f("exif_camera_make"), f("exif_camera_model")] if x) or None}

def get(r):
    p = f"{OUT}/{r['photo_id']}.jpg"
    if os.path.exists(p):
        return 1
    for attempt in range(5):
        try:
            with urllib.request.urlopen(r["photo_image_url"] + "?w=512&q=70&fm=jpg", timeout=30) as resp:
                data = resp.read()
            open(p, "wb").write(data)
            return 1
        except Exception:
            time.sleep(2 * (attempt + 1))
    return 0

json.dump({r["photo_id"] + ".jpg": meta(r) for r in rows}, open(f"{OUT}/_meta.json", "w"))
ok = 0
with ThreadPoolExecutor(24) as ex:
    for i, k in enumerate(ex.map(get, rows)):
        ok += k
        if i % 1000 == 0:
            print(i, ok, flush=True)
print("done", ok, "/", len(rows))
