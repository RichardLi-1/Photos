"""Publishable demo photos from the YFCC100M (OpenAI subset) mirror on Hugging Face: CC BY 2.0 only.

Same outputs as fetch_pass.py, so index.py / demo/build_bundle.py work unchanged:
  images/<key>.webp, images/_meta.json (date, lat, lon, place), credits.csv (creator, license, Flickr page)

Each chunk = metadata/metadata_NNN.jsonl.gz + data/NNN.zip (about 19 photos, about 25% "Attribution License").
Chunks are streamed in memory; only thumbnails of CC BY photos are kept. Resumable (done chunks in state.json).

  python experiments/fetch_yfcc_hf.py --out data/yfcc --chunks 1-1000 --geonames
"""
import argparse, csv, gzip, io, json, os, sys, time, urllib.parse, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from fetch_pass import LICENSE, Geo, make_thumb, save_json  # noqa: E402  (same thumbnails, places, license)

BASE = "https://huggingface.co/datasets/dalle-mini/YFCC100M_OpenAI_subset/resolve/main"
UA = "photosearch-demo-fetch/1.0 (+research demo; YFCC100M CC BY subset)"


def get(url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # flaky network: back off and retry
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    return b""


def chunk(idx: int, out: str, side: int, quality: int) -> list[dict]:
    meta = [json.loads(l) for l in gzip.GzipFile(fileobj=io.BytesIO(get(f"{BASE}/metadata/metadata_{idx:03d}.jsonl.gz")))]
    keep = {m["key"]: m for m in meta if m.get("licensename") == "Attribution License"}
    if not keep:
        return []
    z = zipfile.ZipFile(io.BytesIO(get(f"{BASE}/data/{idx:03d}.zip")))
    recs = []
    for name in z.namelist():
        key = os.path.splitext(os.path.basename(name))[0]
        m = keep.get(key)
        if m is None:
            continue
        try:
            make_thumb(z.read(name), os.path.join(out, "images", f"{key}.webp"), side, quality)
        except Exception:
            continue  # undecodable / too small
        recs.append(m)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", default="1-1000")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--short-side", type=int, default=384)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--geonames", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.chunks.partition("-")
    todo = range(int(lo), int(hi or lo) + 1)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    sp = os.path.join(a.out, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {"done": [], "recs": {}}
    done = set(state["done"])
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(chunk, i, a.out, a.short_side, a.quality): i for i in todo if i not in done}
        for n, f in enumerate(futs):
            i = futs[f]
            try:
                for m in f.result():
                    state["recs"][m["key"]] = m
                state["done"].append(i)
            except Exception as e:
                print(f"chunk {i}: {e}", flush=True)
            if n % 50 == 0:
                save_json(sp, state)
                print(f"{len(state['done'])} chunks, {len(state['recs'])} CC BY photos, {time.time() - t0:.0f}s", flush=True)
    save_json(sp, state)

    geo = Geo(a.out) if a.geonames else None
    side, rows = {}, []
    for key, m in state["recs"].items():
        fn = f"{key}.webp"
        if not os.path.exists(os.path.join(a.out, "images", fn)):
            continue
        date = (m.get("datetaken") or "")[:19]
        try:
            ok_date = 1990 <= int(date[:4]) <= 2016
        except ValueError:
            ok_date = False
        lat = float(m["latitude"]) if m.get("latitude") not in (None, "") else None
        lon = float(m["longitude"]) if m.get("longitude") not in (None, "") else None
        place = geo.place(lat, lon) if geo and lat is not None and lon is not None else None
        side[fn] = {k: v for k, v in (("date", date.replace(" ", "T") if ok_date else None), ("lat", lat), ("lon", lon),
                                       ("place", place)) if v is not None}
        rows.append({"file": fn, "hash": key, "creator": urllib.parse.unquote_plus(m.get("unickname") or "").strip() or "unknown",
                     "license": LICENSE[0], "license_url": LICENSE[1], "source": "YFCC100M (Flickr)",
                     "source_url": m.get("pageurl") or "", "flickr_url": m.get("pageurl") or "",
                     "title": urllib.parse.unquote_plus(m.get("title") or "")[:200], "date_taken": date if ok_date else "",
                     "lat": lat if lat is not None else "", "lon": lon if lon is not None else ""})
    save_json(os.path.join(a.out, "images", "_meta.json"), side)
    with open(os.path.join(a.out, "credits.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
        w.writeheader(); w.writerows(rows)
    print(f"done: {len(rows)} CC BY photos, {sum(1 for v in side.values() if 'place' in v)} with places, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
