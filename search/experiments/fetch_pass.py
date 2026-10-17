#!/usr/bin/env python3
"""Stream a PASS subset (publishable demo photos) straight from Zenodo: thumbnails + metadata + credits, never a tar.

PASS (Asano et al., NeurIPS 2021; https://www.robots.ox.ac.uk/~vgg/research/pass/) is 1.44M YFCC100M photos without
people, every one under Flickr's "Attribution License" (= CC BY 2.0, per YFCC100M's license URLs); the compilation
is CC BY 4.0. Zenodo record 6615455: PASS.0.tar .. PASS.9.tar (~18.7 GB each) + pass_metadata.csv (152 MB:
,unickname,datetaken,licensename,hash,latitude,longitude).

How: each tar is read sequentially over HTTP and parsed on the fly (ustar/pax/GNU long names). Kept members are
decoded and re-encoded as a small WebP (short side --short-side px, enough for SigLIP2's 384x384 input); everything
else is discarded as it streams. Progress (byte offset of the next tar header per tar) is saved, so a re-run resumes
with an HTTP Range request. Then pass_metadata.csv is streamed once for the kept hashes.

Outputs in --out:
  images/<hash>.webp      thumbnails (index these: PHOTOSEARCH_INDEX=data/index-pass python index.py data/pass/images=pass)
  images/_meta.json       index.py sidecar: {file: {date, lat, lon, place}}
  credits.csv             file, hash, creator, license, license_url, source, source_url, flickr_url, title, date_taken, lat, lon
  state.json              resume offsets + throughput

Flickr links: PASS metadata has no Flickr photo id, and the public YFCC100M SQLite (multimedia-commons S3) has no hash
column, so the photo page is not derivable from PASS alone. credits.csv links the YFCC100M copy on AWS instead
(source_url); pass --photoid-map hash_to_photoid.tsv (from the YFCC100M Webscope "hash" file) to fill flickr_url.
Places: --geonames reverse-geocodes GPS offline against GeoNames cities15000 (CC BY 4.0; one 3 MB download).

usage:
  python experiments/fetch_pass.py --out data/pass --n 500                          # test (~65 MB streamed + CSV)
  python experiments/fetch_pass.py --out data/pass --n 50000 --tars 0-9 --geonames  # full demo set, resumable
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

RECORD = "https://zenodo.org/records/6615455/files"
TAR_URL = RECORD + "/PASS.{}.tar?download=1"
CSV_URL = RECORD + "/pass_metadata.csv?download=1"
AWS = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/{}/{}/{}.jpg"
GEONAMES = "https://download.geonames.org/export/dump/cities15000.zip"
LICENSE = ("CC BY 2.0", "https://creativecommons.org/licenses/by/2.0/")
UA = "photosearch-demo-fetch/1.0 (+research demo; PASS subset)"


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


class HTTPStream:
    """Sequential reader over HTTP that reconnects with `Range: bytes=pos-` after errors (Zenodo answers 206)."""

    def __init__(self, url: str, start: int = 0, timeout: int = 60, retries: int = 40):
        self.url, self.pos, self.timeout, self.retries = url, start, timeout, retries
        self.r, self.downloaded, self.reconnects = None, 0, -1

    def _open(self):
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": UA, **({"Range": f"bytes={self.pos}-"} if self.pos else {})})
                r = urllib.request.urlopen(req, timeout=self.timeout)
                if self.pos and r.status != 206:
                    raise IOError(f"server ignored Range (HTTP {r.status})")
                self.r, self.reconnects = r, self.reconnects + 1
                return
            except (urllib.error.URLError, OSError, IOError) as e:
                wait = min(120, 5 * (attempt + 1))
                if isinstance(e, urllib.error.HTTPError) and e.code == 416:
                    raise EOFError("range past end")
                print(f"  [http] {type(e).__name__}: {e}; retry in {wait}s", flush=True)
                time.sleep(wait)
        raise IOError(f"gave up on {self.url}")

    def read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            if self.r is None:
                self._open()
            try:
                b = self.r.read(min(n - len(out), 1 << 20))
            except (OSError, urllib.error.URLError) as e:
                print(f"  [http] read error at {self.pos}: {e}; reconnecting", flush=True)
                self.close()
                continue
            if not b:
                if len(out) == 0:
                    return b""
                self.close()  # short read mid-member: reconnect and continue at self.pos
                continue
            out += b
            self.pos += len(b)
            self.downloaded += len(b)
        return bytes(out)

    def close(self):
        if self.r is not None:
            try:
                self.r.close()
            except Exception:
                pass
            self.r = None


# ---------------------------------------------------------------- tar parsing (streaming)

def _num(field: bytes) -> int:
    if field and field[0] & 0x80:  # GNU base-256
        v = int.from_bytes(field, "big") & ((1 << (8 * len(field) - 1)) - 1)
        return v
    s = field.split(b"\0", 1)[0].strip()
    return int(s, 8) if s else 0


def _pax(data: bytes) -> dict:
    out, i = {}, 0
    while i < len(data):
        sp = data.index(b" ", i)
        n = int(data[i:sp])
        k, _, v = data[sp + 1:i + n - 1].partition(b"=")
        out[k.decode()] = v.decode("utf-8", "replace")
        i += n
    return out


def tar_members(s: HTTPStream):
    """Yield (header_offset, name, size, pad) for regular files; the caller must then read size + pad bytes."""
    long_name = None
    while True:
        hdr_off = s.pos
        h = s.read(512)
        if len(h) < 512 or h == b"\0" * 512:
            return
        size, typ = _num(h[124:136]), h[156:157]
        if long_name:
            name = long_name
        else:
            name = h[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
            prefix = h[345:500].split(b"\0", 1)[0].decode("utf-8", "replace") if h[257:262] == b"ustar" else ""
            if prefix:
                name = prefix + "/" + name
        long_name = None
        pad = (-size) % 512
        if typ in (b"L",):
            long_name = s.read(size + pad)[:size].split(b"\0", 1)[0].decode("utf-8", "replace")
            continue
        if typ in (b"x", b"g"):
            meta = _pax(s.read(size + pad)[:size])
            if typ == b"x" and "path" in meta:
                long_name = meta["path"]
            continue
        if typ in (b"0", b"\0", b"7"):
            yield hdr_off, name, size, pad
        else:
            s.read(size + pad)


# ---------------------------------------------------------------- thumbnails

def make_thumb(data: bytes, dst: str, short_side: int, quality: int):
    from PIL import Image, ImageOps
    im = Image.open(io.BytesIO(data))
    w0, h0 = im.size
    if min(w0, h0) < 128:
        raise ValueError("too small")
    im.draft("RGB", (short_side * w0 // min(w0, h0), short_side * h0 // min(w0, h0)))  # fast JPEG DCT downscale
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if min(w, h) > short_side:
        f = short_side / min(w, h)
        im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
    tmp = dst + ".tmp"
    im.save(tmp, "WEBP", quality=quality, method=4)
    os.replace(tmp, dst)
    return w0, h0


# ---------------------------------------------------------------- places (offline reverse geocoding)

class Geo:
    def __init__(self, cache_dir: str):
        import numpy as np
        path = os.path.join(cache_dir, "cities15000.zip")
        if not os.path.exists(path):
            print("downloading GeoNames cities15000 (CC BY 4.0) ...", flush=True)
            req = urllib.request.Request(GEONAMES, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(path + ".tmp", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(path + ".tmp", path)
        import pycountry
        names, cc, lat, lon = [], [], [], []
        with zipfile.ZipFile(path) as z, z.open("cities15000.txt") as f:
            for line in io.TextIOWrapper(f, "utf-8"):
                p = line.rstrip("\n").split("\t")
                names.append(p[1]); lat.append(float(p[4])); lon.append(float(p[5])); cc.append(p[8])
        country = {}
        for c in set(cc):
            e = pycountry.countries.get(alpha_2=c)
            country[c] = (getattr(e, "common_name", None) or e.name) if e else c
        self.names, self.cc, self.country = names, cc, country
        r = np.radians
        self.xyz = np.stack([np.cos(r(lat)) * np.cos(r(lon)), np.cos(r(lat)) * np.sin(r(lon)), np.sin(r(lat))], 1)
        self.np = np

    def place(self, lat: float, lon: float, city_km: float = 25, country_km: float = 150) -> str | None:
        np = self.np
        v = np.array([math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
                      math.cos(math.radians(lat)) * math.sin(math.radians(lon)), math.sin(math.radians(lat))])
        i = int(np.argmax(self.xyz @ v))
        km = 6371 * math.acos(max(-1.0, min(1.0, float(self.xyz[i] @ v))))
        if km <= city_km:
            return f"{self.names[i]}, {self.country[self.cc[i]]}"
        if km <= country_km:
            return self.country[self.cc[i]]
        return None


# ---------------------------------------------------------------- main

def parse_tars(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return [t for t in out if 0 <= t <= 9]


def save_json(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def fetch_images(a, state):
    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    tars = parse_tars(a.tars)
    quota = math.ceil(a.n / len(tars))
    members_path = os.path.join(a.out, "members.jsonl")
    lock = threading.Lock()
    t0, total_dl, kept_now, errors = time.time(), 0, 0, 0

    def work(h, data, tar, off):
        nonlocal errors
        dst = os.path.join(img_dir, f"{h}.webp")
        try:
            w, hh = make_thumb(data, dst, a.short_side, a.quality)
        except Exception as e:
            with lock:
                errors += 1
            return None
        rec = {"hash": h, "tar": tar, "offset": off, "bytes": len(data), "w": w, "h": hh, "thumb_bytes": os.path.getsize(dst)}
        with lock, open(members_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(a.workers) as pool:
        for tar in tars:
            st = state["tars"].setdefault(str(tar), {"offset": 0, "kept": 0, "eof": False})
            if st["eof"] or st["kept"] >= quota:
                continue
            s = HTTPStream(TAR_URL.format(tar), st["offset"])
            print(f"tar {tar}: resuming at byte {st['offset']:,} ({st['kept']} kept, quota {quota})", flush=True)
            pending, last_save = [], time.time()
            try:
                for off, name, size, pad in tar_members(s):
                    data = s.read(size + pad)[:size]
                    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    h = os.path.basename(name).split(".", 1)[0]
                    if os.path.exists(os.path.join(img_dir, f"{h}.webp")):
                        st["offset"] = s.pos
                        continue
                    pending.append(pool.submit(work, h, data, tar, off))
                    while len(pending) > a.workers * 4:
                        if pending.pop(0).result():
                            st["kept"] += 1; kept_now += 1
                    st["offset"] = s.pos  # next header; members up to here are written or in flight
                    if st["kept"] + len(pending) >= quota:
                        break
                    if time.time() - last_save > 10:
                        for p in pending:
                            if p.result():
                                st["kept"] += 1; kept_now += 1
                        pending = []
                        dt = time.time() - t0
                        state["throughput"] = {"images_per_s": round(kept_now / dt, 2),
                                               "mb_per_s": round((total_dl + s.downloaded) / dt / 1e6, 2)}
                        save_json(os.path.join(a.out, "state.json"), state)
                        last_save = time.time()
                        print(f"  tar {tar}: {st['kept']}/{quota} kept, {(total_dl + s.downloaded) / 1e6:.0f} MB streamed, "
                              f"{kept_now / dt:.1f} img/s", flush=True)
                        if free_gb(a.out) < a.min_free_gb:
                            sys.exit(f"stopping: free disk {free_gb(a.out):.1f} GB < {a.min_free_gb} GB (state saved; re-run resumes)")
                else:
                    st["eof"] = True
            finally:
                for p in pending:
                    if p.result():
                        st["kept"] += 1; kept_now += 1
                total_dl += s.downloaded
                s.close()
                state["streamed_bytes"] = state.get("streamed_bytes", 0) + s.downloaded
                save_json(os.path.join(a.out, "state.json"), state)
    dt = time.time() - t0
    state["throughput"] = {"images_per_s": round(kept_now / dt, 2) if dt else None,
                           "mb_per_s": round(total_dl / dt / 1e6, 2) if dt else None,
                           "bytes_per_image": round(total_dl / kept_now) if kept_now else None, "errors": errors,
                           "seconds": round(dt, 1), "kept": kept_now}
    save_json(os.path.join(a.out, "state.json"), state)
    print(f"images: {kept_now} new in {dt:.0f}s ({state['throughput']}), {errors} undecodable", flush=True)


def fetch_meta(a, state):
    recs = {}
    mp = os.path.join(a.out, "members.jsonl")
    for line in open(mp):
        r = json.loads(line)
        if os.path.exists(os.path.join(a.out, "images", f"{r['hash']}.webp")):
            recs[r["hash"]] = r
    print(f"metadata for {len(recs)} photos: streaming pass_metadata.csv ...", flush=True)
    t0 = time.time()
    if a.csv:
        text = open(a.csv, newline="", encoding="utf-8")
        s = None
    else:
        s = HTTPStream(CSV_URL)

        class _Lines(io.RawIOBase):
            def readable(self):
                return True

            def readinto(self, b):
                d = s.read(len(b)) if s else b""
                b[:len(d)] = d
                return len(d)
        text = io.TextIOWrapper(io.BufferedReader(_Lines(), 1 << 20), "utf-8", newline="")
    meta, seen = {}, 0
    for row in csv.DictReader(text):
        seen += 1
        if row["hash"] in recs:
            meta[row["hash"]] = row
            if len(meta) == len(recs) and not a.csv_full:
                break
    if s:
        state["csv_streamed_bytes"] = s.downloaded
        s.close()
    print(f"  scanned {seen:,} rows in {time.time() - t0:.0f}s, matched {len(meta)}/{len(recs)}", flush=True)

    photoid = {}
    if a.photoid_map:
        for line in open(a.photoid_map):
            h, _, pid = line.rstrip("\n").partition("\t")
            if h in recs:
                photoid[h] = pid
    geo = Geo(a.out) if a.geonames else None
    side, rows, dropped = {}, [], 0
    for h, r in recs.items():
        m = meta.get(h)
        if not m or m.get("licensename") != "Attribution License":
            dropped += 1  # no credit / unexpected license: not publishable
            os.remove(os.path.join(a.out, "images", f"{h}.webp"))
            continue
        creator = urllib.parse.unquote_plus(m.get("unickname") or "").strip() or "unknown"
        date = (m.get("datetaken") or "")[:19]
        try:
            ok_date = 1990 <= int(date[:4]) <= 2016
        except ValueError:
            ok_date = False
        lat = float(m["latitude"]) if m.get("latitude") else None
        lon = float(m["longitude"]) if m.get("longitude") else None
        place = geo.place(lat, lon) if geo and lat is not None and lon is not None else None
        fn = f"{h}.webp"
        side[fn] = {k: v for k, v in (("date", date.replace(" ", "T") if ok_date else None), ("lat", lat), ("lon", lon),
                                       ("place", place)) if v is not None}
        flickr = f"https://www.flickr.com/photo.gne?id={photoid[h]}" if h in photoid else ""
        rows.append({"file": fn, "hash": h, "creator": creator, "license": LICENSE[0], "license_url": LICENSE[1],
                     "source": "PASS / YFCC100M", "source_url": AWS.format(h[:3], h[3:6], h), "flickr_url": flickr,
                     "title": "", "date_taken": date if ok_date else "", "lat": lat if lat is not None else "",
                     "lon": lon if lon is not None else ""})
    save_json(os.path.join(a.out, "images", "_meta.json"), side)
    with open(os.path.join(a.out, "credits.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
        w.writeheader(); w.writerows(rows)
    state["meta"] = {"photos": len(rows), "dropped_no_credit": dropped, "with_gps": sum(1 for r in rows if r["lat"] != ""),
                     "with_date": sum(1 for r in rows if r["date_taken"]), "with_place": sum(1 for v in side.values() if "place" in v),
                     "seconds": round(time.time() - t0, 1)}
    save_json(os.path.join(a.out, "state.json"), state)
    print(f"credits.csv: {len(rows)} photos ({state['meta']})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True, help="photos to keep in total (split evenly over --tars)")
    ap.add_argument("--tars", default="0", help="e.g. 0 | 0-9 | 0,3,7")
    ap.add_argument("--short-side", type=int, default=384)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--workers", type=int, default=2, help="decode/encode threads")
    ap.add_argument("--geonames", action="store_true", help="offline reverse geocoding for place names")
    ap.add_argument("--csv", help="a local pass_metadata.csv instead of streaming it")
    ap.add_argument("--csv-full", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--photoid-map", help="TSV hash<TAB>flickr_photo_id (YFCC100M Webscope) to add Flickr links")
    ap.add_argument("--meta-only", action="store_true", help="skip the image pass (re-run metadata/credits)")
    ap.add_argument("--min-free-gb", type=float, default=10.0, help="stop when free disk drops below this (repo rule: 10)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    need = a.n * 40e3 / 1e9
    if free_gb(a.out) - need < a.min_free_gb:
        sys.exit(f"refusing: {free_gb(a.out):.1f} GB free, ~{need:.1f} GB needed for {a.n} thumbnails, floor {a.min_free_gb} GB")
    sp = os.path.join(a.out, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {"tars": {}, "args": {}}
    state["args"] = {k: v for k, v in vars(a).items() if k not in ("csv", "photoid_map")}
    if not a.meta_only:
        fetch_images(a, state)
    fetch_meta(a, state)


if __name__ == "__main__":
    main()
