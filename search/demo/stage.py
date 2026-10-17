#!/usr/bin/env python3
"""Assemble a minimal Docker build context for the demo (no repo-root context: search/data is gigabytes).

usage (from the repo root):
  python3 search/demo/stage.py --bundle demo-bundle --out build/demo [--target fly|cloudrun|hf]
  python3 search/demo/stage.py --bundle demo-bundle --export-thumbs build/thumbs   # files <id>.<ext> for R2 / a CDN

Writes OUT/{Dockerfile, requirements.txt, app/, bundle/} plus the target's config (fly.toml | cloudrun.env.yaml |
README.md with the Spaces header). Bundle files are hard-linked when possible (no extra disk).
Only the photosearch modules the demo imports are copied; none of them imports torch.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SEARCH)
PHOTOSEARCH = ("search.py", "library.py", "query.py", "places.py", "explain.py", "tools.py", "cleanup.py", "guard.py")
DEMO = ("demo_server.py", "engine.py", "textenc.py", "visits.py")


def link_or_copy(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_thumbs(bundle: str, out: str):
    with np.load(os.path.join(bundle, "thumbs.npz"), allow_pickle=False) as z:
        ids, offs, lens, mime = z["ids"], z["offsets"], z["lengths"], str(z["mime"])
    ext = {"image/avif": "avif", "image/webp": "webp"}.get(mime, "jpg")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(bundle, "thumbs.pack"), "rb") as f:
        for i, o, n in zip(ids, offs, lens):
            f.seek(int(o))
            with open(os.path.join(out, f"{int(i)}.{ext}"), "wb") as g:
                g.write(f.read(int(n)))
    print(f"{len(ids)} thumbnails -> {out} (*.{ext}, {mime}); upload with e.g. "
          f"`rclone copy {out} r2:BUCKET --header-upload 'Content-Type: {mime}'`, then build the bundle with "
          f"--no-thumbs or drop thumbs.* from the staged bundle and set DEMO_THUMB_BASE_URL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out")
    ap.add_argument("--target", choices=("fly", "cloudrun", "hf"), default="cloudrun")
    ap.add_argument("--no-thumbs", action="store_true", help="leave thumbs.* out (served from DEMO_THUMB_BASE_URL)")
    ap.add_argument("--export-thumbs", metavar="DIR")
    a = ap.parse_args()
    if a.export_thumbs:
        return export_thumbs(a.bundle, a.export_thumbs)
    if not a.out:
        sys.exit("--out required")
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"{a.out} exists and is not empty")
    web = os.path.join(ROOT, "web", "dist")
    if not os.path.isfile(os.path.join(web, "index.html")):
        sys.exit("web/dist missing: cd web && npm run build")
    if not os.path.isfile(os.path.join(a.bundle, "manifest.json")):
        sys.exit(f"{a.bundle} is not a bundle (no manifest.json): run search/demo/build_bundle.py")

    o = a.out
    for d in (o, os.path.join(o, "app", "photosearch"), os.path.join(o, "app", "demo")):
        os.makedirs(d, exist_ok=True)
    shutil.copy2(os.path.join(HERE, "Dockerfile"), os.path.join(o, "Dockerfile"))
    shutil.copy2(os.path.join(HERE, "requirements-demo.txt"), os.path.join(o, "requirements.txt"))
    for m in PHOTOSEARCH:
        shutil.copy2(os.path.join(SEARCH, "photosearch", m), os.path.join(o, "app", "photosearch", m))
    shutil.copy2(os.path.join(SEARCH, "mock_agent.py"), os.path.join(o, "app", "mock_agent.py"))
    for m in DEMO:
        shutil.copy2(os.path.join(HERE, m), os.path.join(o, "app", "demo", m))
    shutil.copytree(web, os.path.join(o, "app", "web"))
    for dp, _, fs in os.walk(a.bundle):
        for fn in fs:
            rel = os.path.relpath(os.path.join(dp, fn), a.bundle)
            if a.no_thumbs and rel.startswith("thumbs."):
                continue
            link_or_copy(os.path.join(dp, fn), os.path.join(o, "bundle", rel))
    dep = os.path.join(HERE, "deploy")
    if a.target == "fly":
        shutil.copy2(os.path.join(dep, "fly.toml"), os.path.join(o, "fly.toml"))
    elif a.target == "cloudrun":
        shutil.copy2(os.path.join(dep, "cloudrun.env.yaml"), os.path.join(o, "cloudrun.env.yaml"))
    else:
        shutil.copy2(os.path.join(dep, "hf-space-README.md"), os.path.join(o, "README.md"))
        with open(os.path.join(o, ".gitattributes"), "w") as f:  # Spaces: big binaries through LFS/Xet
            f.write("*.onnx filter=lfs diff=lfs merge=lfs -text\n*.bin filter=lfs diff=lfs merge=lfs -text\n"
                    "*.pack filter=lfs diff=lfs merge=lfs -text\n*.npz filter=lfs diff=lfs merge=lfs -text\n"
                    "*.sqlite filter=lfs diff=lfs merge=lfs -text\n")
    with open(os.path.join(o, ".dockerignore"), "w") as f:
        f.write("**/__pycache__\n*.pyc\n.git\nfly.toml\ncloudrun.env.yaml\nREADME.md\n")
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fs in os.walk(o) for fn in fs)
    print(f"staged {o} for {a.target}: {total / 1e6:.0f} MB of context (+ ~200 MB base image and wheels)")


if __name__ == "__main__":
    main()
