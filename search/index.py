"""Index photo folders.  usage: python index.py [--no-embed] [--no-ocr] FOLDER[=ALBUM] ...

Runs fully offline once models are downloaded (HF_HUB_OFFLINE=1).
"""
import argparse, os, sys

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never contact the hub while indexing; set HF_HUB_OFFLINE=0 for the first model download
from photosearch.library import Library  # noqa: E402

INDEX = os.environ.get("PHOTOSEARCH_INDEX", os.path.join(os.path.dirname(__file__), "data", "index"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()
    lib = Library(INDEX)
    for f in a.folders:
        folder, _, album = f.partition("=")
        print("added", lib.add_folder(folder, album or None), "from", folder, flush=True)
    lib.assign_sessions()
    log = lambda m: print(m, flush=True)
    if not a.no_ocr:
        lib.ocr_pending(log)
    if not a.no_embed:
        lib.embed_pending(log=log)
    print("total", lib.db.execute("SELECT count(*) FROM photos").fetchone()[0], "photos;", len(lib.ids), "embedded")
    sys.exit(0)
