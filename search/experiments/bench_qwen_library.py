"""Whole-library Qwen3-VL-Embedding index + eval.

  embed            Library.embed_pending into data/index/vectors__qwen3vl-emb-2b.npz (resumable; DB opened read-only)
  eval [ARGS...]   experiments/eval_search.py unchanged, but with Library defaulting to the Qwen vectors
"""
import functools, glob, os, runpy, sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
from index import INDEX  # noqa: E402
import photosearch.library as L  # noqa: E402

NAME = "qwen3vl-emb-2b"

if __name__ == "__main__":
    cmd = sys.argv.pop(1)
    if cmd == "embed":  # optional N: stop after N images (multiple of 512) and exit, releasing the ML budget
        lib = L.Library(INDEX, model=NAME, readonly=True)
        n, it = int(sys.argv[1]) if len(sys.argv) > 1 else None, lib.encoder.iter_images
        lib.encoder.iter_images = lambda items, bs=8: it(items[:n], bs)
        lib.embed_pending(chunk=512, log=lambda m: print(m, flush=True))
    else:
        L.Library = functools.partial(L.Library, model=NAME, readonly=True)
        before = set(glob.glob(f"{HERE}/results/eval_plain_*.json"))
        runpy.run_path(f"{HERE}/eval_search.py", run_name="__main__")
        for f in set(glob.glob(f"{HERE}/results/eval_plain_*.json")) - before:
            os.rename(f, f.replace("eval_plain_", f"eval_plain_{NAME}_"))
