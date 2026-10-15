"""Qwen3-VL-Embedding checks + benchmarks.

  validate   20 Flickr images/captions: prompt vs chat template, batch/order consistency, R@1, throughput
  bench      hotels + flickr (bench.py tasks; bench.py itself skips flickr for non-CLIP names)
  fuse       z-scored score fusion with siglip2 on the cached embeddings (no model load)
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench import CACHE, D, flickr_data, flickr_task, hotels_task  # noqa: E402
from photosearch import encoders  # noqa: E402

NAME = "qwen3vl-emb-2b"


def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def validate(n=20):
    import mlx.core as mx
    files, caps, cap_img = flickr_data()
    paths = [f"{D}/flickr1k/images_flickr_1k_test/{f}" for f in files[:n]]
    texts = [caps[int(np.flatnonzero(cap_img == i)[0])] for i in range(n)]
    t = time.time(); e = encoders.get(NAME); print(f"load {time.time() - t:.1f}s  peak RSS {rss_gb():.2f}GB")

    msgs = lambda instr, content: [{"role": "system", "content": [{"type": "text", "text": instr}]},
                                   {"role": "user", "content": content}]
    ref_t = e.proc.apply_chat_template(msgs(e.QUERY, [{"type": "text", "text": "a dog"}]), add_generation_prompt=True, tokenize=False)
    ref_i = e.proc.apply_chat_template(msgs(e.DOC, [{"type": "image", "image": "x"}]), add_generation_prompt=True, tokenize=False)
    print("text prompt == chat template:", ref_t == e._prompt(e.QUERY, "a dog"))
    print("image prompt == chat template:", ref_i == e._prompt(e.DOC, "<|vision_start|><|image_pad|><|vision_end|>"))

    T0 = e.texts(texts[:1])                      # fresh process, before any image
    t = time.time(); I = e.images(paths, bs=8); mx.synchronize(); ti = time.time() - t
    I1 = np.concatenate([e.images([p], bs=1) for p in paths[:4]])
    t = time.time(); T = e.texts(texts); tt = time.time() - t
    T1 = np.concatenate([e.texts([x]) for x in texts[:4]])
    print(f"NaN: img {np.isnan(I).any()} txt {np.isnan(T).any()}")
    print(f"image batch-vs-single cos min {np.min((I[:4] * I1).sum(1)):.4f}")
    print(f"text batch-vs-single cos min {np.min((T[:4] * T1).sum(1)):.4f}")
    print(f"text before-vs-after images cos {float(T0[0] @ T[0]):.4f}")
    S = T @ I.T
    print(f"20x20 t2i R@1 {np.mean(S.argmax(1) == np.arange(n)):.2f}  i2t R@1 {np.mean(S.argmax(0) == np.arange(n)):.2f}  "
          f"diag mean {np.diag(S).mean():.3f} off-diag mean {(S.sum() - np.trace(S)) / (n * n - n):.3f}")
    print(f"throughput: images {n / ti:.2f}/s (bs 8, incl. decode)  texts {n / tt:.1f}/s  peak RSS {rss_gb():.2f}GB  mlx peak {mx.get_peak_memory() / 1e9:.2f}GB")


def resumable(key, items, fn, chunk=200):
    """Fill bench.py's cache in chunks so a guard abort only loses the current chunk."""
    final, part = f"{CACHE}/{key}__{NAME}.npy", f"{CACHE}/{key}__{NAME}.partial.npy"
    if os.path.exists(final):
        return
    X = np.load(part) if os.path.exists(part) else None
    for i in range(0 if X is None else len(X), len(items), chunk):
        t = time.time(); x = fn(items[i:i + chunk])
        X = x if X is None else np.concatenate([X, x]); np.save(part, X)
        print(f"  {key} {len(X)}/{len(items)}  {len(x) / (time.time() - t):.2f}/s", flush=True)
    np.save(final, X); os.remove(part)


def bench():
    from bench import hotels_subset
    enc = None

    def enc_fn():
        nonlocal enc
        enc = enc or encoders.get(NAME)
        return enc
    files, caps, _ = flickr_data()
    resumable("hotels2400", [f"{D}/hotels/images/{f}" for f in hotels_subset()[0]], lambda x: enc_fn().images(x))
    resumable("flickr_img", [f"{D}/flickr1k/images_flickr_1k_test/{f}" for f in files], lambda x: enc_fn().images(x))
    resumable("flickr_txt", caps, lambda x: enc_fn().texts(x), chunk=1000)
    t = time.time(); r = hotels_task(NAME, enc_fn); print(r, f"{time.time() - t:.0f}s", flush=True)
    t = time.time(); r = flickr_task(NAME, enc_fn); print(r, f"{time.time() - t:.0f}s", flush=True)
    print(f"peak RSS {rss_gb():.2f}GB")


def zs(S):
    return (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)


def fuse():
    from fusion import fuse as hfuse, run
    run(NAME); run("siglip2-b16-384")
    for w in (0.25, 0.5, 1.0, 2.0):
        run(f"fuse siglip2 + {w}*qwen", hfuse(["siglip2-b16-384", NAME], [1, w]))
    _, caps, cap_img = flickr_data()
    L = lambda m, k: np.load(f"{CACHE}/flickr_{k}__{m}.npy").astype(np.float32)
    sims = {m: L(m, "txt") @ L(m, "img").T for m in ("siglip2-b16-384", NAME)}

    def r1(S):
        rank = (S > S[np.arange(len(caps)), cap_img][:, None]).sum(1)
        return " ".join(f"R@{k}={(rank < k).mean():.3f}" for k in (1, 5, 10))
    for m, S in sims.items():
        print(f"flickr {m:28s} {r1(S)}")
    for w in (0.5, 1.0, 2.0):
        print(f"flickr fuse siglip2 + {w}*qwen     {r1(zs(sims['siglip2-b16-384']) + w * zs(sims[NAME]))}")


if __name__ == "__main__":
    {"validate": validate, "bench": bench, "fuse": fuse}[sys.argv[1]]()
