"""Cache frozen SigLIP2-B/16-384 features for Hotels-50K train shards + the hotels2400 benchmark.

Per image (fp16): pooled (MAP head, L2-normed = what the library stores), mean & GeM of patch tokens,
and a 4x4 grid of region tokens (avg-pooled 24x24 patch map) for learned pooling heads.

usage: python train/extract.py [--test] [--train] [--singles]
Test-set hotels (data/hotels/gt.json) are never extracted from the train shards.
"""
import argparse, glob, io, json, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from photosearch import encoders as E, guard  # noqa: E402

D = f"{ROOT}/data"
OUT = f"{D}/hotels_train/feats"


class Rows(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]
        im = Image.open(io.BytesIO(x)) if isinstance(x, bytes) else Image.open(x)
        return self.tf(im.convert("RGB"))


@torch.inference_mode()
def feats(enc, x):
    t = enc.model.visual.trunk
    tok = t.forward_features(x)                       # (B, 576, 768), final norm applied
    pooled = enc.model.visual.head(t.forward_head(tok)).float()
    tok = tok.float()
    g = int(tok.shape[1] ** 0.5)
    grid = F.adaptive_avg_pool2d(tok.transpose(1, 2).reshape(len(tok), -1, g, g), 4).flatten(2).transpose(1, 2)
    gem = tok.clamp(min=1e-6).pow(3).mean(1).pow(1 / 3)
    return {"pooled": F.normalize(pooled, dim=-1), "mean": tok.mean(1), "gem": gem, "grid": grid}


def run(enc, items, path, bs=16):
    if os.path.exists(path):
        print("exists", path); return
    out = {k: [] for k in ("pooled", "mean", "gem", "grid")}
    dl = torch.utils.data.DataLoader(Rows(items, enc.tf), batch_size=bs, num_workers=2)
    t0 = time.time()
    for i, x in enumerate(dl):
        guard.check()
        for k, v in feats(enc, x.to(E.DEVICE, E.DTYPE)).items():
            out[k].append(v.cpu().numpy().astype(np.float16))
        if i % 50 == 0:
            n = (i + 1) * bs
            print(f"  {os.path.basename(path)} {n}/{len(items)} {n / (time.time() - t0):.1f} img/s", flush=True)
    keep = out if "test" in path else {k: v for k, v in out.items() if k != "grid"}  # grid only for test (disk)
    np.savez(path + ".tmp.npz", **{k: np.concatenate(v) for k, v in keep.items()})
    os.replace(path + ".tmp.npz", path)


def train_rows(singles: bool):
    gt = set(int(v) for v in json.load(open(f"{D}/hotels/gt.json")).values())
    shards = [f"{D}/hotels_train/raw/data/train-{i:05d}-of-00011.parquet" for i in range(3)]  # fixed indices: raw shards are deleted once extracted
    import pandas as pd
    mp = f"{OUT}/train_{'single' if singles else 'multi'}_meta.parquet"
    if os.path.exists(mp):
        return shards, pd.read_parquet(mp)
    meta = pd.concat([pq.read_table(f, columns=["image_id", "hotel_id", "source"]).to_pandas().assign(shard=i, row=np.arange(pq.ParquetFile(f).metadata.num_rows))
                      for i, f in enumerate(shards)], ignore_index=True)
    meta = meta[~meta.hotel_id.isin(gt)].drop_duplicates("image_id")  # NO test leakage
    n = meta.hotel_id.map(meta.hotel_id.value_counts())
    return shards, meta[(n == 1) if singles else (n >= 2)]


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--test", action="store_true"); a.add_argument("--train", action="store_true"); a.add_argument("--singles", action="store_true")
    a = a.parse_args()
    os.makedirs(OUT, exist_ok=True)
    enc = E.get("siglip2-b16-384")  # guard.acquire(1)
    if a.test:
        sys.path.insert(0, f"{ROOT}/experiments")
        from bench import hotels_subset
        files, labels = hotels_subset()
        run(enc, [f"{D}/hotels/images/{f}" for f in files], f"{OUT}/test2400.npz")
        ref = np.load(f"{D}/cache/hotels2400__siglip2-b16-384.npy")
        mine = np.load(f"{OUT}/test2400.npz")["pooled"].astype(np.float32)
        print("pooled vs bench cache cos:", float((ref * mine).sum(1).mean()))
    for flag, tag in ((a.train, "multi"), (a.singles, "single")):
        if not flag:
            continue
        shards, meta = train_rows(tag == "single")
        if not os.path.exists(f"{OUT}/train_{tag}_meta.parquet"):
            meta.to_parquet(f"{OUT}/train_{tag}_meta.parquet")
        for si, f in enumerate(shards):
            m = meta[meta.shard == si]
            C = 1024  # resumable chunks: a guard abort loses at most one chunk
            todo = [c for c in range(0, len(m), C) if not os.path.exists(f"{OUT}/train_{tag}_{si}_{c // C:03d}.npz")]
            if not todo or not os.path.exists(f):
                continue
            col = pq.read_table(f, columns=["image"]).column("image")
            imgs = [col[int(r)].as_py()["bytes"] for r in m.row]
            del col
            print(f"shard {si}: {len(imgs)} imgs", flush=True)
            for c in todo:
                run(enc, imgs[c:c + C], f"{OUT}/train_{tag}_{si}_{c // C:03d}.npz")
            del imgs
