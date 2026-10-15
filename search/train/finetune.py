"""UNTESTED DRAFT (not run: the raw train images were deleted to save disk; re-download a shard first). Light fine-tune: last N SigLIP2-B/16-384 vision blocks + attention-pool + projection head, SupCon on hotel_id.

usage: python train/finetune.py --init data/models/hotel_head.pt --blocks 2 --epochs 2 --name hotel_ft
Same train/val split as train_head.py (test hotels never used). Frozen blocks run in fp16 under no_grad,
trainable ones in fp32 on MPS. Evaluates val each epoch and hotels2400 at the end.
"""
import argparse, io, json, os, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/experiments"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photosearch import guard, heads  # noqa: E402

FEATS = f"{ROOT}/data/hotels_train/feats"
IMGS = f"{ROOT}/data/hotels_train/images"


def dump_images(meta):
    """Write the multi-hotel training JPEGs to disk once, so loader workers don't hold shards in RAM."""
    os.makedirs(IMGS, exist_ok=True)
    shards = sorted(__import__("glob").glob(f"{ROOT}/data/hotels_train/raw/data/*.parquet"))
    for si, f in enumerate(shards):
        m = meta[meta.shard == si]
        if all(os.path.exists(f"{IMGS}/{i}.jpg") for i in m.image_id):
            continue
        col = pq.read_table(f, columns=["image"]).column("image")
        for r, iid in zip(m.row, m.image_id):
            open(f"{IMGS}/{iid}.jpg", "wb").write(col[int(r)].as_py()["bytes"])
        del col


class DS(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


class Model(torch.nn.Module):
    def __init__(self, enc, n_train, head):
        super().__init__()
        self.t = enc.model.visual.trunk
        self.k = len(self.t.blocks) - n_train
        for p in self.t.parameters():
            p.requires_grad = False
        self.train_mods = torch.nn.ModuleList([*self.t.blocks[self.k:], self.t.norm, self.t.attn_pool])
        self.train_mods.float()
        for p in self.train_mods.parameters():
            p.requires_grad = True
        self.head = head

    def pooled(self, x):
        t = self.t
        with torch.no_grad():
            x = t.patch_embed(x); x = t._pos_embed(x); x = t.norm_pre(x)
            for b in t.blocks[:self.k]:
                x = b(x)
        x = x.float()
        for b in t.blocks[self.k:]:
            x = b(x)
        return F.normalize(t.attn_pool(t.norm(x)), dim=-1)

    def forward(self, x):
        p = self.pooled(x)
        return p, self.head(p)


def embed(model, paths, tf, bs=32):
    model.eval(); P, Z = [], []
    dl = torch.utils.data.DataLoader(DS(paths, tf), batch_size=bs, num_workers=2)
    with torch.no_grad():
        for x in dl:
            guard.check()
            p, z = model(x.to("mps", torch.float16))
            P.append(p.cpu().numpy()); Z.append(z.cpu().numpy())
    return np.concatenate(P), np.concatenate(Z)


def main():
    from train_head import supcon, pk_batches
    from bench_head import metrics, fmt
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=f"{ROOT}/data/models/hotel_head.pt"); ap.add_argument("--name", default="hotel_ft")
    ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--P", type=int, default=32); ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--t", type=float, default=0.05); ap.add_argument("--max_steps", type=int, default=0)
    a = ap.parse_args()
    guard.acquire(2)
    from photosearch import encoders as E
    meta = pd.read_parquet(f"{FEATS}/train_multi_meta.parquet")
    gt = set(int(v) for v in json.load(open(f"{ROOT}/data/hotels/gt.json")).values())
    assert not (set(meta.hotel_id) & gt)
    dump_images(meta)
    hs = np.unique(meta.hotel_id.values)
    val_h = set(np.random.default_rng(123).choice(hs, int(0.05 * len(hs)), replace=False).tolist())
    isv = meta.hotel_id.isin(val_h).values
    tr, va = meta[~isv], meta[isv]
    enc = E.get("siglip2-b16-384")
    head = heads.load(a.init)
    model = Model(enc, a.blocks, head.float()).to("mps")
    norm = enc.tf.transforms[-1]
    aug = T.Compose([T.RandomResizedCrop(384, scale=(0.5, 1.0), ratio=(0.75, 1.33), interpolation=T.InterpolationMode.BICUBIC),
                     T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), norm])
    opt = torch.optim.AdamW([{"params": model.train_mods.parameters(), "lr": a.lr},
                             {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.05)
    vpaths = [f"{IMGS}/{i}.jpg" for i in va.image_id]
    yva = va.hotel_id.values

    def val(tag):
        P, Z = embed(model, vpaths, enc.tf)
        S0, S1 = P @ P.T, Z @ Z.T
        print(fmt(f"val {tag} pooled", metrics(S0, yva)) + " | " + fmt("head", metrics(S1, yva)), flush=True)
        return metrics(S1, yva)["mAP"]
    best = val("init")
    heads.save(head, f"{ROOT}/data/models/{a.name}_head.pt")
    ytr, ptr = tr.hotel_id.values, np.array([f"{IMGS}/{i}.jpg" for i in tr.image_id])
    rng = np.random.default_rng(0); step = 0; t0 = time.time()
    for ep in range(a.epochs):
        batches = list(pk_batches(ytr, a.P, a.K, rng))
        model.train()
        for bi, idx in enumerate(batches):
            guard.check()
            x = torch.stack([aug(Image.open(p).convert("RGB")) for p in ptr[idx]]).to("mps", torch.float16)
            _, z = model(x)
            loss = supcon(z.cpu(), torch.from_numpy(ytr[idx]), a.t)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if bi % 20 == 0:
                print(f"ep {ep} step {bi}/{len(batches)} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        m = val(f"ep{ep}")
        if m > best:
            best = m
            torch.save({k: v for k, v in model.train_mods.state_dict().items()}, f"{ROOT}/data/models/{a.name}_blocks.pt")
            heads.save(head, f"{ROOT}/data/models/{a.name}_head.pt", blocks=a.blocks)
        if a.max_steps and step >= a.max_steps:
            break
    # final test on hotels2400 with the best weights
    if os.path.exists(f"{ROOT}/data/models/{a.name}_blocks.pt"):
        model.train_mods.load_state_dict(torch.load(f"{ROOT}/data/models/{a.name}_blocks.pt"))
        model.head.load_state_dict(heads.load(f"{ROOT}/data/models/{a.name}_head.pt").state_dict())
    from bench import hotels_subset
    files, labels = hotels_subset()
    P, Z = embed(model, [f"{ROOT}/data/hotels/images/{f}" for f in files], enc.tf)
    np.savez(f"{FEATS}/test2400_{a.name}.npz", pooled=P, head=Z)
    X0 = np.load(f"{FEATS}/test2400.npz")["pooled"].astype(np.float32)
    print(fmt("TEST raw siglip2", metrics(X0 @ X0.T, labels)))
    print(fmt("TEST ft pooled", metrics(P @ P.T, labels)))
    print(fmt("TEST ft head", metrics(Z @ Z.T, labels)))
    R, S = X0 @ X0.T, Z @ Z.T
    np.fill_diagonal(R, -np.inf); np.fill_diagonal(S, -np.inf)
    for w in (0.5, 1, 2):
        print(fmt(f"TEST raw + {w}*ft head", metrics(heads.fuse(R, S, w), labels)), flush=True)


if __name__ == "__main__":
    main()
