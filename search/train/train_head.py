"""Train a small projection head on cached SigLIP2 features with a metric-learning loss on hotel_id.

usage: python train/train_head.py --name hotel_head [--inputs pooled] [--dim 512] [--depth 2] [--loss supcon] ...
Train data: Hotels-50K train shards minus every test hotel (extract.py drops them; asserted again here).
5% of train hotels are held out for early stopping and choosing the fusion weight. Test numbers are printed
at the end only for reporting; nothing is selected on them.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/experiments")
from photosearch import guard, heads  # noqa: E402
from bench_head import metrics, fmt, test_data, eval_head  # noqa: E402

FEATS = f"{ROOT}/data/hotels_train/feats"


def load_train(tags=("multi",)):
    feats, ys, srcs = {}, [], []
    for tag in tags:
        meta = pd.read_parquet(f"{FEATS}/train_{tag}_meta.parquet")
        for si in sorted(meta.shard.unique()):
            parts = sorted(glob.glob(f"{FEATS}/train_{tag}_{si}_*.npz"))
            m = meta[meta.shard == si].iloc[:sum(len(np.load(p)["pooled"]) for p in parts)]  # partial shards ok
            if not len(m):
                continue
            for p in parts:
                z = np.load(p)
                for k in ("pooled", "mean", "gem"):
                    feats.setdefault(k, []).append(z[k])
            ys.append(m.hotel_id.values); srcs.append(m.source.values)
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    y, src = np.concatenate(ys), np.concatenate(srcs)
    gt = set(int(v) for v in json.load(open(f"{ROOT}/data/hotels/gt.json")).values())
    assert not (set(y.tolist()) & gt), "test hotel leaked into training"
    return feats, y, src


def supcon(z, y, t):
    """Multi-positive InfoNCE (supervised contrastive) over the batch."""
    s = z @ z.T / t
    eye = torch.eye(len(z), dtype=torch.bool)
    s = s.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None]) & ~eye
    logp = s - s.logsumexp(1, keepdim=True)
    has = pos.any(1)
    return -(logp * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


def multisim(z, y, a=2.0, b=50.0, lam=0.5, eps=0.1):
    """Multi-similarity loss (Wang et al. 2019) with hard pair mining."""
    s = z @ z.T
    eye = torch.eye(len(z), dtype=torch.bool)
    pos = (y[:, None] == y[None]) & ~eye
    neg = y[:, None] != y[None]
    loss = []
    for i in range(len(z)):
        sp, sn = s[i][pos[i]], s[i][neg[i]]
        if not len(sp):
            continue
        sn = sn[sn + eps > sp.min()]
        sp = sp[sp - eps < sn.max()] if len(sn) else sp[:0]
        if not len(sp) or not len(sn):
            continue
        loss.append(torch.log1p(torch.exp(-a * (sp - lam)).sum()) / a + torch.log1p(torch.exp(b * (sn - lam)).sum()) / b)
    return torch.stack(loss).mean() if loss else s.sum() * 0


def pk_batches(y, P, K, rng, weight=None):
    by = pd.Series(np.arange(len(y))).groupby(y).apply(list)
    hs = rng.permutation(len(by))
    for i in range(0, len(hs) - P + 1, P):
        idx = []
        for h in hs[i:i + P]:
            ii = by.iloc[h]
            idx += list(rng.choice(ii, min(K, len(ii)), replace=False))
        yield np.array(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hotel_head"); ap.add_argument("--inputs", default="pooled")
    ap.add_argument("--dim", type=int, default=512); ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=2); ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--loss", default="supcon"); ap.add_argument("--t", type=float, default=0.05)
    ap.add_argument("--P", type=int, default=256); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--featdrop", type=float, default=0.0); ap.add_argument("--singles", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    guard.acquire(1)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    names = tuple(a.inputs.split(","))
    feats, y, src = load_train(("multi", "single") if a.singles else ("multi",))
    X = heads.inputs_from(feats, names)
    hs = np.unique(y); val_h = set(np.random.default_rng(123).choice(hs, int(0.05 * len(hs)), replace=False).tolist())
    isv = np.isin(y, list(val_h))
    Xtr, ytr, Xva, yva = X[~isv], y[~isv], X[isv], y[isv]
    print(f"train {len(Xtr)} imgs / {len(np.unique(ytr))} hotels; val {len(Xva)} / {len(val_h)}", flush=True)

    h = heads.Head(X.shape[1], a.dim, a.hidden, a.depth, a.drop, names)
    with torch.no_grad():  # init projection: identity if square, else PCA of train features
        if a.dim == X.shape[1]:
            h.proj.weight.copy_(torch.eye(a.dim))
        else:
            _, _, vt = np.linalg.svd(Xtr[rng.choice(len(Xtr), min(20000, len(Xtr)), replace=False)] - Xtr.mean(0), full_matrices=False)
            h.proj.weight.copy_(torch.from_numpy(vt[:a.dim].copy()))
    opt = torch.optim.AdamW(h.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(np.unique(ytr)) // a.P)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)

    def val():
        Z = h.embed(Xva); return metrics(Z @ Z.T, yva)
    base = metrics(Xva[:, :768] @ Xva[:, :768].T if names[0] == "pooled" else Xva @ Xva.T, yva)
    print(fmt("val raw", base), flush=True)
    best, best_ep, t0 = -1, -1, time.time()
    for ep in range(a.epochs):
        h.train(); tot = n = 0
        for idx in pk_batches(ytr, a.P, a.K, rng):
            guard.check()
            x = Xt[idx]
            if a.noise:
                x = x + a.noise * torch.randn_like(x) / x.shape[1] ** 0.5
            if a.featdrop:
                x = F.dropout(x, a.featdrop)
            z = h(x)
            loss = supcon(z, Yt[idx], a.t) if a.loss == "supcon" else multisim(z, Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); n += 1
        m = val()
        if m["mAP"] > best:
            best, best_ep = m["mAP"], ep
            heads.save(h, f"{ROOT}/data/models/{a.name}.pt", args=vars(a))
        print(f"ep {ep:2d} loss {tot / max(n, 1):.3f} " + fmt("val", m) + f" ({time.time() - t0:.0f}s)", flush=True)
    h = heads.load(f"{ROOT}/data/models/{a.name}.pt")
    # fusion weight chosen on val, never on test
    Z = h.embed(Xva); S1 = Z @ Z.T; S0 = Xva[:, :768] @ Xva[:, :768].T
    np.fill_diagonal(S1, -np.inf); np.fill_diagonal(S0, -np.inf)
    ws = [0.25, 0.5, 1, 2, 4, 8]
    vm = {w: metrics(heads.fuse(S0, S1, w), yva)["mAP"] for w in ws}
    h.fuse_w = float(max(vm, key=vm.get))
    print(f"best ep {best_ep} val mAP {best:.3f}; val fuse mAP {vm} -> fuse_w {h.fuse_w}")
    heads.save(h, f"{ROOT}/data/models/{a.name}.pt", args=vars(a), val_mAP=float(best))
    feats_t, labels = test_data()
    if names[0] == "pooled" or True:
        Xq = heads.inputs_from(feats_t, ("pooled",)); print(fmt("TEST raw siglip2", metrics(Xq @ Xq.T, labels)))
    for k, m in eval_head(h, feats_t, labels, ws=(h.fuse_w,)).items():
        print(fmt(f"TEST {a.name} {k}", m), flush=True)


if __name__ == "__main__":
    main()
