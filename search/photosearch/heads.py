"""Small learned heads on top of frozen SigLIP2-B/16-384 features ("same place" retrieval).

A head maps an image's cached feature(s) to a new L2-normalized embedding. The `pooled` head takes only the
normalized 768-d SigLIP2 vector the library already stores, so it can be applied to an existing index with
no re-embedding:

    from photosearch import heads
    h = heads.load("data/models/hotel_head.pt")
    Z = h.embed(X)                                  # X: (N, 768) library vectors -> (N, d)
    s = heads.fuse(X @ X[q], Z @ Z[q], h.fuse_w)    # "other photos of this place" scores for query q
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Residual MLP projection: z = norm(P(x) + MLP(x)). P is a linear map (PCA-initialized when dim < in_dim)."""

    def __init__(self, in_dim=768, dim=512, hidden=1024, depth=2, drop=0.1, inputs=("pooled",)):
        super().__init__()
        self.inputs, self.cfg = tuple(inputs), dict(in_dim=in_dim, dim=dim, hidden=hidden, depth=depth, drop=drop, inputs=list(inputs))
        self.proj = nn.Linear(in_dim, dim, bias=False)
        layers, d = [], in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop)]
            d = hidden
        self.mlp = nn.Sequential(*layers, nn.Linear(d, dim)) if depth > 0 else None
        if self.mlp is not None:
            nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)  # start as the (PCA) projection
        self.fuse_w = 1.0

    def forward(self, x):
        z = self.proj(x)
        if self.mlp is not None:
            z = z + self.mlp(x)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def embed(self, X: np.ndarray, bs=8192) -> np.ndarray:
        self.eval()
        return np.concatenate([self(torch.from_numpy(np.asarray(X[i:i + bs], np.float32))).numpy() for i in range(0, len(X), bs)])


def inputs_from(feats: dict, names) -> np.ndarray:
    """Concatenate (separately L2-normalized) cached features, e.g. ('pooled', 'gem')."""
    return np.concatenate([feats[n].astype(np.float32) / np.linalg.norm(feats[n].astype(np.float32), axis=-1, keepdims=True)
                           for n in names], -1)


def zs(s: np.ndarray) -> np.ndarray:
    """Per-query z-score (last axis), ignoring -inf, so two similarity scales can be added."""
    m = np.where(np.isfinite(s), s, np.nan)
    return (s - np.nanmean(m, -1, keepdims=True)) / np.nanstd(m, -1, keepdims=True)


def fuse(s_raw: np.ndarray, s_head: np.ndarray, w: float = 1.0) -> np.ndarray:
    return zs(s_raw) + w * zs(s_head)


def save(h: Head, path: str, **extra):
    torch.save({"cfg": h.cfg, "state": h.state_dict(), "fuse_w": h.fuse_w, **extra}, path)


def load(path: str) -> Head:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    h = Head(**ck["cfg"]); h.load_state_dict(ck["state"]); h.fuse_w = ck.get("fuse_w", 1.0)
    return h.eval()
