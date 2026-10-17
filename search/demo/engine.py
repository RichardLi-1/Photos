"""The demo's search engine: photosearch.search.Searcher (unchanged ranking, parsing, OCR fusion, filters) on top of a
compact, read-only bundle. No torch: vectors and the "same place" space are precomputed and int8-quantized, query text
goes through the int8 ONNX text tower (textenc.py).

Bundle layout (build_bundle.py):
  manifest.json         counts, sizes, quantization, sources
  library.sqlite        photos (no absolute paths), fts, credits
  vectors.npz           ids int64[N], q int8[N,D'], scale f32[D'] (+ basis f32[D',768], mean f32[768] when PCA'd)
  place.npz             q int8[N,512], scale f32[512]      (the trained head's space, Z = head(X))
  concepts.npz          names[C], T f16[C,768]             ("why it matched" prompts, "a photo of <concept>")
  models/               siglip2_tokenizer.bin, siglip2_text_q8.onnx, token_embed.bin, calibration.json
  thumbs.pack + thumbs.npz   all thumbnails in one file + ids/offsets/lengths
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict

import numpy as np

from photosearch.library import Library
from photosearch.search import Searcher


# ---------------------------------------------------------------- quantized matrices

def quantize_cols(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 with one scale per dimension (column). Returns (q int8[N,D], scale f32[D])."""
    X = np.asarray(X, np.float32)
    scale = np.abs(X).max(0) / 127.0
    scale[scale == 0] = 1.0
    return np.clip(np.rint(X / scale), -127, 127).astype(np.int8), scale.astype(np.float32)


def pca_basis(X: np.ndarray, dim: int, sample: int = 30000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(basis f32[dim, D] with orthonormal rows, mean f32[D]) from a sample of rows."""
    X = np.asarray(X, np.float32)
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(sample, len(X)), replace=False)]
    mean = S.mean(0)
    _, _, Vt = np.linalg.svd(S - mean, full_matrices=False)
    return Vt[:dim].astype(np.float32), mean.astype(np.float32)


class QMatrix:
    """Read-only stand-in for the float32 matrices Searcher keeps (S.X, S.Z).

    Supports what photosearch uses: `M @ v`, `M[row]`, `M[rows]`, len(M), M.shape. Rows are stored as int8 with a
    per-dimension scale, optionally in a PCA basis (then `M[row]` is reconstructed to the full dimension).
    `M @ v` folds the scale (and basis) into the query, then multiplies int8 rows in chunks, so RAM stays ~1 byte
    per value instead of 4 (dense=True keeps a dequantized float32 copy instead: faster, 4x the RAM).
    """

    def __init__(self, q: np.ndarray, scale: np.ndarray, basis: np.ndarray | None = None, mean: np.ndarray | None = None,
                 dense: bool = False, chunk: int = 8192):
        self.q, self.scale, self.basis, self.mean, self.chunk = q, scale.astype(np.float32), basis, mean, chunk
        self.full_dim = basis.shape[1] if basis is not None else q.shape[1]
        self.shape = (q.shape[0], self.full_dim)
        self.dtype = np.float32
        self._dense = (q.astype(np.float32) * self.scale) if dense else None

    def __len__(self):
        return self.q.shape[0]

    def _rows(self, idx) -> np.ndarray:
        r = (self._dense[idx] if self._dense is not None else self.q[idx].astype(np.float32) * self.scale)
        if self.basis is not None:
            r = r @ self.basis + self.mean
        return r.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._rows(np.array([int(idx)]))[0]
        return self._rows(np.asarray(idx))

    def __matmul__(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, np.float32)
        w = self.basis @ v if self.basis is not None else v
        const = self.mean @ v if self.mean is not None else 0.0  # PCA centring: x ~ mean + x' B
        if self._dense is not None:
            return self._dense @ w + const
        w = w * (self.scale if w.ndim == 1 else self.scale[:, None])
        out = np.empty((len(self),) + w.shape[1:], np.float32)
        for a in range(0, len(self), self.chunk):
            out[a:a + self.chunk] = self.q[a:a + self.chunk].astype(np.float32) @ w
        return out + const

    def astype(self, dtype, copy=True):  # anything that insists on a real array gets one (only for small N)
        return self._rows(np.arange(len(self))).astype(dtype)

    @property
    def nbytes(self) -> int:
        return self.q.nbytes + (self._dense.nbytes if self._dense is not None else 0)


def load_qmatrix(path: str, dense: bool = False) -> tuple[np.ndarray | None, QMatrix]:
    with np.load(path, allow_pickle=False) as z:
        ids = z["ids"] if "ids" in z else None
        basis = z["basis"] if "basis" in z else None
        mean = z["mean"] if "mean" in z else None
        return ids, QMatrix(z["q"], z["scale"], basis, mean, dense=dense)


# ---------------------------------------------------------------- duck-typed Library / encoder

class _Scalar(float):
    def exp(self):  # explain.py reads float(model.logit_scale.exp()) (open_clip stores log(scale))
        return self


class DemoEncoder:
    """encoders.Encoder look-alike: texts() via ONNX, concept prompts from the bundle, no image tower."""
    name = "siglip2-b16-384 (onnx int8)"
    has_text = True

    def __init__(self, model_dir: str, concepts: dict[str, np.ndarray] | None = None):
        from textenc import TextEncoder
        self.text = TextEncoder(model_dir)
        cal = self.text.calibration
        scale, bias = cal.get("logit_scale", 112.846), cal.get("logit_bias", -16.772)

        class _M:  # Explainer reads model.logit_scale.exp() and model.logit_bias
            logit_scale, logit_bias = _Scalar(scale), bias
        self.model = _M()
        self.concepts = concepts or {}

    def texts(self, texts: list[str], bs: int = 16) -> np.ndarray:
        if texts and all(t in self.concepts for t in texts):
            return np.stack([self.concepts[t] for t in texts]).astype(np.float32)
        return self.text.texts(texts, bs)

    def images(self, *a, **k):
        raise NotImplementedError("search by uploaded photo is turned off in the public demo")

    iter_images = images


class DemoLibrary(Library):
    """Library over a copied, read-only bundle sqlite. Never touches encoders.py (torch)."""

    def __init__(self, bundle: str, db_path: str, encoder: DemoEncoder):
        self.dir, self.model_name = bundle, "siglip2-b16-384"
        self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self._vec_path = os.path.join(bundle, "vectors.npz")
        self.ids, self.X, self.row_of = np.zeros(0, np.int64), np.zeros((0, 0), np.float32), {}
        self._enc = encoder


class DemoSearcher(Searcher):
    """Searcher whose X / Z are QMatrix objects loaded from the bundle (never re-read: the bundle is immutable)."""

    def __init__(self, lib: DemoLibrary, ids: np.ndarray, X: QMatrix, Z: QMatrix | None):
        self._bundle = (ids, X, Z)
        super().__init__(lib)

    def refresh(self):
        from photosearch.places import Places
        ids, X, Z = self._bundle
        meta = {r[0]: r[1:] for r in self.lib.db.execute("SELECT id, session, taken, place, album, ocr IS NOT NULL, lat, lon FROM photos")}
        m = [meta.get(int(i), (None,) * 7) for i in ids]
        self.session = np.array([-1 if r[0] is None else int(r[0]) for r in m], dtype=np.int64)
        self.taken = np.array([np.nan if r[1] is None else r[1] for r in m], dtype=np.float64)
        place_rows = defaultdict(list)
        for n, r in enumerate(m):
            if r[2]:
                place_rows[r[2]].append(n)
        self.album = np.array([r[3] or "" for r in m], dtype=object)
        self.has_text = np.array([bool(r[4]) for r in m], dtype=bool)
        self.lat = np.array([np.nan if r[5] is None else r[5] for r in m], dtype=np.float64)
        self.lon = np.array([np.nan if r[6] is None else r[6] for r in m], dtype=np.float64)
        row_of = {int(i): n for n, i in enumerate(ids)}
        self.lib.ids, self.lib.X, self.lib.row_of = ids, X, row_of
        self.ids, self.X, self.Z, self.place_rows, self.places = ids, X, Z, place_rows, Places(place_rows)
        self._has_deleted = False

    def _place_space(self, X):  # precomputed in the bundle
        return self.Z


def load_concepts(path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    with np.load(path, allow_pickle=False) as z:
        return {f"a photo of {n}": v for n, v in zip(z["names"].tolist(), z["T"].astype(np.float32))}


def open_bundle(bundle: str, db_path: str | None = None, dense: bool = False) -> DemoSearcher:
    """Load a bundle (see module docstring). db_path: a writable copy of library.sqlite if the caller needs one."""
    db_path = db_path or os.path.join(bundle, "library.sqlite")
    enc = DemoEncoder(os.path.join(bundle, "models"), load_concepts(os.path.join(bundle, "concepts.npz")))
    ids, X = load_qmatrix(os.path.join(bundle, "vectors.npz"), dense=dense)
    Z = load_qmatrix(os.path.join(bundle, "place.npz"), dense=dense)[1] if os.path.exists(os.path.join(bundle, "place.npz")) else None
    return DemoSearcher(DemoLibrary(bundle, db_path, enc), ids, X, Z)


def manifest(bundle: str) -> dict:
    p = os.path.join(bundle, "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else {}
