"""Torch-free SigLIP2-B/16-384 text encoder: HF `tokenizers` + int8 ONNX text tower + a quantized token-embedding table.

Same pipeline as the Android app (ml/android/convert_siglip2_onnx.py), in Python:
  open_clip "canonicalize" clean -> Gemma BPE (tokenizer.json, <eos>, pad 0 to 64) -> table lookup -> ONNX tower -> unit vector

Model dir layout (written by build_bundle.py):
  siglip2_tokenizer.bin     Gemma BPE as sorted arrays (ml/android/export_tokenizer.py; from timm/ViT-B-16-SigLIP2-384,
                            Apache-2.0). Fallback: tokenizer.json via HF `tokenizers` (same ids, ~470 MB more RSS)
  siglip2_text_q8.onnx      token_embeds float[N,64,768] -> embedding float[N,768] (unit length)
  token_embed.bin           'SGP8' (int8 rows reachable after clean(), ~137 MB), 'SGE8' (all rows, Android's) or
                            'SGE4' (int4; measured too lossy), memory-mapped
  calibration.json          logit_scale / logit_bias (SigLIP sigmoid head)

Deps: numpy, onnxruntime, ftfy. No torch.
"""
from __future__ import annotations

import html
import json
import os
import string
import struct
import threading
from collections import OrderedDict

import numpy as np

CONTEXT_LENGTH, DIM = 64, 768
_PUNCT = str.maketrans("", "", string.punctuation)


def clean(text: str) -> str:
    """open_clip.tokenizer._clean_canonicalize: ftfy + html unescape, '_' -> ' ', ASCII punctuation removed, lower,
    whitespace collapsed."""
    try:
        import ftfy
        text = ftfy.fix_text(text)
    except ImportError:  # ftfy only fixes mojibake; the demo image ships it, keep working without it
        pass
    text = html.unescape(html.unescape(text)).strip()
    text = text.replace("_", " ").translate(_PUNCT).lower()
    return " ".join(text.split()).strip()


# ---------------------------------------------------------------- token-embedding tables

def write_q8(path: str, W: np.ndarray):
    """'SGE8' | u32 rows | u32 dim | f32 scale[rows] | i8 q[rows*dim]. value = q * scale (same as Android)."""
    scale = np.abs(W).max(1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(W / scale[:, None]), -127, 127).astype(np.int8)
    with open(path, "wb") as f:
        f.write(b"SGE8" + struct.pack("<II", *W.shape))
        f.write(scale.astype("<f4").tobytes())
        f.write(q.tobytes())


def reachable_ids(tokenizer_json: dict) -> np.ndarray:
    """Token ids that can occur after clean(): lower case, no ASCII punctuation, single spaces (BPE only merges
    substrings of its input), plus byte-fallback and pad/eos/bos/unk. ~70% of SigLIP2's 256k vocabulary."""
    vocab = tokenizer_json["model"]["vocab"]
    added = {t["id"] for t in tokenizer_json["added_tokens"]}
    keep = []
    for tok, i in vocab.items():
        if i in added:
            ok = tok in ("<pad>", "<eos>", "<bos>", "<unk>")
        elif tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
            ok = True
        else:
            s = tok.replace("\u2581", " ")
            ok = (s == s.lower() and not any(c in string.punctuation for c in s) and "  " not in s
                  and not any(c.isspace() and c != " " for c in s))
        if ok:
            keep.append(i)
    return np.array(sorted(keep), dtype=np.int64)


def write_pruned_q8(path: str, W: np.ndarray, keep: np.ndarray, fallback: int):
    """'SGP8' | u32 rows | u32 kept | u32 dim | i32 remap[rows] | f32 scale[kept] | i8 q[kept*dim].

    Same int8 rows as 'SGE8' for the kept ids; any other id reads row remap[fallback] (<unk>), which clean() makes
    unreachable in practice (measure.py checks 5.9k texts)."""
    rows, dim = W.shape
    remap = np.full(rows, -1, np.int32)
    remap[keep] = np.arange(len(keep), dtype=np.int32)
    remap[remap < 0] = remap[fallback]
    Wk = W[keep].astype(np.float32)
    scale = np.abs(Wk).max(1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(Wk / scale[:, None]), -127, 127).astype(np.int8)
    with open(path, "wb") as f:
        f.write(b"SGP8" + struct.pack("<III", rows, len(keep), dim))
        f.write(remap.astype("<i4").tobytes())
        f.write(scale.astype("<f4").tobytes())
        f.write(q.tobytes())


def write_q4(path: str, W: np.ndarray, group: int = 32, chunk: int = 16384):
    """'SGE4' | u32 rows | u32 dim | u32 group | f16 scale[rows, dim/group] | u8 packed[rows*dim/2].

    Symmetric int4 in [-7, 7] per group of `group` dims; two values per byte (low nibble = even column).
    """
    rows, dim = W.shape
    assert dim % group == 0 and group % 2 == 0
    with open(path, "wb") as f:
        f.write(b"SGE4" + struct.pack("<III", rows, dim, group))
        scales, packed = [], []
        for a in range(0, rows, chunk):
            w = W[a:a + chunk].astype(np.float32).reshape(-1, dim // group, group)
            s = np.abs(w).max(-1) / 7.0
            s[s == 0] = 1.0
            s = s.astype(np.float16)
            q = np.clip(np.rint(w / s.astype(np.float32)[..., None]), -7, 7).astype(np.int8).reshape(-1, dim)
            u = (q + 8).astype(np.uint8)
            packed.append((u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8))
            scales.append(s)
        f.write(np.concatenate(scales).astype("<f2").tobytes())
        for p in packed:
            f.write(p.tobytes())


class TokenTable:
    """Memory-mapped quantized table: only the rows a query touches are paged in (RSS stays small)."""

    def __init__(self, path: str):
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        magic = bytes(raw[:4])
        if magic == b"SGE8":
            rows, dim = struct.unpack("<II", bytes(raw[4:12]))
            self.kind, off = 8, 12
            self.scale = np.frombuffer(raw, "<f4", rows, off)
            self.q = np.frombuffer(raw, np.int8, rows * dim, off + 4 * rows).reshape(rows, dim)
        elif magic == b"SGP8":
            rows, kept, dim = struct.unpack("<III", bytes(raw[4:16]))
            self.kind, off = 8, 16
            self.remap = np.frombuffer(raw, "<i4", rows, off)
            off += 4 * rows
            self.scale = np.frombuffer(raw, "<f4", kept, off)
            self.q = np.frombuffer(raw, np.int8, kept * dim, off + 4 * kept).reshape(kept, dim)
        elif magic == b"SGE4":
            rows, dim, group = struct.unpack("<III", bytes(raw[4:16]))
            self.kind, self.group, off = 4, group, 16
            ng = dim // group
            self.scale = np.frombuffer(raw, "<f2", rows * ng, off).reshape(rows, ng)
            self.q = np.frombuffer(raw, np.uint8, rows * dim // 2, off + 2 * rows * ng).reshape(rows, dim // 2)
        else:
            raise ValueError(f"{path}: unknown token table format {magic!r}")
        self.rows, self.dim = rows, dim
        self.remap = getattr(self, "remap", None)

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        flat = ids.reshape(-1)
        if self.remap is not None:
            flat = self.remap[flat]
        if self.kind == 8:
            out = self.q[flat].astype(np.float32) * self.scale[flat][:, None]
        else:
            p = self.q[flat]
            v = np.empty((len(flat), self.dim), np.int8)
            v[:, 0::2] = (p & 0x0F).astype(np.int8) - 8
            v[:, 1::2] = (p >> 4).astype(np.int8) - 8
            s = self.scale[flat].astype(np.float32)
            out = (v.reshape(len(flat), -1, self.group) * s[..., None]).reshape(len(flat), self.dim)
        return out.reshape(*ids.shape, self.dim).astype(np.float32)


# ---------------------------------------------------------------- tokenizer

class GemmaBPE:
    """SigLIP2's Gemma BPE from the compact 'SGTK' file (ml/android/export_tokenizer.py), ~10 MB of RAM.

    HF `tokenizers` holds the 256k vocab + 580k merges in ~470 MB of RSS; this keeps them as sorted numpy arrays.
    Same algorithm as tokenizer.json: ' ' -> '\u2581', one word, chars -> ids (byte fallback <0xXX>), repeatedly merge
    the lowest-rank adjacent pair (leftmost first), then '<eos>', truncate to 64 keeping <eos>, pad with 0.
    """

    def __init__(self, path: str):
        raw = open(path, "rb").read()
        assert raw[:4] == b"SGTK", path
        ver, self.ctx, self.pad, self.eos, self.unk = struct.unpack_from("<5I", raw, 4)
        off = 24
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        cps = np.frombuffer(raw, "<i4", n, off); off += 4 * n
        cid = np.frombuffer(raw, "<i4", n, off); off += 4 * n
        self.char = dict(zip(cps.tolist(), cid.tolist()))
        self.byte = np.frombuffer(raw, "<i4", 256, off).tolist(); off += 4 * 256
        (m,) = struct.unpack_from("<I", raw, off); off += 4
        self.keys = np.frombuffer(raw, "<i8", m, off); off += 8 * m
        self.rank = np.frombuffer(raw, "<i4", m, off); off += 4 * m
        self.merged = np.frombuffer(raw, "<i4", m, off)

    def _lookup(self, left: np.ndarray, right: np.ndarray):
        k = (left.astype(np.int64) << 32) | right.astype(np.int64)
        i = np.searchsorted(self.keys, k)
        i = np.minimum(i, len(self.keys) - 1)
        ok = self.keys[i] == k
        return np.where(ok, self.rank[i], np.iinfo(np.int32).max), self.merged[i]

    def encode(self, text: str) -> list[int]:
        syms: list[int] = []
        for ch in text.replace(" ", "\u2581"):
            i = self.char.get(ord(ch))
            if i is not None:
                syms.append(i)
            else:
                bs = [self.byte[b] for b in ch.encode("utf-8")]
                syms.extend(bs if all(b >= 0 for b in bs) else [self.unk])
        s = np.array(syms, np.int64)
        while len(s) > 1:
            r, mg = self._lookup(s[:-1], s[1:])
            j = int(np.argmin(r))  # lowest rank, leftmost on ties
            if r[j] == np.iinfo(np.int32).max:
                break
            s = np.concatenate([s[:j], [mg[j]], s[j + 2:]])
        ids = s.tolist()[: self.ctx - 1] + [self.eos]
        return ids + [self.pad] * (self.ctx - len(ids))


# ---------------------------------------------------------------- encoder

class TextEncoder:
    def __init__(self, model_dir: str, threads: int | None = None, cache: int = 2048):
        import onnxruntime as ort
        self.dir = model_dir
        sgtk = os.path.join(model_dir, "siglip2_tokenizer.bin")
        if os.path.exists(sgtk):
            self.bpe, self.tok = GemmaBPE(sgtk), None
        else:  # HF tokenizers + tokenizer.json: identical ids, ~470 MB more RSS
            from tokenizers import Tokenizer
            self.bpe, self.tok = None, Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
            self.tok.enable_truncation(CONTEXT_LENGTH)
            self.tok.enable_padding(length=CONTEXT_LENGTH, pad_id=0, pad_token="<pad>")
        table = next(p for p in ("token_embed.bin", "siglip2_token_embed_q8.bin") if os.path.exists(os.path.join(model_dir, p)))
        self.table = TokenTable(os.path.join(model_dir, table))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or int(os.environ.get("ORT_THREADS", "0")) or min(4, os.cpu_count() or 1)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = False  # small, variable batches: don't keep per-shape arenas around
        so.enable_cpu_mem_arena = False  # measured: -50 MB RSS for ~1 ms per query
        self.sess = ort.InferenceSession(os.path.join(model_dir, "siglip2_text_q8.onnx"), so, providers=["CPUExecutionProvider"])
        cal = os.path.join(model_dir, "calibration.json")
        self.calibration = json.load(open(cal)) if os.path.exists(cal) else {}
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_n, self._lock = cache, threading.Lock()

    def ids(self, texts: list[str]) -> np.ndarray:
        if self.bpe is not None:
            return np.array([self.bpe.encode(clean(t)) for t in texts], dtype=np.int64)
        return np.array([e.ids for e in self.tok.encode_batch([clean(t) for t in texts])], dtype=np.int64)

    def _run(self, texts: list[str]) -> np.ndarray:
        out = self.sess.run(None, {"token_embeds": self.table(self.ids(texts))})[0].astype(np.float32)
        return out / np.linalg.norm(out, axis=-1, keepdims=True)

    def texts(self, texts: list[str], bs: int = 16) -> np.ndarray:
        """Unit-length float32 [N, 768]. Single queries are LRU-cached in memory (never written anywhere)."""
        res: list[np.ndarray | None] = [None] * len(texts)
        todo = []
        with self._lock:
            for n, t in enumerate(texts):
                v = self._cache.get(t)
                if v is not None:
                    self._cache.move_to_end(t)
                    res[n] = v
                else:
                    todo.append(n)
        for a in range(0, len(todo), bs):
            part = todo[a:a + bs]
            E = self._run([texts[n] for n in part])
            with self._lock:
                for n, v in zip(part, E):
                    res[n] = v
                    self._cache[texts[n]] = v
                while len(self._cache) > self._cache_n:
                    self._cache.popitem(last=False)
        return np.stack(res) if res else np.zeros((0, DIM), np.float32)
