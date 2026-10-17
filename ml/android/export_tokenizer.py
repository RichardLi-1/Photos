#!/usr/bin/env python3
"""Export the SigLIP2 (Gemma) BPE tokenizer for the Kotlin implementation (android/.../ml/GemmaTokenizer.kt).

tokenizer.json: normalizer ' ' -> '▁', no pre-split (after normalizing there are no spaces left, so the whole string is
one word), BPE over Unicode scalars with ~580k ranked merges and byte fallback (<0xXX>), then '<eos>', truncated /
padded (pad=0) to 64. open_clip first applies its 'canonicalize' clean (ftfy, html unescape, ASCII punctuation removed,
lowercase, whitespace collapsed); Kotlin reimplements that in TextClean.kt.

Writes android/app/src/main/assets/models/siglip2_tokenizer.bin (little endian):
  'SGTK' u32 version=1 | u32 ctx | u32 pad | u32 eos | u32 unk
  u32 nChars | i32 codepoint[nChars] (sorted) | i32 id[nChars]
  i32 byteId[256] (-1 = none)
  u32 nMerges | i64 key[n] = left<<32|right (sorted) | i32 rank[n] | i32 mergedId[n]
and android/app/src/test/resources/tokenizer_fixtures.json [{text, clean, ids}] from open_clip's own tokenizer.

usage:  PYTHONPATH=ml/android/.pylib search/.venv/bin/python ml/android/export_tokenizer.py
"""
import json
import os
import struct

import numpy as np

from common import ASSETS, CONTEXT_LENGTH, HF_REPO, PARITY_TEXTS, TEST_RES, TOKENIZER_EXTRA, tokenizer


def main():
    from huggingface_hub import hf_hub_download
    from open_clip.tokenizer import _clean_canonicalize
    tj = json.load(open(hf_hub_download(HF_REPO, "tokenizer.json")))
    m = tj["model"]
    assert m["type"] == "BPE" and m["byte_fallback"] and not m["ignore_merges"] and m["fuse_unk"], "unexpected model"
    assert tj["normalizer"] == {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}, tj["normalizer"]
    vocab = m["vocab"]

    # Added (special) tokens are matched before BPE. After canonicalize, text has no ASCII punctuation, tabs or
    # newlines; list the ones that could still occur so we know the Kotlin side can ignore them.
    reachable = [t["content"] for t in tj["added_tokens"] if t["content"] == _clean_canonicalize(t["content"]).replace(" ", "▁") and t["content"]]
    print("added tokens reachable after cleaning:", reachable[:20], len(reachable))

    merges = [tuple(x) if isinstance(x, list) else tuple(x.split(" ", 1)) for x in m["merges"]]
    left = np.array([vocab[a] for a, _ in merges], dtype=np.int64)
    right = np.array([vocab[b] for _, b in merges], dtype=np.int64)
    merged = np.array([vocab[a + b] for a, b in merges], dtype=np.int32)
    keys = (left << 32) | right
    rank = np.arange(len(merges), dtype=np.int32)
    order = np.lexsort((rank, keys))
    keys, rank, merged = keys[order], rank[order], merged[order]
    first = np.concatenate([[True], keys[1:] != keys[:-1]])  # keep the best rank if a pair repeats
    keys, rank, merged = keys[first], rank[first], merged[first]

    chars = sorted((ord(k), v) for k, v in vocab.items() if len(k) == 1)
    byte_ids = [vocab.get(f"<0x{i:02X}>", -1) for i in range(256)]
    os.makedirs(ASSETS, exist_ok=True)
    with open(os.path.join(ASSETS, "siglip2_tokenizer.bin"), "wb") as f:
        f.write(b"SGTK" + struct.pack("<5I", 1, CONTEXT_LENGTH, vocab["<pad>"], vocab["<eos>"], vocab["<unk>"]))
        f.write(struct.pack("<I", len(chars)))
        f.write(np.array([c for c, _ in chars], "<i4").tobytes())
        f.write(np.array([i for _, i in chars], "<i4").tobytes())
        f.write(np.array(byte_ids, "<i4").tobytes())
        f.write(struct.pack("<I", len(keys)))
        f.write(keys.astype("<i8").tobytes())
        f.write(rank.astype("<i4").tobytes())
        f.write(merged.astype("<i4").tobytes())

    texts = PARITY_TEXTS + TOKENIZER_EXTRA
    ids = tokenizer()(texts).tolist()
    os.makedirs(TEST_RES, exist_ok=True)
    fx = [{"text": t, "clean": _clean_canonicalize(t), "ids": i} for t, i in zip(texts, ids)]
    json.dump(fx, open(os.path.join(TEST_RES, "tokenizer_fixtures.json"), "w"), ensure_ascii=False, indent=0)
    print(f"merges {len(keys)}, single-scalar tokens {len(chars)}, fixtures {len(fx)}")


if __name__ == "__main__":
    main()
