#!/usr/bin/env python3
"""Export the SigLIP2 (Gemma) BPE tokenizer for the Swift implementation in ios/Core/ML/SigLIPTokenizer.swift.

The HF tokenizer.json is: normalizer ' ' -> '▁', no pre-split (the whole string is one word), BPE over Unicode scalars
with 580k ranked merges and byte fallback (<0xXX>), then '<eos>' appended, truncated/padded (pad=0) to 64.
open_clip first applies its 'canonicalize' clean fn (lowercase, ASCII punctuation removed, whitespace collapsed).

Writes (ios/Core/Models/, gitignored, regenerate with this script):
  siglip2_merges.bin      little-endian: u32 n | u64 key[n] (left<<32|right, sorted) | u32 rank[n] | u32 merged_id[n]
  siglip2_tokenizer.json  {context_length, pad_id, eos_id, unk_id, chars: {scalar: id}, bytes: [256 ids, -1 = none]}
and the test fixture ios/PhotosTests/siglip2_token_fixtures.json ({text, ids} for SAMPLE_TEXTS, from HF `tokenizers`).

usage:  ml/.venv/bin/python ml/export_siglip2_tokenizer.py
"""
import json
import os

import numpy as np

from siglip2_common import CONTEXT_LENGTH, FIXTURES, OUT, SAMPLE_TEXTS, tokenize, tokenizer_json_path


def main():
    tj = json.load(open(tokenizer_json_path()))
    m = tj["model"]
    assert m["type"] == "BPE" and m["byte_fallback"] and not m["ignore_merges"], "unexpected tokenizer model"
    assert tj["normalizer"] == {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}, tj["normalizer"]
    vocab = m["vocab"]

    merges = m["merges"]
    left = np.array([vocab[a] for a, _ in merges], dtype=np.uint64)
    right = np.array([vocab[b] for _, b in merges], dtype=np.uint64)
    merged = np.array([vocab[a + b] for a, b in merges], dtype=np.uint32)
    keys = (left << np.uint64(32)) | right
    rank = np.arange(len(merges), dtype=np.uint32)
    # a pair can appear once only; keep the best rank if the file ever repeats one
    order = np.lexsort((rank, keys))
    keys, rank, merged = keys[order], rank[order], merged[order]
    first = np.concatenate([[True], keys[1:] != keys[:-1]])
    keys, rank, merged = keys[first], rank[first], merged[first]

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "siglip2_merges.bin"), "wb") as f:
        f.write(np.uint32(len(keys)).tobytes())
        f.write(keys.astype("<u8").tobytes())
        f.write(rank.astype("<u4").tobytes())
        f.write(merged.astype("<u4").tobytes())

    chars = {str(ord(k)): v for k, v in vocab.items() if len(k) == 1}
    meta = {
        "context_length": CONTEXT_LENGTH,
        "pad_id": vocab["<pad>"], "eos_id": vocab["<eos>"], "unk_id": vocab["<unk>"],
        "chars": chars,
        "bytes": [vocab.get(f"<0x{i:02X}>", -1) for i in range(256)],  # -1: no byte token -> <unk>
    }
    json.dump(meta, open(os.path.join(OUT, "siglip2_tokenizer.json"), "w"), separators=(",", ":"))

    fixtures = [{"text": t, "ids": ids} for t, ids in zip(SAMPLE_TEXTS, tokenize(SAMPLE_TEXTS))]
    json.dump(fixtures, open(FIXTURES, "w"), ensure_ascii=False, indent=0)
    print(f"merges {len(keys)}, single-scalar tokens {len(chars)}, fixtures {len(fixtures)} -> {OUT}")


if __name__ == "__main__":
    main()
