#!/usr/bin/env python3
"""Convert SigLIP2-B/16-384 (open_clip ViT-B-16-SigLIP2-384 / webli) to Core ML mlprograms for iOS 18 (fp16).

  SigLIP2ImageEncoder.mlpackage  input  "image": RGB 384x384 image (squash-resize the photo, no crop);
                                          normalized in-model: x = pixel/127.5 - 1 (SigLIP mean = std = 0.5)
                                 output "embedding": float32 (1,768), L2-normalized
  SigLIP2TextEncoder.mlpackage   input  "input_ids": int32 (1,64) from SigLIPTokenizer (pad 0, '<eos>'=1 last)
                                 output "embedding": float32 (1,768), L2-normalized
  siglip2_meta.json              logit_scale (exp'd) / logit_bias for calibrated probabilities, sizes, ids
  siglip2_concepts.json/.f16     the explain.py concept vocabulary + fp16 (n,768) text embeddings of
                                 "a photo of {concept}" (for the "why it matched" chips)

usage (repo root; one part per process keeps peak RAM low; each holds 2 units of the shared ML guard):
  ml/.venv/bin/python ml/convert_siglip2.py --part image
  ml/.venv/bin/python ml/convert_siglip2.py --part text
  ml/.venv/bin/python ml/convert_siglip2.py --part concepts
"""
import argparse
import json
import os
import shutil

import numpy as np

from siglip2_common import (CONTEXT_LENGTH, DIM, IMAGE_PKG, IMAGE_SIZE, OUT, ROOT, TEXT_PKG, calibration, guard,
                            load_model, tokenize, towers)


def _convert(module, example, inputs, path):
    import coremltools as ct
    import torch
    traced = torch.jit.trace(module, example)
    guard.check()
    ml = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    ml.author = "SigLIP2-B/16-384 (open_clip ViT-B-16-SigLIP2-384, webli), converted by ml/convert_siglip2.py"
    ml.license = "Apache-2.0 (SigLIP2 weights)"
    guard.check()
    if os.path.exists(path):
        shutil.rmtree(path)
    ml.save(path)
    print("saved", path)


def convert_image(model):
    import coremltools as ct
    import torch
    image_tower, _ = towers(model)
    _convert(image_tower, torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE) * 2 - 1,
             [ct.ImageType(name="image", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE), scale=1 / 127.5, bias=[-1, -1, -1],
                           color_layout=ct.colorlayout.RGB)],
             IMAGE_PKG)


def convert_text(model):
    import coremltools as ct
    import torch
    _, text_tower = towers(model)
    ids = torch.tensor(tokenize(["a photo of a dog"]), dtype=torch.int32)
    _convert(text_tower, ids, [ct.TensorType(name="input_ids", shape=(1, CONTEXT_LENGTH), dtype=np.int32)], TEXT_PKG)


def export_concepts(model):
    import torch
    from photosearch.explain import CONCEPTS  # single source of truth: the prototype's vocabulary (read-only)
    _, text_tower = towers(model)
    prompts = [f"a photo of {c}" for c in CONCEPTS]
    X = torch.cat([text_tower(torch.tensor(tokenize(prompts[i:i + 64]))) for i in range(0, len(prompts), 64)])
    X.numpy().astype("<f2").tofile(os.path.join(OUT, "siglip2_concepts.f16"))
    json.dump({"prompt": "a photo of {}", "dim": DIM, "concepts": CONCEPTS},
              open(os.path.join(OUT, "siglip2_concepts.json"), "w"), indent=0)
    print(f"concepts: {len(CONCEPTS)}")


def write_meta(model):
    meta = {"model": "ViT-B-16-SigLIP2-384/webli", "image_size": IMAGE_SIZE, "context_length": CONTEXT_LENGTH,
            "dim": DIM, "resize": "squash, bicubic", "normalize": "x/127.5 - 1 (in-model)", **calibration(model)}
    json.dump(meta, open(os.path.join(OUT, "siglip2_meta.json"), "w"), indent=1)
    print(meta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["image", "text", "concepts"], required=True)
    a = p.parse_args()
    guard.acquire(2)
    os.makedirs(OUT, exist_ok=True)
    model, _ = load_model()
    guard.check()
    write_meta(model)
    {"image": convert_image, "text": convert_text, "concepts": export_concepts}[a.part](model)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
