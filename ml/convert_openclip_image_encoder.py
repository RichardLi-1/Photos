#!/usr/bin/env python3
"""
convert_openclip_image_encoder.py

Core idea (for coremltools==9.0):
- Use torch.export ExportedProgram directly (NO torch.jit.trace)
- Decompose aten._native_multi_head_attention into primitive ops
- Convert ExportedProgram with coremltools using source="pytorch"
- Let coremltools infer outputs (do NOT pass outputs=)

IMPORTANT version note:
- This approach is intended for coremltools 9.0 with a newer torch export API.
  If you are on torch 2.2.x, upgrade torch/torchvision (e.g., torch==2.7.0) or this may fail.

Input contract:
- "input": float32 tensor shape (1, 3, 224, 224), already CLIP-normalized
Output:
- single tensor embedding (1, D) (D=512 for ViT-B-32)

Usage:
  python convert_openclip_image_encoder.py \
    --model "ViT-B-32" \
    --pretrained "laion2b_s34b_b79k" \
    --out ImageEncoder.mlpackage
"""

import argparse
import numpy as np
import torch
import coremltools as ct
import open_clip
from torch._decomp import get_decompositions


class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.clip = clip_model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: float32 (1,3,224,224), pre-normalized in Swift
        with torch.no_grad():
            y = self.clip.encode_image(x)
            return y.to(torch.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name")
    p.add_argument("--pretrained", required=True, help="OpenCLIP pretrained tag")
    p.add_argument("--out", default="ImageEncoder.mlpackage", help="Output .mlpackage path")
    p.add_argument("--img_size", type=int, default=224, help="Input size (default 224)")
    p.add_argument("--deployment", choices=["iOS16", "iOS17", "iOS18"], default="iOS17")
    p.add_argument("--precision", choices=["float32", "float16"], default="float32")
    args = p.parse_args()

    torch.set_grad_enabled(False)

    # Build OpenCLIP model (CPU for export/conversion)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.model,
        pretrained=args.pretrained,
        device="cpu",
    )
    clip_model.eval()

    wrapper = ImageEncoderWrapper(clip_model).cpu().eval()

    example = torch.randn(1, 3, args.img_size, args.img_size, dtype=torch.float32)

    # Decompose native attention into primitives to avoid coremltools op gaps
    aten_native_mha = torch.ops.aten._native_multi_head_attention.default
    decomps = get_decompositions([aten_native_mha])

    # Export graph and apply decompositions
    exported = torch.export.export(wrapper, (example,))
    exported = exported.run_decompositions(decomps)

    deployment_map = {"iOS16": ct.target.iOS16, "iOS17": ct.target.iOS17, "iOS18": ct.target.iOS18}
    minimum_deployment_target = deployment_map[args.deployment]

    compute_precision = ct.precision.FLOAT32 if args.precision == "float32" else ct.precision.FLOAT16

    # Convert ExportedProgram directly (NO outputs=, let coremltools infer)
    mlmodel = ct.convert(
        exported,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input", shape=example.shape, dtype=np.float32)],
        compute_precision=compute_precision,
        minimum_deployment_target=minimum_deployment_target,
    )

    mlmodel.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
