"""Image/text encoders behind one interface. All outputs are L2-normalized float32."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from . import guard

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32


def load_image(path: str) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)  # phone photos are stored rotated
    return im.convert("RGB")


class _Images(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]
        return self.tf(x if isinstance(x, Image.Image) else load_image(x))


def _norm(x: torch.Tensor) -> np.ndarray:
    return F.normalize(x.float(), dim=-1).cpu().numpy()


class Encoder:
    name: str
    has_text = False

    def _batches(self, items, tf, bs):
        workers = 0 if items and isinstance(items[0], Image.Image) else 1
        return torch.utils.data.DataLoader(_Images(items, tf), batch_size=bs, num_workers=workers)

    @torch.inference_mode()
    def images(self, items, bs=32) -> np.ndarray:
        return np.concatenate(list(self.iter_images(items, bs)))

    def iter_images(self, items, bs=32):
        """Yield one embedding batch at a time from a single loader (worker startup is expensive)."""
        with torch.inference_mode():
            for x in self._batches(items, self.tf, bs):
                guard.check()
                yield self._embed(x.to(DEVICE, DTYPE))


class OpenClip(Encoder):
    """CLIP-family dual encoders (text + image in one space)."""
    has_text = True

    def __init__(self, arch: str, pretrained: str):
        import open_clip
        self.name = f"{arch}:{pretrained}"
        self.model, _, self.tf = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=DEVICE)
        self.model.to(DTYPE).eval()
        self.tok = open_clip.get_tokenizer(arch)

    def _embed(self, x):
        return _norm(self.model.encode_image(x))

    @torch.inference_mode()
    def texts(self, texts, bs=256) -> np.ndarray:
        out = [_norm(self.model.encode_text(self.tok(texts[i:i + bs]).to(DEVICE))) for i in range(0, len(texts), bs)]
        return np.concatenate(out)


class Dino(Encoder):
    """Self-supervised ViT (DINOv2/v3 via timm). Image-only; strong at instance/place matching.

    pool: 'cls' | 'gem' (GeM over patch tokens) | 'cls+gem' (concat, each normalized).
    """

    def __init__(self, timm_name: str, res: int = 448, pool: str = "cls+gem", p: float = 3.0):
        import timm
        from torchvision import transforms as T
        self.name = f"{timm_name}@{res}:{pool}"
        self.model = timm.create_model(timm_name, pretrained=True, num_classes=0, dynamic_img_size=True).to(DEVICE, DTYPE).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.tf = T.Compose([T.Resize((res, res), interpolation=T.InterpolationMode.BICUBIC),
                             T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
        self.pool, self.p, self.res = pool, p, res

    def tokens(self, x):
        t = self.model.forward_features(x).float()  # pool in fp32: GeM's pow(3) overflows fp16
        n = self.model.num_prefix_tokens
        return t[:, 0], t[:, n:]  # cls, patches

    def pools(self, cls, patches) -> dict[str, torch.Tensor]:
        gem = patches.clamp(min=1e-6).pow(self.p).mean(1).pow(1 / self.p)
        return {"cls": cls, "mean": patches.mean(1), "gem": gem,
                "cls+gem": torch.cat([F.normalize(cls, dim=-1), F.normalize(gem, dim=-1)], -1),
                "cls+mean": torch.cat([F.normalize(cls, dim=-1), F.normalize(patches.mean(1), dim=-1)], -1)}

    def _embed(self, x):
        return _norm(self.pools(*self.tokens(x))[self.pool])


class QwenVLEmbed(Encoder):
    """Qwen3-VL-Embedding on MLX: VLM, last-token pooling. Runs fully local (weights from the HF cache).

    Format follows the official Qwen3VLEmbedder: system turn = instruction, user turn = image or text,
    then the generation prompt; the embedding is the hidden state of the final token ("\\n" after
    "assistant"). Documents (images) use the default instruction; text queries get the retrieval one.
    Images are resized to <= max_tokens visual tokens (one token per 32x32 px).
    """
    has_text = True
    DOC = "Represent the user's input."
    QUERY = "Retrieve images or text relevant to the user's query."

    def __init__(self, repo="mlx-community/Qwen3-VL-Embedding-2B-4bit", max_tokens=320):
        import os
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from mlx_vlm.encoder_loader import load_encoder_model
        from transformers import AutoProcessor
        path = Path(snapshot_download(repo, local_files_only=os.environ.get("PHOTOSEARCH_ALLOW_DOWNLOAD") != "1"))
        self.name, self.max_px = f"{repo}@{max_tokens}", max_tokens * 32 * 32
        import mlx.core as mx
        mx.set_cache_limit(512 << 20)  # every batch has a new padded shape: an uncapped buffer cache grew to 11 GB
        self.model = load_encoder_model(path, model_remapping={"qwen3_vl": "qwen3_vl_embedding"})
        self.proc = AutoProcessor.from_pretrained(path)
        self.proc.tokenizer.padding_side = "right"

    @staticmethod
    def _prompt(instr, content):
        return f"<|im_start|>system\n{instr}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def _resize(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        s = min(1.0, (self.max_px / (w * h)) ** 0.5)
        w, h = max(32, int(w * s) // 32 * 32), max(32, int(h * s) // 32 * 32)  # multiples of 32: processor won't resize again
        return im.resize((w, h), Image.BICUBIC)

    def _run(self, **enc) -> np.ndarray:
        import mlx.core as mx
        lm = self.model.language_model  # mlx-vlm caches M-RoPE positions and only resets them for image inputs:
        lm._position_ids = lm._rope_deltas = None  # stale image positions would silently corrupt later text batches
        out = self.model(**{k: mx.array(v) for k, v in enc.items()}).text_embeds
        return np.array(out.astype(mx.float32))

    def iter_images(self, items, bs=8):
        from concurrent.futures import ThreadPoolExecutor
        def prep(x):
            return self._resize(x if isinstance(x, Image.Image) else load_image(x))
        prompt = self._prompt(self.DOC, "<|vision_start|><|image_pad|><|vision_end|>")
        with ThreadPoolExecutor(4) as pool:
            ims = pool.map(prep, items)  # lazily prefetches while the GPU runs
            for i in range(0, len(items), bs):
                guard.check()
                batch = [next(ims) for _ in range(min(bs, len(items) - i))]
                enc = self.proc(text=[prompt] * len(batch), images=batch, padding=True, return_tensors="np")
                yield _norm_np(self._run(**{k: enc[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}))

    def images(self, items, bs=8) -> np.ndarray:
        return np.concatenate(list(self.iter_images(items, bs)))

    def texts(self, texts, bs=64, instruction=QUERY) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            enc = self.proc.tokenizer([self._prompt(instruction, t) for t in texts[i:i + bs]], padding=True, return_tensors="np")
            out.append(_norm_np(self._run(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])))
        return np.concatenate(out)


def _norm_np(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


REGISTRY = {
    "clip-b32": lambda: OpenClip("ViT-B-32", "laion2b_s34b_b79k"),  # what the iOS app ships today
    "mobileclip2-s2": lambda: OpenClip("MobileCLIP2-S2", "dfndr2b"),
    "mobileclip2-b": lambda: OpenClip("MobileCLIP2-B", "dfndr2b"),
    "siglip2-b16-384": lambda: OpenClip("ViT-B-16-SigLIP2-384", "webli"),
    "siglip2-so400m-384": lambda: OpenClip("ViT-SO400M-16-SigLIP2-384", "webli"),
    "dinov2-b": lambda: Dino("vit_base_patch14_reg4_dinov2.lvd142m"),
    "dinov3-b": lambda: Dino("vit_base_patch16_dinov3.lvd1689m"),
    "dinov3-l": lambda: Dino("vit_large_patch16_dinov3.lvd1689m"),
    "qwen3vl-emb-2b": lambda: QwenVLEmbed("mlx-community/Qwen3-VL-Embedding-2B-4bit"),
}


def get(name: str) -> Encoder:
    guard.acquire(2 if name.startswith(("qwen", "siglip2-so400m", "dinov3-l")) else 1)  # ML budget, see guard.py
    if name in REGISTRY:
        return REGISTRY[name]()
    if name.startswith(("dinov3-b@", "dinov3-l@", "dinov2-b@")):  # e.g. dinov3-b@336:gem
        base, rest = name.split("@")
        res, pool = rest.split(":")
        timm_name = {"dinov3-b": "vit_base_patch16_dinov3.lvd1689m", "dinov3-l": "vit_large_patch16_dinov3.lvd1689m",
                     "dinov2-b": "vit_base_patch14_reg4_dinov2.lvd142m"}[base]
        return Dino(timm_name, int(res), pool)
    raise KeyError(name)
