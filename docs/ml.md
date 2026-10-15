# Models

All models run locally. Benchmark tables and their reasoning live in the lab notebook, [search/NOTES.md](../search/NOTES.md). This page lists what is used where and summarises what was tried.

## In use

| Model | Used for | Where |
|---|---|---|
| **SigLIP2-B/16-384** (open_clip `ViT-B-16-SigLIP2-384`/`webli`, 768-d) | Image + text embeddings, explain concepts, cleanup zero-shot | Mac engine (`siglip2-b16-384`). iOS: Core ML fp16 (`ml/convert_siglip2.py`, Swift tokenizer). Android: ONNX int8 (`ml/android/`, Kotlin tokenizer). |
| **Same-place head** (ours: `search/data/models/hotel_head.pt`) | "Same room/place" photo search (`Query.mode="place"`) | Mac engine (`photosearch/heads.py`). Not on mobile yet. |
| Qwen3.5-4B, 4-bit MLX (`mlx-community/Qwen3.5-4B-4bit`) | Ask: query plan + visual verification | Mac (`photosearch/ask.py`, `mlx-vlm`) |
| Apple Vision text recognition | OCR. Fast at index time, accurate for re-runs. | Mac (`ocr.py`), iOS (`TextRecognizer.swift`, fast) |
| ML Kit text recognition (bundled) | OCR, off by default (telemetry) | Android |

**Port parity** (cosine vs PyTorch fp32; target ≥ 0.99, checked by `ml/check_siglip2_parity.py`):

- Android int8 ONNX: image min 0.996, text min 0.9988 (`ml/android/parity.json`).
- iOS Core ML: the iOS agent reports 0.9998, but that figure isn't recorded in the repo.

## Own model: the same-place head

A 2-layer residual MLP (768 → 1024 → 512) on the **frozen, already stored** SigLIP2 vectors, so nothing is re-embedded.

- **Loss and data:** multi-similarity loss on `hotel_id`, with P×K batches. Trained on 25.8k Hotels-50K train photos from 10k hotels. **Every test hotel is excluded.**
- **Cost:** about 45 min of feature extraction plus about 1 min of CPU training.

| | hotels2400 R@1 | mAP | Whole library R@1 | Eval same_room hit@1 / MRR |
|---|---|---|---|---|
| SigLIP2 raw | 0.196 | 0.113 | 0.162 | 0 / 0.07 |
| **+ head** | **0.359** | **0.252** | **0.306** | **0.25 / 0.30** |

To reproduce:

```sh
train/run_extract.sh --train --test
.venv/bin/python train/train_head.py --name hotel_head --epochs 20 --loss multisim --lr 3e-4
experiments/bench_head*.py
```

These flags match the `ms` run in `train/sweep.sh`, because the script's default loss is SupCon. The shipped file's exact command isn't recorded.

`train/sweep.sh` runs the hyper-parameter sweep. `train/finetune.py` (last blocks) is untested. Next steps and options, such as more shards, on-device personalisation and a Core ML export, are in NOTES.md.

## Tried and not shipped

| Model | Result | Verdict |
|---|---|---|
| CLIP ViT-B/32 (`laion2b`) | Hotels R@1 0.165, Flickr 0.667 | The old iOS model; replaced by SigLIP2 |
| MobileCLIP2-S2 | Hotels R@1 0.093, Flickr 0.748 | Weak on rooms; research-only license |
| DINOv2 / DINOv3-B | Hotels R@1 0.034 / 0.107 | Inconclusive: an fp16 GeM overflow bug, fixed since (fp32 pooling); not re-run |
| Qwen3-VL-Embedding-2B (MLX 4-bit) | Flickr 0.810 (ties SigLIP2), Hotels R@1 0.155 (worse). Fusion with SigLIP2: Flickr 0.849 (+3 points) | Not shipped: about 3 GB, 1.6 img/s, and only text gains |
| Qwen3-VL-Reranker-2B on the SigLIP2 top-30 | Hotels mAP 0.119 → 0.194 (+63%) at **40 s/query**. Flickr R@5 0.93 → 0.98 at 13 s/query | Not shipped: too slow for interactive search |

The Qwen code stays in place: `encoders.QwenVLEmbed`, `photosearch/rerank.py`, and `experiments/bench_qwen*.py`. Its weights were deleted from the HF cache to free disk, so re-download them before re-running (`PHOTOSEARCH_ALLOW_DOWNLOAD=1` for the reranker). Two gotchas were fixed in the code:

- mlx-vlm caches M-RoPE positions, so text embedded after images came out as garbage.
- The MLX buffer cache was uncapped, so it is now limited to 512 MB.

## Re-running and changing models

Commands are in [development.md](development.md#benchmarks-and-evaluation).

- **Add an encoder:** register it in `encoders.REGISTRY` and set its guard weight in `encoders.get()`. Run `bench.py`, then record the result in NOTES.md.
- **Change the default:** change `MODEL` in `server.py` and `tools.py`, plus the `Library` default, then re-index. Vectors are stored per model (`vectors__<model>.npz`). The head only fits SigLIP2-B/16-384 vectors, so a new encoder needs a newly trained head.

Licenses for every model and dataset are in [data-and-licenses.md](data-and-licenses.md).
