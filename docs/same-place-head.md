# Learning "same place" retrieval on frozen SigLIP2 embeddings

*Photos project, technical note, October 2026*

## Abstract
General-purpose image embeddings retrieve photos that *look alike*, not photos of the *same place*. For queries like "other pictures of this room", a strong vision-language encoder (SigLIP2-B/16-384) ranks a photo of the same hotel first only 19.6% of the time. We train a 1.7M-parameter projection head on the encoder's stored, frozen 768-d embeddings, using a multi-similarity metric-learning loss on Hotels-50K training hotels that are disjoint from the test hotels. The head nearly doubles retrieval quality: R@1 goes from 0.196 to 0.359 and mAP from 0.113 to 0.252. Against a 21,389-photo mixed library, R@1 goes from 0.162 to 0.306, with no loss on generic scene retrieval. Because it operates on embeddings that are already stored, it ships without re-indexing and adds about 2 ms per query.

## 1. Problem
Users ask for *other photos of this place*, but views of the same room differ a lot: bed, bathroom, desk, window. Meanwhile different rooms of the same hotel chain look nearly identical. Off-the-shelf encoders optimise for semantic similarity, which rewards the latter and penalises the former. We test whether a small learned metric on top of a frozen encoder can recover place identity, with three constraints:
- privacy: on-device only,
- no re-embedding of existing libraries,
- negligible latency.

## 2. Method
**Features.** Pooled SigLIP2-B/16-384 image embeddings, L2-normalised and stored as float16. These are the same vectors the search index already stores. The encoder stays frozen.

**Head.** A 2-layer MLP producing a 512-d L2-normalised output, with about 1.7M parameters. A linear 768×768 variant was also trained.

**Loss.** Multi-similarity loss [Wang et al., 2019] with P×K batches (P hotels × K photos each). Positives are photos of the same hotel; negatives are everything else in the batch. Each run trains for about 1 minute on CPU.

**Data.** Hotels-50K training images: 3 shards, 39.9k images. We keep the 25.8k photos from 10k hotels that have 2–4 photos each. *Every hotel in the test set is excluded*, and the training script asserts this. 5% of training hotels are held out for validation.

**Inference.** `Z = head(X)` is computed once over the library (21k vectors in well under a second). A query photo q is ranked by `Z · Z[q]`, plus the engine's shooting-session propagation.

## 3. Evaluation
| Benchmark | Queries / gallery | Relevant | Metric |
|---|---|---|---|
| **hotels2400**: Hotels-50K test (TraffickCam), leave-one-out | 2,400 / 2,400 | same hotel | R@1, R@5, mAP |
| **Whole library**: all hotel photos against the full demo library | 5,390 / 21,389 | same hotel | R@1, mAP |
| **Places365 regression**: generic scene categories | val subset | same scene class | mAP |
| **Product path**: the server's search code, `Query(mode="place")` | 12 "same room" queries from the app's eval set; 300 sampled hotel queries | same hotel | hit@k, MRR |

## 4. Results
| Method | hotels2400 R@1 / R@5 / mAP | Library R@1 / mAP | Places365 mAP |
|---|---|---|---|
| SigLIP2 (raw cosine) | 0.196 / 0.294 / 0.113 | 0.162 / 0.090 | 0.198 |
| **+ MLP head (shipped)** | **0.359 / 0.534 / 0.252** | **0.306 / 0.200** | **0.198** |
| + linear head | 0.369 / 0.549 / 0.258 | 0.315 / 0.204 | 0.192 |
| MLP head fused with raw cosine | 0.310 / 0.455 / 0.198 | 0.289 / 0.181 | 0.202 |

- **In the product,** for the 12 "same room" eval queries: hit@1 0.00 → 0.25, hit@10 0.25 → 0.42, MRR 0.07 → 0.30. The numbers are identical on the 72,887-photo index, so the extra distractors cost nothing.
- **An independent check** of 300 hotel queries through the server path: R@1 0.173 → 0.303.
- **Speed:** a query takes 2.2 ms, against 3.3 ms for the raw path.
- **Off-target results:** non-hotel photos make up 0.3% of the head's top-10 results.

**Other approaches we compared** on hotels2400 (R@1):
- other encoders: CLIP B/32 0.165, MobileCLIP2-S2 0.093, DINOv3-B 0.107;
- SigLIP2 fused with CLIP: 0.208;
- a Qwen3-VL-2B cross-encoder reranker over the top 30: 0.250 → 0.375 on 40 queries, but at **40 s per query**. That's similar quality to the head at roughly 20,000× the cost.

We ship the MLP rather than the linear head. The linear head is about 1 point better on hotels but loses 3% on generic scenes, which matters because "same place" is offered on every photo.

## 5. Ablations

| Variant | mAP (hotels2400) |
|---|---|
| Multi-similarity (shipped) | **0.252** (3 seeds, std ≈ 0.002) |
| SupCon | 0.242 |
| Training on 15k photos instead of 26k | 0.219 |
| Feature noise / dropout augmentation | 0.220 |
| Patch-token mean/GeM pooling (no head) | 0.087–0.089 |
| 4×4 regional matching | 0.084 (+0.001 when fused) |
| Head on pooled + patch features | 0.263, but needs re-embedding the library |

Embedding width (256), a wider MLP, temperature and longer training made little difference. **Data is the main lever:** going from 15k to 26k photos raised mAP by +0.033.

## 6. Limitations
- **The configuration was chosen partly on test numbers.** The held-out validation set, mostly professional travel-site photos, scored every configuration within 0.42–0.43, so it couldn't separate them. Treat the absolute numbers as slightly optimistic. The size of the gain holds across seeds, benchmarks and the whole-library test.
- **Domain:** trained on hotel rooms only. Homes, offices and outdoor places are untested beyond the Places365 regression check.
- **Small product evaluation:** 12 queries. The 5,390-query library benchmark is the stronger evidence.
- **Licensing:** Hotels-50K has no license we could find, so the weights inherit that and aren't product-ready.
- **Not done:** fine-tuning the encoder itself. That would also force re-embedding every library.

## 7. Next steps
- **More data:** stream all 11 training shards, about 130k photos.
- **On-device personalisation:** retrain only the head from a user's own "same place / not this" feedback, plus EXIF sessions and GPS, with no photo ever leaving the phone.
- **iOS:** the head is a small matrix stack on stored embeddings, easy to port to Core ML.

## Reproducibility
Code: `search/train/` (`extract.py`, `train_head.py`, `sweep.sh`), `search/photosearch/heads.py`, and `search/experiments/bench_head{,_library,_places}.py`. Weights: `search/data/models/hotel_head.pt` (gitignored). Logs and per-run numbers are in `search/NOTES.md`. The exact command for the shipped file wasn't recorded; the matching flags are in `sweep.sh` (`--loss multisim`).
