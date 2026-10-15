# search — status (paused 2026-10-15)

Goal: on-device-ish semantic photo search. Any question, text or photo input ("other photos of this room").

## Layout
- `photosearch/encoders.py` — every embedding model behind one interface (`encoders.get("siglip2-b16-384")`)
- `photosearch/ocr.py` — Apple Vision OCR
- `experiments/bench.py ENCODER...` — benchmarks; embeddings cached in `data/cache/`
- `experiments/prepare_data.py`, `experiments/fetch_demo.py` — datasets / demo photos (resumable)

## Run
`cd web && npm run build`, then `cd search && .venv/bin/python server.py` and open http://127.0.0.1:8000 (local only).
UI dev: keep the server running and use `cd web && npm run dev` (http://127.0.0.1:5173, proxies /api and /photo).
Real liquid-glass refraction renders in Chrome/Edge only; Safari gets frosted glass.
Ask: a question in the pill (trailing "?" or how/what/when…, see `web/src/ask.ts`) calls `/api/ask`; `api_ask.py` adds `/api/ask/status` for the progress line.
Browse wall = rows of N (`Virtuoso`); ~2.5 screens down a pinned Story item opens one tile into the product panel (`web/src/components/Story.tsx`).

## Results so far
| encoder | hotels R@1 | hotels mAP | flickr t2i R@1 |
|---|---|---|---|
| clip-b32 (current iOS model) | 0.165 | 0.097 | 0.667 |
| mobileclip2-s2 | 0.093 | 0.053 | 0.748 |
| dinov3-b@336:cls+gem | 0.107 | 0.057 | — |
| **siglip2-b16-384** | **0.196** | **0.113** | **0.818** |
| qwen3vl-emb-2b (MLX 4bit, 320 vis tok) | 0.155 | 0.089 | 0.810 |
| z-fuse siglip2 + 1.0·qwen | 0.188 | 0.110 | 0.849 |
| dinov2-b@336:cls+gem | 0.034 | 0.022 | — |

- SigLIP2-B/16-384 wins both tasks: +19% relative on rooms, +15 points R@1 on text. Current default.
- DINO results are probably a bug, not the model. GeM cubes values in fp16, which overflows past 65504. DINOv2 at 0.034 is implausibly bad. Redo DINO pooling in fp32 (and try `cls` / `mean`) before judging it.
- Demo photos: 15k Unsplash in `data/demo/unsplash/` (778 MB, metadata in `_meta.json`).

- hotels = 2,400 Hotels-50K room photos; find others from the same hotel. Hard: same-chain rooms look identical.
- Apple Vision OCR: 110 ms/img fast, 370 ms accurate. Reads menus and signs. Cheap enough to index everything.
- This Mac is swapping ~10 GB (Edge + VS Code), so ML runs ~10x slow (29 img/s for B/32). Close Edge before long runs.

**Search box (2026-10-15)**: `photosearch/query.py` parses dates ("sept 29 2025", "summer 2018", "from March to May 2021", "last
christmas") and places (`places.py`: pycountry names in 29 languages, codes, US states, ~250 cities/regions, plus the library's
own place names) into HARD filters; the rest is ranked visually (prompt ensemble "x" + "a photo of x"). No visual text left →
filter matches newest first. ~3 ms/query, `experiments/eval_parse.py` = 41 hand-written cases. FTS now ranks OCR only
(no album/place votes); exact OCR phrases go to the top. `eval_search.py --parse` (91 queries, hit@1 / P@10 / MRR):
| | ALL | time | place+object | place |
|---|---|---|---|---|
| before (plain) | .593 / .365 / .643 | .00 / .03 / .05 | .83 / .58 / .86 | 1.0 / .95 / 1.0 |
| after (--parse) | .736 / .499 / .783 | 1.0 / .88 / 1.0 | .67 / .69 / .79 | 1.0 / .98 / 1.0 |
- Remaining place/time "misses" are mostly ground-truth gaps: eval regexes lack aliases ("Кёльн, Германия", "Norvège").
- OCR redone in Vision accurate mode (`Library.ocr_accurate`; fast text kept in table `ocr_fast`) on the 2,427 photos with
  fast text + 1,500 others: 2,427 -> 209 photos with any text (garbage dropped), legible 10 -> 70 (`experiments/eval_ocr.py`);
  text_in_photo hit@1 .93 -> 1.0. Re-runs of the eval reuse cached query embeddings (no model load).

**Agent tools (2026-10-15)**: `photosearch/tools.py`. Tools: search_photos, get_photos, explain_photo, list/get/create/rename/delete_album,
add_to/remove_from_album, trash_photos, restore_photos, list_trash, purge_trash, find_cleanup_candidates. Call `tools.call(name, args)` in-process, or
`GET /api/tools/schema` and `POST /api/tools/{name}` (`api_tools.py`, plus /api/albums|trash|cleanup). Plug in an LLM with `agent_stub.py` (loop = TODO).
Safety: files are never touched. Trash sets `photos.deleted_at`, purge only sets `purged_at` (`purge_files_hook` is unwired). Destructive tools
default to dry_run and return a single-use 15-minute confirm_token bound to the exact ids; a real run needs `dry_run=false` + that token.
Per-call caps (trash 200, purge 500, albums 1000). `audit_log` is written in the same transaction as each change.
Cleanup (`photosearch/cleanup.py`): `python -m photosearch.cleanup stats` precomputes `photo_stats` (resumable, CPU only, ~200 img/s).
Candidates combine sha1, SigLIP cosine (bursts ≥.95 same session+camera ≤120 s, else ≥.978), edge-width blur, histogram and SigLIP zero-shot
with a "defect vs on purpose" veto. Demo library: 36 exact / 124 near-dup groups, 5 blurry, 2 whiteboard, 1 overexposed. Sample precision:
exact 16/16, near-dup 19/24, blurry 5/5. Synthetic recall: screenshot/receipt/document 10/10, but meme/accidental/overexposed are weak.

**Own model: "same room" head (2026-10-15)**: `train/`, `photosearch/heads.py`, weights `data/models/hotel_head.pt` (6.8 MB). A residual MLP 768->1024->512
on the STORED frozen SigLIP2 vectors (no re-embedding), multi-similarity loss on hotel_id, P x K batches. Trained on 25.8k Hotels-50K train photos (10k hotels from
3 HF shards; all 1,577 test hotels excluded). Cost: ~45 min feature extraction (MPS) + ~1 min CPU training. Use the head alone; fusing with raw siglip hurts.
| | hotels2400 R@1 / R@5 / mAP | whole library (5,390 hotel queries vs 21k photos) R@1 / mAP | Places365 scene mAP |
|---|---|---|---|
| siglip2 raw | .196 / .294 / .113 | .162 / .090 | .198 |
| + hotel_head (MLP) | **.359 / .534 / .252** | **.306 / .200** | .198 (no regression) |
| + hotel_head_linear (one 768x768 matrix) | .369 / .549 / .258 | .315 / .204 | .192 |
- SupCon .242 < multi-sim .252 (3 seeds, std ~.002). Patch-mean/GeM inputs add +.01 but need re-embedding. Raw GeM/mean/region-Chamfer pooling all lose to pooled.
- Data is the lever: mAP .219 at 15k photos -> .252 at 26k. Next: stream all 11 shards one at a time (~130k photos). Fine-tuning blocks (`train/finetune.py`) is an untested draft.
- On-device personalisation (private): the user's "same place / not this" taps plus EXIF sessions/GPS clusters as free labels; retrain only this head (1.7M params, seconds on-device), photos never leave the device.
- iOS: the head is a matmul or small MLP on the stored embedding (Core ML or plain Accelerate). Or distil SigLIP2+head into MobileCLIP2-S2 with a feature-regression loss on unlabeled user photos.
- Data needed: many places, each with several photos (hotels, homes, venues, landmarks), phone-camera domain like TraffickCam; hard negatives from same-chain/same-style rooms.

## Next
1. Fix DINO pooling (fp32), rerun `dinov3-b@336:cls` etc. Delete stale `data/cache/*dino*` first
2. On the cached embeddings: fuse DINO+SigLIP scores, query expansion, patch-level re-rank for "same room"
3. Try Qwen3-VL-Embedding-2B (LLM text side, should handle complex queries)
4. Engine: SQLite + FTS5 (OCR, metadata) + vector matrices; hybrid ranking with metadata filters
5. Ask: local VLM (Qwen3-VL-4B / Qwen3.5 via mlx-vlm) plans the query, then looks at a contact sheet of the top hits
6. Demo: index Unsplash (`fetch_demo.py`, 15k) + hotels + flickr; bare HTML front end on FastAPI
7. Search box: same_room is still 0.25 hit@10 (needs DINO / re-rank); "a photo of" template alone trades P@10 for hit@1, ensemble kept
8. Parser: no "near X" radius, no "recently"; bare "Manhattan" also matches Manhattan Beach, CA

## Qwen3-VL (2026-10-15)
- Embedding-2B: ties SigLIP2 on text (0.810 vs 0.818), worse on rooms; 1.6 img/s, ~3 GB. Fusion +3 pts flickr only. Not shipped.
- Reranker-2B on SigLIP2 top-30: flickr n=100 R@1 0.81→0.82, R@5 0.93→0.98 (13 s/query); hotels n=40 R@1 0.25→0.375, mAP 0.119→0.194 (40 s/query).
- Gotchas fixed: mlx-vlm caches M-RoPE positions (text after images = garbage); uncapped MLX buffer cache grew to 11 GB.
- Repro: `experiments/bench_qwen.py validate|bench|fuse`, `bench_qwen_rerank.py --flickr 100 --hotels 40 --eval`, `bench_qwen_library.py embed [N] / eval`.
