# Development

The target machine is an Apple Silicon Mac: OCR uses Apple Vision through pyobjc, and Ask uses MLX. Docker doesn't apply for that reason. The engine was developed on a 16 GB M4.

## Mac engine + web (Makefile)

```sh
make setup                      # search/.venv from search/requirements.txt, SigLIP2 download, web build
make index DIR=~/Pictures/Trip  # OCR + embeddings into search/data/index   [INDEX=search/data/index-xl]
make start                      # http://127.0.0.1:8000 + watchdog; serves search/data/index-xl if built
make stage [INDEX=...]          # staging copy on http://127.0.0.1:8100 (default: the 21k index)
make dev                        # Vite on http://127.0.0.1:5173 with hot reload, proxies /api and /photo to :8000
make service / make unservice   # per-user launchd agent that keeps `make start` running (logs: search/data/server.log)
make test                       # pytest, model-free (~290 tests in search/tests)
make check [ARGS=--ios|--slow]  # tests + web typecheck/build/unit tests (+ iOS build / real-SigLIP smoke test)
```

- **Indexing** is incremental. `index.py FOLDER=ALBUM` names the source folder.
- **Cleanup stats:** `.venv/bin/python -m photosearch.cleanup stats` precomputes them on the CPU and can resume.
- **Ask** needs `mlx-community/Qwen3.5-4B-4bit` in the HF cache. Run it once with `HF_HUB_OFFLINE=0` to download it.
- **Same-place mode** needs `search/data/models/hotel_head.pt`. It is gitignored, so recreate it with `train/` (see [ml.md](ml.md)). Without the file, place mode falls back to plain similarity.
- **XL demo data:** `experiments/fetch_unsplash_more.py` (Unsplash Lite rows 15k–25k) and `experiments/fetch_xl.py places coco`. Both check free disk first. Then index into `search/data/index-xl`.
- **Web:** `npm run build` writes `web/dist`, and `npm test` runs `web/tests/*.test.ts`. Real liquid-glass refraction (an SVG displacement map inside `backdrop-filter`) works only in Chromium; Safari and Firefox get frosted glass. `Glass.tsx` records the Chromium quirks:
  - use only one `feImage` per filter
  - size the map in px, not %
  - put filter primitives directly under `<filter>`

  Add `?stripes` to the URL for a high-contrast backdrop.

## iOS

The project is `ios/Photos.xcodeproj` (iOS 18.5, Xcode 16.4). Generate the Core ML models with `ml/` first (see the README's ml section). Build through the guard:

```sh
python3 ios/scripts/guarded_xcodebuild.py -project ios/Photos.xcodeproj -scheme Photos \
  -destination 'generic/platform=iOS Simulator' build
```

`ios/Features/Shared/GlassBackground.swift` is the one place to switch to `.glassEffect()` on Xcode 26. A simulator end-to-end run and the unit tests haven't been verified yet.

## Android

See [android/README.md](../android/README.md). In short:

1. Build the models: `ml/android/convert_siglip2_onnx.py all`, then `export_tokenizer.py`.
2. Build and test: `python3 tools/guarded.py -- ./gradlew assembleDebug testDebugUnitTest`. It needs JDK 17+ and `local.properties`. `-PbundleModels=false` keeps the ~380 MB of models out of the APK.

The app hasn't been run on a device yet.

## Resource safety

**Why this exists:** the 16 GB dev Mac **kernel-panicked** when an embedding job and three model-loading jobs ran at once. macOS keeps swap on the data disk: once the disk is nearly full, swap can't grow and the machine freezes. Later, an uncapped MLX buffer cache grew to 11 GB and caused swap storms; the Qwen encoders now cap it at 512 MB.

**Rules:**

- Run one heavy ML job at a time.
- Keep about 10 GB of disk free.
- Run every model load through the guard.

**`photosearch/guard.py`:**

- **`acquire(weight)`** reserves units of a machine-wide job budget of `GUARD_SLOTS` units (default **2**). The units are flock'd files in `/tmp/photosearch-ml.slot*`. The budget allows one heavy job or two light ones.

  | Holder | Units |
  |---|---|
  | SigLIP2 encoder users: `index.py`, accurate OCR, `train/extract.py`, `train_head.py`, iOS build | 1 |
  | Qwen embedding, reranker, SO400M, DINOv3-L, `train/finetune.py`, `ml/convert_siglip2.py`, parity checks, Android builds | 2 |
  | Ask's VLM | all of it |

  - A **queue ticket** makes waiters first come, first served, so 1-unit jobs can't starve a 2-unit job.
  - Grabs are all-or-nothing, which prevents deadlocks.
  - Waiters give up after 3 h with `ResourceError`.
  - **The UI server is exempt:** `python server.py` on port 8000 holds a separate reserved slot (`/tmp/photosearch-ml.server`) for its own SigLIP, so search never queues behind batch jobs. Test or staging servers on other ports don't get this slot.
- **`check()`** runs between batches and is rate-limited to once every 10 s. It pauses while swap exceeds 16 GB, free disk drops below **6.5 GB**, or free memory drops below 12%. After 5 min it raises `ResourceError`, which the server returns as **503**. Tune it with `GUARD_MAX_SWAP_GB`, `GUARD_MIN_DISK_GB` and `GUARD_MIN_MEM_FREE_PCT`.
- **`search/watchdog.py`** is the last resort, started by `scripts/start.sh` if none is running. Every 20 s it checks for swap above 18 GB, free disk below **5 GB** or free memory below **8%**. If any holds, it sends SIGTERM to the newest guarded pid. It never stops the :8000 server. It logs to `search/data/watchdog.log`.

## Benchmarks and evaluation

Run one ML job at a time and record the results in `search/NOTES.md`.

**Model benchmarks** (results cached in `data/cache/<task>__<encoder>.npy`):

```sh
cd search
.venv/bin/python experiments/prepare_data.py                      # hotels/rOxford parquet -> images + gt.json
.venv/bin/python experiments/bench.py siglip2-b16-384 clip-b32 --tasks hotels,flickr,roxford
.venv/bin/python experiments/bench_head.py data/models/hotel_head.pt          # hotels2400 with the head
.venv/bin/python experiments/bench_head_library.py data/models/hotel_head.pt  # 5,390 hotel queries vs the library
.venv/bin/python experiments/bench_head_places.py data/models/hotel_head.pt   # regression: generic scenes
.venv/bin/python experiments/bench_qwen.py validate|bench|fuse; experiments/bench_qwen_rerank.py --flickr 100 --hotels 40
```

**Whole-engine eval** (no model loads when the query embeddings are cached):

```sh
.venv/bin/python experiments/eval_search.py --parse [--place]   # the search box; --place = trained head for photo queries
.venv/bin/python experiments/eval_search.py --parse --eval-set experiments/eval_set_xl.json   # with PHOTOSEARCH_INDEX=data/index-xl
.venv/bin/python experiments/eval_parse.py                      # 41 hand-written parser cases, no models
.venv/bin/python experiments/eval_ocr.py                        # fast vs accurate OCR
.venv/bin/python experiments/make_eval.py                       # rebuild eval_set.json from metadata, never from embeddings
```

`eval_search.py` reports hit@1, hit@10, P@10, R@60 and MRR for each query type (`object/scene`, `caption`, `place`, `place+object`, `time`, `text_in_photo`, `same_room`). It writes `experiments/results/eval_<mode>_<ts>.json`. Other modes: `--ask-plan` uses the LLM planner, and `--embed` chooses between the visual, original and template text embeddings.
