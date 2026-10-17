# Photo Search for Android (private, on-device)

Semantic photo search that runs entirely on the phone: SigLIP2-B/16-384 (ONNX Runtime, int8), OCR fused with RRF, date phrases ("dogs last summer"), "more like this", and "why it matched" chips. Kotlin, Jetpack Compose, Material 3. minSdk 30 (Android 11), needed for `MediaStore.createTrashRequest` (system-confirmed, restorable trash); compileSdk/targetSdk 35.

**Models** (offline, from the HF cache; usage is in each script's docstring): `ml/android/convert_siglip2_onnx.py all`, then `export_tokenizer.py`. They write `app/bigassets/models/` (image tower 97 MB, text tower 87 MB, int8 token table 198 MB) and `app/src/main/assets/models/` (tokenizer, 284 concept vectors, calibration). Parity vs PyTorch fp32 on 20 images and 20 texts, as min cosine: image 0.996, text 0.9988. `ml/android/parity.json` has the details.

**Build.** `./gradlew assembleDebug testDebugUnitTest` (JDK 17+; `local.properties` sdk.dir). On the shared Mac, run it through `python3 tools/guarded.py -- ./gradlew …`.
- `-PbundleModels=false` leaves the ~380 MB of models out of the APK. Then `adb push app/bigassets/models /sdcard/Android/data/app.photosearch/files/`.
- Unit tests cover the tokenizer (ids identical to open_clip on 30 strings), text cleaning, bicubic resize vs PIL, cosine/RRF/search, date parsing, fp16, and the explainer.

**Run.** Allow photo access: all photos, or "selected photos" on Android 14+. Indexing runs in WorkManager. Type in the floating capsule, or tap the camera button to search with a photo. Tap a photo for "why it matched" and "More like this". Back clears a search, with a predictive-back preview.

**Architecture** (`app/src/main/java/app/photosearch/`):
- `ml/`: the Gemma BPE tokenizer and open_clip text cleaning in Kotlin. Token embeddings are looked up from a memory-mapped int8 table. The ORT towers are memory-mapped from the APK. Images are decoded at a short side of at least 768 px, then a PIL-compatible bicubic resize (cos ≥ 0.995 vs open_clip on 12 MP inputs; bilinear: 0.967).
- `data/`: SQLite holds metadata, OCR text (FTS4) and albums. Embeddings go to an append-only fp16 file in `noBackupFilesDir`, loaded into `VectorIndex` for brute-force cosine search.
- `index/`: `IndexWorker` is resumable and works in batches of 16. It requires battery-not-low (plus charging if you choose), re-runs on a MediaStore content trigger and nightly while idle and charging, and shows a foreground notification for long runs.
- `search/`: `Searcher` (a port of search.py: cosine top-K + RRF with OCR), `DateParser`, and `Explainer` (SigLIP's calibrated sigmoid over the explain.py concepts).
- `tools/PhotoTools`: `schemas()` and `call(name, args)`, with the same tool names and arguments as `search/photosearch/tools.py`. Albums are app-private collections. `trash_photos` does a dry run plus a confirm token, then shows the system trash dialog. There is no agent loop.
- `ui/`: full-bleed grid plus a translucent capsule. On Android 12+ the capsule blurs the grid with RenderEffect; older versions get a near-solid surface.

**Privacy.**
- The app has no INTERNET permission. Network permissions that libraries add are stripped, and `verifyNoNetwork<Variant>` fails the build if one survives.
- No analytics. Backup and device transfer are excluded.
- OCR uses ML Kit's bundled model, which sends usage metrics to Google with no documented opt-out. It is therefore **off by default**, and when you enable it the OS blocks the upload because the app has no network permission.
