# iOS app

An on-device version of the search engine (SwiftUI, iOS 18.5+, Xcode 16.4). No network in Release, and photos, the index and queries never leave the phone.

## Status (2026-10-15)
- **Builds:** generic simulator, 0 Swift warnings (`make check ARGS=--ios`).
- **Model:** SigLIP2-B/16-384 as two fp16 Core ML mlprograms: image encoder 177 MB, text encoder 539 MB, of which the 256k-token embedding table is most.
  - Parity vs PyTorch fp32, measured by the iOS agent and not recorded in the repo: image min cosine 0.99915 (mean 0.99978), text min 0.99999.
  - logit_scale 112.846, logit_bias −16.772.
  - Convert with `ml/convert_siglip2.py`; check parity with `ml/check_siglip2_parity.py`.
- **Tokenizer:** the Swift BPE matches HF/open_clip token ids on 20 fixture strings (`ios/PhotosTests/siglip2_token_fixtures.json`).
- **Not yet verified:** the simulator end-to-end self-test and the XCTest run (see below), and on-device ANE speed and memory.

## Architecture
- `Core/ML`: `SigLIPTokenizer`, `SigLIPEncoder`.
- `Core/Storage`: `LibraryStore`, an actor over sqlite3 with FTS5 and Float16 embeddings, in Application Support, excluded from backup.
- `Core/Indexing`: PhotoKit with `isNetworkAccessAllowed=false`; Vision OCR; `Indexer`, which is resumable and backs off on thermal state, Low Power Mode, battery < 20% and low disk; BGProcessingTask.
- `Core/Search`:
  - vDSP cosine, plus RRF with OCR hits;
  - `QueryParser`, using NSDataDetector for dates;
  - `ConceptExplainer` for the "why it matched" chips.
- `Core/Tools/PhotoTools`: the same tool names as `search/photosearch/tools.py`. Trash goes through PhotoKit's system confirmation, and there's no agent loop.
- `Features`:
  - grid, search capsule and match card;
  - `GlassBackground`, the single glass modifier.

## Apple guidelines checklist
SF Symbols · Dynamic Type everywhere (text styles, `ViewThatFits`) · Dark Mode · safe areas · 44 pt targets ·
VoiceOver labels/traits on all controls and photo cells · Reduce Motion · Reduce Transparency / Increase Contrast fall back from glass ·
standard navigation · haptics only for errors · `PrivacyInfo.xcprivacy` (no tracking; UserDefaults CA92.1, DiskSpace E174.1,
FileTimestamp C617.1) · usage descriptions · limited-library support (`presentLimitedLibraryPicker`) · no network in Release
(CLGeocoder opt-in, labelled) · PhotoKit deletion confirmation · BGProcessingTask registered in Info.plist · String Catalogs ·
placeholder icon (light/dark/tinted) + launch screen · no private APIs · `ITSAppUsesNonExemptEncryption = NO`.
Still to do: an Accessibility Inspector pass on a device.

## When Xcode 26 / iOS 26 is available
- Replace the body of `ios/Features/Shared/GlassBackground.swift` with `content.glassEffect(.regular.interactive(), in: shape)` behind `#available(iOS 26, *)`. Optionally wrap the capsule and pills in `GlassEffectContainer`.
- Move `PlaceResolver` from CLGeocoder (deprecated in iOS 26) to MapKit reverse geocoding.
