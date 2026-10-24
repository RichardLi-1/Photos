# Apple Photos integration (Mac app)

## Goal
1. Make Apple Photos' own search bar find almost any photo. To do that, the Mac app writes many keywords into each photo.
2. Add a **Show in Photos** button to every result.

## How
Photos' AppleScript interface lets other apps read and write a media item's `keywords` and call `spotlight` (reveal a photo). Photos search already matches keywords. PhotoKit has no keyword API.

## Tagging
**Vocabulary** (built offline, shipped in the app):
- About 20k concepts: WordNet visual nouns, Open Images labels, Places365 scenes, and hand lists of actions, events, attributes and styles.
- Each concept stores its synonyms and its parents (dog → canine, pet, animal). Abstract parents such as "entity", "object" and "whole" are stop-listed.
- One SigLIP2 text vector per concept: the mean of "a photo of {lemma}" over its lemmas.
- Per-concept mean and std of scores over a fixed reference set (the 21k demo library). This makes scores comparable across concepts and keeps them independent of library size.

**Per photo** (from the stored embedding, no re-encoding):
1. Score all concepts. This is one matrix multiply, a few seconds for 50k photos.
2. Keep a concept if `p ≥ floor`, its `z ≥ z_min`, and it is in the photo's top `K`. The z-check drops concepts that score high on every photo.
3. Add each kept concept's synonyms and parents.
4. Lowercase, dedupe and cap at `MAX` keywords.

`floor`, `z_min`, `K` and `MAX` are tuned offline (see Eval). The goal is recall: a wrong parent like "animal" costs little, while a missing word means the photo is never found.

## Writing to Photos
- **Calls:** batched AppleScript (about 50 photos per call) that reads the keywords, merges them in Swift, then writes them.
- **Ledger:** our SQLite table `photo_keywords(asset_id, keyword, state)`, where `state` is `added` or `user_removed`.
- **Merge:**
  - Never touch keywords we didn't add.
  - A keyword the user already had is not recorded as ours.
  - If one of our keywords is missing from Photos, the user deleted it. Mark it `user_removed` and never re-add it.
  - Write only when the set changes.
- **Remove all:** strip every `added` keyword and clear the ledger.
- **When it runs:**
  - After indexing, and again when the vocabulary version changes (diff only).
  - Low priority, resumable, paused while Photos is frontmost.
- **Scope:** Photos-library photos only. Folder photos are skipped.
- **Permissions:**
  - `com.apple.security.automation.apple-events`
  - `scripting-targets` for `com.apple.Photos`: `library.read-write` and `spotlight`
  - `NSAppleEventsUsageDescription`
- **Not touched:** titles and captions.

## Show in Photos
`spotlight media item id "<localIdentifier>"`. The button appears in the match card for Photos-library results.

## Settings
- **"Add keywords to Photos"**: off by default. It shows a note that keywords sync via iCloud Photos and are end-to-end encrypted only with Advanced Data Protection.
- Count of tagged photos.
- **Remove all keywords** button.

## Verify first (spike, on a test library of about 200 photos, never the real one)
1. A sandboxed app can set `keywords` with the scripting-targets entitlement.
2. The AppleScript media item id equals `PHAsset.localIdentifier`.
3. How Photos search matches keywords:
   - whole word or prefix?
   - plurals?
   - multi-word keywords ("hot air balloon")?
   - does "dog beach" AND across keywords?
4. Limits:
   - 150 keywords on one photo
   - 30k distinct keywords in the library
   - write speed per photo
   - search speed afterwards
5. Keywords sync to iPhone.

Items 3 and 4 shape the vocabulary: whether to add plurals, split phrases, and set `MAX`.

## Eval (`search/experiments/eval_tags.py`, offline, using stored embeddings)
The script simulates Photos keyword search using the matching rules found in the spike.
- **Findability:** % of Flickr1k photos where at least one caption noun (lemmatized) is a keyword, plus % of caption nouns covered. Target ≥ 95%.
- **Precision:** hand-check the keywords on 50 random library photos. Target ≥ 70% correct.
- **Compare:** run the 91 `eval_set.json` queries through simulated keyword search versus our semantic search.

## Files
| File | Change |
|---|---|
| `ml/convert_siglip2.py --part tags` | builds `siglip2_tags.json` and `.f16` (vocab, synonyms, parents, ref stats), about 30 MB |
| `ios/Core/Search/Tagger.swift` | scoring and selection; shares math with `ConceptExplainer` |
| `desktop/macos/Sources/Engine/PhotosKeywords.swift` | AppleScript bridge, ledger, merge, remove all, spotlight |
| `desktop/macos/Sources/UI/` | settings toggle and Show in Photos button |
| `desktop/macos/Resources/` | entitlements and Info.plist |
