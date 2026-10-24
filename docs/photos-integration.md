# Apple Photos integration (Mac app)

**Goal:** Photos' own search bar finds almost any photo, and every result in our app can be opened in Photos.

## Decisions

### What gets tagged
- **To make almost every photo findable:** tag each photo with its top `K` concepts from a vocabulary of about 20k. The vocabulary combines WordNet visual nouns, Open Images labels, Places365 scenes, and hand lists of actions, events, attributes and styles.
- **To match the word people actually type:** add each concept's synonyms (dog → puppy, doggy).
- **To match broad searches:** add each concept's parents (dog → canine, pet, animal). Stop-list abstract parents such as "entity" and "object".
- **To raise recall without adding wrong tags:** grow the keyword list through synonyms and parents, not by lowering the score threshold.
- **To treat synonyms the same:** use one text vector per concept, the mean of "a photo of {lemma}" over its lemmas.
- **To compare scores across concepts:** z-score each concept against a fixed reference set (the 21k demo library). Results then don't depend on the size of the user's library.
- **To drop concepts that match every photo:** keep a concept only if `p ≥ floor` and `z ≥ z_min`, then cap at `MAX` keywords.
- **To avoid re-encoding photos:** score the stored SigLIP2 embeddings. That's one matrix multiply, a few seconds for 50k photos.

### How it talks to Photos
- **To write keywords at all:** use AppleScript (`keywords` on a media item), because PhotoKit has no keyword API. Batch about 50 photos per call.
- **To stay sandboxed:** request the `scripting-targets` entitlement for `com.apple.Photos` with the `library.read-write` and `spotlight` groups, plus `NSAppleEventsUsageDescription`.
- **To open a result in Photos:** call `spotlight media item id "<localIdentifier>"`. The button appears only on Photos-library results.

### Protecting the user's library
- **To never damage the user's data:** record every keyword we add in a ledger, `photo_keywords(asset_id, keyword, state)`. Only ever change keywords in that ledger. Never touch titles or captions.
- **To respect the user's edits:** if one of our keywords is missing from Photos, the user deleted it. Mark it `user_removed` and never re-add it.
- **To make it undoable:** a **Remove all keywords** button strips every keyword in the ledger.
- **To keep Photos responsive:**
  - run at low priority, resumably
  - pause while Photos is frontmost
  - write only when a photo's keyword set changes
- **To respect privacy:** keyword writing is off by default. The toggle notes that keywords sync via iCloud Photos and are end-to-end encrypted only with Advanced Data Protection.

## Verify first (on a test library of about 200 photos, never the real one)
1. A sandboxed app can set `keywords`.
2. The AppleScript media item id equals `PHAsset.localIdentifier`.
3. How Photos search matches keywords: prefix, plurals, multi-word phrases, and AND across keywords.
4. Limits: 150 keywords per photo, 30k distinct keywords, write speed, and search speed.
5. How cluttered Keyword Manager (⌘K) gets.
6. Keywords sync to iPhone.

Items 3–5 decide whether to add plurals, how to split phrases, and how big `MAX` and the vocabulary should be.

## Eval (`search/experiments/eval_tags.py`, offline)
The script simulates Photos keyword search using the matching rules found in step 3.
- **Findability:** at least one caption noun is a keyword on ≥ 95% of Flickr1k photos.
- **Precision:** ≥ 70% of keywords are correct when hand-checked on 50 random photos.
- **Comparison:** the 91 `eval_set.json` queries through simulated keyword search versus our semantic search.

Tune `floor`, `z_min`, `K` and `MAX` against these targets.

## Files
| File | Change |
|---|---|
| `ml/convert_siglip2.py --part tags` | `siglip2_tags.json` and `.f16`: vocab, synonyms, parents, reference stats (about 30 MB) |
| `ios/Core/Search/Tagger.swift` | scoring and selection; shares math with `ConceptExplainer` |
| `desktop/macos/Sources/Engine/PhotosKeywords.swift` | AppleScript bridge, ledger, merge, remove all, spotlight |
| `desktop/macos/Sources/UI/` | settings toggle and Show in Photos button |
| `desktop/macos/Resources/` | entitlements and Info.plist |
