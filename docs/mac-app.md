# Mac app: the rest of the features

**Goal:** the Mac app does everything the web demo does: search, Ask, commands, clean-up, albums. Apple Photos stays the place where photos live. Keywords and Show in Photos are covered in [photos-integration.md](photos-integration.md).

**Today the app has:** search, drop-an-image search, More Like This, Same Place, Why it matched, folders, and the Photos library.

## Decisions

### One search field
- **To keep one place to type:** the existing search field routes by text, using the web's rules (`ask.ts` and `agent.ts` ported to Swift):
  - plain text → search
  - a question ("what…", trailing "?") → Ask
  - a command ("make an album of…", "delete…") → Agent

### Ask and commands
- **To stay private without a 4B local model:** use Apple's Foundation Models framework instead of Qwen.
  - The on-device model plans: it turns the question or command into tool calls.
  - Private Cloud Compute answers what the on-device model can't, such as long contexts and checking photos.
- **To let the user opt out of PCC:** Settings → "Use Private Cloud Compute for harder questions", on by default. With it off, Ask answers from on-device results only.
- **To reuse one toolset:** the model calls `PhotoTools` from `ios/Core` (search, albums, delete), the same tools the iOS app uses.
- **To answer with evidence:** an Ask card shows the answer plus the photos it is based on. Each photo opens its match card.
- **To work on older macOS:** Ask and commands appear only where Foundation Models exists (macOS 27). On macOS 15–26 the field only searches.

### Changing photos
- **To never lose a photo:**
  - Changes apply only to Photos-library photos, through PhotoKit.
  - Deletes show the system confirmation and go to Recently Deleted (30 days).
  - Folder photos are read-only. You can search them and Show in Finder, nothing else.
- **To never act on a guess:** every command that changes something shows a preview card ("Delete 14 photos", with thumbnails) and runs only after you click Confirm. This matches the web Executor's dry run plus confirm token.
- **To use real albums:** albums are Photos albums (PhotoKit), not our own. **Save as Album** on any result set creates one.
- **To act on several photos:** multi-select in the grid (⌘-click or shift-click). The context menu offers Add to Album, Show in Photos, and Delete.

### Clean up
- **To find photos you may not need:** add a **Clean Up** sidebar section with groups for exact duplicates, near-duplicates and bursts, blurry, screenshots and documents, and accidental shots. It ports `search/photosearch/cleanup.py` (sha1, SigLIP cosine, edge-width blur, zero-shot with the "defect vs on purpose" veto).
- **To keep the best one:** each group pre-selects the best photo to keep (sharpest, then largest) and offers **Delete Others**, which follows the delete rule above.
- **To stay cheap:** stats are computed after indexing on CPU, resumably, with the same pause rules as the indexer.

### Match card
- **To make tags useful in the app too:** show the photo's keywords as chips. Clicking a chip searches for it.
- **To jump to Photos:** add **Show in Photos** next to Show in Finder.

### Settings window (⌘,)
- **Sources:** folders and the Photos library toggle (moved out of the sidebar).
- **Photos:** Add keywords, tagged count, Remove all keywords.
- **Ask:** the PCC toggle.

## Verify first
1. Foundation Models works from our sandboxed app with no network entitlement, including PCC.
   - Note: this Mac is on macOS 15.5, so it needs macOS 27 to test.
2. PCC access:
   - It requires a paid developer account, the Small Business Program, and the PCC entitlement.
   - The app is currently signed ad hoc with no account.
3. Can Foundation Models take image input, on-device and on PCC? If not, Ask checks photos using their keywords and OCR text instead of pixels.
4. On-device tool calling handles `PhotoTools` reliably.

## Eval
- **Ask:** the question subset of `search/experiments/eval_set.json`. The correct photo should be among the evidence photos.
- **Commands:** 30 hand-written commands.
  - The tool plan is correct.
  - 100% of changing commands stop at a preview card.
- **Clean up:** reuse the `cleanup.py` sample-precision checks on the demo library.

## Files
| File | Change |
|---|---|
| `ios/Core/Search/Router.swift` | search / question / command routing |
| `ios/Core/Agent/` | Foundation Models planner, PCC escalation, preview and confirm executor |
| `ios/Core/Cleanup/` | port of `cleanup.py` |
| `ios/Core/Tools/PhotoTools.swift` | add Save as Album; dry-run previews |
| `desktop/macos/Sources/UI/` | Ask card, preview card, Clean Up section, multi-select, keyword chips, Settings window |
