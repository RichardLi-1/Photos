# Two-minute demo

Setup: `make start`, then open http://127.0.0.1:8000 in Chrome or Edge (real liquid glass). Everything runs on this Mac.
`make start` serves the 72,887-photo index when it's built; `make stage` serves the curated 21,389-photo index on :8100.

1. **Browse (10 s).** "72,887 public images, all searchable on-device." Scroll about 2.5 screens: one tile opens into the **scroll story**, then folds back into the wall.
2. **Search (20 s).** Just start typing (type-to-search): `dog on a beach at sunset` ↵. The results take milliseconds. Then try `flowers in germany`: the bubble shows the chips 📍 Germany “flowers”. `sept 29 2025` gives the friendly empty state with the library's date span.
3. **Match card (10 s).** Click any result: the card says what the photo shares with the query. **More like this →** runs a photo-to-photo search.
4. **Same place (15 s).** Search `hotel bathroom`, click a result, then **Same place →**: other photos of that same hotel room (bedroom, desk, view). This is our own trained model. It ranks the right hotel first about twice as often as plain similarity (whole library R@1 0.16 → 0.31).
5. **Photo search (10 s).** Drag a photo onto the page (or use the camera button) to search by image. Add words to refine it.
6. **Ask (20 s).** `how many dogs did we see in the snow?` ↵. The Ask ↵ chip appears for questions. A local VLM plans, searches and checks the top hits; verified photos get a ✓. The first question loads a ~3 GB model.
7. **Agent: album (15 s).** Click **Agent** in the pill (or just type a command, and **Agent ↵** lights up). `make an album of beach photos in Australia` ↵: the place becomes a filter, and the wall shows the album. Click **Albums** above the pill to see it listed.
8. **Agent: cleanup (20 s).** `clean up duplicates` ↵ finds the exact-duplicate groups (55 on the 73k index, 36 on the 21k one). The preview pairs keep ✓ with remove ✕. Point out "nothing changed yet". Then press **Confirm** (Enter works too): "Done: N photos moved to Recently Deleted."
9. **Undo (10 s).** Click **Undo** to restore them all. Optionally click **Recently Deleted** to show that it's empty again.
10. **Graceful (5 s).** `clean up screenshots` gets "none found" plus a suggestion to try near-duplicates.

Talking points: no cloud calls, files are never deleted (trash is a flag), every destructive step is previewed and confirmed, and every change can be undone.
After the demo, `undo` or `delete the album …` (confirm) leaves the library as it was.
