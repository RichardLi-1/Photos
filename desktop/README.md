# Photo Search for Mac

**Download:** [PhotoSearch-macOS.zip](https://github.com/RichardLi-1/Photos/releases/latest/download/PhotoSearch-macOS.zip) (Apple Silicon, macOS 15+, roughly 700 MB because the models ship inside). Check it against the `.sha256` file on the release page.

**Open it.** Unzip, then drag **PhotoSearch.app** to Applications. It's signed ad hoc, not by an Apple Developer ID, so the first launch is blocked:
1. Double-click it. macOS says it "can't be opened". Click **Done**.
2. Open **System Settings → Privacy & Security**, scroll down to "PhotoSearch was blocked", and click **Open Anyway**. Confirm with your password.
3. Or, in Terminal: `xattr -dr com.apple.quarantine /Applications/PhotoSearch.app`

**What it does.** Add a folder (or drop one on the window, or turn on your Photos library) and type what you remember: "dog on the beach", "menu with prices", "receipts last month". Click a photo to see why it matched, then **More like this** or **Same place**. Same place is our own head, trained on hotel photos to recognise the same room. Drop an image onto the window to search with it. Indexing runs at a low priority, picks up where it left off, and pauses when the Mac is hot or in Low Power Mode.

**Privacy.** It's the same engine as the iOS app (`ios/Core`: SigLIP2 on Core ML and the Neural Engine, Apple Vision OCR, SQLite), and it runs entirely on your Mac. The app is sandboxed *without* network permission, so macOS itself refuses any connection. It can only read folders you pick (read-only), and it never copies, moves or changes a photo. The index lives in `~/Library/Containers/app.photosearch.mac`. Delete that folder to forget everything.

**Build it yourself.** You need Xcode 16 and xcodegen, plus the models from `ml/convert_siglip2.py` and `ml/export_place_head.py`. Then run `desktop/macos/scripts/build.sh` to get `dist/PhotoSearch-macOS.zip`, and `desktop/macos/scripts/self_test.sh` to index 20 photos headlessly and run three searches.
