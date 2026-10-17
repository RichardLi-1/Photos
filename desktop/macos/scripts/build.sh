#!/bin/bash
# Build, ad-hoc sign and package Photo Search for macOS (Apple Silicon) → dist/PhotoSearch-macOS.zip (+ .sha256).
#
#   desktop/macos/scripts/build.sh            # build + package
#   desktop/macos/scripts/build.sh --clean    # …and delete the DerivedData afterwards
#
# Needs: Xcode 16+, xcodegen, the Core ML models from ml/convert_siglip2.py (ios/Core/Models/), and the place head from
# ml/export_place_head.py (desktop/macos/Models/; exported automatically if missing). No network is used.
#
# Notarizing later (needs a paid Apple Developer account; removes the Gatekeeper "Open Anyway" step):
#   1. Sign with the Developer ID instead of ad hoc (hardened runtime is already on):
#        codesign --force --sign "Developer ID Application: NAME (TEAMID)" --options runtime --timestamp \
#          --entitlements desktop/macos/Resources/PhotoSearch.entitlements PhotoSearch.app
#   2. ditto -c -k --keepParent PhotoSearch.app PhotoSearch-macOS.zip
#      xcrun notarytool submit PhotoSearch-macOS.zip --keychain-profile NOTARY --wait
#      (one-time: xcrun notarytool store-credentials NOTARY --apple-id … --team-id … --password <app-specific>)
#   3. xcrun stapler staple PhotoSearch.app, then zip again (ditto) and recompute the SHA-256.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MAC="$ROOT/desktop/macos"
DD="${DERIVED_DATA:-$MAC/build/DerivedData}"
DIST="$ROOT/dist"
APP="$DD/Build/Products/Release/PhotoSearch.app"
ZIP="$DIST/PhotoSearch-macOS.zip"

free_gb=$(df -g "$HOME" | awk 'NR==2 {print $4}')
if [ "$free_gb" -lt 10 ]; then
  echo "Only ${free_gb} GB free on $HOME; the build needs ~2.5 GB and the repo rule is to stop below 10 GB." >&2
  exit 1
fi

for f in SigLIP2ImageEncoder.mlpackage SigLIP2TextEncoder.mlpackage siglip2_tokenizer.json siglip2_merges.bin \
         siglip2_meta.json siglip2_concepts.json siglip2_concepts.f16; do
  [ -e "$ROOT/ios/Core/Models/$f" ] || { echo "missing ios/Core/Models/$f: run ml/convert_siglip2.py first (see docs/ios.md)" >&2; exit 1; }
done
if [ ! -f "$MAC/Models/place_head.f32" ]; then
  "$ROOT/search/.venv/bin/python" "$ROOT/ml/export_place_head.py"
fi

if command -v xcodegen >/dev/null; then (cd "$MAC" && xcodegen generate --spec project.yml --quiet); fi

python3 "$MAC/scripts/guarded.py" --weight 1 -- xcodebuild -project "$MAC/PhotoSearch.xcodeproj" -scheme PhotoSearch \
  -configuration Release -destination 'platform=macOS,arch=arm64' -derivedDataPath "$DD" build -quiet

# Re-sign explicitly (ad-hoc, hardened runtime, our entitlements) so the shipped signature never depends on Xcode defaults.
codesign --force --sign - --options runtime --timestamp=none --entitlements "$MAC/Resources/PhotoSearch.entitlements" "$APP"
codesign --verify --strict --verbose=1 "$APP"
ents="$(codesign -d --entitlements - --xml "$APP" 2>/dev/null)"
if grep -q "network" <<<"$ents"; then echo "refusing to ship: a network entitlement is present" >&2; exit 1; fi
grep -q "com.apple.security.app-sandbox" <<<"$ents" || { echo "refusing to ship: not sandboxed" >&2; exit 1; }

mkdir -p "$DIST"
[ -f "$DIST/.gitignore" ] || echo "*" > "$DIST/.gitignore"   # release artifacts go to GitHub Releases, not git
rm -f "$ZIP" "$ZIP.sha256"
ditto -c -k --keepParent --sequesterRsrc "$APP" "$ZIP"   # ditto keeps the signature intact (zip -r can break it)
(cd "$DIST" && shasum -a 256 "$(basename "$ZIP")" > "$(basename "$ZIP").sha256")
echo "app: $(du -sh "$APP" | cut -f1)   zip: $(du -h "$ZIP" | cut -f1)   sha256: $(cut -d' ' -f1 "$ZIP.sha256")"

if [ "${1:-}" = "--clean" ]; then rm -rf "$DD"; echo "removed $DD"; fi
