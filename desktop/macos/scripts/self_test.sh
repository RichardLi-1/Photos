#!/bin/bash
# Headless end-to-end test of the packaged app: unzip dist/PhotoSearch-macOS.zip to a temp dir, stream N photos into
# the sandboxed app on stdin, index them, run text queries, More like this and Same place, check the network is blocked.
#
#   desktop/macos/scripts/self_test.sh [IMAGES_DIR] [N]     (default: search/data/flickr1k/images_flickr_1k_test, 20)
#   APP=/path/PhotoSearch.app desktop/macos/scripts/self_test.sh   # test an app bundle instead of the zip
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
IMAGES="${1:-$ROOT/search/data/flickr1k/images_flickr_1k_test}"
N="${2:-20}"
TMP=""
if [ -z "${APP:-}" ]; then
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/photosearch-selftest.XXXXXX")"
  trap 'rm -rf "$TMP"' EXIT
  ditto -x -k "$ROOT/dist/PhotoSearch-macOS.zip" "$TMP"
  APP="$TMP/PhotoSearch.app"
fi
codesign --verify --strict "$APP"
files=$(cd "$IMAGES" && ls | grep -iE '\.(jpe?g|png|heic)$' | sort | head -n "$N")
# shellcheck disable=SC2086
COPYFILE_DISABLE=1 tar -cf - -C "$IMAGES" $files | \
  python3 "$ROOT/desktop/macos/scripts/guarded.py" --weight 1 -- "$APP/Contents/MacOS/PhotoSearch" --self-test --tar - \
    --query "a dog running in a garden" --query "snowmobiles in a snowy forest" --query "a couple sitting on the beach"
