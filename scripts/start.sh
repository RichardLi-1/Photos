#!/usr/bin/env bash
# Run the local server (API + web app) with the resource watchdog.
#
#   scripts/start.sh                         production: http://127.0.0.1:8000, the 73k index (search/data/index-xl) if built,
#                                            else search/data/index
#   scripts/start.sh --stage [INDEX_DIR]     staging:    http://127.0.0.1:8100, default the 21k index (search/data/index)
#
# Both bind to 127.0.0.1 only. Ctrl-C stops the server and its watchdog.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/search"
PY=.venv/bin/python

if [ "${1:-}" = "--stage" ]; then
  export PORT="${PORT:-8100}"
  export PHOTOSEARCH_INDEX="${2:-$ROOT/search/data/index}"
  echo "staging: index $PHOTOSEARCH_INDEX on :$PORT"
else
  export PORT="${PORT:-8000}"
  default="$ROOT/search/data/index"; [ -f "$ROOT/search/data/index-xl/library.sqlite" ] && default="$ROOT/search/data/index-xl"
  export PHOTOSEARCH_INDEX="${PHOTOSEARCH_INDEX:-$default}"
fi
[ -f "$PHOTOSEARCH_INDEX/library.sqlite" ] || { echo "no index at $PHOTOSEARCH_INDEX (run: make index DIR=...)"; exit 1; }
[ -f "$ROOT/web/dist/index.html" ] || (cd "$ROOT/web" && npm run build)

watchdog_pid=""
if ! pgrep -f "python watchdog.py" >/dev/null; then  # one watchdog per machine is enough
  WATCHDOG_LOG="$ROOT/search/data/watchdog.log" "$PY" watchdog.py & watchdog_pid=$!
fi
trap '[ -n "$watchdog_pid" ] && kill "$watchdog_pid" 2>/dev/null || true' EXIT

echo "http://127.0.0.1:$PORT"
"$PY" server.py
