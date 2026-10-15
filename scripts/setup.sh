#!/usr/bin/env bash
# One-time setup on a Mac (Apple Silicon): Python env, model download, web build.
#
#   scripts/setup.sh
#
# The only network use in the whole project is here: pip/npm packages and the SigLIP2 weights (Hugging Face).
# After this, everything runs offline (HF_HUB_OFFLINE=1 is the default for the server and indexer).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3.11}"

echo "==> Python env (search/.venv)"
[ -x "$ROOT/search/.venv/bin/python" ] || "$PYTHON" -m venv "$ROOT/search/.venv"
"$ROOT/search/.venv/bin/pip" install -q --upgrade pip
"$ROOT/search/.venv/bin/pip" install -q -r "$ROOT/search/requirements.txt"

echo "==> SigLIP2 weights (one-time download, ~1.5 GB, through the ML guard)"
cd "$ROOT/search"
HF_HUB_OFFLINE=0 .venv/bin/python -c "from photosearch import encoders; encoders.get('siglip2-b16-384'); print('model ready')"

echo "==> Web app (web/dist)"
cd "$ROOT/web"
npm install --no-fund --no-audit
npm run build

echo
echo "Done. Next:"
echo "  make index DIR=~/Pictures/SomeFolder   # index photos (OCR + embeddings, local)"
echo "  make start                             # http://127.0.0.1:8000"
