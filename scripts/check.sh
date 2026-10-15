#!/usr/bin/env bash
# One-command check for the repo.
#
#   scripts/check.sh            Python tests (fast, model-free) + web typecheck/build/unit tests
#   scripts/check.sh --ios      ...plus an iOS build for the generic simulator (no simulator is booted),
#                               run under the machine-wide ML guard so it never overlaps a heavy ML job
#   scripts/check.sh --slow     ...plus the real-SigLIP smoke test (one model load, through the guard)
#
# Never touches the server on :8000 and never writes web/dist (the check build goes to node_modules/.cache).
# Env: IOS_GUARD_WAIT_S (default 120) = how long --ios waits for a free guard slot before giving up.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/search/.venv/bin/python"
IOS=0 SLOW=0
for a in "$@"; do
  case "$a" in
    --ios) IOS=1 ;;
    --slow) SLOW=1 ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a (try --help)" >&2; exit 2 ;;
  esac
done

SUMMARY=() FAILED=0
step() {  # step NAME CMD...   (runs CMD, records PASS/FAIL/SKIP with timing)
  local name="$1"; shift
  local t0=$SECONDS
  printf '\n\033[1m==> %s\033[0m\n' "$name"
  "$@"; local rc=$?
  local dt=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then SUMMARY+=("PASS  $name (${dt}s)")
  elif [ $rc -eq 75 ]; then SUMMARY+=("SKIP  $name (guard busy / not available)")
  else SUMMARY+=("FAIL  $name (exit $rc, ${dt}s)"); FAILED=1; fi
}

python_tests() {
  [ -x "$PY" ] || { echo "no venv at search/.venv"; return 1; }
  (cd "$ROOT/search" && "$PY" -m pytest tests -rxX "$@")
}

web_check() {
  [ -d "$ROOT/web/node_modules" ] || { echo "web/node_modules missing: run (cd web && npm ci)"; return 75; }
  (cd "$ROOT/web" && npm run --silent check)
}

ios_build() {
  command -v xcodebuild >/dev/null || { echo "xcodebuild not found"; return 75; }
  local dd="${TMPDIR:-/tmp}/photos-check-derived"
  # Hold one guard slot for the whole build: the wrapper takes the flock, marks it inheritable and exec()s xcodebuild,
  # so the slot belongs to the xcodebuild pid (and guard.holders()/watchdog.py see it) until the build exits.
  "$PY" - xcodebuild -project "$ROOT/ios/Photos.xcodeproj" -scheme Photos \
      -destination 'generic/platform=iOS Simulator' -derivedDataPath "$dd" \
      -quiet CODE_SIGNING_ALLOWED=NO build <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["CHECK_ROOT"], "search"))
from photosearch import guard
try:
    guard.acquire(1, wait_s=float(os.environ.get("IOS_GUARD_WAIT_S", 120)))
except guard.ResourceError as e:
    print(f"[check] {e}", file=sys.stderr)
    sys.exit(75)
for f in guard._held:
    os.set_inheritable(f.fileno(), True)
os.execvp(sys.argv[1], sys.argv[1:])
PY
}

export CHECK_ROOT="$ROOT"
step "python tests (fast, model-free)" python_tests
[ $SLOW -eq 1 ] && step "python smoke test (real SigLIP, guarded)" python_tests -m slow
step "web: typecheck + build + unit tests" web_check
[ $IOS -eq 1 ] && step "iOS build (generic simulator, guarded)" ios_build

printf '\n\033[1m==> summary\033[0m\n'
printf '  %s\n' "${SUMMARY[@]}"
[ $IOS -eq 0 ] && echo "  (iOS build skipped; pass --ios)"
exit $FAILED
