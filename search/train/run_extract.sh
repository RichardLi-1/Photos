#!/bin/zsh
# Retry extraction until it completes (the guard aborts it when the machine is under pressure; chunks resume).
cd "$(dirname $0)/.."
for i in {1..30}; do
  .venv/bin/python train/extract.py "$@" && exit 0
  echo "[run_extract] attempt $i aborted; retrying in 120s"; sleep 120
done
