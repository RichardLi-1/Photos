#!/usr/bin/env python3
"""Run a heavy command (xcodebuild, the model-loading self-test) only while holding the machine-wide ML budget.

usage: python3 desktop/macos/scripts/guarded.py [--weight 1] -- <command...>
Same pattern as ios/scripts/guarded_xcodebuild.py; the budget and pressure checks live in search/photosearch/guard.py.
"""
import os
import subprocess
import sys

os.environ.setdefault("GUARD_MIN_DISK_GB", "10")  # repo rule: stop if free disk < 10 GB
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "search"))
from photosearch import guard  # noqa: E402  (read-only import; owned by search/)

args = sys.argv[1:]
weight = 1
if args[:1] == ["--weight"]:
    weight, args = int(args[1]), args[2:]
if args[:1] == ["--"]:
    args = args[1:]
guard.acquire(weight)      # waits for free slots
guard.check(force=True)    # then swap / disk / memory
sys.exit(subprocess.call(args))
