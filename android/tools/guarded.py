#!/usr/bin/env python3
"""Run a heavy command (Gradle build/tests, emulator) only while holding the machine-wide ML budget.

usage: python3 android/tools/guarded.py [--weight 2] -- ./gradlew assembleDebug
The budget/pressure checks live in search/photosearch/guard.py (stdlib only); this just wraps a subprocess with them.
"""
import os
import subprocess
import sys

os.environ.setdefault("GUARD_MIN_DISK_GB", "10")  # repo rule: stop if free disk < 10 GB
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "search"))
from photosearch import guard  # noqa: E402

args = sys.argv[1:]
weight = 2
if args[:1] == ["--weight"]:
    weight, args = int(args[1]), args[2:]
if args[:1] == ["--"]:
    args = args[1:]
guard.acquire(weight)  # waits for free slots, then checks swap / disk / memory
sys.exit(subprocess.call(args))
