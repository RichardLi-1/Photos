#!/usr/bin/env python3
"""Run xcodebuild under the shared ML/RAM guard (search/photosearch/guard.py): it holds 1 unit of the machine-wide
budget, so it never runs alongside model conversion or other heavy jobs, and refuses to start when disk/memory is low.

usage: python3 ios/scripts/guarded_xcodebuild.py <xcodebuild args...>
e.g.   python3 ios/scripts/guarded_xcodebuild.py -project ios/Photos.xcodeproj -scheme Photos \
         -destination 'generic/platform=iOS Simulator' -derivedDataPath /tmp/dd build
"""
import os
import subprocess
import sys

os.environ.setdefault("GUARD_MIN_DISK_GB", "10")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "search"))
from photosearch import guard  # noqa: E402

guard.acquire(1)
guard.check(force=True)
sys.exit(subprocess.call(["xcodebuild", *sys.argv[1:]]))
