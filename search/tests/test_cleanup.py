"""photosearch.cleanup: only the stable, model-free parts (per-image stats, quality ordering, trash-aware flatten)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

cleanup = pytest.importorskip("photosearch.cleanup")


def _save(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path, "PNG")
    return str(path)


@pytest.fixture
def imgs(tmp_path):
    rng = np.random.default_rng(0)
    sharp = np.kron(rng.integers(0, 2, (16, 16)) * 255, np.ones((16, 16)))[..., None].repeat(3, 2)  # hard-edged checks
    return {
        "dark": _save(tmp_path / "dark.png", np.full((128, 128, 3), 6)),
        "bright": _save(tmp_path / "bright.png", np.full((128, 128, 3), 250)),
        "sharp": _save(tmp_path / "sharp.png", sharp),
        "blurry": str(tmp_path / "blurry.png"),
        "_sharp_arr": sharp,
    }


def test_image_stats_basics(imgs):
    st = cleanup.image_stats(imgs["dark"])
    assert st["sha1"] == hashlib.sha1(open(imgs["dark"], "rb").read()).hexdigest()
    assert (st["w"], st["h"]) == (128, 128) and st["bytes"] > 0
    assert st["frac_dark"] > 0.9 and cleanup.image_stats(imgs["bright"])["frac_bright"] > 0.9


def test_sharpness_orders_sharp_above_blurry(imgs):
    from PIL import ImageFilter
    Image.fromarray(imgs["_sharp_arr"].astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)).save(imgs["blurry"])
    s, b = cleanup.image_stats(imgs["sharp"]), cleanup.image_stats(imgs["blurry"])
    assert s["lap_var"] > b["lap_var"]
    assert cleanup.quality(s) > cleanup.quality(b)


def test_quality_of_missing_stats():
    assert cleanup.quality({}) < -1e6


def test_flatten_drops_trashed_and_reelects_keep():
    cands = {"near_duplicate": [{"photo_ids": [1, 2, 3], "keep": 1, "remove": [2, 3], "confidence": 0.9}],
             "blurry": [{"photo_ids": [4], "confidence": 0.8}, {"photo_ids": [5], "confidence": 0.7}]}
    out = cleanup.flatten(cands, None, exclude={1, 5})
    grp = next(c for c in out if "keep" in c)
    assert grp["photo_ids"] == [2, 3] and grp["keep"] == 2 and grp["remove"] == [3]
    assert [c["photo_ids"] for c in out if "keep" not in c] == [[4]]
    assert cleanup.flatten(cands, "near_duplicate", exclude={1, 2}) == []    # group of one vanishes
    assert cands["near_duplicate"][0]["photo_ids"] == [1, 2, 3]               # input untouched
