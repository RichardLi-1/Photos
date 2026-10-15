"""Optional: load the real SigLIP2 once, through the machine-wide ML guard.  Run with:  pytest tests -m slow

Not sandboxed: uses the real guard slots and pressure checks, so it queues behind (never alongside) other ML jobs.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.mark.slow
def test_siglip_text_image_agree():
    from photosearch import encoders, guard
    try:
        guard.acquire(1, wait_s=60)  # fail fast instead of queueing for hours; encoders.get re-acquires (no-op)
    except guard.ResourceError as e:
        pytest.skip(f"ML budget busy: {e}")
    assert guard._held, "model must only load while holding a guard slot"
    enc = encoders.get("siglip2-b16-384")
    assert len(guard._held) == 1

    ims = [Image.new("RGB", (96, 96), c) for c in ((220, 30, 30), (30, 30, 220))]
    X = enc.images(ims, bs=2)
    T = enc.texts(["a plain red square", "a plain blue square"])
    assert X.shape == (2, T.shape[1]) and X.dtype == np.float32
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-3) and np.allclose(np.linalg.norm(T, axis=1), 1, atol=1e-3)
    sim = T @ X.T
    assert sim[0, 0] > sim[0, 1] and sim[1, 1] > sim[1, 0]
    assert getattr(enc.model, "logit_bias", None) is not None  # explain.py relies on SigLIP's sigmoid head
