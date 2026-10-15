"""photosearch.explain: word_hits / stopwords, and the Explainer's concept ranking with the fake encoder."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from photosearch.explain import CONCEPTS, STOP, Explainer, word_hits
from photosearch.search import Searcher


@pytest.mark.parametrize("q,text,want", [
    ("lunch menu", "Lunch MENU pasta", ["lunch", "menu"]),
    ("show me photos of the menu", "today's menu", ["menu"]),        # stopwords dropped
    ("menu", "Menus of the day", ["menu"]),                           # prefix at a word start
    ("enu", "menu", []),                                              # ...but not mid-word
    ("menu menu MENU", "menu", ["menu"]),                             # de-duplicated, lower-cased
    ("go to an ox", "go to an ox", []),                               # words under 3 chars ignored
    ("paris", None, []),
    ("paris", "", []),
    ("c++ (menu) [x] .*", "menu", ["menu"]),                          # regex metacharacters are inert
    ("café", "Café de Flore", ["café"]),                              # unicode words
    ("louvre ticket", "Louvre ticket admission", ["louvre", "ticket"]),
    ("ticket louvre", "Louvre ticket admission", ["ticket", "louvre"]),  # query order
])
def test_word_hits(q, text, want):
    assert word_hits(q, text) == want


def test_stopwords():
    for w in ("the", "photo", "photos", "show", "find", "with", "near"):
        assert w in STOP
    for w in ("menu", "beach", "paris"):
        assert w not in STOP
    assert word_hits(" ".join(STOP), " ".join(STOP)) == []


def test_concepts_clean():
    assert len(CONCEPTS) > 200 and len(set(CONCEPTS)) == len(CONCEPTS)
    assert all(c == c.strip() and c for c in CONCEPTS)


@pytest.fixture
def E(ro_lib):
    return Explainer(Searcher(ro_lib))


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_probabilities_in_range(E, fake_encoder):
    p = E.p(fake_encoder.text_vec("red"))
    assert p.shape == (len(CONCEPTS),) and ((p >= 0) & (p <= 1)).all()
    assert (E.scale, E.bias) == (100.0, -10.0)       # fallbacks when the model has no logit_scale / logit_bias


def test_tags_find_the_colour(E, fake_encoder):
    red = fake_encoder.image_vec(Image.new("RGB", (4, 4), (255, 0, 0)))
    tags = E.tags(red, k=4)
    assert tags[0] == "red tones" and len(tags) <= 4 and set(tags) <= set(CONCEPTS)


def test_shared_with_tags_is_a_subset(E, fake_encoder):
    v = fake_encoder.image_vec(Image.new("RGB", (4, 4), (255, 0, 0)))
    out = E.shared_with_tags(["red tones", "blue tones", "not a concept"], v)
    assert out == ["red tones"]
    assert E.shared_with_tags([], v) == []


def test_shared_with_image_needs_both(E, fake_encoder):
    red, blue = (fake_encoder.image_vec(Image.new("RGB", (4, 4), c)) for c in ((255, 0, 0), (0, 0, 255)))
    assert "red tones" in E.shared_with_image(red, red)
    assert "red tones" not in E.shared_with_image(red, blue)


def test_matched_text(E, fake_encoder):
    out = E.matched_text("blue tones", fake_encoder.text_vec("blue tones"))
    assert out and out[0] == "blue tones"


def test_top_floor(E):
    assert E._top(np.full(len(CONCEPTS), 1e-6), k=5, floor=0.001) == []
