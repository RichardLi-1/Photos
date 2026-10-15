"""Explain a match in words: which visual concepts a result shares with the query (zero-shot, same embedding space)."""
from __future__ import annotations

import re

import numpy as np

CONCEPTS = sorted(set("""
dog, cat, bird, horse, cow, sheep, deer, bear, elephant, giraffe, zebra, lion, tiger, cheetah, rhino, monkey, fish,
coral reef, butterfly, bee, insect, snake, turtle, duck, swan, owl, eagle, seagull, penguin, whale, dolphin, jellyfish,
person, man, woman, child, baby, crowd, couple, family, portrait, selfie, smiling face, hands, silhouette of a person,
wedding, bride, musician, guitar, piano, dancer, athlete, soccer, basketball, baseball, tennis, running, cycling,
skateboarding, surfing, swimming, skiing, snowboarding, hiking, climbing, camping, tent, yoga, martial arts, fishing,
beach, ocean waves, sea cliff, island, palm trees, sand dunes, desert, mountains, snowy peaks, glacier, volcano,
valley, hills, meadow, field of grass, wheat field, farm, forest, pine trees, autumn leaves, jungle, river, lake,
waterfall, canyon, rock formation, cave, coastline, harbor, pier, lighthouse, bridge, road, highway, winding road,
railway, train, city skyline, skyscrapers, city street at night, alley, market, shop front, cafe, restaurant, bar,
church, cathedral, temple, mosque, castle, ruins, monument, statue, fountain, park, garden, playground, stadium,
house, cabin, apartment building, old town, village, neon lights, street sign, graffiti, mural, construction site,
car, vintage car, truck, bus, motorcycle, scooter, bicycle, boat, sailboat, ship, airplane, hot air balloon, helicopter,
bedroom, bed, headboard, pillows, hotel room, bathroom, bathtub, shower curtain, sink, toilet, tiled floor,
kitchen, kitchenette, microwave, living room, sofa, armchair, desk, office chair, dining table, chairs, carpet,
patterned carpet, wooden floor, curtains, window with a view, lamp, ceiling light, mirror, framed picture on wall,
television, wardrobe, door, staircase, hallway, balcony, pool, gym, lobby, beige walls, white walls, brick wall,
laptop, phone, camera, book, notebook, newspaper, document, receipt, menu, sign with text, poster, map, clock,
cup of coffee, tea, wine glass, beer, cocktail, breakfast, pizza, burger, sushi, noodles, salad, cake, dessert,
fruit, vegetables, bread, cooked meal on a plate, ice cream, chocolate, candles, flowers, roses, sunflowers, tulips,
daisies, houseplant, cactus, tree, christmas tree, fireworks, balloons, gift, toy, shoes, hat, glasses, jacket,
dress, jewelry, watch, bag, umbrella, flag, sunset, sunrise, golden hour, night sky, stars, milky way, moon, clouds,
fog, rain, snow, storm, rainbow, blue sky, overcast sky, reflection in water, shadows, smoke, fire, light rays,
aerial view, close-up, macro shot, wide landscape, black and white photo, vintage film look, long exposure,
red tones, blue tones, green tones, pastel colors, warm light, dark moody photo, bright and airy
""".replace("\n", " ").split(", ")) - {""})
CONCEPTS = [c.strip() for c in CONCEPTS if c.strip()]


class Explainer:
    """Uses SigLIP's sigmoid head, so each concept score is a calibrated P(photo shows concept)."""

    def __init__(self, searcher):
        self.s = searcher
        enc = searcher.lib.encoder
        self.T = enc.texts([f"a photo of {c}" for c in CONCEPTS]).astype(np.float32)
        m = enc.model
        self.scale = float(m.logit_scale.exp()) if hasattr(m, "logit_scale") else 100.0
        self.bias = float(m.logit_bias) if getattr(m, "logit_bias", None) is not None else -10.0

    def p(self, v: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.scale * (self.T @ v) + self.bias)))

    def _top(self, score, k, floor):
        """Top-k above an absolute floor and within 15x of the best (SigLIP probs are small but well ranked)."""
        order = np.argsort(-score)[:k]
        best = score[order[0]] if len(order) else 0
        return [CONCEPTS[i] for i in order if score[i] >= floor and score[i] >= best / 15]

    def tags(self, v: np.ndarray, k: int = 4, floor: float = 0.002) -> list[str]:
        return self._top(self.p(v), k, floor)

    def shared_with_image(self, v_ref: np.ndarray, v: np.ndarray, k: int = 4) -> list[str]:
        return self._top(np.minimum(self.p(v_ref), self.p(v)), k, 0.001)  # present in BOTH photos

    def shared_with_tags(self, tags: list[str], v: np.ndarray, k: int = 4) -> list[str]:
        p, idx = self.p(v), np.array([CONCEPTS.index(t) for t in tags if t in CONCEPTS], dtype=int)
        mask = np.full(len(CONCEPTS), -1.0); mask[idx] = p[idx]
        return self._top(mask, k, 0.001)

    def matched_text(self, q: str, v: np.ndarray, k: int = 3) -> list[str]:
        """Concepts close in meaning to the query that the photo clearly shows."""
        rel = self.T @ self.s.embed_text(q)
        rel = (rel - rel.mean()) / (rel.std() + 1e-6)
        return self._top(np.where(rel > 2.0, self.p(v), -1.0), k, 0.001)


STOP = set("the and for with from that this are was were you your our photo photos picture pictures image images show find "
           "some any all into onto over under near".split())


def word_hits(q: str, text: str | None) -> list[str]:
    if not text:
        return []
    low = text.lower()
    return [w for w in dict.fromkeys(re.findall(r"\w{3,}", q.lower())) if w not in STOP and re.search(rf"\b{re.escape(w)}", low)]
