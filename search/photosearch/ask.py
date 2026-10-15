"""Answer free-form questions about the library with a LOCAL vision-language model (mlx-vlm). Nothing leaves the machine.

  1. plan    question -> visual queries, literal text to find, date/place filters, answer type
  2. recall  hybrid retrieval for each query, fused
  3. verify  the VLM looks at numbered contact sheets of the top hits, keeps true matches, answers
"""
from __future__ import annotations

import json
import re
import time
from datetime import date

from PIL import Image, ImageDraw, ImageOps

from . import guard
from .encoders import load_image
from .search import Filters, Query, Searcher

VLM = "mlx-community/Qwen3.5-4B-4bit"

PLAN = """You turn questions about someone's personal photo library into a search plan. Today is {today} ({weekday}).
Return ONLY a JSON object with these keys:
  "queries": 1-3 short visual descriptions of what matching photos look like. The first one must be a faithful
     visual paraphrase of the question. Never invent details the user did not mention. Leave out dates, places and
     counting words, because those go in the other keys.
  "text_in_photo": a few distinctive words that would be literally written in the photo (signs, menus, documents) or null,
  "after": "YYYY-MM-DD" or null, "before": "YYYY-MM-DD" or null. Resolve relative dates against today: "last summer"
     = the most recent June 1 - Sept 1 that has already ended, "last year" = the previous calendar year, "in 2019" =
     2019-01-01 to 2020-01-01. Use null if the question has no time.
  "place": a city/country/place name to filter on or null,
  "answer_type": "photos" (they want to see photos) | "answer" (a fact, read from photos) | "count" (how many).
{image_note}
Question: {question}"""

IMAGE_NOTE = """The user attached a photo. Add key "image_role": "same_place" (other photos of that place/room/object),
"similar" (photos like it), or "modify" (like it but changed as described; then put the change in "queries")."""

VERIFY_REF = """The first image is the user's REFERENCE photo. The second image shows {n} numbered candidate photos from their library.
Question: {question}
Judge each candidate against the reference: for "same place/room/object" questions, look for the same distinctive details
(furniture, decor, patterns, layout, fixtures). Different views of the same room count as matches.
Return ONLY JSON: {{"matches": [numbers of candidates that satisfy the question], "answer": "a short direct answer, or null"}}"""

VERIFY = """These are {n} numbered photos from the user's library (number in the top-left corner).
Question: {question}
Look carefully at each photo. Return ONLY JSON: {{"matches": [numbers of photos that satisfy the question], "answer": "a short direct answer to the question based only on what you see, or null"}}"""


class Asker:
    def __init__(self, searcher: Searcher, model: str = VLM):
        self.s, self.model_name = searcher, model
        self._m = None

    def _load(self):
        if self._m is None:
            from mlx_vlm import load
            guard.acquire(3)  # the ~3 GB VLM on top of whatever this process already holds
            self._m = load(self.model_name)
        return self._m

    def _gen(self, prompt: str, images: list[Image.Image] = (), max_tokens: int = 400) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, proc = self._load()
        guard.check(force=True)
        p = apply_chat_template(proc, model.config, prompt, num_images=len(images), enable_thinking=False)
        r = generate(model, proc, p, image=list(images) or None, max_tokens=max_tokens, temperature=0.0, verbose=False)
        return getattr(r, "text", r)

    @staticmethod
    def _json(s: str) -> dict:
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
        m = re.search(r"\{.*\}", s, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}

    def plan(self, question: str, has_image: bool) -> dict:
        raw = self._gen(PLAN.format(today=date.today().isoformat(), weekday=date.today().strftime("%A"), question=question,
                                    image_note=IMAGE_NOTE if has_image else ""), max_tokens=300)
        p = self._json(raw)
        p.setdefault("queries", [question])
        p["queries"] = [q for q in p["queries"] if isinstance(q, str) and q.strip()][:3] or [question]
        p.setdefault("answer_type", "photos")
        return p

    def recall(self, plan: dict, image=None, k: int = 60) -> list[dict]:
        f = Filters(after=plan.get("after"), before=plan.get("before"), place=plan.get("place"))
        runs = []
        role = plan.get("image_role")
        if image is not None and role in ("same_place", "similar", None):
            runs.append(self.s.search(Query(image=image, filters=f, k=k, mode="place" if role == "same_place" else "similar")))
        texts = list(plan["queries"]) + ([plan["text_in_photo"]] if plan.get("text_in_photo") else [])
        for t in texts:
            if image is not None and role == "modify":
                runs.append(self.s.search(Query(text=t, image=image, text_weight=0.6, filters=f, k=k)))
            elif image is None or role != "same_place":
                runs.append(self.s.search(Query(text=t, filters=f, k=k)))
        if not any(runs) and f.active():  # filters too strict (e.g. no GPS) -> retry without them
            plan = {**plan, "after": None, "before": None, "place": None, "_relaxed": True}
            return self.recall(plan, image, k)
        score, by = {}, {}
        for run in runs:
            for n, r in enumerate(run):
                score[r["id"]] = score.get(r["id"], 0) + 1 / (60 + n)
                by[r["id"]] = r
        return [by[i] for i in sorted(score, key=score.get, reverse=True)][:k]

    def verify(self, question: str, cands: list[dict], sheet: int = 12, ref=None) -> tuple[list[int], str | None]:
        matches, answers = [], []
        for start in range(0, len(cands), sheet):
            chunk = cands[start:start + sheet]
            imgs = [contact_sheet([c["path"] for c in chunk])]
            if ref is not None:
                ref_img = ref if isinstance(ref, Image.Image) else load_image(self.s.lib.get([ref])[0]["path"])
                imgs.insert(0, ImageOps.contain(ref_img, (600, 600)))
            raw = self._gen((VERIFY_REF if ref is not None else VERIFY).format(n=len(chunk), question=question), imgs)
            j = self._json(raw)
            for n in j.get("matches") or []:
                if isinstance(n, int) and 1 <= n <= len(chunk):
                    matches.append(chunk[n - 1]["id"])
            if j.get("answer"):
                answers.append(str(j["answer"]))
        return matches, (answers[0] if answers else None)

    def ask(self, question: str, image=None, verify_n: int = 24) -> dict:
        t0 = time.time()
        plan = self.plan(question, image is not None)
        cands = self.recall(plan, image)
        t1 = time.time()
        matches, answer = self.verify(question, cands[:verify_n], ref=image) if cands else ([], None)
        keep = set(matches)
        ranked = [c for c in cands if c["id"] in keep] + [c for c in cands if c["id"] not in keep]
        for c in ranked:
            c["verified"] = c["id"] in keep
        if plan["answer_type"] == "count":
            answer = f"{len(matches)} matching photos among the top {min(verify_n, len(cands))} checked" + (f" — {answer}" if answer else "")
        return {"plan": plan, "answer": answer, "results": ranked,
                "timing": {"plan+recall_s": round(t1 - t0, 1), "verify_s": round(time.time() - t1, 1)}}


def contact_sheet(paths: list[str], cols: int = 4, cell: int = 300) -> Image.Image:
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        im = ImageOps.contain(load_image(p), (cell - 6, cell - 6))
        x, y = (i % cols) * cell, (i // cols) * cell
        sheet.paste(im, (x + 3, y + 3))
        d.rectangle([x + 3, y + 3, x + 45, y + 35], fill="yellow")
        d.text((x + 10, y + 6), str(i + 1), fill="black", font_size=26)
    return sheet
