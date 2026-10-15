"""Score retrieval on eval_set.json.  usage: python experiments/eval_search.py [--parse | --ask-plan] [--limit N_PER_TYPE] [--types a,b]

--parse: the search box path (query.parse: dates/places become hard filters). Query embeddings are cached in data/cache,
so re-runs only load the text model (via the guard) for unseen query texts.
"""
import argparse, collections, json, os, sys, time
from datetime import datetime

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from index import INDEX  # noqa: E402
from photosearch.library import Library  # noqa: E402
from photosearch.search import Query, Searcher  # noqa: E402

HERE = os.path.dirname(__file__)
K = 60
METRICS = ("hit@1", "hit@10", "P@10", "R@60", "MRR")


def score(ranked: list[int], rel: set[int]) -> dict:
    hits = [i in rel for i in ranked[:K]]
    first = hits.index(True) if any(hits) else None
    return {"hit@1": float(any(hits[:1])), "hit@10": float(any(hits[:10])), "P@10": sum(hits[:10]) / 10,
            "R@60": sum(hits) / min(len(rel), K),  # capped: 1.0 = top-60 is all relevant when |rel| > 60
            "MRR": 0.0 if first is None else 1 / (first + 1)}


def table(rows: list[dict]) -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    by["ALL"] = rows
    out = {}
    print(f"\n{'type':14s} {'n':>3s} {'|rel|':>5s} " + " ".join(f"{m:>6s}" for m in METRICS))
    for t, rs in by.items():
        out[t] = {"n": len(rs), **{m: round(float(np.mean([r["metrics"][m] for r in rs])), 3) for m in METRICS}}
        med = int(np.median([r["n_rel"] for r in rs]))
        print(f"{t:14s} {len(rs):3d} {med:5d} " + " ".join(f"{out[t][m]:6.3f}" for m in METRICS))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask-plan", action="store_true", help="LLM query plan (filters) + recall, no visual verification")
    ap.add_argument("--parse", action="store_true", help="rule-based query parsing + hard filters (the server's search box)")
    ap.add_argument("--embed", choices=["visual", "original", "template"], default=None,
                    help="--parse: override what the search box embeds (default: its prompt ensemble of the visual text)")
    ap.add_argument("--limit", type=int, help="first N queries of each type")
    ap.add_argument("--types", help="comma-separated subset of query types")
    ap.add_argument("--eval-set", default=f"{HERE}/eval_set.json")
    ap.add_argument("--place", action="store_true", help="photo queries use the trained same-place head (Query.mode='place')")
    a = ap.parse_args()

    E = json.load(open(a.eval_set))
    qs = [q for q in E["queries"] if not a.types or q["type"] in a.types.split(",")]
    if a.limit:
        qs = [q for n, q in enumerate(qs) if sum(x["type"] == q["type"] for x in qs[:n]) < a.limit]

    lib = Library(INDEX, readonly=True)
    if a.ask_plan:
        lib.encoder  # waits for the machine-wide ML lock BEFORE snapshotting the vectors
    for attempt in range(10):  # the indexer may be rewriting the npz
        try:
            s = Searcher(lib); break
        except Exception as e:
            print("vectors not readable yet:", e); time.sleep(5)
    cache_path = f"{HERE}/../data/cache/eval_text__{lib.model_name}.npz"
    cache = dict(zip(*(lambda z: (z["t"].tolist(), list(z["X"])))(np.load(cache_path)))) if os.path.exists(cache_path) else {}
    n_cached = len(cache)

    def embed_text(t):
        t = typed[0] if a.embed == "original" and typed else f"a photo of {t}" if a.embed == "template" else t
        return cached(t)

    def cached(t, _orig=s.embed_text):
        if t not in cache:
            cache[t] = _orig(t)
        return cache[t]
    s.embed_text = embed_text
    if a.embed:  # variants replace the ensemble with one embedding
        s.embed_prompted = embed_text
    typed = []
    n_photos = lib.db.execute("SELECT count(*) FROM photos").fetchone()[0]
    emb = set(s.ids.tolist())
    all_rel = set().union(*(q["relevant"] for q in qs))
    print(f"embedded {len(emb)}/{n_photos} photos; {len(all_rel & emb)}/{len(all_rel)} relevant photos embedded")
    if a.ask_plan:
        from photosearch.ask import Asker
        asker = Asker(s)

    rows, skipped, t0 = [], [], time.time()
    for q in qs:
        rel, img = set(q["relevant"]) & emb, q.get("image_id")
        if not rel or (img is not None and img not in emb):
            skipped.append(q["id"]); continue
        t = time.time()
        plan = None
        if a.ask_plan:
            plan = asker.plan(q["query"], img is not None)
            res = asker.recall(plan, image=img, k=K)
        elif img is not None:  # the "more like this photo" button: image only
            res = s.search(Query(image=img, k=K, mode="place" if a.place else "similar"))
        elif a.parse:  # the search box: dates/places parsed out as hard filters
            typed[:] = [q["query"]]
            sq = Query(text=q["query"], k=K, parse=True)
            res = s.search(sq)
            plan = sq.parsed
        else:  # the plain search box: query text as typed, no filters
            res = s.search(Query(text=q["query"], k=K))
        ranked = [r["id"] for r in res]
        rows.append({"id": q["id"], "type": q["type"], "query": q["query"], "n_rel": len(rel),
                     "rel_coverage": round(len(rel) / len(q["relevant"]), 3), "metrics": score(ranked, rel),
                     "top10": ranked[:10], "rel_ranks": [n for n, i in enumerate(ranked) if i in rel][:10],
                     "plan": plan, "secs": round(time.time() - t, 2)})
    if skipped:
        print(f"skipped {len(skipped)} queries with nothing embedded: {', '.join(skipped)}")
    summary = table(rows)

    if len(cache) > n_cached:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, t=np.array(list(cache)), X=np.stack(list(cache.values())))
    mode = "ask-plan" if a.ask_plan else ("parse" + ("-embed-" + a.embed if a.embed else "")) if a.parse else "plain"
    os.makedirs(f"{HERE}/results", exist_ok=True)
    out = f"{HERE}/results/eval_{mode}_{datetime.now():%Y%m%d-%H%M%S}.json"
    json.dump({"mode": mode, "eval_set": E["meta"], "embedded": len(emb), "photos": n_photos, "skipped": skipped,
               "secs": round(time.time() - t0, 1), "summary": summary, "queries": rows}, open(out, "w"), indent=1)
    print(f"\n{len(rows)} queries in {time.time() - t0:.0f}s -> {out}")
