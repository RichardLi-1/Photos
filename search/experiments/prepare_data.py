"""Extract benchmark images to plain folders + write ground-truth JSON."""
import io, json, os
import pandas as pd, pyarrow.parquet as pq
from PIL import Image

D = os.path.join(os.path.dirname(__file__), "..", "data")

def hotels():
    out = f"{D}/hotels/images"; os.makedirs(out, exist_ok=True)
    t = pq.read_table(f"{D}/hotels/unoccluded-0.parquet").to_pandas()
    gt = {}
    for r in t.itertuples():
        fn = r.filename
        p = f"{out}/{fn}"
        if not os.path.exists(p):
            with open(p, "wb") as f: f.write(r.image["bytes"])
        gt[fn] = str(r.hotel_id)
    json.dump(gt, open(f"{D}/hotels/gt.json", "w"))
    print("hotels", len(gt))

def roxford():
    out = f"{D}/roxford/images"; os.makedirs(out, exist_ok=True)
    for split in ["corpus", "query"]:
        t = pd.read_parquet(f"{D}/roxford/{split}.parquet")
        sub = f"{out}/{split}"; os.makedirs(sub, exist_ok=True)
        for r in t.itertuples():
            Image.open(io.BytesIO(r.image if isinstance(r.image, bytes) else r.image["bytes"])).convert("RGB").save(f"{sub}/{r.id}.jpg", quality=95)
        print("roxford", split, len(t))
    q = pd.read_parquet(f"{D}/roxford/qrels.parquet")
    gt = {qid: {"pos": g[g.score == 1]["corpus-id"].tolist(), "eval": g["corpus-id"].tolist()} for qid, g in q.groupby("query-id")}
    json.dump(gt, open(f"{D}/roxford/gt.json", "w"))

if __name__ == "__main__":
    hotels(); roxford()
