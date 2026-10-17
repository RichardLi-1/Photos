# Deploying the public demo

The product runs on users' devices. The hosted demo is a read-only showcase of search on **licensed PASS photos** (YFCC100M, CC BY 2.0 per photo; see [data-and-licenses.md](data-and-licenses.md)). It is torch-free (int8 ONNX text tower, int8 vectors, precomputed same-place space): about 350 MB RAM, 50 ms per query on an M4, and a 0.55 GB image plus about 10 KB per photo. Numbers and how they were measured: `search/demo/measure.py`, `search/demo/results/`.

**Pick:** **Google Cloud Run** (scale to zero, 1 vCPU / 1 GiB). Fallback: **Fly.io** shared-cpu-1x 1 GB with auto-stop.

| Monthly cost (20k photos; a visit = 10 searches + 150 thumbnails ≈ 2 MB) | 1k visits | 50k visits |
|---|---|---|
| Cloud Run (free tier covers CPU and requests; egress about $0.12/GB, unverified) | ≈ $0.2 | ≈ $15 (≈ $3 with thumbnails on R2) |
| Fly.io 1 GB, auto-stop (egress $0.02/GB in NA/EU) | ≈ $1 | ≈ $9 |
| HF Spaces Docker (needs PRO since 2026) | $9 | $9 |

**Steps** (Mac, one heavy job at a time, keep 10 GB free):
1. `cd search && .venv/bin/python experiments/fetch_pass.py --out data/pass --n 20000 --tars 0-9 --geonames`. Streams about 2.5 GB, writes about 0.6 GB, takes about 1 h at Zenodo's ~0.8 MB/s, and can be resumed.
2. `PHOTOSEARCH_INDEX=data/index-pass .venv/bin/python index.py data/pass/images=pass` runs OCR and SigLIP2, guarded.
3. `.venv/bin/python demo/build_bundle.py --index data/index-pass --credits data/pass/credits.csv --out ../demo-bundle`. This ships Same place as precomputed int8 Z from `hotel_head.pt`; the head's weights are not shipped. Its Hotels-50K training data is still unverified, so add `--head none` until that is cleared.
4. `python3 search/demo/stage.py --bundle demo-bundle --out build/demo --target cloudrun`, or `--target fly` / `--target hf`.
5. Deploy one of these:
   - **Cloud Run:** run `gcloud run deploy photos-demo --source build/demo --region us-central1 --memory 1Gi --cpu 1 --concurrency 8 --min-instances 0 --max-instances 2 --cpu-boost --allow-unauthenticated --env-vars-file build/demo/cloudrun.env.yaml`. Then set a billing budget alert.
   - **Fly:** `cd build/demo && fly launch --no-deploy --copy-config --name <app>`, then edit `DEMO_ALLOWED_HOSTS`, then `fly deploy`.
   - **HF:** `hf upload <user>/<space> build/demo . --repo-type space`, then set the Variables listed in its README.
6. Check that `DEMO_ALLOWED_HOSTS` includes the real hostname (any other Host gets 421). On Cloud Run, deploy once with `DEMO_IP_DEBUG=1` and confirm that `X-Demo-Client-Key` shows your own IP.

To offload thumbnails, run `stage.py --export-thumbs DIR`, upload them to an R2 bucket on a custom domain (r2.dev is rate-limited), set `DEMO_THUMB_BASE_URL`, and stage with `--no-thumbs`.

**Off in the demo, and why:**
- **Ask:** needs a local VLM on a Mac (MLX). It falls back to plain search results with a note.
- **Search by uploaded photo:** this would upload visitors' photos, and the image tower would add 97 MB. "More like this" and "Same place" still work on demo photos.
- **Agent changes:** albums, trash, purge and restore would mutate a library everyone shares, so they are refused before they run. Cleanup is also off, since it needs on-device image stats. Read-only tools work. Sessions live in memory (30 min).
- **OCR:** precomputed on the Mac (Apple Vision); the server never runs it.

**Privacy:** the server writes nothing at runtime except a temp copy of the sqlite file. There is no access or query log, and per-IP rate limits are kept in memory. Its only outbound requests are the optional visit cards below. A CSP, `no-referrer` and same-origin CORP are set. Credits are at `/credits`.

## Analytics (hosted demo only)

Ported from Transit Planner's visitor tracking. Code: `web/src/visits.ts`, `search/demo/visits.py`. The browser part starts only when `/api/stats` returns a `demo` object, and only the demo server returns one. The local app, iOS, Android and desktop collect nothing.

- **Visit cards:** one per page load, plus one for a bounce under 10 s. The browser posts to `/api/track`, and the server adds the IP and an approximate location, then posts to Discord in the background. The location is the Fly edge region (`fly-region`), not a geo-IP lookup.
- **Referral**, checked in this order: `?l` LinkedIn, `?r` Resume, `?t` Twitter/X, `?e` Email, `?g` GitHub, `?c` Cover Letter, then `utm_source`, then an AI-assistant referrer, then the source stored from an earlier visit. Tracking params are then removed from the address bar. AI landings go to their own channel (teal cards), on the first visit only.
- **AI fetchers** (`ChatGPT-User`, `GPTBot`, `ClaudeBot`, `PerplexityBot`, ...): each fetch writes one `ai_fetch` JSON line to the Fly logs, with the path only. User-triggered fetches (`*-User`) also post to Discord.
- **What is sent:** event label, device and OS, path, landing URL, referral params and referrer, IP, and the edge region. **Never sent:** search text (`?q=` is removed from every URL), photos or photo IDs, or agent messages.
- **Skipped:** `localhost`, any browser that once opened a URL with `?m` (stored as `skip_tracking`), and holding `/` while the page loads.

| Env (all optional; none set = off) | Used for |
|---|---|
| `DISCORD_REGULAR_VISITS_WEBHOOK_URL` | Visits with no referral; falls back to `DISCORD_WEBHOOK_URL` |
| `DISCORD_REFERRAL_VISITS_WEBHOOK_URL` | Referral and other query-param visits; falls back to `DISCORD_WEBHOOK_URL` |
| `DISCORD_AI_VISITS_WEBHOOK_URL` | AI-assistant visits and `*-User` fetches; falls back to referral |
| `DISCORD_WEBHOOK_URL` | Fallback for every channel: one webhook is enough |

Fill in `search/demo/deploy/.env` from `.env.example` (gitignored), then run `fly secrets import -a photos-demo < search/demo/deploy/.env`. Webhook URLs stay on the server.
