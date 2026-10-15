# Privacy model

**Requirement:** photos, embeddings, OCR text and queries never leave the device. All ML runs locally, on the user's Mac or phone. No cloud LLM or vision API is used.

## What can touch the network

| Component | When | Notes |
|---|---|---|
| `scripts/setup.sh` (`make setup`) | Once | pip and npm packages, plus the SigLIP2 weights from Hugging Face. |
| Other model weights (Ask VLM, benchmark encoders, reranker) | First use, if not cached | Download them online once. `server.py` and `index.py` default to `HF_HUB_OFFLINE=1`. `rerank.py` reads only the local cache unless `PHOTOSEARCH_ALLOW_DOWNLOAD=1`. |
| Dataset scripts (`search/experiments/fetch_*.py`) | Only when you run them | Demo and benchmark data only. They never read the library. |
| iOS place names (`CLGeocoder`, `ios/Core/Indexing/PlaceResolver.swift`) | Only if the user turns it on (off by default) | Sends coordinates to Apple. It is the app's only networked feature. |
| Android | Never | No INTERNET permission. Library-added network permissions are stripped, and `verifyNoNetwork*` fails the build if any survives. ML Kit OCR would send usage metrics with no opt-out, so it is **off by default**, and the missing permission blocks the upload anyway. |

**Fully offline Mac:** after `make setup`, and after running Ask once online if you want it, everything runs with `HF_HUB_OFFLINE=1`. `server.py` forces it, along with `HF_HUB_DISABLE_TELEMETRY=1`. `index.py`, `eval_search.py` and the Qwen scripts default to it. OCR is Apple Vision, with no download. The web bundle loads nothing external, and `index.html` sets `referrer: no-referrer`.

## Network exposure (`search/server.py`)

- Binds to **127.0.0.1** only: :8000, :8100 for staging, Vite dev on :5173. `access_log=False`, so queries are not logged. `/docs` and `/openapi.json` are disabled.
- **Local-only middleware** runs on every route:

  | Check | Rejected with |
  |---|---|
  | `Host` not on the allow-list (127.0.0.1/localhost on the server port or 5173) | 421, blocks DNS rebinding |
  | `Sec-Fetch-Site` not same-origin/none | 403 |
  | A non-GET with a foreign `Origin` | 403 |
  | Body over 25 MiB | 413 |

  Responses carry `Cross-Origin-Resource-Policy: same-origin` and `nosniff`. `/api/ask` and the tool routes accept only JSON or multipart, never `text/plain` cross-site simple requests.
- **Input caps:**
  - uploads: 20 MiB, 50 MP, JPEG/PNG/WebP/HEIF/GIF only
  - query or question: 300 characters
  - explain tags: 32
  - `k` ≤ 500
- **Remaining gap: there is no auth token.** Any local process or user that can reach 127.0.0.1 can call the API, including the mutating tool routes. The `X-Actor` header in the audit log is self-declared.
- `scripts/install-service.sh` installs a per-user launchd agent (no root). `backend/` (legacy) uses Postgres on localhost:5433.

## What is stored where

| Data | Location |
|---|---|
| Original photos | Left in place and referenced by absolute path, never copied or modified. |
| Metadata, OCR text (plus `ocr_fast`), FTS index, image stats, albums, trash flags, `audit_log` | `search/data/index[-xl]/library.sqlite` (+ `-wal`/`-shm`). Override with `PHOTOSEARCH_INDEX`. |
| Embeddings, cleanup prompt vectors | `vectors__<model>.npz` and `cleanup_prompts__<model>.npz` in the index directory |
| Same-place head | `search/data/models/hotel_head.pt` (override with `PHOTOSEARCH_HEAD`) |
| Logs | `search/data/server.log` (launchd), `search/data/watchdog.log`, indexing progress logs. None of them contain queries. |
| Agent sessions, pending previews, undo records, confirm tokens | Server memory only. Sessions expire after 30 min, tokens after 15 min. |
| Thumbnails | Server memory (LRU, 96 MiB) plus the browser cache (`private, max-age=3600`) |
| Uploaded query photos | Memory only |
| Model weights | `~/.cache/huggingface/hub/` |
| Guard slot files (pid + first 100 chars of the command line) | `/tmp/photosearch-ml.*` |
| iOS | SQLite in Application Support, excluded from backup |
| Android | SQLite, plus embeddings in `noBackupFilesDir`. Backup and device transfer are excluded. |

`search/data/` is gitignored. The index contains OCR'd text from receipts, documents and screenshots, so treat it as sensitive as the photos. The `audit_log` stores each mutating call's arguments (up to 4,000 characters). Any future sync must be end-to-end encrypted.

## Agent tools

`search/photosearch/tools.py` has 15 tools. They are reachable in-process (`tools.call`), over HTTP (`api_tools.py`), and through the mock agent (`api_agent.py`). The mobile `PhotoTools` reuse the same names and arguments for the subset they implement. On iOS that is `search_photos`, `create_album`, `add_to_album` and `trash_photos`.

**What the tools can do:**

- Search.
- Read metadata.
- Explain a photo.
- Create, rename and delete albums, and add or remove photos. Deleting an album removes only the album.
- **Trash** photos (soft delete).
- Restore photos.
- List the trash.
- **Purge** the trash.
- Suggest cleanup candidates.

**What they can't do:**

- **Delete, move or modify a file.** Trash sets `photos.deleted_at`, and trashed photos disappear from search, browse and albums. Purge only sets `purged_at`, after which the tools can no longer restore the photo. The hook that would free disk space, `purge_files_hook()`, is deliberately unwired. On Android, trash goes through the system's own restorable trash dialog.
- **Act destructively in one step.** `trash_photos`, `purge_trash`, `delete_album` and `remove_from_album` default to `dry_run=true`. The preview returns a single-use HMAC `confirm_token` tied to the exact ids, valid for 15 min. The mock agent's `Executor` writes the confirmation text from the dry run, not from the planner, and runs the step only after `/api/agent/confirm`.
- **Exceed per-call caps:** 200 trashed, 500 purged, 1000 per album change, 200 results.
- **Act unrecorded.** Every mutating call, including dry runs and failures, is written to `audit_log`. The agent records an undo for each mutation it runs.

Cleanup only proposes candidates, each with reasons and a confidence score. A user or agent decides.
