# Photos

**Search every photo you own by just describing it, and keep all of it on your own device.**

How many times have you been on your iPhone, unable to find the photos of the room you used to live in, and the search features didn't help either? Your 64,243 photos are clogging up your iCloud storage and don't leave much room for error in organization. This is unacceptable. Let's improve search and agentify your photos experience.

You remember the moment, not the filename. "The dog on the beach that sunset in Bali." "The menu with the coffee prices." "Every other picture of that hotel room." Photos finds them in milliseconds, and it does it without uploading a single image, embedding or query anywhere. The models, the index and the answers all live on your Mac or your phone. That isn't a setting you can turn off. It's how the whole thing is built.

```sh
make setup                            # once: Python env, model download, web build (the only time anything touches the network)
make index DIR=~/Pictures/SomeTrip    # read, embed and OCR your photos, locally
make start                            # http://127.0.0.1:8000, best in Chrome or Edge
```

Then just start typing. For a guided two-minute tour, follow [docs/demo.md](docs/demo.md).

---

### What happens when you type "beach photos from Bali in 2019"

The query parser ([`query.py`](search/photosearch/query.py)) notices that *Bali* is a place and *2019* is a date. Both become hard filters, so nothing from Lisbon or 2016 can sneak in. "Bali" also matches Indonesian place names through aliases in 29 languages. What's left, "beach photos", goes to **SigLIP2**, which turns words and images into the same kind of vector. Your library was turned into those vectors once, at index time, so ranking 72,887 photos is a single matrix multiply that takes a few milliseconds. Words that actually appear *inside* photos (signs, menus, receipts, read on-device by Apple Vision) get their own vote, so "stop sign" finds photos that literally say STOP.

Tap a result and a small glass card tells you *why* it matched ("beach · sunset · palm trees"), using the same model's calibrated probabilities instead of guesswork. Tap **Same place** and a model we trained ourselves takes over. It's a small head on top of SigLIP2, taught on hotel photos what makes two photos the *same room*, not just similar rooms. Drop in a photo instead of text and it all works the same way.

Ask a real question ("how many photos have a hot air balloon?") and a local vision-language model (Qwen3.5-4B via MLX) plans the search, looks at the top hits itself, and answers with the photos it checked. Ask it to *do* something ("clean up my duplicates", "make an album of Japan") and the agent shows you exactly what would change. It does nothing until you confirm, every change has Undo, and no file is ever deleted: trash is just a flag.

### What we measured

We didn't pick these models on vibes. Every claim below comes from benchmarks in [`search/experiments/`](search/experiments) and is logged in [`search/NOTES.md`](search/NOTES.md):

- **Text → photo:** SigLIP2-B/16-384 finds the right Flickr photo first 82% of the time, against 67% for the CLIP model we started with.
- **The search box,** on 91 realistic queries: right photo first 59% → **74%**, right photo in the top ten 75% → **89%**. Date questions went from essentially random to 100% in the top ten.
- **"Same place",** our own trained head: the right hotel first 19.6% → **35.9%**, and across the whole library 16% → **31%**. It costs about 2 ms and needed no re-indexing. ([write-up](docs/same-place-head.md))
- **Things we tried and dropped:** bigger Qwen3-VL embedding and re-ranking models. They tied on text and were 10× slower, or bought quality at 40 seconds a query. We kept the code for a bigger machine.

### What's in the repo

The heart is [`search/`](search): the Python engine (`photosearch/`), the local server that the [`web/`](web) demo talks to, the agent tools and mock agent, model training in `train/`, and 288 tests that run in about ten seconds without loading a model (`make test`). [`ios/`](ios) and [`android/`](android) are the same idea on a phone: SigLIP2 converted to Core ML and ONNX ([`ml/`](ml)), indexing in the background, and no network at all. [`desktop/`](desktop) is a Mac app you can [download](desktop/README.md) and try without an app store: the same Swift core, sandboxed without network permission. [`backend/`](backend) is the original API stub, kept for history.

The longer story lives in [`docs/`](docs): [architecture](docs/architecture.md), [privacy](docs/privacy.md), [development](docs/development.md), [models](docs/ml.md), [iOS](docs/ios.md), and [data & licenses](docs/data-and-licenses.md).

### What isn't done yet

We'd rather tell you than have you find out:
- **The phone apps** build and their unit tests pass, but neither has run end to end on a simulator or device yet.
- **The glass:** real refraction renders in Chrome and Edge; Safari gets frosted glass. Native Liquid Glass on iOS needs Xcode 26, and the swap is one modifier.
- **Ask** needs a Mac with MLX, so a hosted demo would ship search, Same place and the agent without it.
- **The agent's brain is a mock:** rules, not a model. The safety layer around it is real. Plug your own planner in through [`agent_stub.py`](search/agent_stub.py) and it inherits preview, confirm and undo.
- **The demo photos** (Unsplash Lite, Places365, COCO, Hotels-50K) are fine for running locally, **not** for publishing. See [data & licenses](docs/data-and-licenses.md) before you host anything.

### Two ground rules

**Nothing leaves the device.** The server binds to `127.0.0.1` and refuses requests from other websites. The models run offline after setup. The iOS app has no network use, and the Android app doesn't even request the internet permission. Only the hosted web demo records visits ([deploy.md](docs/deploy.md#analytics-hosted-demo-only)).

**Be kind to the machine.** This was built on a 16 GB Mac that kernel-panicked when too many models ran at once. So every model load goes through a shared budget ([`guard.py`](search/photosearch/guard.py)), and a [watchdog](search/watchdog.py) steps in before disk or memory run out. Run heavy jobs one at a time, and let the guard queue the rest.
