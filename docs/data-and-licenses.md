# Data and licenses

Checked on 2026-10-15 against the model and dataset cards and terms pages linked below. **This is not legal advice.** Have the terms reviewed before anything ships. "Unverified" means no license statement was found.

## Datasets

| Dataset | Used for | Source used here | License / terms | Demo | Product |
|---|---|---|---|---|---|
| **Unsplash Lite** (25k photos) | Demo libraries: 15k in `index`, all 25k in `index-xl`. Also the eval set. | `unsplash-research-dataset-lite` zip + `images.unsplash.com` ([terms](https://github.com/unsplash/datasets/blob/master/TERMS.md)) | Lite grant: use to train ML models "for your internal business purposes". It bars publishing or redistributing any portion (§3A–B), and disputed images must be deleted (§2C). | OK on your own machine. A **public hosted demo** that displays the images is not clearly allowed. | Don't bundle or redistribute. |
| **Hotels-50K** test ("unoccluded" split, 5,390 images) | "Same room" benchmark + demo album `hotel-rooms` | HF mirror [`alperctnkaya/hotels-50k-test`](https://huggingface.co/datasets/alperctnkaya/hotels-50k-test). Original: [GWUvision/Hotels-50K](https://github.com/GWUvision/Hotels-50K) | **Unverified.** Neither the mirror nor the GitHub repo states a license. The images come from travel websites and the TraffickCam app (anti-trafficking research). | Internal benchmarking only. Avoid showing them publicly. | No. |
| **Hotels-50K** train (3 of 11 shards; 25.8k photos used) | Training our same-place head (`search/train/` → `hotel_head.pt`) | HF mirror [`alperctnkaya/hotel-images-200k-50k-hotels`](https://huggingface.co/datasets/alperctnkaya/hotel-images-200k-50k-hotels) | **Unverified** (empty card). Same sources as the test set. | Research only. | `hotel_head.pt` carries the same uncertainty. Don't ship it until this is cleared; retrain on licensed or user-owned data instead. |
| **Flickr30k** 1k test | Text→image benchmark + demo album `flickr` | HF mirror [`nlphuji/flickr_1k_test_image_text_retrieval`](https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval) (no license on the card) | Original: "non-commercial research and/or educational purposes". Images must follow the Flickr Terms of Use ([source](https://shannon.cs.illinois.edu/DenotationGraph/)). | Benchmarking OK. Public display is questionable. | No. |
| **rOxford** (Revisited Oxford, medium) | Landmark benchmark | HF mirror [`mteb/r-oxford-medium-multi`](https://huggingface.co/datasets/mteb/r-oxford-medium-multi) (empty card) | Oxford Buildings images are from Flickr: they must "respect Flickr's Terms and Conditions" and the VGG Terms of Access ([source](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)). No license is stated for the rOxford annotations. | Benchmarking OK. | No. |
| **Places365** val (36,500 images, 256 px) | XL demo index, head regression check (`bench_head_places.py`) | `data.csail.mit.edu` via `experiments/fetch_xl.py` | "The copyright of all the images belongs to the image owners" ([CSAILVision/places365](https://github.com/CSAILVision/places365)). The download page's non-commercial research/education terms are **unverified** because the site was unreachable when checked. | Local demo only. Don't show it publicly. | No. |
| **COCO** val2017 (5,000 images) | XL demo index | `images.cocodataset.org` via `fetch_xl.py` | Annotations are CC BY 4.0. For images: "The COCO Consortium does not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use." Per-image licenses vary ([terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)). | Local demo OK. Public display needs a per-image license check. | No, unless filtered to commercially licensed images. |

For any public demo, use photos you own or that are clearly licensed, such as images fetched individually under the Unsplash License. Keep the benchmark datasets internal.

## Models

| Model | License | Caveats |
|---|---|---|
| **SigLIP2** (`timm/ViT-B-16-SigLIP2-384`, open_clip `webli`) | Apache-2.0 ([card](https://huggingface.co/timm/ViT-B-16-SigLIP2-384)) | Fine for a demo or a product. Keep the license notice. The Core ML (iOS) and ONNX int8 (Android) conversions are derivatives, under the same license. |
| **Qwen3-VL-Embedding-2B** (`mlx-community/Qwen3-VL-Embedding-2B-4bit`) | Apache-2.0: [upstream card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) and the MLX conversion card | Evaluated and not shipped (weights deleted). The license would allow a product. |
| **Qwen3-VL-Reranker** (2B evaluated, `mlx-community/Qwen3-VL-Reranker-2B-4bit`) | Apache-2.0 ([upstream card](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)). The MLX repo's card was not checked. | Evaluated and not shipped (too slow). |
| **Qwen3.5-4B** (`mlx-community/Qwen3.5-4B-4bit`, used by Ask) | Upstream [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) is Apache-2.0 | **The MLX 4-bit repo has no model card or license**, so its provenance is unverified. For a product, convert the upstream weights yourself. |
| CLIP ViT-B/32 `laion2b_s34b_b79k` (old iOS model, replaced) | MIT ([card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)) | The card calls it a research output: "any deployed use case … is currently out of scope". This is one reason iOS moved to SigLIP2. |
| MobileCLIP2 (benchmarked) | `apple-amlr`: "exclusively for Research Purposes", with no commercial or product use ([card](https://huggingface.co/apple/MobileCLIP2-S2)) | Benchmarking only. Don't ship it. |
| DINOv3 (benchmarked) | Meta "DINOv3 License", a custom license ([card](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m)) | Terms not reviewed (**unverified**). |
| DINOv2 (benchmarked) | Apache-2.0 per upstream (**not re-checked**) | n/a |
| Apple Vision OCR | Part of macOS/iOS | Used through the OS API. No weights are shipped. |
| ML Kit text recognition (Android, bundled) | Google ML Kit terms (**not reviewed**) | Sends usage metrics to Google with no opt-out, so it is off by default and blocked by the missing INTERNET permission. |
| Same-place head (`hotel_head.pt`, ours) | Our code. The weights derive from SigLIP2 (Apache-2.0) and Hotels-50K train (unverified). | See the Hotels-50K train row: not product-ready. |
