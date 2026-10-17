---
title: Private Photo Search (demo)
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8080
pinned: false
license: apache-2.0
short_description: Semantic photo search demo (SigLIP2, ONNX, CPU); photos from PASS (CC BY)
---

Hosted demo of a private, local-first photo search. On your own device nothing leaves it; here, queries run on this
Space's CPU and are not logged. Ask, search-by-upload and anything that changes photos are off in the demo.
Photo credits: `/credits` (PASS / YFCC100M, CC BY; place names from GeoNames, CC BY 4.0). Model: SigLIP2 (Apache-2.0).

Space settings → Variables: `DEMO_ALLOWED_HOSTS=<user>-<space>.hf.space`, `DEMO_FRAME_ANCESTORS=https://huggingface.co`,
`DEMO_CLIENT_IP_HEADER=x-forwarded-for`, `ORT_THREADS=2`.
