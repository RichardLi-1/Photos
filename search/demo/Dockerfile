# Public photo-search demo: FastAPI + onnxruntime + numpy, read-only bundle, no torch.
# Build context = the directory written by `search/demo/stage.py OUT` (not the repo root):
#   OUT/Dockerfile  OUT/requirements.txt  OUT/app/{photosearch,mock_agent.py,demo,web}  OUT/bundle/
#   docker build -t photos-demo OUT && docker run --rm -p 8080:8080 -e DEMO_ALLOWED_HOSTS=localhost photos-demo
FROM python:3.11.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=1 \
    ORT_THREADS=1 \
    DEMO_BUNDLE=/bundle \
    DEMO_WEB=/app/web \
    HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
# onnxruntime ships its quantization/transformers tooling and tests: not needed to run a model
RUN pip install --no-compile -r /tmp/requirements.txt \
 && rm -rf /tmp/requirements.txt \
    /usr/local/lib/python3.11/site-packages/onnxruntime/tools \
    /usr/local/lib/python3.11/site-packages/onnxruntime/transformers \
    /usr/local/lib/python3.11/site-packages/onnxruntime/quantization \
    /usr/local/lib/python3.11/site-packages/onnxruntime/datasets \
 && find /usr/local/lib/python3.11 -depth -name '__pycache__' -type d -exec rm -rf {} + \
 && useradd --uid 1000 --create-home --shell /usr/sbin/nologin app

COPY --chown=1000:1000 app /app
COPY --chown=1000:1000 bundle /bundle

# uid 1000: required by Hugging Face Spaces, harmless elsewhere. The bundle is read-only at runtime; the server
# writes only a temp copy of library.sqlite (~10 MB) under /tmp.
USER 1000
EXPOSE 8080
CMD ["python", "/app/demo/demo_server.py"]
