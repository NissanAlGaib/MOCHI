# MOCHI gateway.
#
# Two decisions shape this image, both measured rather than assumed.
#
# **CPU-only torch.** The installed development build is torch 2.6.0+cu124,
# of which 4.3 GB out of 4.4 GB is CUDA kernels. Stage II runs on CPU in any
# deployment that also serves a local model - an 8 GB card holding a 7B target
# at an 8k context has no room for a second resident model, and Stage II costs
# about 200 ms per request there. Installing from the CPU wheel index drops
# roughly 4 GB from the image for no loss of function.
#
# **The model is mounted, not copied.** models/e5-fine-tuned is 468 MB, and
# baking it in would mean rebuilding to change a checkpoint and shipping
# weights inside an image people may push to a registry. docker-compose mounts
# it read-only instead, so the same image serves whichever checkpoint is
# mounted - which is also how the e5-fine-tuned / e5-banking comparison would
# be run in production.

FROM python:3.11-slim AS base

# libgomp1: onnxruntime (RapidOCR) links against OpenMP and fails at import
# without it. libglib2.0-0 and libgl1: Pillow and opencv-python-headless,
# pulled in by RapidOCR, need them to load at all. The error without these is
# an ImportError at request time rather than at build time, which is the worst
# place to discover a missing library.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so a source change does not reinstall torch.
COPY requirements.txt ./

# torch and transformers are commented out in requirements.txt on purpose -
# they are only needed when Stage II is enabled, and the project keeps the
# base install light. The gateway is not useful without Stage II (Stage I
# alone measured 5.9% recall), so the image installs them explicitly, from the
# CPU index.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4.0" \
    && pip install "transformers>=4.45.0" \
    && pip install -r requirements.txt

COPY mochi/ ./mochi/
COPY training/ ./training/

# Stage II imports training.model to rebuild the classifier head, so training/
# is a runtime dependency despite the name. eval/ is not - it is the
# evaluation harness and never runs in production.

# Non-root. The gateway needs no write access beyond its log directory.
RUN useradd --create-home --uid 10001 mochi \
    && mkdir -p /app/logs \
    && chown -R mochi:mochi /app/logs
USER mochi

ENV MOCHI_HOST=0.0.0.0 \
    MOCHI_PORT=8000 \
    MOCHI_LOG_PATH=/app/logs/mochi.jsonl \
    MOCHI_ENABLE_STAGE2=true \
    MOCHI_STAGE2_DEVICE=cpu \
    MOCHI_STAGE2_MODEL_DIR=/app/models/e5-fine-tuned

EXPOSE 8000

# Stage II loads a 468 MB model and scores a warm-up string before the app
# accepts traffic, so start_period covers a slow first boot rather than a
# per-request cost.
#
# Note what this does and does not prove. /health returns 200 from config
# alone and never touches the detector, so a passing healthcheck means the
# process is up, not that Stage II is live. That gap is covered at startup
# instead: app.py scores ["warmup"] eagerly, so a missing or unreadable model
# mount raises there and the container exits. With restart: unless-stopped
# that shows up as a restart loop, which is the intended loud failure -
# serving Stage I alone (5.9% recall) while an operator believes Stage II is
# running would be the quiet one.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "mochi.gateway.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
