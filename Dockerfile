# ── Stage 1: Build dependencies ───────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for numpy/scipy compilation (if needed from source)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime image ────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="quant-engineering"
LABEL description="SDCF Valuation API — Bottazzi et al. (2019)"

# Non-root user for security
RUN useradd -m -u 1000 sdcf
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=sdcf:sdcf sdcf/ ./sdcf/

USER sdcf

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "sdcf.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--access-log"]
