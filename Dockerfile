# aurora-lens — multi-stage build
# Stage 1: install dependencies and spaCy model (separate RUNs — clear build logs)
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project and install (include redis for session backend)
COPY pyproject.toml ./
COPY aurora_lens/ ./aurora_lens/

RUN pip install --no-cache-dir -e ".[spacy,proxy,redis]"
RUN python -m spacy download en_core_web_sm

# Stage 2: minimal runtime
FROM python:3.12-slim

# Non-root user + gosu for privilege drop in entrypoint
RUN groupadd -r aurora && useradd -r -g aurora aurora && \
    apt-get update && apt-get install -y --no-install-recommends gosu && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /data && chown aurora:aurora /data

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app (for config discovery; proxy runs via -m)
COPY --chown=aurora:aurora aurora_lens/ ./aurora_lens/
COPY --chown=aurora:aurora pyproject.toml ./
COPY --chown=aurora:aurora aurora-lens.yaml ./aurora-lens.yaml

# Config mounted at runtime
ENV AURORA_LENS_CONFIG=/app/aurora-lens.yaml

# Entrypoint runs as root to fix volume ownership, then drops to aurora
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "aurora_lens.proxy", "--config", "/app/aurora-lens.yaml", "--host", "0.0.0.0"]
