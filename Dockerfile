# aurora-lens — multi-stage build
# Stage 1: install dependencies and spaCy model (separate RUNs — clear build logs)
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-editable install, same rule as the README. The build backend and licence
# files have to be in the context or the wheel build cannot run.
COPY pyproject.toml aurora_lens_build_backend.py release_guard.py setup.py ./
COPY LICENSE NOTICE LICENSING.md README.md ./
COPY aurora_lens/ ./aurora_lens/

RUN pip install --no-cache-dir ".[spacy,proxy,redis]"
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

# The installed package is already in site-packages. aurora-lens.yaml is not
# baked into the image: copy aurora-lens.yaml.example to aurora-lens.yaml and
# mount it at /app/aurora-lens.yaml before starting the container.
COPY --chown=aurora:aurora pyproject.toml ./

# Config mounted at runtime. The container does not start correctly without it.
ENV AURORA_LENS_CONFIG=/app/aurora-lens.yaml

# Entrypoint runs as root to fix volume ownership, then drops to aurora
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "aurora_lens.proxy", "--config", "/app/aurora-lens.yaml", "--host", "0.0.0.0"]
