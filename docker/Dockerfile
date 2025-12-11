# Multi-stage build for arxiv-corpus

# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install package and dependencies
RUN pip install --no-cache-dir build \
    && python -m build --wheel \
    && pip install --no-cache-dir dist/*.whl

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Stage 2: Runtime
FROM python:3.12-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PDF processing dependencies
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/arxiv-corpus /usr/local/bin/arxiv-corpus

# Copy configuration
COPY config/ ./config/

# Create data directories
RUN mkdir -p data/raw data/processed data/output

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ARXIV_CORPUS_CONFIG=/app/config/default.yaml

# Entry point
ENTRYPOINT ["arxiv-corpus"]
CMD ["--help"]
