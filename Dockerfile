FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ARTIFACT_CACHE_ROOT=/app/runtime/model-releases

WORKDIR /app

# Install locked dependencies while native build tools are available, then
# remove the build-only packages from the runtime image.
COPY requirements.lock ./
COPY pyproject.toml ./
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    libc-dev \
    && pip install uv \
    && uv pip install --system -r requirements.lock \
    && apt-get purge -y --auto-remove gcc g++ libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application and artifacts
COPY src/ ./src/
COPY config_file/ ./config_file/
COPY templates/ ./templates/
COPY app.py .

# Set permissions
RUN mkdir -p /app/runtime/model-releases \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Healthcheck uses liveness; readiness is exposed separately at /ready.
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/live || exit 1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
