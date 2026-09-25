FROM node:20-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim

ARG BUILD_COMMIT_SHA=unknown
ARG BUILD_TIME
ARG UV_VERSION=0.11.15

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ARTIFACT_CACHE_ROOT=/app/runtime/model-releases \
    BUILD_COMMIT_SHA=${BUILD_COMMIT_SHA} \
    BUILD_TIME=${BUILD_TIME}

WORKDIR /app

# Install locked dependencies while native build tools are available, then
# remove the build-only packages from the runtime image.
COPY requirements.lock ./
COPY pyproject.toml ./
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    gcc \
    g++ \
    libc-dev \
    && pip install "uv==${UV_VERSION}" \
    && uv pip install --system -r requirements.lock \
    && apt-get purge -y --auto-remove gcc g++ libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy the API and the compiled frontend. The final image contains no Node.js
# runtime or frontend development dependencies.
COPY src/ ./src/
COPY app.py .
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# Persist an immutable UTC image-build timestamp after the source copy so a
# new commit cannot reuse a timestamp from an older cached image layer.
RUN if [ -n "${BUILD_TIME}" ]; then \
      printf '%s\n' "${BUILD_TIME}"; \
    else \
      date -u +%Y-%m-%dT%H:%M:%SZ; \
    fi > /app/build-time.txt

# Set permissions
RUN mkdir -p /app/runtime/model-releases \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Readiness is the traffic and deployment gate.
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ready || exit 1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
