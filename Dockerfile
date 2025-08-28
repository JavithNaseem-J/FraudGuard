FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    libc-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv \
    && apt-get purge -y --auto-remove gcc g++ libc-dev

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy Poetry files and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r <(uv pip compile pyproject.toml)

# Copy application and artifacts
COPY src/ ./src/
COPY config_file/ ./config_file/
COPY templates/ ./templates/
COPY artifacts/ ./artifacts/
COPY app.py .
COPY main.py .

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Healthcheck (verify /health endpoint exists in app.py)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]