FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY requirements.txt .
COPY setup.py .
COPY src/ ./src/

# Install Python dependencies with UV (much faster than pip)
RUN uv pip install --system -r requirements.txt

COPY config_file/ ./config_file/
COPY templates/ ./templates/
COPY app.py .
COPY main.py .

# Install the application in editable mode
RUN uv pip install --system -e .

# Create necessary directories and set permissions
RUN mkdir -p logs artifacts models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]