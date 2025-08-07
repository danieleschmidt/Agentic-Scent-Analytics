# Multi-stage Dockerfile for Sentiment Analyzer Pro
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY requirements-minimal.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black flake8 mypy

# Copy source code
COPY . .

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Development command
CMD ["uvicorn", "sentiment_analyzer.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Install only production dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn

# Copy source code
COPY sentiment_analyzer/ ./sentiment_analyzer/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install -e .

# Create directories for logs and models
RUN mkdir -p /var/log/sentiment-analyzer /app/models && \
    chown -R appuser:appuser /var/log/sentiment-analyzer /app/models /app

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for dependencies\n\
echo "Waiting for dependencies..."\n\
sleep 10\n\
\n\
# Start the application\n\
if [ "$WORKER_MODE" = "true" ]; then\n\
    echo "Starting worker mode..."\n\
    python -m sentiment_analyzer.utils.async_processor\n\
else\n\
    echo "Starting API server..."\n\
    gunicorn sentiment_analyzer.api.main:app \\\n\
        --bind 0.0.0.0:8000 \\\n\
        --workers 4 \\\n\
        --worker-class uvicorn.workers.UvicornWorker \\\n\
        --worker-connections 1000 \\\n\
        --max-requests 1000 \\\n\
        --max-requests-jitter 100 \\\n\
        --timeout 60 \\\n\
        --keep-alive 2 \\\n\
        --access-logfile /var/log/sentiment-analyzer/access.log \\\n\
        --error-logfile /var/log/sentiment-analyzer/error.log \\\n\
        --log-level info\n\
fi' > /app/start.sh && chmod +x /app/start.sh

USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["/app/start.sh"]

# Minimal stage for lightweight deployment
FROM python:3.11-alpine as minimal

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal system dependencies
RUN apk add --no-cache curl gcc musl-dev

# Create app user
RUN addgroup -S appuser && adduser -S appuser -G appuser

WORKDIR /app

# Copy minimal requirements
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy only essential source files
COPY sentiment_analyzer/core/ ./sentiment_analyzer/core/
COPY sentiment_analyzer/api/ ./sentiment_analyzer/api/
COPY sentiment_analyzer/__init__.py ./sentiment_analyzer/
COPY setup.py .

RUN pip install -e . && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Minimal command
CMD ["python", "-m", "uvicorn", "sentiment_analyzer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Worker stage for background processing
FROM production as worker

USER root

# Install worker-specific dependencies if needed
RUN pip install celery[redis]

USER appuser

# Worker startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting Sentiment Analyzer worker..."\n\
python -c "\n\
import asyncio\n\
from sentiment_analyzer.utils.async_processor import get_global_processor, start_global_processor\n\
\n\
async def run_worker():\n\
    processor = get_global_processor()\n\
    await start_global_processor()\n\
    print(\"Worker started, waiting for tasks...\")\n\
    try:\n\
        while True:\n\
            await asyncio.sleep(1)\n\
    except KeyboardInterrupt:\n\
        print(\"Shutting down worker...\")\n\
        await processor.stop()\n\
\n\
if __name__ == \"__main__\":\n\
    asyncio.run(run_worker())\n\
"' > /app/worker.sh && chmod +x /app/worker.sh

CMD ["/app/worker.sh"]