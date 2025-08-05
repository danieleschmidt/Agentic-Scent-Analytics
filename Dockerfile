# Multi-stage Docker build for Agentic Scent Analytics
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r agentic && useradd -r -g agentic agentic

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio black flake8 mypy coverage

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e ".[dev,industrial,llm]"

# Set ownership
RUN chown -R agentic:agentic /app

# Switch to non-root user
USER agentic

# Default command for development
CMD ["python", "-m", "agentic_scent.cli", "demo"]

# Production stage
FROM base as production

# Copy only necessary files
COPY agentic_scent/ ./agentic_scent/
COPY examples/ ./examples/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R agentic:agentic /app

# Copy production configuration
COPY docker/production.env /app/.env

# Switch to non-root user
USER agentic

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import agentic_scent; print('OK')" || exit 1

# Expose ports
EXPOSE 8000 8090

# Default command for production
CMD ["python", "-m", "agentic_scent.cli", "status"]

# Minimal runtime stage
FROM python:3.11-alpine as minimal

# Install minimal system dependencies
RUN apk add --no-cache gcc musl-dev

# Create user
RUN addgroup -S agentic && adduser -S agentic -G agentic

# Set working directory
WORKDIR /app

# Copy minimal requirements
COPY requirements-minimal.txt .

# Install minimal dependencies
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy core modules only
COPY agentic_scent/core/ ./agentic_scent/core/
COPY agentic_scent/sensors/base.py ./agentic_scent/sensors/base.py
COPY agentic_scent/sensors/mock.py ./agentic_scent/sensors/mock.py
COPY agentic_scent/sensors/__init__.py ./agentic_scent/sensors/__init__.py
COPY agentic_scent/__init__.py ./agentic_scent/__init__.py

# Set ownership
RUN chown -R agentic:agentic /app

# Switch to non-root user
USER agentic

# Health check
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=2 \
    CMD python -c "import agentic_scent.core.config; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "from agentic_scent.core.config import get_config; print('Agentic Scent Analytics - Minimal Runtime Ready')"]