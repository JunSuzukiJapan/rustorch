# RusTorch Python Bindings Dockerfile
# Multi-stage build for Python bindings testing and development

# Build stage
FROM rust:1.81-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    clang \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    maturin \
    numpy \
    typing-extensions

# Set working directory
WORKDIR /usr/src/rustorch-python

# Copy Python bindings source
COPY . .

# Build Python bindings
RUN maturin build --release --no-default-features

# Install the built wheel
RUN pip3 install target/wheels/*.whl

# Test stage
FROM python:3.9-slim as test

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    typing-extensions

# Copy the built wheel from builder stage
COPY --from=builder /usr/src/rustorch-python/target/wheels/*.whl /tmp/

# Install RusTorch Python bindings
RUN pip install /tmp/*.whl

# Copy test files
COPY tests/ /app/tests/
COPY python/ /app/python/

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/python
ENV RUST_LOG=info

# Create non-root user
RUN useradd -m -u 1001 rustorch
RUN chown -R rustorch:rustorch /app
USER rustorch

# Health check - test basic import
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import rustorch; print('RusTorch import successful')"

# Default command - run Phase 5 tests
CMD ["python3", "tests/test_phase5_autograd.py"]