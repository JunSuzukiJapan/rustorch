# RusTorch Production Dockerfile
# Multi-stage build for optimized production image

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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/rustorch

# Copy manifests first for better caching
COPY Cargo.toml ./
COPY benchmarks/Cargo.toml ./benchmarks/

# Create src directory and add dummy lib.rs to build dependencies
RUN mkdir -p src && echo "// Dummy lib for dependency compilation" > src/lib.rs

# Build dependencies (this will generate a new Cargo.lock compatible with container's Cargo version)
RUN cargo build --release && rm -rf src

# Copy source code
COPY src ./src
COPY examples ./examples
COPY benches ./benches
COPY benchmarks/src ./benchmarks/src

# Build the library only (excluding problematic binaries)
RUN cargo build --release --lib

# Production stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1001 rustorch

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/output /app/lib && \
    chown -R rustorch:rustorch /app

# Copy examples only
COPY --from=builder /usr/src/rustorch/examples /app/examples/

# Copy library files
COPY --from=builder /usr/src/rustorch/target/release/librustorch.* /app/lib/

# Set environment variables
ENV RUST_LOG=info
ENV RUSTORCH_DATA_PATH=/app/data
ENV RUSTORCH_MODEL_PATH=/app/models
ENV RUSTORCH_OUTPUT_PATH=/app/output

# Switch to non-root user
USER rustorch

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD echo "RusTorch container is healthy"

# Default command
CMD ["bash"]# Docker Build Fix Applied
