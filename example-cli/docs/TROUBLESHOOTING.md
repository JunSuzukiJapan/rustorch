# RusTorch CLI - Troubleshooting Guide

Common issues and their solutions.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Model Loading](#model-loading)
- [Backend Problems](#backend-problems)
- [Performance Issues](#performance-issues)
- [Session Management](#session-management)
- [Build Errors](#build-errors)

## Installation Issues

### Rust Version Too Old

**Problem:**
```bash
error: package `rustorch-cli` requires at least Rust 1.75
```

**Solution:**
```bash
# Update Rust
rustup update stable

# Verify version
rustc --version
# Should show: rustc 1.75.0 or higher
```

### Missing Dependencies

**Problem:**
```bash
error: linking with `cc` failed
```

**Solution:**
```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt-get install build-essential

# Fedora/RHEL
sudo dnf install gcc
```

## Model Loading

### File Not Found

**Problem:**
```bash
Error: Model file not found: model.gguf
```

**Solution:**
```bash
# Use absolute path
rustorch-cli --model /full/path/to/model.gguf

# Or verify current directory
ls -la model.gguf

# Check permissions
ls -l model.gguf
# Should be readable
```

### Unsupported Format

**Problem:**
```bash
Error: Unsupported model format: bin
```

**Solution:**
```bash
# Check file extension
file model.bin

# Supported formats:
# - .gguf (recommended)
# - .safetensors
# - .onnx

# Convert or obtain correct format
```

### Corrupted Model File

**Problem:**
```bash
Error: Failed to parse model format
```

**Solution:**
```bash
# Verify file integrity
md5sum model.gguf

# Re-download if corrupted
wget https://example.com/model.gguf

# Check file size
ls -lh model.gguf
```

## Backend Problems

### Metal Not Available

**Problem:**
```bash
Warning: Backend 'metal' may not be available
```

**Solution:**
```bash
# Rebuild with Metal support
cargo build --release --features metal

# Verify macOS version (requires 10.13+)
sw_vers

# Check Metal support
system_profiler SPDisplaysDataType | grep Metal
```

### CUDA Not Available

**Problem:**
```bash
Error: CUDA backend not available
```

**Solution:**
```bash
# Install CUDA toolkit
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version

# Rebuild with CUDA
cargo build --release --features cuda

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Backend Mismatch

**Problem:**
```bash
Error: Backend 'cuda' not available in this build
```

**Solution:**
```bash
# Check available backends
rustorch-cli --version

# Rebuild with required feature
cargo build --release --features cuda

# Use available backend
rustorch-cli --backend cpu
```

## Performance Issues

### Slow Inference

**Problem:** Generation takes too long

**Solutions:**

1. **Use GPU Backend:**
```bash
# Metal on macOS
rustorch-cli --backend metal

# CUDA on Linux/Windows
rustorch-cli --backend cuda
```

2. **Reduce Token Limit:**
```bash
rustorch-cli --max-tokens 512
```

3. **Use Quantized Model:**
```bash
# GGUF Q4 quantization
rustorch-cli --model model-q4.gguf
```

4. **Optimize Temperature:**
```bash
# Lower temperature = faster
rustorch-cli --temperature 0.2
```

### High Memory Usage

**Problem:** CLI uses too much memory

**Solutions:**

1. **Reduce Context:**
```bash
rustorch-cli --max-tokens 256
```

2. **Clear History:**
```bash
You> /clear
```

3. **Use Smaller Model:**
```bash
# Use 7B instead of 13B parameter model
rustorch-cli --model small-model.gguf
```

4. **Monitor Memory:**
```bash
# macOS
top -pid $(pgrep rustorch-cli)

# Linux
htop -p $(pgrep rustorch-cli)
```

### CPU at 100%

**Problem:** High CPU usage

**Solutions:**

1. **Use GPU:**
```bash
rustorch-cli --backend metal
```

2. **Reduce Concurrency:**
```bash
# Set environment variable
export RAYON_NUM_THREADS=4
rustorch-cli
```

## Session Management

### Cannot Save Session

**Problem:**
```bash
Error: Failed to save session
```

**Solution:**
```bash
# Check directory exists
mkdir -p ~/.rustorch/history

# Check permissions
ls -ld ~/.rustorch/history

# Use full path
You> /save /full/path/to/session.json

# Check disk space
df -h
```

### Cannot Load Session

**Problem:**
```bash
Error: Failed to load session
```

**Solution:**
```bash
# Verify file exists
ls -la session.json

# Check JSON format
cat session.json | jq .

# Try with verbose mode
rustorch-cli --verbose
You> /load session.json
```

### Corrupted History

**Problem:**
```bash
Error: Invalid JSON in history file
```

**Solution:**
```bash
# Backup corrupted file
cp session.json session.json.bak

# Validate JSON
jsonlint session.json

# Start fresh session
rm session.json
rustorch-cli
```

## Build Errors

### Compilation Fails

**Problem:**
```bash
error: could not compile `rustorch-cli`
```

**Solutions:**

1. **Clean Build:**
```bash
cargo clean
cargo build --release
```

2. **Update Dependencies:**
```bash
cargo update
cargo build --release
```

3. **Check Rust Version:**
```bash
rustup show
rustup update stable
```

### Feature Flags Conflict

**Problem:**
```bash
error: features `metal` and `cuda` cannot be used together
```

**Solution:**
```bash
# Build separately
cargo build --release --features metal
cargo build --release --features cuda

# Or use default
cargo build --release
```

### Linker Errors

**Problem:**
```bash
error: linking with `cc` failed
```

**Solution:**
```bash
# macOS
xcode-select --install

# Linux
sudo apt-get install build-essential pkg-config

# Verify linker
which cc
```

## Runtime Errors

### Panic on Startup

**Problem:**
```bash
thread 'main' panicked at 'assertion failed'
```

**Solution:**
```bash
# Run with backtrace
RUST_BACKTRACE=1 rustorch-cli

# Enable debug logging
rustorch-cli --log-level debug

# Check configuration
rustorch-cli --config /dev/null
```

### Segmentation Fault

**Problem:**
```bash
Segmentation fault (core dumped)
```

**Solution:**
```bash
# Run with debug build
cargo build
./target/debug/rustorch-cli

# Check memory limits
ulimit -a

# Try with different backend
rustorch-cli --backend cpu
```

## Common Error Messages

### "No such file or directory"

**Cause:** Invalid model path

**Fix:**
```bash
# Use absolute path
rustorch-cli --model $(pwd)/model.gguf

# Check existence
ls -la model.gguf
```

### "Permission denied"

**Cause:** Insufficient file permissions

**Fix:**
```bash
# Make executable
chmod +x target/release/rustorch-cli

# Fix model permissions
chmod 644 model.gguf
```

### "Out of memory"

**Cause:** Model too large for available RAM

**Fix:**
```bash
# Reduce max tokens
rustorch-cli --max-tokens 256

# Use quantized model
rustorch-cli --model model-q4.gguf

# Close other applications
```

### "Connection refused"

**Cause:** Network issues (if using remote models)

**Fix:**
```bash
# Check network
ping example.com

# Verify firewall
sudo ufw status

# Use local model
rustorch-cli --model local-model.gguf
```

## Debugging Tips

### Enable Verbose Logging

```bash
# Maximum verbosity
RUST_LOG=debug rustorch-cli --verbose

# Specific module
RUST_LOG=rustorch_cli::model=trace rustorch-cli

# Save logs to file
rustorch-cli --verbose 2>&1 | tee debug.log
```

### Check System Resources

```bash
# Memory
free -h

# Disk space
df -h

# CPU
lscpu

# GPU (NVIDIA)
nvidia-smi

# GPU (macOS)
system_profiler SPDisplaysDataType
```

### Verify Installation

```bash
# Check version
rustorch-cli --version

# Check features
rustorch-cli --version | grep Features

# Test basic functionality
echo "test" | rustorch-cli --model /dev/null || true
```

## Getting Help

### Collect Debug Information

```bash
# System info
uname -a
rustc --version
cargo --version

# Build info
rustorch-cli --version

# Run with debug
RUST_BACKTRACE=full rustorch-cli --verbose --log-level trace
```

### Report Issues

When reporting issues, include:

1. **System Information:**
   - OS and version
   - Rust version
   - RusTorch CLI version

2. **Steps to Reproduce:**
   - Exact commands run
   - Model format and size
   - Backend used

3. **Error Output:**
   - Full error message
   - Backtrace if available
   - Log output

4. **Configuration:**
   - Command-line flags
   - Config file (if used)

### Resources

- [GitHub Issues](https://github.com/JunSuzukiJapan/rustorch/issues)
- [Documentation](README.md)
- [Examples](EXAMPLES.md)
- [Implementation Plan](IMPLEMENTATION_PLAN.md)

## Advanced Debugging

### Memory Profiling

```bash
# Using valgrind
valgrind --tool=massif ./target/release/rustorch-cli

# Using heaptrack
heaptrack ./target/release/rustorch-cli
```

### Performance Profiling

```bash
# Using perf (Linux)
perf record ./target/release/rustorch-cli
perf report

# Using Instruments (macOS)
instruments -t "Time Profiler" ./target/release/rustorch-cli
```

### Core Dump Analysis

```bash
# Enable core dumps
ulimit -c unlimited

# Analyze with gdb
gdb ./target/release/rustorch-cli core

# Print backtrace
(gdb) bt
```

## Prevention

### Best Practices

1. **Always use release builds for production**
2. **Keep Rust toolchain updated**
3. **Use appropriate backend for hardware**
4. **Monitor memory usage**
5. **Save sessions regularly**
6. **Use quantized models when possible**
7. **Test with verbose mode first**

### Regular Maintenance

```bash
# Update Rust
rustup update

# Clean old builds
cargo clean

# Update dependencies
cargo update

# Run tests
cargo test
```
