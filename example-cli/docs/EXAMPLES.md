# RusTorch CLI - Usage Examples

Comprehensive examples for using the RusTorch CLI effectively.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Model Management](#model-management)
- [Backend Selection](#backend-selection)
- [Session Management](#session-management)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)

## Basic Usage

### Starting the CLI

```bash
# Default mode (CPU backend, dummy model)
rustorch-cli

# With verbose logging
rustorch-cli --verbose

# Quiet mode
rustorch-cli --log-level error
```

### Simple Conversation

```bash
$ rustorch-cli

You> Hello!
Assistant> I understand you said: "Hello!"

You> What is Rust?
Assistant> That's an interesting point about: What is Rust?

You> /exit
Goodbye!
```

### Multiline Input

```bash
You> Write a function that \
...> calculates the factorial \
...> of a number in Rust
Assistant> [Generated response with code...]
```

## Model Management

### Loading Different Models

```bash
# Load GGUF model
rustorch-cli --model ~/models/llama-7b-q4.gguf

# Load Safetensors model
rustorch-cli --model ~/models/model.safetensors

# Load ONNX model
rustorch-cli --model ~/models/model.onnx
```

### Switching Models During Session

```bash
You> /model ~/models/llama-7b.gguf
Loading model: /Users/user/models/llama-7b.gguf
Model path updated.

You> /model
Current model: /Users/user/models/llama-7b.gguf
Usage: /model <path>
```

## Backend Selection

### Using Different Backends

```bash
# CPU (default)
rustorch-cli --backend cpu

# Metal (macOS)
rustorch-cli --backend metal

# CUDA (NVIDIA GPU)
rustorch-cli --backend cuda
```

### Switching Backends During Session

```bash
You> /backend metal
Backend switched to: metal
Note: Backend switch will take effect for new models.

You> /backend cuda
Warning: Backend 'cuda' may not be available on this system
Attempting to switch anyway...
Backend switched to: cuda
```

### Backend-Specific Builds

```bash
# Build with Metal support
cargo build --release --features metal
./target/release/rustorch-cli --backend metal

# Build with CUDA support
cargo build --release --features cuda
./target/release/rustorch-cli --backend cuda

# Build with multiple backends
cargo build --release --features "metal cuda"
```

## Session Management

### Saving Conversations

```bash
# Save to default location
You> /save
Conversation saved to: ~/.rustorch/history/session.json

# Save to specific file
You> /save chat-2024.json
Conversation saved to: chat-2024.json
```

### Loading Previous Conversations

```bash
# Load from default location
You> /load
Conversation loaded from: ~/.rustorch/history/session.json

# Load from specific file
You> /load chat-2024.json
Conversation loaded from: chat-2024.json
```

### Auto-Save Feature

```bash
# Enable auto-save
rustorch-cli --auto-save

# Saves automatically on exit
You> /exit
Saving session...
Goodbye!
```

## Advanced Features

### Generation Parameters

```bash
# Creative mode (high temperature)
rustorch-cli --temperature 1.0 --top-p 0.95

# Precise mode (low temperature)
rustorch-cli --temperature 0.2 --top-p 0.85

# Maximum length responses
rustorch-cli --max-tokens 4096

# Short responses
rustorch-cli --max-tokens 256
```

### System Prompts

```bash
# Set system prompt via command line
rustorch-cli --system "You are a helpful coding assistant"

# Set during session
You> /system You are a helpful AI that explains concepts simply
System prompt updated.
```

### Statistics and Monitoring

```bash
You> /stats
╔════════════════════════════════════════════════════════════╗
║                     Statistics                             ║
╚════════════════════════════════════════════════════════════╝

  Messages:        10
  Total tokens:    450
  Backend:         cpu
  Model:           dummy-model
```

### Configuration Display

```bash
You> /config
╔════════════════════════════════════════════════════════════╗
║                   Configuration                            ║
╚════════════════════════════════════════════════════════════╝

  Model:           dummy-model
  Backend:         cpu
  Max tokens:      2048
  Temperature:     0.70
  Top-p:           0.90
  Top-k:           40
```

## Configuration

### Command-Line Configuration

```bash
# Full configuration example
rustorch-cli \
  --model ~/models/llama-7b.gguf \
  --backend metal \
  --max-tokens 2048 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 40 \
  --system "You are a helpful assistant" \
  --auto-save \
  --verbose
```

### Configuration File

Create `~/.rustorch/config.toml`:

```toml
[model]
default = "~/models/llama-7b.gguf"
format = "gguf"

[backend]
default = "metal"
fallback = ["cpu"]

[generation]
max_tokens = 2048
temperature = 0.7
top_p = 0.9
top_k = 40

[ui]
colored = true
show_progress = true
stream_tokens = true

[session]
auto_save = true
save_path = "~/.rustorch/history"
```

Use configuration file:

```bash
rustorch-cli --config ~/.rustorch/config.toml
```

## Use Cases

### Code Generation

```bash
# Optimized for code
rustorch-cli --temperature 0.2 --max-tokens 1024

You> Write a Rust function for binary search
Assistant> [Clean, focused code generation...]
```

### Creative Writing

```bash
# Optimized for creativity
rustorch-cli --temperature 0.9 --top-p 0.95 --max-tokens 2048

You> Write a short story about time travel
Assistant> [Creative narrative generation...]
```

### Technical Explanations

```bash
# Balanced settings
rustorch-cli --temperature 0.5 --max-tokens 1024

You> Explain how neural networks work
Assistant> [Clear technical explanation...]
```

### Interactive Debugging

```bash
# Load previous session for context
rustorch-cli --load debug-session.json

You> Continue analyzing the error from earlier
Assistant> [Continues from previous context...]
```

## Troubleshooting Examples

### Checking Model Format

```bash
# Verify model file
$ file model.bin
model.bin: data

# Check with verbose mode
$ rustorch-cli --model model.bin --verbose
ERROR: Unsupported model format
```

### Memory Issues

```bash
# Reduce token limit
rustorch-cli --max-tokens 512

# Use quantized model
rustorch-cli --model model-q4.gguf
```

### Backend Problems

```bash
# Try different backend
rustorch-cli --backend cpu

# Check feature flags
$ rustorch-cli --version
rustorch-cli 0.1.0
Features: cpu, metal
```

## Performance Optimization

### Fastest Inference

```bash
# Metal on macOS
cargo build --release --features metal
rustorch-cli --backend metal --max-tokens 512

# CUDA on Linux/Windows
cargo build --release --features cuda
rustorch-cli --backend cuda --batch-size 1
```

### Memory Efficiency

```bash
# Small context window
rustorch-cli --max-tokens 256

# Quantized model
rustorch-cli --model model-q4.gguf
```

## Batch Processing

### Processing Multiple Prompts

```bash
# Create input file
cat > prompts.txt << EOF
Explain quantum computing
What is Rust?
How do neural networks work?
EOF

# Process (future feature)
rustorch-cli --batch prompts.txt --output results.json
```

## Integration Examples

### Scripting

```bash
#!/bin/bash
# Simple wrapper script

MODEL="$HOME/models/llama-7b.gguf"
BACKEND="metal"

rustorch-cli \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --temperature 0.7 \
  --max-tokens 1024
```

### Environment Variables

```bash
# Set default model
export RUSTORCH_MODEL="~/models/llama-7b.gguf"
export RUSTORCH_BACKEND="metal"

rustorch-cli
```

## Tips and Best Practices

### For Best Results

1. **Model Selection**: Use GGUF Q4 quantization for balance
2. **Temperature**: 0.2 for code, 0.7 for general, 0.9 for creative
3. **Context**: Use system prompts to set behavior
4. **Sessions**: Save important conversations
5. **Backend**: Use GPU when available

### Common Workflows

**Code Review:**
```bash
rustorch-cli --temperature 0.3 --system "You are a code reviewer"
You> Review this Rust code: [paste code]
```

**Learning:**
```bash
rustorch-cli --temperature 0.5 --max-tokens 1024
You> Explain [concept] like I'm a beginner
```

**Brainstorming:**
```bash
rustorch-cli --temperature 0.9 --top-p 0.95
You> Generate 10 project ideas for [topic]
```

## Next Steps

- See [README.md](../README.md) for installation
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for roadmap
