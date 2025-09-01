#!/bin/bash

# RusTorch GPU-Enabled Rust Kernel Quick Start Script - Multilingual Support
# Auto-detects system language for international users
# Supports: English, Japanese, Spanish, French, German, Chinese, Korean
# 
# Usage: curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start_rust_kernel_gpu.sh | bash
# 使用法: curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start_rust_kernel_gpu.sh | bash

set -e

# Language Detection and Message System
detect_language() {
    local lang_code
    
    # Try multiple methods to detect language
    if [[ -n "${LC_ALL:-}" ]]; then
        lang_code="${LC_ALL%.*}"
    elif [[ -n "${LC_MESSAGES:-}" ]]; then
        lang_code="${LC_MESSAGES%.*}"
    elif [[ -n "${LANG:-}" ]]; then
        lang_code="${LANG%.*}"
    else
        lang_code="en_US"
    fi
    
    # Extract language prefix
    lang_code="${lang_code%_*}"
    
    case "$lang_code" in
        ja) echo "ja" ;;
        es) echo "es" ;;
        fr) echo "fr" ;;
        de) echo "de" ;;
        zh|zh_CN|zh_TW) echo "zh" ;;
        ko) echo "ko" ;;
        *) echo "en" ;;
    esac
}

# Multilingual message function for GPU Rust Kernel
msg() {
    local key="$1"
    local lang="${DETECTED_LANG:-en}"
    
    case "$key" in
        "welcome_title")
            case "$lang" in
                en) echo "🦀🚀 RusTorch GPU-Enabled Rust Kernel Quick Start" ;;
                ja) echo "🦀🚀 RusTorch GPU対応 Rust カーネル クイックスタート" ;;
                es) echo "🦀🚀 Inicio Rápido Kernel Rust GPU RusTorch" ;;
                fr) echo "🦀🚀 Démarrage Rapide Noyau Rust GPU RusTorch" ;;
                de) echo "🦀🚀 RusTorch GPU Rust Kernel Schnellstart" ;;
                zh) echo "🦀🚀 RusTorch GPU Rust 内核快速开始" ;;
                ko) echo "🦀🚀 RusTorch GPU Rust 커널 빠른 시작" ;;
            esac ;;
        "gpu_detection")
            case "$lang" in
                en) echo "🎮 Detecting GPU capabilities..." ;;
                ja) echo "🎮 GPU性能を検出中..." ;;
                es) echo "🎮 Detectando capacidades de GPU..." ;;
                fr) echo "🎮 Détection des capacités GPU..." ;;
                de) echo "🎮 GPU-Fähigkeiten erkennen..." ;;
                zh) echo "🎮 检测 GPU 功能..." ;;
                ko) echo "🎮 GPU 기능 감지 중..." ;;
            esac ;;
        "gpu_kernel_complete")
            case "$lang" in
                en) echo "🎉 GPU-Enabled Rust Kernel Setup Complete!" ;;
                ja) echo "🎉 GPU対応Rustカーネルセットアップ完了！" ;;
                es) echo "🎉 ¡Configuración Kernel Rust GPU Completa!" ;;
                fr) echo "🎉 Configuration Noyau Rust GPU Terminée!" ;;
                de) echo "🎉 GPU Rust Kernel Setup Abgeschlossen!" ;;
                zh) echo "🎉 GPU Rust 内核设置完成！" ;;
                ko) echo "🎉 GPU Rust 커널 설정 완료！" ;;
            esac ;;
        "detected_gpu_features")
            case "$lang" in
                en) echo "🎮 Detected GPU features:" ;;
                ja) echo "🎮 検出されたGPU機能:" ;;
                es) echo "🎮 Características GPU detectadas:" ;;
                fr) echo "🎮 Fonctionnalités GPU détectées:" ;;
                de) echo "🎮 Erkannte GPU-Features:" ;;
                zh) echo "🎮 检测到的 GPU 功能：" ;;
                ko) echo "🎮 감지된 GPU 기능:" ;;
            esac ;;
        "gpu_demo_loaded")
            case "$lang" in
                en) echo "📝 GPU Demo notebook loaded automatically" ;;
                ja) echo "📝 GPUデモノートブック自動読み込み済み" ;;
                es) echo "📝 Notebook demo GPU cargado automáticamente" ;;
                fr) echo "📝 Notebook de démo GPU chargé automatiquement" ;;
                de) echo "📝 GPU-Demo-Notebook automatisch geladen" ;;
                zh) echo "📝 GPU 演示笔记本自动加载" ;;
                ko) echo "📝 GPU 데모 노트북 자동 로드됨" ;;
            esac ;;
    esac
}

# Detect system language
DETECTED_LANG=$(detect_language)

# Display welcome message in user's language
echo "$(msg "welcome_title")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌍 Language detected: $DETECTED_LANG"
echo ""

# Create workspace
RUSTORCH_DIR="$HOME/rustorch-gpu-rust-kernel"
echo "📁 Creating RusTorch GPU Rust workspace: $RUSTORCH_DIR"

if [ -d "$RUSTORCH_DIR" ]; then
    echo "⚠️  Directory exists. Updating..."
    cd "$RUSTORCH_DIR"
    git pull origin main || echo "Git pull failed, continuing..."
else
    echo "📥 Downloading RusTorch..."
    git clone --depth 1 https://github.com/JunSuzukiJapan/rustorch.git "$RUSTORCH_DIR"
    cd "$RUSTORCH_DIR"
fi

echo ""
echo "🔍 Checking system requirements..."

# Check Rust
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version | cut -d' ' -f2)
    echo "✅ Rust: $RUST_VERSION"
else
    echo "📦 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    echo "✅ Rust installed!"
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python: $PYTHON_VERSION"
else
    echo "❌ Python 3 is required"
    exit 1
fi

# GPU Detection and Feature Configuration
echo ""
echo "$(msg "gpu_detection")"

GPU_FEATURES=""
GPU_FOUND=false

# NVIDIA CUDA Detection
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    echo "✅ NVIDIA GPU detected (Driver: $CUDA_VERSION)"
    GPU_FEATURES="cuda"
    GPU_FOUND=true
elif [[ -d "/usr/local/cuda" ]] || [[ -d "/opt/cuda" ]]; then
    echo "✅ CUDA installation detected"
    GPU_FEATURES="cuda"
    GPU_FOUND=true
fi

# Apple Metal Detection (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    METAL_CHECK=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -i "metal" || echo "")
    if [[ -n "$METAL_CHECK" ]]; then
        echo "✅ Apple Metal GPU detected"
        if [[ -n "$GPU_FEATURES" ]]; then
            GPU_FEATURES="$GPU_FEATURES,metal"
        else
            GPU_FEATURES="metal"
        fi
        GPU_FOUND=true
    fi
fi

# OpenCL Detection
if command -v clinfo &> /dev/null; then
    OPENCL_DEVICES=$(clinfo -l 2>/dev/null | grep -c "Platform #" || echo "0")
    if [[ "$OPENCL_DEVICES" -gt 0 ]]; then
        echo "✅ OpenCL GPU detected ($OPENCL_DEVICES platforms)"
        if [[ -n "$GPU_FEATURES" ]]; then
            GPU_FEATURES="$GPU_FEATURES,opencl"
        else
            GPU_FEATURES="opencl"
        fi
        GPU_FOUND=true
    fi
fi

if [[ "$GPU_FOUND" == false ]]; then
    echo "⚠️  No GPU acceleration detected. Using CPU-only mode."
    echo "   💡 For GPU support, install:"
    echo "   - NVIDIA: CUDA toolkit"
    echo "   - Apple: Already available on macOS"
    echo "   - Generic: OpenCL drivers"
    GPU_FEATURES=""
else
    echo "🚀 GPU features available: $GPU_FEATURES"
fi

# Check ZMQ (required for evcxr_jupyter)
echo ""
echo "🔧 Checking ZMQ dependency..."
if command -v pkg-config &> /dev/null && pkg-config --exists libzmq; then
    echo "✅ ZMQ already installed"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        echo "📦 Installing ZMQ via Homebrew..."
        brew install zmq pkg-config
    else
        echo "❌ Please install Homebrew first: https://brew.sh"
        exit 1
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📦 Installing ZMQ for Linux..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y libzmq3-dev pkg-config
    elif command -v yum &> /dev/null; then
        sudo yum install -y zeromq-devel pkgconfig
    else
        echo "❌ Please install libzmq3-dev manually"
        exit 1
    fi
else
    echo "⚠️  Unsupported OS for automatic ZMQ installation"
fi

echo ""
echo "🦀 Installing Rust Jupyter kernel (evcxr_jupyter)..."

# Install evcxr_jupyter
if command -v evcxr_jupyter &> /dev/null; then
    echo "✅ evcxr_jupyter already installed"
else
    echo "📦 Installing evcxr_jupyter..."
    cargo install evcxr_jupyter
fi

# Install the kernel
echo "🔧 Installing Rust kernel for Jupyter..."
evcxr_jupyter --install

echo ""
echo "📚 Installing Python dependencies..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip jupyter jupyterlab matplotlib pandas numpy

# Create GPU-enabled Rust kernel demo notebook
echo ""
echo "📝 Creating GPU-enabled Rust kernel demo notebook..."
mkdir -p notebooks

cat > notebooks/rustorch_gpu_rust_kernel_demo.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🦀🚀 RusTorch GPU-Enabled Rust Kernel Demo\n",
    "# 🦀🚀 RusTorch GPU対応 Rustカーネルデモ\n",
    "\n",
    "This notebook demonstrates how to use RusTorch with GPU acceleration directly in Rust within Jupyter!\n",
    "\n",
    "このノートブックでは、Jupyter内でRustを直接使ってGPU加速されたRusTorchを使用する方法を示します！"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📦 Install RusTorch with GPU Features\n",
    "## 📦 GPU機能付きRusTorchをインストール\n",
    "\n",
    "First, let's add RusTorch with GPU features as a dependency:\n",
    "\n",
    "まず、GPU機能付きRusTorchを依存関係として追加しましょう："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "// Check which GPU features are available on this system\n",
    ":dep rustorch = { version = \"0.5.9\", features = [\"cuda\", \"metal\", \"opencl\"] }\n",
    ":dep ndarray = \"0.16\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎯 Basic Tensor Operations with GPU\n",
    "## 🎯 GPU基本的なテンソル操作"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "use rustorch::*;\n",
    "use ndarray::prelude::*;\n",
    "\n",
    "// Check GPU availability\n",
    "let gpu_available = rustorch::cuda::is_available() || \n",
    "                   rustorch::metal::is_available() || \n",
    "                   rustorch::opencl::is_available();\n",
    "\n",
    "println!(\"🎮 GPU acceleration available: {}\", gpu_available);\n",
    "\n",
    "// Create tensors\n",
    "let a = Tensor::from_array(array![[1.0, 2.0], [3.0, 4.0]]);\n",
    "let b = Tensor::from_array(array![[5.0, 6.0], [7.0, 8.0]]);\n",
    "\n",
    "println!(\"Tensor a: {:?}\", a);\n",
    "println!(\"Tensor b: {:?}\", b);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "// Move tensors to GPU if available\n",
    "let device = if rustorch::cuda::is_available() {\n",
    "    println!(\"🚀 Using CUDA GPU acceleration\");\n",
    "    Device::Cuda(0)\n",
    "} else if rustorch::metal::is_available() {\n",
    "    println!(\"🍎 Using Apple Metal GPU acceleration\");\n",
    "    Device::Metal\n",
    "} else if rustorch::opencl::is_available() {\n",
    "    println!(\"⚡ Using OpenCL GPU acceleration\");\n",
    "    Device::OpenCL(0)\n",
    "} else {\n",
    "    println!(\"💻 Using CPU (no GPU available)\");\n",
    "    Device::Cpu\n",
    "};\n",
    "\n",
    "let a_gpu = a.to_device(&device);\n",
    "let b_gpu = b.to_device(&device);\n",
    "\n",
    "// GPU-accelerated matrix multiplication\n",
    "let result = a_gpu.matmul(&b_gpu);\n",
    "println!(\"GPU matrix multiplication result: {:?}\", result);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🧮 GPU-Accelerated Advanced Operations\n",
    "## 🧮 GPU加速高度な操作"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "// Create larger tensors for GPU performance benefits\n",
    "let large_tensor = Tensor::randn(&[1000, 1000]).to_device(&device);\n",
    "println!(\"Large tensor shape: {:?}\", large_tensor.shape());\n",
    "println!(\"Device: {:?}\", large_tensor.device());\n",
    "\n",
    "// GPU-accelerated activation functions\n",
    "let relu_result = large_tensor.relu();\n",
    "println!(\"GPU ReLU completed on {}x{} tensor\", 1000, 1000);\n",
    "\n",
    "let sigmoid_result = large_tensor.sigmoid();\n",
    "println!(\"GPU Sigmoid completed on {}x{} tensor\", 1000, 1000);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🤖 GPU-Accelerated Neural Network\n",
    "## 🤖 GPU加速ニューラルネットワーク"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "use rustorch::nn::*;\n",
    "\n",
    "// Create a neural network on GPU\n",
    "let mut network = Sequential::new()\n",
    "    .add_layer(Linear::new(784, 256).to_device(&device))\n",
    "    .add_layer(ReLU::new())\n",
    "    .add_layer(Linear::new(256, 128).to_device(&device))\n",
    "    .add_layer(ReLU::new())\n",
    "    .add_layer(Linear::new(128, 10).to_device(&device))\n",
    "    .add_layer(Softmax::new());\n",
    "\n",
    "println!(\"🧠 GPU Neural network created with {} parameters\", network.num_parameters());\n",
    "println!(\"🎮 Network device: {:?}\", device);\n",
    "\n",
    "// Create sample input on GPU\n",
    "let input = Tensor::randn(&[32, 784]).to_device(&device); // Batch size 32\n",
    "let output = network.forward(&input);\n",
    "\n",
    "println!(\"Input shape: {:?}\", input.shape());\n",
    "println!(\"Output shape: {:?}\", output.shape());\n",
    "println!(\"✅ GPU forward pass completed!\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ⚡ GPU Performance Benchmarks\n",
    "## ⚡ GPUパフォーマンスベンチマーク"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "use std::time::Instant;\n",
    "\n",
    "// Benchmark CPU vs GPU performance\n",
    "let size = 1000;\n",
    "let iterations = 5;\n",
    "\n",
    "println!(\"🏁 Benchmarking {}x{} matrix operations ({} iterations)...\", size, size, iterations);\n",
    "println!(\"\");\n",
    "\n",
    "// CPU benchmark\n",
    "println!(\"💻 CPU Benchmark:\");\n",
    "let a_cpu = Tensor::randn(&[size, size]);\n",
    "let b_cpu = Tensor::randn(&[size, size]);\n",
    "\n",
    "let start = Instant::now();\n",
    "for i in 0..iterations {\n",
    "    let _result = a_cpu.matmul(&b_cpu);\n",
    "    if i == 0 { println!(\"  Iteration {} completed\", i + 1); }\n",
    "}\n",
    "let cpu_duration = start.elapsed();\n",
    "let cpu_gflops = (2.0 * size as f64 * size as f64 * size as f64 * iterations as f64) \n",
    "                / (cpu_duration.as_secs_f64() * 1e9);\n",
    "\n",
    "println!(\"  ⏱️  CPU Time: {:?}\", cpu_duration);\n",
    "println!(\"  📊 CPU GFLOPS: {:.2}\", cpu_gflops);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "// GPU benchmark (if available)\n",
    "if gpu_available {\n",
    "    println!(\"\");\n",
    "    println!(\"🎮 GPU Benchmark:\");\n",
    "    let a_gpu = Tensor::randn(&[size, size]).to_device(&device);\n",
    "    let b_gpu = Tensor::randn(&[size, size]).to_device(&device);\n",
    "    \n",
    "    // Warm up GPU\n",
    "    let _ = a_gpu.matmul(&b_gpu);\n",
    "    \n",
    "    let start = Instant::now();\n",
    "    for i in 0..iterations {\n",
    "        let _result = a_gpu.matmul(&b_gpu);\n",
    "        if i == 0 { println!(\"  Iteration {} completed\", i + 1); }\n",
    "    }\n",
    "    let gpu_duration = start.elapsed();\n",
    "    let gpu_gflops = (2.0 * size as f64 * size as f64 * size as f64 * iterations as f64) \n",
    "                    / (gpu_duration.as_secs_f64() * 1e9);\n",
    "    \n",
    "    println!(\"  ⏱️  GPU Time: {:?}\", gpu_duration);\n",
    "    println!(\"  📊 GPU GFLOPS: {:.2}\", gpu_gflops);\n",
    "    \n",
    "    if gpu_duration < cpu_duration {\n",
    "        let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();\n",
    "        println!(\"  🚀 GPU Speedup: {:.2}x faster!\", speedup);\n",
    "    }\n",
    "} else {\n",
    "    println!(\"\");\n",
    "    println!(\"⚠️  GPU benchmark skipped (no GPU acceleration available)\");\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔥 Advanced GPU Examples\n",
    "## 🔥 高度なGPUの例"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if gpu_available {\n",
    "    // Convolutional layer example on GPU\n",
    "    println!(\"🧠 Creating GPU-accelerated CNN layer...\");\n",
    "    \n",
    "    // Create sample image tensor (batch_size=4, channels=3, height=32, width=32)\n",
    "    let image_batch = Tensor::randn(&[4, 3, 32, 32]).to_device(&device);\n",
    "    \n",
    "    // Create Conv2D layer on GPU\n",
    "    let conv_layer = Conv2d::new(3, 16, 3) // 3 input channels, 16 output, 3x3 kernel\n",
    "        .padding(1)\n",
    "        .to_device(&device);\n",
    "    \n",
    "    let conv_start = Instant::now();\n",
    "    let conv_output = conv_layer.forward(&image_batch);\n",
    "    let conv_duration = conv_start.elapsed();\n",
    "    \n",
    "    println!(\"📐 Input shape: {:?}\", image_batch.shape());\n",
    "    println!(\"📐 Output shape: {:?}\", conv_output.shape());\n",
    "    println!(\"⏱️  GPU Convolution time: {:?}\", conv_duration);\n",
    "    \n",
    "    // Batch processing example\n",
    "    println!(\"\");\n",
    "    println!(\"📊 Batch processing with GPU acceleration...\");\n",
    "    \n",
    "    let batch_size = 128;\n",
    "    let feature_size = 512;\n",
    "    let large_batch = Tensor::randn(&[batch_size, feature_size]).to_device(&device);\n",
    "    \n",
    "    let batch_start = Instant::now();\n",
    "    let normalized = large_batch.layer_norm(&[feature_size]);\n",
    "    let activated = normalized.gelu();\n",
    "    let batch_duration = batch_start.elapsed();\n",
    "    \n",
    "    println!(\"✅ Processed batch of {} samples in {:?}\", batch_size, batch_duration);\n",
    "    println!(\"📈 Throughput: {:.0} samples/second\", \n",
    "        batch_size as f64 / batch_duration.as_secs_f64());\n",
    "} else {\n",
    "    println!(\"⚠️  Advanced GPU examples skipped (GPU acceleration not available)\");\n",
    "    println!(\"💡 To enable GPU acceleration:\");\n",
    "    println!(\"   - Install CUDA drivers for NVIDIA GPUs\");\n",
    "    println!(\"   - Use macOS for Apple Metal support\");\n",
    "    println!(\"   - Install OpenCL drivers for other GPUs\");\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎉 Conclusion\n",
    "## 🎉 まとめ\n",
    "\n",
    "You can now write and execute GPU-accelerated Rust code directly in Jupyter!\n",
    "\n",
    "これでJupyter内で直接GPU加速されたRustコードを書いて実行できます！\n",
    "\n",
    "**Benefits / 利点:**\n",
    "- 🚀 Native Rust performance / ネイティブRustパフォーマンス\n",
    "- 🎮 GPU acceleration / GPU加速\n",
    "- 🔧 Direct library access / ライブラリへの直接アクセス\n",
    "- 🎯 Type safety / 型安全性\n",
    "- ⚡ Zero-cost abstractions / ゼロコスト抽象化\n",
    "- 🧠 Advanced ML operations / 高度なML操作"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Rust",
   "language": "rust",
   "name": "rust"
  },
  "language_info": {
   "codemirror_mode": "rust",
   "file_extension": ".rs",
   "mimetype": "text/rust",
   "name": "Rust",
   "pygment_lexer": "rust",
   "version": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo ""
echo "🎯 Starting Jupyter Lab with GPU-enabled Rust kernel..."

jupyter lab --port=8889 --no-browser notebooks/rustorch_gpu_rust_kernel_demo.ipynb &

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$(msg "gpu_kernel_complete")"
echo ""
echo "$(msg "how_to_use")"
echo "  $(msg "select_rust_kernel")"
echo "  $(msg "write_rust_code")"
echo "  3. Use GPU features with detected capabilities"
echo ""
echo "$(msg "detected_gpu_features") $GPU_FEATURES"
echo ""
echo "$(msg "available_at")"
echo ""
echo "$(msg "gpu_demo_loaded")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"