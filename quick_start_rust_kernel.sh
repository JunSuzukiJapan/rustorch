#!/bin/bash

# RusTorch Rust Kernel Quick Start Script
# RusTorch Rust カーネル クイックスタートスクリプト
# 
# Usage: curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start_rust_kernel.sh | bash
# 使用法: curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start_rust_kernel.sh | bash

set -e

echo "🦀 RusTorch Rust Kernel Quick Start"
echo "🦀 RusTorch Rust カーネル クイックスタート"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create workspace
RUSTORCH_DIR="$HOME/rustorch-rust-kernel"
echo "📁 Creating RusTorch Rust workspace: $RUSTORCH_DIR"

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

# Check ZMQ (required for evcxr_jupyter)
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

# Create Rust kernel demo notebook
echo ""
echo "📝 Creating Rust kernel demo notebook..."
mkdir -p notebooks

cat > notebooks/rustorch_rust_kernel_demo.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🦀 RusTorch with Rust Kernel Demo\n",
    "# 🦀 RusTorch Rustカーネルデモ\n",
    "\n",
    "This notebook demonstrates how to use RusTorch directly in Rust within Jupyter!\n",
    "\n",
    "このノートブックでは、Jupyter内でRustを直接使ってRusTorchを使用する方法を示します！"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📦 Install RusTorch\n",
    "## 📦 RusTorchをインストール\n",
    "\n",
    "First, let's add RusTorch as a dependency:\n",
    "\n",
    "まず、RusTorchを依存関係として追加しましょう："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    ":dep rustorch = \"0.5.7\"\n",
    ":dep ndarray = \"0.16\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎯 Basic Tensor Operations\n",
    "## 🎯 基本的なテンソル操作"
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
    "// Matrix multiplication\n",
    "let result = a.matmul(&b);\n",
    "println!(\"Matrix multiplication result: {:?}\", result);\n",
    "\n",
    "// Element-wise operations\n",
    "let sum = &a + &b;\n",
    "println!(\"Element-wise sum: {:?}\", sum);\n",
    "\n",
    "let product = &a * &b;\n",
    "println!(\"Element-wise product: {:?}\", product);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🧮 Advanced Operations\n",
    "## 🧮 高度な操作"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "// Create a random tensor\n",
    "let random_tensor = Tensor::randn(&[3, 3]);\n",
    "println!(\"Random tensor: {:?}\", random_tensor);\n",
    "\n",
    "// Apply activation functions\n",
    "let relu_result = random_tensor.relu();\n",
    "println!(\"ReLU result: {:?}\", relu_result);\n",
    "\n",
    "let sigmoid_result = random_tensor.sigmoid();\n",
    "println!(\"Sigmoid result: {:?}\", sigmoid_result);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🤖 Neural Network Example\n",
    "## 🤖 ニューラルネットワークの例"
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
    "// Create a simple neural network\n",
    "let mut network = Sequential::new()\n",
    "    .add_layer(Linear::new(784, 128))\n",
    "    .add_layer(ReLU::new())\n",
    "    .add_layer(Linear::new(128, 10))\n",
    "    .add_layer(Softmax::new());\n",
    "\n",
    "println!(\"Neural network created with {} parameters\", network.num_parameters());\n",
    "\n",
    "// Create sample input\n",
    "let input = Tensor::randn(&[1, 784]); // Batch size 1, 784 features\n",
    "let output = network.forward(&input);\n",
    "\n",
    "println!(\"Input shape: {:?}\", input.shape());\n",
    "println!(\"Output shape: {:?}\", output.shape());\n",
    "println!(\"Output probabilities: {:?}\", output);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ⚡ Performance Benchmarks\n",
    "## ⚡ パフォーマンスベンチマーク"
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
    "// Benchmark matrix multiplication\n",
    "let size = 500;\n",
    "let a = Tensor::randn(&[size, size]);\n",
    "let b = Tensor::randn(&[size, size]);\n",
    "\n",
    "println!(\"🏁 Benchmarking {}x{} matrix multiplication...\", size, size);\n",
    "\n",
    "let start = Instant::now();\n",
    "let result = a.matmul(&b);\n",
    "let duration = start.elapsed();\n",
    "\n",
    "println!(\"✅ Completed in: {:?}\", duration);\n",
    "println!(\"📊 Result shape: {:?}\", result.shape());\n",
    "println!(\"📈 Throughput: {:.2} GFLOPS\", \n",
    "    (2.0 * size as f64 * size as f64 * size as f64) / (duration.as_secs_f64() * 1e9));"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎉 Conclusion\n",
    "## 🎉 まとめ\n",
    "\n",
    "You can now write and execute Rust code directly in Jupyter!\n",
    "\n",
    "これでJupyter内で直接Rustコードを書いて実行できます！\n",
    "\n",
    "**Benefits / 利点:**\n",
    "- 🚀 Native Rust performance / ネイティブRustパフォーマンス\n",
    "- 🔧 Direct library access / ライブラリへの直接アクセス\n",
    "- 🎯 Type safety / 型安全性\n",
    "- ⚡ Zero-cost abstractions / ゼロコスト抽象化"
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
echo "🎯 Starting Jupyter Lab with Rust kernel..."

jupyter lab --port=8888 --no-browser notebooks/rustorch_rust_kernel_demo.ipynb

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Rust Kernel Setup Complete!"
echo "🎉 Rustカーネルセットアップ完了！"
echo ""
echo "📋 How to use:"
echo "📋 使用方法:"
echo "  1. Select 'Rust' kernel in Jupyter"
echo "  1. Jupyterで'Rust'カーネルを選択"
echo "  2. Write Rust code directly in cells"
echo "  2. セル内に直接Rustコードを記述"
echo "  3. Use :dep rustorch = \"0.5.7\" to add RusTorch"
echo "  3. :dep rustorch = \"0.5.7\" でRusTorchを追加"
echo ""
echo "🚀 Available at: http://localhost:8888"
echo "🚀 利用可能: http://localhost:8888"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"