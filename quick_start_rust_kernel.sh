#!/bin/bash

# RusTorch Rust Kernel Quick Start Script - Multilingual Support
# Auto-detects system language for international users
# Supports: English, Japanese, Spanish, French, German, Chinese, Korean
# 
# Usage: curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start_rust_kernel.sh | bash
# 使用法: curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start_rust_kernel.sh | bash

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

# Multilingual message function for Rust Kernel
msg() {
    local key="$1"
    local lang="${DETECTED_LANG:-en}"
    
    case "$key" in
        "welcome_title")
            case "$lang" in
                en) echo "🦀 RusTorch Rust Kernel Quick Start" ;;
                ja) echo "🦀 RusTorch Rust カーネル クイックスタート" ;;
                es) echo "🦀 Inicio Rápido Kernel Rust RusTorch" ;;
                fr) echo "🦀 Démarrage Rapide Noyau Rust RusTorch" ;;
                de) echo "🦀 RusTorch Rust Kernel Schnellstart" ;;
                zh) echo "🦀 RusTorch Rust 内核快速开始" ;;
                ko) echo "🦀 RusTorch Rust 커널 빠른 시작" ;;
            esac ;;
        "rust_kernel_complete")
            case "$lang" in
                en) echo "🎉 Rust Kernel Setup Complete!" ;;
                ja) echo "🎉 Rustカーネルセットアップ完了！" ;;
                es) echo "🎉 ¡Configuración Kernel Rust Completa!" ;;
                fr) echo "🎉 Configuration Noyau Rust Terminée!" ;;
                de) echo "🎉 Rust Kernel Setup Abgeschlossen!" ;;
                zh) echo "🎉 Rust 内核设置完成！" ;;
                ko) echo "🎉 Rust 커널 설정 완료！" ;;
            esac ;;
        "how_to_use")
            case "$lang" in
                en) echo "📋 How to use:" ;;
                ja) echo "📋 使用方法:" ;;
                es) echo "📋 Cómo usar:" ;;
                fr) echo "📋 Comment utiliser:" ;;
                de) echo "📋 Verwendung:" ;;
                zh) echo "📋 使用方法：" ;;
                ko) echo "📋 사용 방법:" ;;
            esac ;;
        "select_rust_kernel")
            case "$lang" in
                en) echo "1. Select 'Rust' kernel in Jupyter" ;;
                ja) echo "1. Jupyterで'Rust'カーネルを選択" ;;
                es) echo "1. Selecciona el kernel 'Rust' en Jupyter" ;;
                fr) echo "1. Sélectionnez le noyau 'Rust' dans Jupyter" ;;
                de) echo "1. Wählen Sie 'Rust' Kernel in Jupyter" ;;
                zh) echo "1. 在 Jupyter 中选择 'Rust' 内核" ;;
                ko) echo "1. Jupyter에서 'Rust' 커널을 선택" ;;
            esac ;;
        "write_rust_code")
            case "$lang" in
                en) echo "2. Write Rust code directly in cells" ;;
                ja) echo "2. セル内に直接Rustコードを記述" ;;
                es) echo "2. Escribe código Rust directamente en las celdas" ;;
                fr) echo "2. Écrivez du code Rust directement dans les cellules" ;;
                de) echo "2. Schreiben Sie Rust-Code direkt in Zellen" ;;
                zh) echo "2. 在单元格中直接编写 Rust 代码" ;;
                ko) echo "2. 셀에 직접 Rust 코드를 작성" ;;
            esac ;;
        "add_rustorch")
            case "$lang" in
                en) echo "3. Use :dep rustorch = \"0.5.11\" to add RusTorch" ;;
                ja) echo "3. :dep rustorch = \"0.5.11\" でRusTorchを追加" ;;
                es) echo "3. Usa :dep rustorch = \"0.5.11\" para añadir RusTorch" ;;
                fr) echo "3. Utilisez :dep rustorch = \"0.5.11\" pour ajouter RusTorch" ;;
                de) echo "3. Verwenden Sie :dep rustorch = \"0.5.11\" um RusTorch hinzuzufügen" ;;
                zh) echo "3. 使用 :dep rustorch = \"0.5.11\" 添加 RusTorch" ;;
                ko) echo "3. :dep rustorch = \"0.5.11\"를 사용하여 RusTorch 추가" ;;
            esac ;;
        "available_at")
            case "$lang" in
                en) echo "🚀 Available at: http://localhost:8888" ;;
                ja) echo "🚀 利用可能: http://localhost:8888" ;;
                es) echo "🚀 Disponible en: http://localhost:8888" ;;
                fr) echo "🚀 Disponible à: http://localhost:8888" ;;
                de) echo "🚀 Verfügbar unter: http://localhost:8888" ;;
                zh) echo "🚀 可用地址: http://localhost:8888" ;;
                ko) echo "🚀 사용 가능: http://localhost:8888" ;;
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
    ":dep rustorch = \"0.5.11\"\n",
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
    "use rustorch::prelude::*;\n",
    "\n",
    "// Create tensors\n",
    "let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);\n",
    "let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);\n",
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
    "// Create special tensors (with explicit type annotations)\n",
    "let zeros: Tensor<f32> = Tensor::zeros(&[3, 3]);\n",
    "let ones: Tensor<f32> = Tensor::ones(&[3, 3]);\n",
    "let random: Tensor<f32> = Tensor::randn(&[3, 3]);\n",
    "\n",
    "println!(\"Zeros tensor: {:?}\", zeros);\n",
    "println!(\"Ones tensor: {:?}\", ones);\n",
    "println!(\"Random tensor: {:?}\", random);\n",
    "\n",
    "// Apply activation functions\n",
    "let relu_result = random.relu();\n",
    "println!(\"ReLU result: {:?}\", relu_result);"
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
    "let input: Tensor<f32> = Tensor::randn(&[1, 784]); // Batch size 1, 784 features\n",
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
    "let a: Tensor<f32> = Tensor::randn(&[size, size]);\n",
    "let b: Tensor<f32> = Tensor::randn(&[size, size]);\n",
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
echo "$(msg "rust_kernel_complete")"
echo ""
echo "$(msg "how_to_use")"
echo "  $(msg "select_rust_kernel")"
echo "  $(msg "write_rust_code")"
echo "  $(msg "add_rustorch")"
echo ""
echo "$(msg "available_at")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"