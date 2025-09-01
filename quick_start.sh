#!/bin/bash

# RusTorch Quick Start Script
# RusTorch クイックスタートスクリプト
# 
# Usage: curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start.sh | bash
# 使用法: curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start.sh | bash

set -e

echo "🚀 RusTorch Quick Start"
echo "🚀 RusTorch クイックスタート"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create temporary directory for RusTorch
RUSTORCH_DIR="$HOME/rustorch-jupyter"
echo "📁 Creating RusTorch workspace: $RUSTORCH_DIR"
echo "📁 RusTorchワークスペースを作成: $RUSTORCH_DIR"

if [ -d "$RUSTORCH_DIR" ]; then
    echo "⚠️  Directory exists. Updating..."
    echo "⚠️  ディレクトリが存在します。更新中..."
    cd "$RUSTORCH_DIR"
    git pull origin main || echo "Git pull failed, continuing..."
else
    echo "📥 Downloading RusTorch..."
    echo "📥 RusTorchをダウンロード中..."
    git clone --depth 1 https://github.com/JunSuzukiJapan/rustorch.git "$RUSTORCH_DIR"
    cd "$RUSTORCH_DIR"
fi

# Check system requirements
echo ""
echo "🔍 Checking system requirements..."
echo "🔍 システム要件を確認中..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python: $PYTHON_VERSION"
else
    echo "❌ Python 3 is required. Please install Python 3.8+"
    echo "❌ Python 3が必要です。Python 3.8+をインストールしてください"
    exit 1
fi

# Check Rust (optional for quick start)
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version | cut -d' ' -f2)
    echo "✅ Rust: $RUST_VERSION"
    HAS_RUST=true
else
    echo "⚠️  Rust not found - will use pre-built binaries"
    echo "⚠️  Rustが見つかりません - ビルド済みバイナリを使用します"
    HAS_RUST=false
fi

# Check for pip
if command -v pip3 &> /dev/null; then
    echo "✅ pip3 available"
elif command -v pip &> /dev/null; then
    echo "✅ pip available"
    alias pip3=pip
else
    echo "❌ pip is required. Please install pip"
    echo "❌ pipが必要です。pipをインストールしてください"
    exit 1
fi

echo ""
echo "🛠️  Setting up Python environment..."
echo "🛠️  Python環境をセットアップ中..."

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    echo "📦 仮想環境を作成中..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
echo "🔌 仮想環境をアクティベート中..."
source .venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
echo "📚 Python依存関係をインストール中..."
pip install --upgrade pip
pip install jupyter jupyterlab matplotlib pandas numpy maturin

# Try to install pre-built wheel from PyPI if available
echo ""
echo "🎯 Installing RusTorch..."
echo "🎯 RusTorchをインストール中..."

if pip install rustorch 2>/dev/null; then
    echo "✅ Installed RusTorch from PyPI"
    echo "✅ PyPIからRusTorchをインストールしました"
    RUSTORCH_INSTALLED=true
else
    echo "⚠️  PyPI package not available, building from source..."
    echo "⚠️  PyPIパッケージが利用できません、ソースからビルド中..."
    RUSTORCH_INSTALLED=false
    
    if [ "$HAS_RUST" = true ]; then
        echo "🔧 Building RusTorch Python bindings..."
        echo "🔧 RusTorch Pythonバインディングをビルド中..."
        maturin develop --features python --release
        RUSTORCH_INSTALLED=true
    else
        echo "❌ Cannot build from source without Rust"
        echo "❌ Rustなしではソースからビルドできません"
        echo ""
        echo "📋 Please install Rust first:"
        echo "📋 まずRustをインストールしてください:"
        echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo ""
        echo "Then run this script again."
        echo "その後、このスクリプトを再度実行してください。"
        exit 1
    fi
fi

if [ "$RUSTORCH_INSTALLED" = true ]; then
    echo ""
    echo "🧪 Testing RusTorch installation..."
    echo "🧪 RusTorchインストールをテスト中..."
    
    python3 -c "
import rustorch
print('✅ RusTorch imported successfully!')
print('✅ RusTorchのインポートに成功しました！')
print(f'📍 RusTorch version: {rustorch.__version__ if hasattr(rustorch, \"__version__\") else \"unknown\"}')
" || echo "⚠️  Import test failed, but installation may still work"

    echo ""
    echo "🎉 Setup Complete!"
    echo "🎉 セットアップ完了！"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "🚀 Starting Jupyter Lab with RusTorch demo..."
    echo "🚀 RusTorchデモ付きJupyter Labを起動中..."
    echo ""
    echo "📋 Available notebooks:"
    echo "📋 利用可能なノートブック:"
    echo "   • rustorch_demo.ipynb - Basic tensor operations"
    echo "   • webgpu_ml_demo.ipynb - WebGPU acceleration demo"
    echo "   • webgpu_performance_demo.ipynb - Performance benchmarks"
    echo ""
    echo "🌐 Jupyter Lab will open at: http://localhost:8888"
    echo "🌐 Jupyter Labは次のURLで開きます: http://localhost:8888"
    echo "🛑 Press Ctrl+C to stop / 停止するにはCtrl+Cを押してください"
    echo ""
    
    # Launch Jupyter Lab
    exec jupyter lab --port=8888 --no-browser
fi