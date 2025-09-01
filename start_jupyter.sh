#!/bin/bash

# RusTorch Jupyter Lab Launcher for macOS
# RusTorch用Jupyter Lab起動スクリプト（macOS）

set -e

echo "🚀 Starting RusTorch Jupyter Lab..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    
    echo "📚 Installing dependencies..."
    source .venv/bin/activate
    pip install maturin numpy jupyter jupyterlab matplotlib pandas
else
    echo "✅ Virtual environment found"
    source .venv/bin/activate
fi

# Check if RusTorch Python bindings are installed
if ! python -c "import rustorch" 2>/dev/null; then
    echo "🔧 Building RusTorch Python bindings..."
    maturin develop --features python
else
    echo "✅ RusTorch Python bindings available"
fi

echo "🎯 Launching Jupyter Lab..."
echo "📍 Access URL will be displayed below"
echo "🛑 Press Ctrl+C to stop"
echo ""

# Launch Jupyter Lab with demo notebook
jupyter lab --port=8888 --no-browser notebooks/rustorch_demo.ipynb