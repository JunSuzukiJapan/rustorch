#!/bin/bash

# RusTorch Jupyter Quick Launcher
# 簡単Jupyter起動スクリプト

set -e

echo "🚀 RusTorch Jupyter Quick Launcher"
echo "🚀 RusTorch Jupyter 簡単起動"
echo ""

# Function to check if setup exists
check_setup() {
    local setup_type=$1
    case $setup_type in
        "python")
            [ -d ".venv" ] && [ -f ".venv/bin/activate" ]
            ;;
        "webgpu")
            [ -d "pkg-webgpu" ] && [ -f "pkg-webgpu/rustorch.js" ]
            ;;
        "rust")
            command -v evcxr_jupyter >/dev/null 2>&1
            ;;
        "hybrid")
            [ -d ".venv-hybrid" ] && command -v evcxr_jupyter >/dev/null 2>&1
            ;;
    esac
}

# Display menu
echo "📓 Available Jupyter Demos:"
echo ""

if check_setup "python"; then
    echo "🐍 [1] Python Demo     - Standard CPU-based ML demos"
else
    echo "🐍 [1] Python Demo     - ⚠️  Setup required (run ./start_jupyter.sh first)"
fi

if check_setup "webgpu"; then
    echo "⚡ [2] WebGPU Demo     - Browser GPU acceleration"
else
    echo "⚡ [2] WebGPU Demo     - ⚠️  Setup required (run ./start_jupyter_webgpu.sh first)"
fi

if check_setup "rust"; then
    echo "🦀 [3] Rust Kernel    - Native Rust in Jupyter"
else
    echo "🦀 [3] Rust Kernel    - ⚠️  Setup required (run ./quick_start_rust_kernel.sh first)"
fi

if check_setup "hybrid"; then
    echo "🦀🐍 [4] Hybrid Demo    - Python + Rust dual-kernel environment"
else
    echo "🦀🐍 [4] Hybrid Demo    - ⚠️  Setup required (run ./start_jupyter_hybrid.sh first)"
fi

echo ""
echo "🌐 [5] Online Binder   - Run immediately in browser (no local setup needed)"
echo "❌ [q] Quit"
echo ""

read -p "Choose demo type [1-5/q]: " choice

case $choice in
    1)
        if check_setup "python"; then
            echo "🐍 Starting Python demo..."
            source .venv/bin/activate
            jupyter lab --port=8888 --no-browser notebooks/
        else
            echo "❌ Python environment not found. Run setup first:"
            echo "   ./start_jupyter.sh"
            exit 1
        fi
        ;;
    2)
        if check_setup "webgpu"; then
            echo "⚡ Starting WebGPU demo..."
            export JUPYTER_ENABLE_UNSAFE_EXTENSION=1
            export JUPYTER_CONFIG_DIR="$(pwd)/.jupyter"
            jupyter lab --port=8888 --no-browser --allow-root --config=.jupyter/jupyter_lab_config.py notebooks/
        else
            echo "❌ WebGPU setup not found. Run setup first:"
            echo "   ./start_jupyter_webgpu.sh"
            exit 1
        fi
        ;;
    3)
        if check_setup "rust"; then
            echo "🦀 Starting Rust kernel demo..."
            jupyter lab --port=8888 --no-browser notebooks/
        else
            echo "❌ Rust kernel not found. Run setup first:"
            echo "   ./quick_start_rust_kernel.sh"
            exit 1
        fi
        ;;
    4)
        if check_setup "hybrid"; then
            echo "🦀🐍 Starting Hybrid demo..."
            source .venv-hybrid/bin/activate
            jupyter lab --port=8888 --no-browser notebooks/
        else
            echo "❌ Hybrid environment not found. Run setup first:"
            echo "   ./start_jupyter_hybrid.sh"
            exit 1
        fi
        ;;
    5)
        echo "🌐 Opening Binder in browser..."
        if command -v open >/dev/null 2>&1; then
            open "https://mybinder.org/v2/gh/JunSuzukiJapan/rustorch/main?urlpath=lab"
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open "https://mybinder.org/v2/gh/JunSuzukiJapan/rustorch/main?urlpath=lab"
        else
            echo "Please open this URL in your browser:"
            echo "https://mybinder.org/v2/gh/JunSuzukiJapan/rustorch/main?urlpath=lab"
        fi
        ;;
    q|Q)
        echo "👋 Goodbye! / さようなら！"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run again and select 1-5 or q."
        exit 1
        ;;
esac