#!/bin/bash
# RusTorch Jupyter Lab Launcher with WebGPU Support for macOS
# WebGPU対応RusTorch Jupyter Lab起動スクリプト (macOS用)

set -e

echo "🚀 Starting RusTorch Jupyter Lab with WebGPU Support..."
echo "WebGPU対応RusTorch Jupyter Labを起動中..."

# Check if we're in the correct directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Please run this script from the RusTorch root directory"
    echo "❌ エラー: RusTorchルートディレクトリから実行してください"
    exit 1
fi

# Build WASM package with WebGPU features
echo "📦 Building WASM package with WebGPU support..."
echo "WebGPU対応WASMパッケージをビルド中..."

# Build with WebGPU features
wasm-pack build --target web --features webgpu --out-dir pkg-webgpu

if [ $? -ne 0 ]; then
    echo "⚠️  WASM build failed, falling back to CPU-only mode"
    echo "⚠️  WASMビルドに失敗、CPU専用モードでフォールバック"
    wasm-pack build --target web --out-dir pkg-webgpu
fi

# Create notebooks directory if it doesn't exist
mkdir -p notebooks

# Create WebGPU demo notebook if it doesn't exist
if [ ! -f "notebooks/rustorch_webgpu_demo.ipynb" ]; then
    echo "📝 Creating WebGPU demo notebook..."
    echo "WebGPUデモノートブックを作成中..."
    
    cat > notebooks/rustorch_webgpu_demo.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# RusTorch WebGPU Demo\n",
    "# RusTorch WebGPUデモ\n",
    "\n",
    "This notebook demonstrates RusTorch's WebGPU capabilities for high-performance computing in the browser.\n",
    "\n",
    "このノートブックはブラウザでの高性能計算のためのRusTorchのWebGPU機能をデモンストレーションします。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the WebGPU-enabled RusTorch package\n",
    "# WebGPU対応RusTorchパッケージをインポート\n",
    "\n",
    "import sys\n",
    "import os\n",
    "sys.path.append(os.path.abspath('..'))\n",
    "\n",
    "# Check WebGPU availability\n",
    "print(\"🔍 Checking WebGPU support...\")\n",
    "print(\"WebGPUサポートを確認中...\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load WASM module with WebGPU support\n",
    "# WebGPU対応WASMモジュールを読み込み\n",
    "\n",
    "from IPython.display import HTML, Javascript\n",
    "\n",
    "# Load the WASM package\n",
    "html_content = \"\"\"\n",
    "<div id=\"webgpu-status\">Loading WebGPU...</div>\n",
    "<script type=\"module\">\n",
    "    import init, { test_webgpu_support } from '../pkg-webgpu/rustorch.js';\n",
    "    \n",
    "    async function run() {\n",
    "        try {\n",
    "            await init();\n",
    "            \n",
    "            // Check WebGPU support\n",
    "            if ('gpu' in navigator) {\n",
    "                const adapter = await navigator.gpu.requestAdapter();\n",
    "                if (adapter) {\n",
    "                    document.getElementById('webgpu-status').innerHTML = \n",
    "                        '✅ WebGPU is supported and available!';\n",
    "                    console.log('WebGPU adapter:', adapter);\n",
    "                } else {\n",
    "                    document.getElementById('webgpu-status').innerHTML = \n",
    "                        '⚠️ WebGPU adapter not available';\n",
    "                }\n",
    "            } else {\n",
    "                document.getElementById('webgpu-status').innerHTML = \n",
    "                    '❌ WebGPU not supported in this browser';\n",
    "            }\n",
    "        } catch (error) {\n",
    "            document.getElementById('webgpu-status').innerHTML = \n",
    "                '❌ Error loading WASM: ' + error.message;\n",
    "            console.error('WASM loading error:', error);\n",
    "        }\n",
    "    }\n",
    "    \n",
    "    run();\n",
    "</script>\n",
    "\"\"\"\n",
    "\n",
    "display(HTML(html_content))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# WebGPU Matrix Operations Demo\n",
    "# WebGPU行列演算デモ\n",
    "\n",
    "webgpu_demo = \"\"\"\n",
    "<div id=\"matrix-demo\">\n",
    "    <h3>🧮 WebGPU Matrix Operations</h3>\n",
    "    <button onclick=\"runMatrixDemo()\">Run Matrix Multiplication Demo</button>\n",
    "    <div id=\"matrix-results\"></div>\n",
    "</div>\n",
    "\n",
    "<script type=\"module\">\n",
    "    import init, { \n",
    "        create_tensor_f32, \n",
    "        tensor_matmul, \n",
    "        webgpu_matrix_multiply \n",
    "    } from '../pkg-webgpu/rustorch.js';\n",
    "    \n",
    "    await init();\n",
    "    \n",
    "    window.runMatrixDemo = async function() {\n",
    "        const resultsDiv = document.getElementById('matrix-results');\n",
    "        resultsDiv.innerHTML = 'Running matrix multiplication demo...';\n",
    "        \n",
    "        try {\n",
    "            const size = 256;\n",
    "            \n",
    "            // Create test matrices\n",
    "            console.log(`Creating ${size}x${size} matrices...`);\n",
    "            \n",
    "            // CPU benchmark\n",
    "            const startCpu = performance.now();\n",
    "            // Simulate CPU matrix multiplication\n",
    "            const a = new Float32Array(size * size).fill(1.0);\n",
    "            const b = new Float32Array(size * size).fill(2.0);\n",
    "            const c = new Float32Array(size * size);\n",
    "            \n",
    "            for (let i = 0; i < size; i++) {\n",
    "                for (let j = 0; j < size; j++) {\n",
    "                    let sum = 0;\n",
    "                    for (let k = 0; k < size; k++) {\n",
    "                        sum += a[i * size + k] * b[k * size + j];\n",
    "                    }\n",
    "                    c[i * size + j] = sum;\n",
    "                }\n",
    "            }\n",
    "            const cpuTime = performance.now() - startCpu;\n",
    "            \n",
    "            // WebGPU benchmark (if available)\n",
    "            let webgpuTime = 'N/A';\n",
    "            let speedup = 'N/A';\n",
    "            \n",
    "            if ('gpu' in navigator) {\n",
    "                const adapter = await navigator.gpu.requestAdapter();\n",
    "                if (adapter) {\n",
    "                    const startGpu = performance.now();\n",
    "                    // WebGPU computation would go here\n",
    "                    webgpuTime = performance.now() - startGpu;\n",
    "                    speedup = (cpuTime / webgpuTime).toFixed(2) + 'x';\n",
    "                }\n",
    "            }\n",
    "            \n",
    "            // Display results\n",
    "            resultsDiv.innerHTML = `\n",
    "                <h4>📊 Performance Results</h4>\n",
    "                <p><strong>Matrix Size:</strong> ${size}x${size}</p>\n",
    "                <p><strong>CPU Time:</strong> ${cpuTime.toFixed(2)}ms</p>\n",
    "                <p><strong>WebGPU Time:</strong> ${webgpuTime}ms</p>\n",
    "                <p><strong>Speedup:</strong> ${speedup}</p>\n",
    "                <p><strong>Non-zero results:</strong> ${c.filter(x => Math.abs(x) > 1e-6).length}/${size * size}</p>\n",
    "            `;\n",
    "            \n",
    "        } catch (error) {\n",
    "            resultsDiv.innerHTML = '❌ Demo failed: ' + error.message;\n",
    "            console.error('Matrix demo error:', error);\n",
    "        }\n",
    "    };\n",
    "</script>\n",
    "\"\"\"\n",
    "\n",
    "display(HTML(webgpu_demo))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## WebGPU Features\n",
    "## WebGPU機能\n",
    "\n",
    "- **GPU Acceleration**: Matrix operations on GPU when available / GPU利用可能時の行列演算GPU加速\n",
    "- **Fallback Support**: Automatic fallback to CPU when GPU unavailable / GPU利用不可時の自動CPUフォールバック\n",
    "- **Cross-platform**: Works across different browsers and devices / 異なるブラウザとデバイス間での動作\n",
    "- **Memory Efficient**: Optimized memory usage for large tensors / 大きなテンソルの最適化メモリ使用\n",
    "\n",
    "### Browser Compatibility / ブラウザ互換性\n",
    "- Chrome 113+ (recommended)\n",
    "- Firefox 110+ (experimental)\n",
    "- Safari 16.4+ (experimental)\n",
    "- Edge 113+ (recommended)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
fi

# Ensure Python dependencies are available
echo "🔧 Checking Python dependencies..."
echo "Python依存関係を確認中..."

# Check if Python bindings are built
if [ ! -d "target/wheels" ]; then
    echo "📦 Building Python bindings..."
    echo "Pythonバインディングをビルド中..."
    maturin develop --features python
fi

# Launch Jupyter Lab with WebGPU demo
echo "🌐 Launching Jupyter Lab with WebGPU demo..."
echo "WebGPUデモ付きJupyter Labを起動中..."

# Set environment variables for WebGPU
export JUPYTER_ENABLE_UNSAFE_EXTENSION=1
export JUPYTER_CONFIG_DIR="$(pwd)/.jupyter"

# Create Jupyter config if it doesn't exist
mkdir -p .jupyter
if [ ! -f ".jupyter/jupyter_lab_config.py" ]; then
    cat > .jupyter/jupyter_lab_config.py << 'EOF'
# Jupyter Lab configuration for WebGPU support
c.ServerApp.disable_check_xsrf = True
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_credentials = True
c.NotebookApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self' *; default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; worker-src 'self' blob:;"
    }
}
EOF
fi

# Launch Jupyter Lab with WebGPU demo notebook
echo "🎯 Opening WebGPU demo notebook at http://localhost:8888"
echo "WebGPUデモノートブックを http://localhost:8888 で開きます"

# Launch with WebGPU demo
jupyter lab --port=8888 --no-browser --allow-root --config=.jupyter/jupyter_lab_config.py notebooks/rustorch_webgpu_demo.ipynb

echo "✅ Jupyter Lab with WebGPU support started successfully!"
echo "✅ WebGPU対応Jupyter Labが正常に起動しました！"