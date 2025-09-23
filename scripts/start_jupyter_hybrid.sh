#!/bin/bash

# RusTorch Hybrid Jupyter Environment
# Rust + Python デュアルカーネル環境

set -e

echo "🦀🐍 RusTorch Hybrid Jupyter Setup"
echo "🦀🐍 RusTorch ハイブリッドJupyter環境"
echo ""

# Create Python virtual environment
if [ ! -d ".venv-hybrid" ]; then
    echo "📦 Creating Python virtual environment..."
    echo "📦 Python仮想環境を作成中..."
    python3 -m venv .venv-hybrid
    
    # Verify creation was successful
    if [ ! -f ".venv-hybrid/bin/activate" ]; then
        echo "❌ Failed to create virtual environment"
        echo "❌ 仮想環境の作成に失敗しました"
        exit 1
    fi
else
    echo "✅ Using existing virtual environment"
    echo "✅ 既存の仮想環境を使用"
fi

# Activate virtual environment
echo "🔌 Activating Python virtual environment..."
echo "🔌 Python仮想環境をアクティベート中..."
source .venv-hybrid/bin/activate

# Upgrade pip and install base packages
echo "📦 Installing Python packages..."
echo "📦 Pythonパッケージをインストール中..."
pip install --upgrade pip
pip install jupyter jupyterlab ipykernel numpy matplotlib seaborn pandas

# Install evcxr (Rust Jupyter kernel)
echo "🦀 Installing Rust Jupyter kernel..."
echo "🦀 Rust Jupyterカーネルをインストール中..."

if ! command -v cargo >/dev/null 2>&1; then
    echo "❌ Cargo not found. Please install Rust first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Install evcxr_jupyter if not already installed
if ! command -v evcxr_jupyter >/dev/null 2>&1; then
    cargo install evcxr_jupyter
fi

# Register Rust kernel
evcxr_jupyter --install

# Get Python version for proper site-packages path
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📍 Detected Python version: $PYTHON_VERSION"

# Create Python wrapper for RusTorch
echo "🔗 Creating Python-RusTorch bridge..."
echo "🔗 Python-RusTorchブリッジを作成中..."

mkdir -p .venv-hybrid/lib/python$PYTHON_VERSION/site-packages/rustorch_bridge

cat > .venv-hybrid/lib/python$PYTHON_VERSION/site-packages/rustorch_bridge/__init__.py << 'EOF'
"""
RusTorch Python Bridge
Allows calling RusTorch functionality from Python notebooks
"""

import subprocess
import json
import tempfile
import os

class RusTorchBridge:
    """Bridge to execute RusTorch code from Python"""
    
    def __init__(self):
        self.project_root = self._find_project_root()
    
    def _find_project_root(self):
        """Find RusTorch project root"""
        current = os.getcwd()
        while current != '/':
            if os.path.exists(os.path.join(current, 'Cargo.toml')):
                with open(os.path.join(current, 'Cargo.toml'), 'r') as f:
                    if 'rustorch' in f.read():
                        return current
            current = os.path.dirname(current)
        return os.getcwd()
    
    def run_rust_code(self, code, return_output=True):
        """Execute Rust code and return results"""
        
        # Create temporary Rust file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
            rust_code = f'''
use rustorch::{{tensor::Tensor, nn::linear::Linear}};

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    {code}
    Ok(())
}}
'''
            f.write(rust_code)
            temp_file = f.name
        
        try:
            # Compile and run
            result = subprocess.run(
                ['rustc', '--extern', f'rustorch={self.project_root}/target/release/deps/librustorch.rlib', 
                 temp_file, '-o', temp_file + '.exe'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return f"Compile Error: {result.stderr}"
            
            # Execute
            exec_result = subprocess.run([temp_file + '.exe'], 
                                       capture_output=True, text=True)
            
            if return_output:
                return exec_result.stdout if exec_result.stdout else exec_result.stderr
            
        finally:
            # Cleanup
            for file_path in [temp_file, temp_file + '.exe']:
                if os.path.exists(file_path):
                    os.unlink(file_path)
        
        return "Execution completed"
    
    def create_tensor(self, data):
        """Create RusTorch tensor from Python data"""
        # This would be implemented with proper Python bindings
        # For now, return a description
        return f"RusTorch Tensor: {data}"

# Global instance
rustorch = RusTorchBridge()

# Convenience functions
def rust(code):
    """Execute Rust code in current notebook"""
    return rustorch.run_rust_code(code)

def tensor(data):
    """Create RusTorch tensor"""
    return rustorch.create_tensor(data)
EOF

# Create rustorch module as alias to rustorch_bridge
echo "🔗 Creating rustorch Python module alias..."
echo "🔗 rustorch Pythonモジュールエイリアスを作成中..."
mkdir -p .venv-hybrid/lib/python$PYTHON_VERSION/site-packages/rustorch
cat > .venv-hybrid/lib/python$PYTHON_VERSION/site-packages/rustorch/__init__.py << 'EOF'
"""
RusTorch Python Module
Direct import compatibility for RusTorch notebooks
"""
# Import everything from rustorch_bridge for compatibility
from rustorch_bridge import *

# Additional RusTorch Python API implementations
import numpy as np
import json
import subprocess
import tempfile
import os

class PyTensor:
    """Python-compatible RusTorch Tensor implementation"""
    
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self.data_array = np.array(data, dtype=np.float32)
        else:
            self.data_array = np.array(data, dtype=np.float32)
        
        if shape:
            self.data_array = self.data_array.reshape(shape)
    
    def shape(self):
        """Return tensor shape"""
        return list(self.data_array.shape)
    
    def data(self):
        """Return tensor data as list"""
        return self.data_array.tolist()
    
    def add(self, other):
        """Tensor addition"""
        if isinstance(other, PyTensor):
            result_data = self.data_array + other.data_array
        else:
            result_data = self.data_array + np.array(other, dtype=np.float32)
        return PyTensor(result_data.tolist())
    
    def matmul(self, other):
        """Matrix multiplication"""
        if isinstance(other, PyTensor):
            result_data = np.dot(self.data_array, other.data_array)
        else:
            result_data = np.dot(self.data_array, np.array(other, dtype=np.float32))
        return PyTensor(result_data.tolist())
    
    def relu(self):
        """ReLU activation function"""
        result_data = np.maximum(0, self.data_array)
        return PyTensor(result_data.tolist())
    
    def sigmoid(self):
        """Sigmoid activation function"""
        result_data = 1 / (1 + np.exp(-self.data_array))
        return PyTensor(result_data.tolist())
    
    def __str__(self):
        return f"PyTensor(shape={self.shape()}, data={self.data()})"
    
    def __repr__(self):
        return self.__str__()

def zeros(shape):
    """Create tensor filled with zeros"""
    return PyTensor(np.zeros(shape).tolist())

def ones(shape):
    """Create tensor filled with ones"""
    return PyTensor(np.ones(shape).tolist())

def randn(shape):
    """Create tensor filled with random normal values"""
    return PyTensor(np.random.randn(*shape).astype(np.float32).tolist())

def tensor(data, shape=None):
    """Create tensor from data"""
    return PyTensor(data, shape)

# For backward compatibility
def create_tensor(data):
    """Create tensor from data (legacy)"""
    return PyTensor(data)
EOF

# Download sample notebooks from repository
echo "📓 Downloading sample notebooks..."
echo "📓 サンプルノートブックをダウンロード中..."

# Download notebook downloader script if needed
if [ ! -f "download_notebooks.sh" ]; then
    curl -sSL "https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/download_notebooks.sh" -o "download_notebooks.sh"
    chmod +x download_notebooks.sh
fi

# Download notebooks using the dedicated script
if [[ -f "./scripts/download_notebooks.sh" ]]; then
    ./scripts/download_notebooks.sh all notebooks
elif [[ -f "./download_notebooks.sh" ]]; then
    ./download_notebooks.sh all notebooks
else
    echo "Warning: download_notebooks.sh not found, continuing with existing notebooks"
fi

# Ensure hybrid directory exists with fallback basic notebooks
mkdir -p notebooks/hybrid

# Python + Rust example notebook
cat > notebooks/hybrid/python_rust_demo.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# RusTorch Hybrid Demo: Python + Rust\n",
    "## 🦀🐍 PythonからRustを呼び出すデモ"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Python セル\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from rustorch_bridge import rustorch, rust, tensor\n",
    "\n",
    "print(\"🐍 Python environment ready!\")\n",
    "print(\"🦀 RusTorch bridge loaded!\")"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Python でデータを準備\n",
    "data = np.random.randn(10, 10)\n",
    "print(f\"Python data shape: {data.shape}\")\n",
    "\n",
    "# RusTorch でテンソル処理\n",
    "rust_result = rust('''\n",
    "    let tensor = Tensor::randn(&[10, 10]);\n",
    "    println!(\"Rust tensor created: {:?}\", tensor.shape());\n",
    "    let result = tensor.matmul(&tensor.transpose(0, 1));\n",
    "    println!(\"Matrix multiplication result shape: {:?}\", result.shape());\n",
    "''')\n",
    "print(rust_result)"
   ],
   "outputs": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Pure Rust example notebook
cat > notebooks/hybrid/pure_rust_demo.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# RusTorch Pure Rust Demo\n",
    "## 🦀 ネイティブRustでの機械学習"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    ":dep rustorch = { path = \"../..\" }"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "use rustorch::tensor::Tensor;\n",
    "use rustorch::nn::linear::Linear;\n",
    "\n",
    "// Create tensors\n",
    "let x = Tensor::randn(&[32, 10]);\n",
    "let y = Tensor::randn(&[32, 1]);\n",
    "\n",
    "println!(\"Input shape: {:?}\", x.shape());\n",
    "println!(\"Target shape: {:?}\", y.shape());"
   ],
   "outputs": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "// Create neural network\n",
    "let mut model = Linear::new(10, 1);\n",
    "\n",
    "// Forward pass\n",
    "let output = model.forward(&x);\n",
    "println!(\"Output shape: {:?}\", output.shape());\n",
    "\n",
    "// Simple training loop would go here\n",
    "println!(\"🦀 Pure Rust ML computation completed!\");"
   ],
   "outputs": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Rust",
   "language": "rust",
   "name": "rust"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo ""
echo -e "🎉 Hybrid Jupyter environment setup complete!"
echo -e "🎉 ハイブリッドJupyter環境のセットアップ完了！"
echo ""
echo "📋 Available kernels / 利用可能なカーネル:"
echo "  🐍 Python 3 - Standard Python with RusTorch bridge"
echo "  🦀 Rust - Native Rust with evcxr kernel"
echo ""
echo "📓 Sample notebooks available in:"
echo "  📁 notebooks/hybrid/python_rust_demo.ipynb"
echo "  📁 notebooks/hybrid/pure_rust_demo.ipynb"
echo "  📁 notebooks/rustorch_demo.ipynb (Basic demo)"
echo "  📁 notebooks/webgpu_ml_demo.ipynb (WebGPU demo)"
echo "  📁 notebooks/en/rustorch_demo_en.ipynb (English version)"
echo "  📁 notebooks/ja/quickstart_ja.md (Japanese quickstart)"
echo "  📁 And many more in language-specific directories!"
echo ""
echo "🚀 Starting Jupyter Lab..."
echo "🚀 Jupyter Labを起動中..."

# Start Jupyter Lab with both kernels
jupyter lab --port=8888 --no-browser notebooks/