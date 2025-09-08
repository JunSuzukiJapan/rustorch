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
fi

# Activate virtual environment
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

# Create Python wrapper for RusTorch
echo "🔗 Creating Python-RusTorch bridge..."
echo "🔗 Python-RusTorchブリッジを作成中..."

mkdir -p .venv-hybrid/lib/python*/site-packages/rustorch_bridge

cat > .venv-hybrid/lib/python*/site-packages/rustorch_bridge/__init__.py << 'EOF'
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

# Download sample notebooks from repository
echo "📓 Downloading sample notebooks..."
echo "📓 サンプルノートブックをダウンロード中..."

# Download notebook downloader script if needed
if [ ! -f "download_notebooks.sh" ]; then
    curl -sSL "https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/download_notebooks.sh" -o "download_notebooks.sh"
    chmod +x download_notebooks.sh
fi

# Download notebooks using the dedicated script
./download_notebooks.sh notebooks

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