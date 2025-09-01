# 🚀 RusTorch Jupyter Quick Start

新規ユーザーがRusTorchを簡単にJupyterで使い始めるためのガイドです。

## 🎯 Option 1: ワンライナーセットアップ（推奨）

**標準版（Python bindings + Jupyter）:**
```bash
curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start.sh | bash
```

**🌟 WebGPU対応版（ブラウザGPU加速 + Jupyter）:**
```bash
curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start_webgpu.sh | bash
```

**標準版の特徴:**
- Python bindingsでJupyter使用
- CPU専用だが高速セットアップ
- 必要環境: Python 3.8+, Git

**WebGPU版の特徴:**
- ブラウザでGPU加速
- インタラクティブなWebGPUデモ
- Jupyter Lab + WebGPUデモサーバー
- 必要環境: Python 3.8+, Git, Rust, wasm-pack（自動インストール）

---

## 🌐 Option 2: ブラウザで即座に試す（Binder）

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JunSuzukiJapan/rustorch/main?urlpath=lab)

**特徴:**
- インストール不要
- ブラウザだけで完全動作
- 5-10分で起動（初回のみ）

---

## 📚 利用可能なノートブック

1. **rustorch_demo.ipynb** - 基本的なテンソル操作
2. **webgpu_ml_demo.ipynb** - WebGPU加速デモ
3. **webgpu_performance_demo.ipynb** - パフォーマンスベンチマーク

---

## 🛠️ 手動セットアップ（上級者向け）

```bash
# 1. リポジトリをクローン
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# 2. Jupyterを起動
./start_jupyter.sh              # 標準版
./start_jupyter_webgpu.sh       # WebGPU対応版
```

---

## 💡 トラブルシューティング

### Rustがインストールされていない場合
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.bashrc
```

### Python仮想環境の問題
```bash
python3 -m pip install --user maturin jupyter
```

### macOS権限エラー
```bash
sudo xcode-select --install
```

---

## 🎮 サンプルコード

```python
import rustorch

# テンソル作成
x = rustorch.tensor([[1, 2], [3, 4]])
y = rustorch.tensor([[5, 6], [7, 8]])

# 行列乗算
result = rustorch.matmul(x, y)
print(result)
```

---

**🎉 数分でRusTorchのパワフルな機能をJupyterで体験できます！**