# RusTorch Jupyter 完全ガイド

Python、Rust、ハイブリッドセットアップでRusTorchをJupyter環境で使用するための最終ガイド。

## 📚 目次

1. [クイックスタート](#-クイックスタート)
2. [インストール方法](#-インストール方法)
3. [環境タイプ](#-環境タイプ)
4. [ハイブリッド環境](#-ハイブリッド環境)
5. [使用例](#-使用例)
6. [高度な機能](#-高度な機能)
7. [トラブルシューティング](#-トラブルシューティング)
8. [移行ガイド](#-移行ガイド)

## 🚀 クイックスタート

### 万能ワンライナー（推奨）

RusTorchをJupyterで始める最も簡単な方法：

```bash
./install_jupyter.sh
```

**実行内容：**
- 🔍 **自動検出** - 環境（OS、CPU、GPU）を自動検出
- 🦀🐍 **ハイブリッドインストール** - Python+Rustデュアルカーネル環境をデフォルト
- 📦 **グローバルランチャー作成** - `rustorch-jupyter`コマンドが使用可能
- ⚡ **ハードウェア最適化** - CUDA、Metal、WebGPU、CPUに最適化

### 次回起動
```bash
rustorch-jupyter          # グローバルコマンド（インストーラー使用後）
# または
./start_jupyter_quick.sh  # 対話式メニュー
```

## 📦 インストール方法

### 1. 万能インストーラー（推奨）

**自動検出インストール：**
```bash
curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/install_jupyter.sh | bash
```

**カスタムインストールパス：**
```bash
RUSTORCH_INSTALL_PATH=/usr/local/bin bash <(curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/install_jupyter.sh)
```

**対話式オプション：**
- `[1]` ハイブリッド環境（デフォルト）- Python + Rustカーネル
- `[2]` GPU最適化 - CUDA/Metal/WebGPU最適化単一環境
- `[q]` キャンセル

### 2. 手動セットアップ

特定の環境タイプを選択：

| 環境タイプ | コマンド | 用途 |
|-----------|---------|------|
| 🦀🐍 **ハイブリッド** | `./start_jupyter_hybrid.sh` | PythonとRustの両方開発 |
| 🐍 **Python** | `./start_jupyter.sh` | Python重点のML開発 |
| ⚡ **WebGPU** | `./start_jupyter_webgpu.sh` | ブラウザGPU加速 |
| 🦀 **Rust** | `./quick_start_rust_kernel.sh` | ネイティブRust開発 |
| 🌐 **オンライン** | [Binder](https://mybinder.org/v2/gh/JunSuzukiJapan/rustorch/main?urlpath=lab) | ローカルセットアップ不要 |

## 🏗️ 環境タイプ

### ハイブリッド環境（デフォルト）
- **適用場面**: フルスタックML開発
- **機能**: Python + Rustカーネル、RusTorchブリッジ、サンプルノートブック
- **ハードウェア**: 利用可能なGPU（CUDA/Metal/CPU）に適応

### Python環境
- **適用場面**: RusTorch機能を求めるPython開発者
- **機能**: RusTorch Pythonバインディング付きPythonカーネル
- **ハードウェア**: CPU/GPU最適化

### WebGPU環境
- **適用場面**: ブラウザベースGPU加速
- **機能**: WebAssembly + WebGPU、Chrome最適化
- **ハードウェア**: WebGPU対応モダンブラウザ

### Rustカーネル環境
- **適用場面**: ネイティブRust開発
- **機能**: evcxrカーネル、RusTorchライブラリ直接アクセス
- **ハードウェア**: ネイティブパフォーマンス、全機能利用可能

## 🦀🐍 ハイブリッド環境

ハイブリッド環境はPythonのエコシステムとRustのパフォーマンスの両方の利点を提供します。

### アーキテクチャ

```
Jupyter Lab
├── Pythonカーネル
│   ├── NumPy, Pandas, Matplotlib
│   ├── RusTorch Pythonバインディング
│   └── rustorch_bridgeモジュール
└── Rustカーネル (evcxr)
    ├── ネイティブRusTorchライブラリ
    ├── ハードウェア直接アクセス
    └── ゼロコスト抽象化
```

### セットアッププロセス

1. **環境検出**
   ```bash
   🔍 環境検出
   ==================================
   OS: macos
   CPU: arm64
   GPU: metal
   WebGPUサポート: false
   
   🎯 デフォルト: ハイブリッド環境
   ```

2. **インストール手順**
   - Python仮想環境作成（`.venv-hybrid`）
   - Pythonパッケージインストール（jupyter、numpy、matplotlibなど）
   - Rust Jupyterカーネルインストール（evcxr）
   - RusTorch Pythonブリッジ作成
   - サンプルノートブック生成

3. **グローバルランチャーセットアップ**
   - `~/bin/`に`rustorch-jupyter`コマンド作成
   - PATH自動追加
   - 任意のディレクトリから動作

### 使用例

**RusTorchを使用したPythonセル：**
```python
import numpy as np
from rustorch_bridge import rust, tensor

# Pythonでデータ準備
data = np.random.randn(100, 100)

# RustでRusTorchを使用して処理
result = rust('''
    let tensor = Tensor::randn(&[100, 100]);
    let result = tensor.matmul(&tensor.transpose(0, 1));
    println!("行列乗算完了: {:?}", result.shape());
''')
```

**ネイティブRustセル：**
```rust
:dep rustorch = "0.6.2"
extern crate rustorch;

use rustorch::tensor::Tensor;
use rustorch::nn::Linear;

let model = Linear::new(784, 10);
let input = Tensor::<f32>::randn(&[32, 784]);
let output = model.forward(&input);
println!("ニューラルネットワーク出力: {:?}", output.shape());
```

## 📊 使用例

### 機械学習パイプライン

**1. データ準備（Python）**
```python
import pandas as pd
import numpy as np

# データ読み込みと前処理
data = pd.read_csv('dataset.csv')
X = data.drop('target', axis=1).values
y = data['target'].values

print(f"データセット形状: {X.shape}")
```

**2. モデル訓練（Rust）**
```rust
use rustorch::nn::{Linear, SGD};
use rustorch::tensor::Tensor;

// RusTorchテンソルに変換
let X_tensor = Tensor::from_vec(X.flatten().to_vec(), vec![X.shape[0], X.shape[1]]);
let y_tensor = Tensor::from_vec(y.to_vec(), vec![y.len()]);

// モデル作成
let mut model = Linear::new(X.shape[1], 1);
let mut optimizer = SGD::new(model.parameters(), 0.01);

// 訓練ループ
for epoch in 0..100 {
    let output = model.forward(&X_tensor);
    let loss = mse_loss(&output, &y_tensor);
    
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    
    if epoch % 10 == 0 {
        println!("エポック {}: 損失 = {:.4}", epoch, loss.item());
    }
}
```

**3. 可視化（Python）**
```python
import matplotlib.pyplot as plt

# Rustモデルから予測取得
predictions = rust_model_predict(X)

# 結果プロット
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('RusTorchモデル予測')
plt.show()
```

## ⚙️ 高度な機能

### GPU加速

インストーラーが自動的にGPUサポートを検出・設定：

**NVIDIA GPU（CUDA）：**
```bash
# 自動検出・設定
cargo build --features cuda
```

**Apple Silicon（Metal）：**
```bash
# 自動検出・設定
cargo build --features metal
```

**WebGPU（ブラウザ）：**
```bash
# WebGPUサポートChrome/Chromium用
./start_jupyter_webgpu.sh
```

### パフォーマンス最適化

**Rustカーネル最適化：**
- **コンパイルキャッシュ**: 初回実行で依存関係コンパイル、以降はキャッシュ使用
- **リリースモード**: 本番パフォーマンス用最適化ビルド
- **SIMD命令**: サポートハードウェアでの自動ベクトル化

**Pythonブリッジ最適化：**
- **ゼロコピー**: PythonとRust間直接メモリ共有
- **バッチ処理**: 効率的な一括操作
- **メモリ管理**: 自動クリーンアップとガベージコレクション

## 🔧 トラブルシューティング

### よくある問題

**1. Rustカーネルコンパイルエラー**
```bash
# BLAS/LAPACK依存関係インストール
# Ubuntu/Debian:
sudo apt-get install libblas-dev liblapack-dev libopenblas-dev

# macOS:
brew install openblas lapack

# Rustキャッシュクリア
cargo clean
```

**2. Pythonブリッジインポートエラー**
```bash
# Python環境再インストール
rm -rf .venv-hybrid
./start_jupyter_hybrid.sh
```

**3. GPU未検出**
```bash
# GPUサポート確認
nvidia-smi  # NVIDIA
system_profiler SPDisplaysDataType  # macOS Metal

# 正しい機能で再インストール
./install_jupyter.sh  # 自動検出再実行
```

## 📈 パフォーマンスベンチマーク

### 環境比較

| 環境 | セットアップ時間 | 起動時間 | 実行速度 | メモリ使用量 |
|------|----------------|----------|----------|-------------|
| ハイブリッド | 3-5分 | 10秒 | ネイティブ+Python | 中程度 |
| Pythonのみ | 1-2分 | 5秒 | Python速度 | 低 |
| Rustのみ | 2-3分 | 15秒 | ネイティブ | 低 |
| WebGPU | 2-3分 | 8秒 | GPU加速 | 中程度 |

### ハードウェア最適化

| ハードウェア | 推奨環境 | 期待パフォーマンス |
|-------------|----------|------------------|
| Apple Silicon M1/M2/M3 | ハイブリッド（Metal） | 5-10倍高速化 |
| NVIDIA GPU | ハイブリッド（CUDA） | 10-100倍高速化 |
| モダンIntel/AMD | ハイブリッド（SIMD） | 2-5倍高速化 |
| 古いハードウェア | Pythonのみ | 標準速度 |

## 🆘 サポート

### ヘルプの取得

- **ドキュメント**: このガイドとAPIドキュメントを確認
- **GitHubイシュー**: [バグ報告と機能リクエスト](https://github.com/JunSuzukiJapan/rustorch/issues)
- **ディスカッション**: [コミュニティサポートと質問](https://github.com/JunSuzukiJapan/rustorch/discussions)

### 有用なコマンド

```bash
# インストール確認
rustorch-jupyter --help

# クイックランチャーメニュー
./start_jupyter_quick.sh

# 環境情報
./install_jupyter.sh --help

# 最新版への更新
git pull origin main
./install_jupyter.sh
```

---

**最終更新**: v0.6.2 - ハイブリッド環境リリース  
**メンテナンス**: RusTorchチーム