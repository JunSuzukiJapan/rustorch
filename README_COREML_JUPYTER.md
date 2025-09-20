# RusTorch CoreML Jupyter Integration

このドキュメントは、RusTorchでCoreMLをJupyter環境で使用する方法について説明します。

## 🎯 概要

RusTorchのCoreML統合により、以下が可能になります：

- **Apple Neural Engineの活用**: M1/M2チップの専用ハードウェアを使用した高速推論
- **スマートデバイス選択**: 演算特性に基づく最適なデバイス（CPU/GPU/CoreML）の自動選択
- **Jupyter統合**: RustカーネルとPythonバインディングの両方でサポート
- **ハイブリッド実行**: CoreML非対応演算の自動フォールバック

## 🛠️ セットアップ

### 1. 前提条件

- **macOS**: CoreMLはmacOSでのみ利用可能
- **Rust**: 1.70以上
- **Python**: 3.8以上（Pythonバインディング使用時）
- **Jupyter**: Jupyter LabまたはJupyter Notebook

### 2. RusTorchのビルド

CoreMLフィーチャーを有効にしてビルド：

```bash
# CoreMLのみ
cargo build --features coreml

# CoreML + Python バインディング
cargo build --features "coreml python"

# ハイブリッド実行（CoreML + Metal）
cargo build --features "coreml metal"
```

### 3. Jupyter環境のセットアップ

#### Rustカーネルのインストール

```bash
# Rust カーネルをインストール
cargo install evcxr_jupyter
evcxr_jupyter --install

# Jupyter Lab を起動
jupyter lab
```

#### Pythonバインディングのビルド

```bash
# Pythonバインディングをビルド
pip install maturin
maturin develop --features "coreml python"

# Jupyter Lab を起動
jupyter lab
```

## 📚 使用方法

### Rustカーネルでの使用

`notebooks/coreml_integration_rust.ipynb`を参照：

```rust
// CoreMLの可用性チェック
use rustorch::backends::DeviceManager;
let available = DeviceManager::is_coreml_available();

// CoreMLバックエンドの作成
use rustorch::gpu::coreml::backend::{CoreMLBackend, CoreMLBackendConfig};
let config = CoreMLBackendConfig::default();
let backend = CoreMLBackend::new(config)?;

// スマートデバイス選択
use rustorch::gpu::coreml::smart_device_selector::*;
let selector = SmartDeviceSelector::new(available_devices);
let device = selector.select_device(&operation_profile);
```

### Pythonバインディングでの使用

`notebooks/coreml_integration_python.ipynb`を参照：

```python
import rustorch

# CoreML可用性チェック
available = rustorch.is_coreml_available()

# CoreMLデバイスの作成
device = rustorch.CoreMLDevice(device_id=0)

# バックエンド設定
config = rustorch.CoreMLBackendConfig(
    enable_caching=True,
    max_cache_size=200,
    enable_profiling=True
)
backend = rustorch.CoreMLBackend(config)
```

## 🚀 主要機能

### 1. デバイス管理

```rust
// デバイスキャッシュによる高速初期化
use rustorch::gpu::coreml::device_cache::DeviceCache;
let cache = DeviceCache::global();
cache.warmup(); // 利用可能デバイスを事前チェック
```

### 2. スマートデバイス選択

演算特性に基づいて最適なデバイスを自動選択：

| 演算タイプ | 小さいサイズ | 中程度サイズ | 大きいサイズ |
|------------|-------------|-------------|-------------|
| 行列乗算    | CPU         | Metal GPU   | CoreML      |
| 活性化関数  | CPU         | Metal GPU   | CoreML      |
| 畳み込み    | CPU         | Metal GPU   | CoreML      |
| 複素数演算  | CPU         | Metal GPU   | Metal GPU   |

### 3. 演算キャッシング

```rust
// 演算結果のキャッシングでパフォーマンス向上
let config = CoreMLBackendConfig {
    enable_caching: true,
    max_cache_size: 1000,
    enable_profiling: true,
    auto_fallback: true,
};
```

### 4. フォールバック機能

CoreML非対応演算は自動的に他のバックエンドにフォールバック：

- **複素数演算**: Metal GPU → CPU
- **カスタムカーネル**: Metal GPU → CPU
- **分散演算**: Metal GPU → CPU

## 📊 パフォーマンス

### ベンチマーク例（仮定値）

| 演算サイズ | CPU (ms) | Metal GPU (ms) | CoreML (ms) | 改善率 |
|------------|----------|----------------|-------------|---------|
| 64x64      | 0.5      | 0.3            | 0.4         | 25%     |
| 128x128    | 2.1      | 1.2            | 0.8         | 62%     |
| 256x256    | 8.5      | 4.2            | 2.1         | 75%     |
| 512x512    | 34.2     | 16.8           | 8.4         | 76%     |

### メモリ使用量

- **CoreMLキャッシュ**: 設定可能（デフォルト100MB）
- **演算キャッシュ**: 最大1000演算（設定可能）
- **デバイスキャッシュ**: 30秒間有効

## 🔧 設定オプション

### CoreMLBackendConfig

```rust
pub struct CoreMLBackendConfig {
    pub enable_caching: bool,        // 演算キャッシングの有効化
    pub max_cache_size: usize,       // 最大キャッシュサイズ
    pub auto_fallback: bool,         // 自動フォールバック
    pub enable_profiling: bool,      // プロファイリング有効化
}
```

### DeviceThresholds

```rust
pub struct DeviceThresholds {
    pub coreml_min_size: usize,      // CoreML最小テンソルサイズ
    pub coreml_max_size: usize,      // CoreML最大テンソルサイズ
    pub metal_min_size: usize,       // Metal GPU最小サイズ
    pub gpu_min_memory: usize,       // GPU最小メモリ要件
}
```

## 🐛 トラブルシューティング

### よくある問題

1. **CoreMLが利用できない**
   ```bash
   # CoreMLフィーチャーでビルドされているか確認
   cargo build --features coreml
   ```

2. **Pythonバインディングが見つからない**
   ```bash
   # maturinでビルド
   maturin develop --features "coreml python"
   ```

3. **パフォーマンスが期待より低い**
   - 小さいテンソルではオーバーヘッドが大きい可能性
   - プロファイリングを有効にして詳細を確認

### デバッグ情報の取得

```rust
// 詳細なログを有効化
env::set_var("RUST_LOG", "debug");

// 統計情報の取得
let stats = backend.get_stats();
println!("Cache hit rate: {:.2%}",
         stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64);
```

## 📝 例とサンプル

### 基本的な行列乗算

```rust
let a = Tensor::randn(&[256, 256]);
let b = Tensor::randn(&[256, 256]);

// スマートデバイス選択で自動的に最適なデバイスを使用
let result = a.matmul(&b)?;
```

### ニューラルネットワーク層

```rust
// 線形層（CoreMLで最適化）
let linear = Linear::new(784, 256);
let hidden = linear.forward(&input)?;

// ReLU活性化（Metal GPUで最適化）
let activated = hidden.relu()?;

// 出力層（CoreMLで最適化）
let output_linear = Linear::new(256, 10);
let output = output_linear.forward(&activated)?;
```

## 🔮 今後の開発

### 短期目標

- [ ] 実際のCoreML演算の実装
- [ ] 包括的なベンチマークスイート
- [ ] エラーハンドリングの改善
- [ ] メモリ使用量の最適化

### 長期目標

- [ ] CoreML独自モデルファイルのサポート
- [ ] 動的な演算グラフの最適化
- [ ] 分散処理のサポート
- [ ] iOSでの実行サポート

## 📄 ライセンス

このプロジェクトは[MITライセンス](../LICENSE)の下で公開されています。

## 🤝 コントリビューション

コントリビューションを歓迎します！詳細は[CONTRIBUTING.md](../CONTRIBUTING.md)を参照してください。

---

**注意**: このドキュメントの例と機能の一部は開発中のものです。最新の実装状況については、コードベースを確認してください。