# RusTorch ベンチマークスイート

このディレクトリには、RusTorchライブラリの性能測定用ベンチマークが含まれています。

## 📊 最新ベンチマーク結果 (2025年8月31日実行)

### 🚀 パフォーマンス概要
- **テンソル作成**: 9.2μs (100要素)
- **GPU行列乗算**: 56ms (大行列、Metal対応)
- **SIMD演算**: 1.0μs-11.5μs (128-2048要素)
- **SVD分解**: 424μs-255ms (4x4-64x64行列)
- **ガンマ関数**: 660-690ns (スカラー演算)
- **正規分布サンプリング**: 1.77μs (100サンプル)
- **FFT**: 1.0μs-61.9μs (4-128点)
- **スループット**: 7800万要素/秒 (マイクロ分布)

## 構造

### アクティブベンチマーク (25個)
- `tensor_ops.rs` - 基本テンソル操作
- `optimization_benchmark.rs` - SIMD最適化演算
- `matrix_decomposition_benchmark.rs` - 行列分解 (SVD/QR/固有値)
- `distributions_benchmark.rs` - 統計分布サンプリング
- `fft_benchmark.rs` - 高速フーリエ変換
- `special_functions_benchmark.rs` - 特殊数学関数
- `nn_benchmark.rs` - ニューラルネットワーク演算
- `multi_gpu_benchmark.rs` - マルチGPU処理
- `memory_pool.rs` - メモリプール管理

## 実行方法

### 全ベンチマーク実行
```bash
cargo bench
```

### 特定のベンチマーク実行
```bash
# テンソル操作ベンチマーク
cargo bench --bench tensor_performance_refactored

# SIMDベンチマーク
cargo bench --bench simd_performance_refactored

# メモリベンチマーク
cargo bench --bench memory_performance_refactored
```

### クイックテスト
```bash
# 短時間でのテスト実行
cargo bench --bench tensor_performance_refactored -- --quick
```

## ベンチマーク設定

### 標準サイズ
- **BENCHMARK_SIZES**: 1K, 4K, 16K, 64K, 256K要素
- **MATRIX_SIZES**: 10K, 250K, 1M, 4M要素（2D行列）
- **QUICK_SIZES**: 64, 256, 1K要素（高速テスト用）

### 設定オプション
- `BenchmarkConfig::default()` - 標準設定（100サンプル、5秒測定）
- `BenchmarkConfig::quick()` - 高速設定（50サンプル、2秒測定）
- `BenchmarkConfig::thorough()` - 詳細設定（200サンプル、10秒測定）

## 測定項目

### テンソル操作
- 要素ごとの加算
- 行列乗算
- テンソル割り当て
- テンソルコピー

### SIMD最適化
- スカラー vs SIMD要素演算
- SIMD行列演算
- ベクトル化操作

### メモリ管理
- 割り当て戦略比較
- ゼロコピー操作
- メモリプール効率

## 結果の解釈

ベンチマーク結果は以下の形式で表示されます：
- **time**: 操作にかかった時間
- **thrpt**: スループット（要素/秒）
- **outliers**: 外れ値の数と種類

性能改善の目安：
- SIMD操作は通常スカラー操作の1.5-3倍高速
- ゼロコピー操作は通常のコピーより大幅に高速
- メモリプール戦略は大量割り当て時に効果的

## 新しいベンチマークの追加

1. `common/`モジュールの適切なファイルに関数を追加
2. 新しいベンチマークファイルを作成し、共通関数を使用
3. `Cargo.toml`の`[[bench]]`セクションに追加

例：
```rust
use common::tensor_ops::bench_elementwise_add;

fn my_benchmark(c: &mut Criterion) {
    bench_elementwise_add::<f32>(
        c,
        "my_group",
        &[(1000, "1K elements")],
        Some(BenchmarkConfig::default()),
    );
}
```
