# RusTorch ベンチマークスイート

このディレクトリには、RusTorchライブラリの性能測定用ベンチマークが含まれています。

## 構造

### 共通ユーティリティ (`common/`)
- `mod.rs` - 共通の設定、サイズ定義、ヘルパー関数
- `tensor_ops.rs` - 基本的なテンソル操作のベンチマーク関数
- `simd_ops.rs` - SIMD最適化操作のベンチマーク関数  
- `memory_ops.rs` - メモリ管理とゼロコピー操作のベンチマーク関数

### リファクタリング済みベンチマーク
- `tensor_performance_refactored.rs` - 基本的なテンソル操作の性能測定
- `simd_performance_refactored.rs` - SIMD最適化の性能測定
- `memory_performance_refactored.rs` - メモリ戦略の性能測定

### レガシーベンチマーク
既存のベンチマークファイルは互換性のために保持されていますが、新しいリファクタリング版の使用を推奨します。

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
