# RusTorch Benchmarks

このディレクトリにはRusTorchのパフォーマンステストと記録が含まれています。

## Directory Structure

- `performance_records/` - パフォーマンステストの結果記録
- `scripts/` - ベンチマーク実行用スクリプト（将来追加予定）

## Performance Records

### 2025-01-27: Comprehensive Backend Comparison
- CPU vs Metal GPU vs CoreML Neural Engine
- Matrix, Convolution, Transformer operations comparison
- File: `performance_records/2025-01-27_comprehensive_backend_comparison.md`

## Running Benchmarks

### Simple Performance Demo
全バックエンドでのパフォーマンス比較:

```bash
# CPU
cd benchmarks && cargo run --bin simple_performance_demo --release -- --backend cpu --benchmark all

# Metal GPU (macOS)
cd benchmarks && cargo run --bin simple_performance_demo --features metal --release -- --backend metal --benchmark all

# CoreML Neural Engine (macOS)
cd benchmarks && cargo run --bin simple_performance_demo --features coreml --release -- --backend coreml --benchmark all
```

### Individual Benchmarks
特定の操作タイプのみテスト:

```bash
# Matrix operations only
cd benchmarks && cargo run --bin simple_performance_demo --features metal --release -- --backend metal --benchmark matrix

# Convolution operations only
cd benchmarks && cargo run --bin simple_performance_demo --features metal --release -- --backend metal --benchmark convolution

# Transformer operations only
cd benchmarks && cargo run --bin simple_performance_demo --features metal --release -- --backend metal --benchmark transformer
```

## Recording Guidelines

新しいパフォーマンス記録を追加する際は:

1. ファイル名フォーマット: `YYYY-MM-DD_test_description.md`
2. システム情報 (OS, ハードウェア) を含める
3. テスト設定 (サイズ、反復回数等) を記録
4. 結果の解釈・分析を含める
5. 再現可能なコマンドを記載

## Hardware Information

パフォーマンステストは以下の環境で実行:
- **OS**: macOS (Darwin 24.6.0)
- **GPU**: Metal対応GPU
- **Neural Engine**: CoreML対応
- **Compilation**: Rust release mode