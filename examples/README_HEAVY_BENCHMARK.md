# Heavy Metal vs CoreML Performance Benchmark

## 概要

`metal_coreml_heavy_benchmark.rs` は、Apple Silicon上でのMetal GPU加速とCoreML Neural Engine最適化の真の性能差を測定するための重いベンチマークです。

## 特徴

### 🚀 本格的なワークロード
- **実行時間**: 約1時間（3つのフェーズ、各20分）
- **メモリ使用量**: 4-8GB+
- **処理規模**: 2048x2048行列、1024x1024画像、大規模Transformer

### 📊 3段階の性能測定

#### Phase 1: 大規模行列演算（20分）
- 2048x2048 行列の並列乗算
- バッチサイズ4での連続処理
- Metal GPU並列処理 vs Neural Engine最適化

#### Phase 2: 深層畳み込みネットワーク（20分）
- 1024x1024入力画像、バッチサイズ8
- 24層の深い畳み込みネットワーク
- チャンネル数: 3→64→128→256→512→1024

#### Phase 3: Transformer風アテンション（20分）
- シーケンス長1024、埋め込み次元512
- 16ヘッドマルチヘッドアテンション
- 12層のTransformerレイヤー

### 📈 測定メトリクス
- **実行時間**: 操作別・フェーズ別の詳細計測
- **スループット**: 秒間演算数
- **メモリ使用量**: ピーク・平均メモリ消費量
- **成功率**: 操作完了率
- **性能比較**: Metal vs CoreML の詳細比較

## 実行方法

### 基本実行
```bash
cargo run --example metal_coreml_heavy_benchmark --features "metal coreml" --release
```

### CI環境での実行
```bash
# CIで明示的に有効化
RUSTORCH_HEAVY_BENCHMARK=1 cargo run --example metal_coreml_heavy_benchmark --features "metal coreml" --release
```

### ベンチマークのスキップ
```bash
# 明示的に無効化
RUSTORCH_SKIP_HEAVY_BENCHMARK=1 cargo run --example metal_coreml_heavy_benchmark --features "metal coreml" --release
```

## CI統合

### 自動スキップ条件
- `CI=true` かつ `RUSTORCH_HEAVY_BENCHMARK` が未設定
- `RUSTORCH_SKIP_HEAVY_BENCHMARK=1` が設定されている場合

### CI実行方法
CIで実行する場合は環境変数を設定：
```yaml
- name: Heavy Metal vs CoreML Benchmark
  if: github.event_name == 'schedule' # Nightly builds only
  run: cargo run --example metal_coreml_heavy_benchmark --features "metal coreml" --release
  env:
    RUSTORCH_HEAVY_BENCHMARK: "1"
```

## 期待される結果

### Metal GPU の強み
- **並列行列演算**: 大規模行列乗算での高いスループット
- **画像処理**: 畳み込み演算での並列処理効率
- **メモリ帯域**: 大容量データの高速転送

### CoreML Neural Engine の強み
- **省電力**: 同等性能での低電力消費
- **専用最適化**: 機械学習演算での特化性能
- **熱効率**: 長時間実行での安定性能

## 注意事項

⚠️ **システムへの影響**
- 約1時間の連続実行
- 大量のメモリ消費（4-8GB+）
- 発熱による熱スロットリングの可能性
- バッテリー消費量の増加

⚠️ **推奨環境**
- Apple Silicon Mac（M1/M2/M3/M4）
- 8GB以上のメモリ
- 良好な冷却環境
- AC電源接続

## トラブルシューティング

### メモリ不足エラー
システムメモリが不足している場合、設定を調整：
```rust
// HeavyBenchmarkConfig の値を小さく設定
matrix_size: 1024,        // 2048 → 1024
image_batch_size: 4,      // 8 → 4
```

### 熱スロットリング
長時間実行中にパフォーマンスが低下する場合：
- システム冷却を改善
- バックグラウンドプロセスを終了
- 室温を下げる

### コンパイルエラー
Feature flagsが正しく設定されていることを確認：
```bash
# 必要なfeatureを有効化
cargo run --example metal_coreml_heavy_benchmark --features "metal coreml" --release
```

## 結果の解釈

### 性能指標
- **ops/sec**: 秒間演算数（高いほど良い）
- **ms/op**: 演算あたりの時間（低いほど良い）
- **Success Rate**: 成功率（100%が理想）
- **Speedup vs CPU**: CPU比でのスピードアップ倍率

### 比較分析
Metal vs CoreMLの性能差を以下の観点で評価：
- 総合スループット
- フェーズ別性能
- メモリ効率
- 長時間実行での安定性

このベンチマークにより、実際のワークロードでのMetal GPU加速とCoreML Neural Engine最適化の性能特性を正確に把握できます。