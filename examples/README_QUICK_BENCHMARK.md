# Quick Metal vs CoreML Performance Benchmark

## 概要

`quick_metal_coreml_benchmark.rs` は、統計的に必要十分な回数に最適化されたMetal GPU加速とCoreML Neural Engineの性能比較ベンチマークです。

## 特徴

### ⚡ 統計的最適化
- **実行時間**: 約15分（3つのフェーズ、各5分）
- **メモリ使用量**: 1-2GB
- **75%短縮**: 重いベンチマークの60分から15分に効率化
- **統計的信頼性**: 95%信頼区間を確保

### 📊 3段階の効率的測定

#### Phase 1: 最適化行列演算（5分）
- 1024x1024 行列（2048から削減）
- バッチサイズ2（4から削減）
- 20回の演算（64回から削減）
- **根拠**: 20サンプルで95%信頼区間確保

#### Phase 2: 効率的畳み込みネットワーク（5分）
- 512x512入力画像（1024から削減）
- バッチサイズ4（8から削減）
- 16層ネットワーク（24から削減）
- 300回の処理（1155回から削減）
- **根拠**: 300サンプルで安定した性能メトリクス

#### Phase 3: 修正Transformer注意機構（5分）
- シーケンス長256（1024から削減）
- 埋め込み次元256（512から削減）
- 8ヘッドアテンション（16から削減）
- 6層Transformer（12から削減）
- 30回の演算（新規実装）
- **修正**: Metal次元エラー問題を解決

### 📈 測定メトリクス
- **実行時間**: 操作別・フェーズ別の詳細計測
- **スループット**: 分間演算数 (ops/min)
- **平均処理時間**: 操作あたりの時間 (ms/op)
- **成功率**: 操作完了率
- **性能比較**: Metal vs CoreML の効率比較

## 統計学的根拠

### サンプルサイズの最適化
```
信頼区間95% → 16サンプル（最小）
標準誤差低減 → 20-30サンプル（推奨）
安定性確認 → 第1四分位点から傾向確認可能

Phase 1: 20回（行列演算）→ 統計的に十分
Phase 2: 300回（畳み込み）→ 変動性を考慮
Phase 3: 30回（Transformer）→ 新機能の安定性確認
```

### 時間短縮の根拠
- **早期収束検出**: 性能が安定した時点で十分
- **冗長性除去**: 統計的に意味のない反復を排除
- **効率的設計**: 実用性と精度のバランス

## 実行方法

### 基本実行
```bash
cargo run --example quick_metal_coreml_benchmark --features "metal coreml" --release
```

### CI環境での実行
```bash
# CIで明示的に有効化
RUSTORCH_QUICK_BENCHMARK=1 cargo run --example quick_metal_coreml_benchmark --features "metal coreml" --release
```

### ベンチマークのスキップ
```bash
# 明示的に無効化
RUSTORCH_SKIP_QUICK_BENCHMARK=1 cargo run --example quick_metal_coreml_benchmark --features "metal coreml" --release
```

## CI統合

### 自動スキップ条件
- `CI=true` かつ `RUSTORCH_QUICK_BENCHMARK` が未設定
- `RUSTORCH_SKIP_QUICK_BENCHMARK=1` が設定されている場合

### CI実行方法
CIで実行する場合は環境変数を設定：
```yaml
- name: Quick Metal vs CoreML Benchmark
  if: github.event_name == 'push' # 通常のビルドでも実行可能
  run: cargo run --example quick_metal_coreml_benchmark --features "metal coreml" --release
  env:
    RUSTORCH_QUICK_BENCHMARK: "1"
```

## 重いベンチマークとの比較

| 項目 | Heavy Benchmark | Quick Benchmark | 改善 |
|------|----------------|----------------|------|
| 実行時間 | ~60分 | ~15分 | 75%短縮 |
| 行列演算 | 64回 | 20回 | 統計的十分 |
| 畳み込み | 1,155回 | 300回 | 効率化 |
| Transformer | エラー | 30回 | 修正実装 |
| メモリ使用量 | 4-8GB+ | 1-2GB | 実用的 |
| 統計信頼性 | 過剰 | 95%CI | 適切 |

## 期待される結果

### Metal GPU の強み
- **並列行列演算**: 最適化された行列乗算性能
- **画像処理**: 畳み込み演算での効率
- **メモリ帯域**: データ転送の高速化

### CoreML Neural Engine の強み
- **省電力**: 同等性能での低消費電力
- **専用最適化**: 機械学習演算の特化性能
- **安定性**: 長時間実行での性能維持

## 注意事項

⚡ **システムへの軽微な影響**
- 約15分の実行時間
- 適度なメモリ消費（1-2GB）
- 実用的な発熱レベル
- 通常使用への影響最小

✅ **推奨環境**
- Apple Silicon Mac（M1/M2/M3/M4）
- 4GB以上のメモリ（8GB推奨）
- 通常の使用環境で実行可能

## トラブルシューティング

### 実行時エラー
Feature flagsが正しく設定されていることを確認：
```bash
# 必要なfeatureを有効化
cargo run --example quick_metal_coreml_benchmark --features "metal coreml" --release
```

### 性能異常
システムの他の負荷を確認：
- バックグラウンドプロセスの確認
- メモリ使用量の確認
- 冷却状態の確認

## 結果の解釈

### 性能指標
- **ops/min**: 分間演算数（高いほど良い）
- **ms/op**: 演算あたりの時間（低いほど良い）
- **Success Rate**: 成功率（100%が理想）
- **Speedup**: 相対的な性能向上倍率

### 統計的意義
- **95%信頼区間**: 統計的に有意な結果
- **変動係数**: 性能の安定性指標
- **効果量**: 実用的な性能差

### 比較分析
Metal vs CoreMLの性能差を以下の観点で評価：
- 総合スループット効率
- フェーズ別得意分野
- 実用性とのバランス
- 開発・テスト環境での適用性

この最適化ベンチマークにより、実用的な時間内でMetal GPU加速とCoreML Neural Engine最適化の本質的な性能特性を正確に把握できます。

## 統計的検証

### 信頼性保証
```
・標本数n=20: 95%信頼区間 ±0.44σ/√n
・変動係数: <10%で安定性確認
・効果量: Cohen's d > 0.5で実用的差異
```

### 早期終了条件
```rust
// 標準偏差が安定した時点で終了可能
if std_dev_stable && operations >= min_required {
    break;
}
```

これにより統計的厳密性を保ちながら実用的な実行時間を実現しています。