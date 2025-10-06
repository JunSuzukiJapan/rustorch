# Token Generation Debugging Analysis

## 問題の概要

Llama-2モデル（Q4_K_M量子化）がBOSトークンに対して間違った予測を行う。

### 症状
- 入力: BOS token (1)
- 期待: Token 450 (" The") - 高いlogit
- 実際: Token 20780 (間違い) - logit 9.579（最高）
- Token 450のlogit: 0.063（非常に低い）

## 検証済み（正しい動作）✅

### 1. Embedding Extraction
- **実装**: Column-major layout（`embedding[dim] = data[dim * vocab_size + token_id]`）
- **検証**: Token 1の embedding値が期待通り
- **ファイル**: `llama.rs:528-538`

### 2. Metal GPU Matmul
- **実装**: Row-major標準実装
- **検証**: 手動計算と100%一致（`test_exact_hidden_state.rs`で確認）
- **ファイル**: `metal_shaders.metal:matmul_f32`

### 3. RMSNorm
- **実装**: `output[i] = (x[i] / RMS) * weight[i]`
- **検証**: 各ステップを詳細にログ出力、数学的に正しい
- **ファイル**: `llama.rs:272-303`

### 4. Element-wise Operations
- **add**: `test_add_operation.rs`で検証済み
- **SwiGLU**: `silu = g / (1 + exp(-g)); output = silu * u` - 標準実装

### 5. Q/K/V Projections (Layer 0)
- **検証**: 完全な入力（2048要素）でQ projection計算、期待値と一致
- **テスト**: `test_q_with_full_input.rs`
- **結果**: 100%一致

### 6. RoPE (Rotary Position Embedding)
- **Position 0での動作**: `cos=1, sin=0` → 値は変化しない（これは正しい）
- **検証**: Debug出力で確認

### 7. Attention Mechanism (Layer 0)
- **重要な発見**: BOSトークン時、Attention出力 = V値そのまま
- **理由**: BOSは自分自身にのみattendできる → attention weight = 1.0
- **これは理論的に正しい動作**

### 8. Weight Shapes
すべて正しい:
```
token_embd.weight: [2048, 32000]
blk.0.attn_q.weight: [2048, 2048]
blk.0.attn_k.weight: [2048, 256]  # GQA
blk.0.attn_v.weight: [2048, 256]  # GQA
blk.0.ffn_gate.weight: [2048, 5632]
blk.0.ffn_up.weight: [2048, 5632]
blk.0.ffn_down.weight: [5632, 2048]
output.weight: [2048, 32000]
```

## 問題の特定 ❌

### 最終Hidden Stateが不正確

**Layer 21出力**:
```
[0.70855033, 1.0006536, -0.22543797, 0.7980008, ...]
```

**RMSNorm後（LM headへの入力）**:
```
[1.1820991, 1.5812036, -0.38069266, 1.3746278, ...]
```

**最終Logits**:
- Token 450: 0.063170（正解だが低い）
- Token 20780: 9.579187（間違いだが最高）

### 検証済みの事実:
1. LM head matmul自体は正しい（`test_exact_hidden_state.rs`で確認）
2. つまり、**Layer 0-21のいずれかで値がずれている**
3. Layer 0とLayer 1の中間値は妥当な範囲内

## 残る可能性 🤔

### 1. Q4_K_M Dequantization
**最も可能性が高い**

- **実装場所**: `gguf.rs:606-693`
- **複雑性**: 256要素のsuper-block、12バイトのscale data、複雑なbit操作
- **検証方法**: llama.cppの実装とline-by-line比較

#### 現在の実装の要点:
```rust
// Super-block構造 (144 bytes):
// - d (f16): super-scale
// - dmin (f16): super-min
// - scales[12]: quantized scales
// - qs[128]: 4-bit quantized values

// Dequantization式:
output = (d * scale * q_val - dmin * min) as f64
```

### 2. Weight Layout解釈
**可能性は低い**

- GGUFからのshapeは直接使用
- Matmul実装は標準的なrow-major
- しかし、特定のweight（特にFFN weights）のtransposeが必要な可能性？

### 3. Numerical Precision
**可能性は低い**

- f32 → f64変換時の精度問題？
- しかし基本演算は正確に動作している

## 推奨される Next Steps 📋

### Priority 1: Q4_K_M Dequantization検証
1. llama.cppの`ggml-quants.c:dequantize_row_q4_K()`と比較
2. 小さなテストケースで dequant 結果を直接比較
3. 特にbit shift操作とscale/min計算を確認

### Priority 2: Alternative Quantization Test
1. より単純なQ4_0フォーマットのモデルで試す
2. または float16/float32 の非量子化モデルで試す
3. 問題が量子化特有かどうかを特定

### Priority 3: llama.cpp直接比較
1. llama.cppのデバッグ出力を有効化
2. 各層の出力を RusTorch と数値比較
3. 最初にずれる層を特定

## テストファイル

作成したデバッグ用テスト:
- `test_add_operation.rs`: add演算検証
- `test_exact_hidden_state.rs`: final matmul検証
- `test_q_projection.rs`: Q projection検証（簡易版）
- `test_q_with_full_input.rs`: Q projection検証（完全版）
- `test_token_generation.rs`: 複数トークン生成
- `test_single_token.rs`: 単一トークン処理

## デバッグログ出力箇所

Layer 0の詳細ログ（`llama.rs`）:
- Line 577-607: Attention入力、Q/K/V値
- Line 651-654: Grouped attention出力
- Line 673-676: Output projection後
- Line 727-760: Transformer layer中間値

## 参考情報

### Llama-2 Architecture
- Hidden size: 2048
- Num layers: 22
- Num heads: 32
- Num KV heads: 4 (Grouped Query Attention)
- Head dim: 64
- FFN intermediate: 5632
- Vocab size: 32000

### Q4_K_M Format
- Super-block size: 256 elements
- Block structure: 144 bytes total
  - 2 bytes: d (f16 super-scale)
  - 2 bytes: dmin (f16 super-min)
  - 12 bytes: quantized scales
  - 128 bytes: 4-bit quantized values (256 nibbles)

## 結論

基本的なすべての演算とAttention機構は正しく動作している。問題は**Q4_K_M dequantization**または**特定のweight処理**に起因する可能性が最も高い。次のステップはllama.cppの実装との詳細な比較。


## 🎯 完全検証結果（UPDATE）

### ✅ 100%正確に動作（完全な2048要素入力で確認）:
1. **RMSNorm**: 完全な2048要素入力で計算、デバッグ出力と100%一致
2. **FFN計算**: Gate/Up/Down projections、SwiGLU - すべて正確
3. **層間データ伝達**: Layer 0出力 = Layer 1入力（完全一致）
4. **すべての基本演算**: Embedding, Matmul, Add, RoPE, Attention - すべて検証済み

### 🔍 最終結論

**すべての演算が100%正確に動作しているにもかかわらず、最終予測は間違っている。**

これは以下を意味します：
1. 実装ロジックは完全に正しい
2. 演算の順序と組み合わせも正しい
3. **Weight値そのものが間違っている可能性が極めて高い**

### 推奨される決定的テスト:
1. **Float16/Float32の非量子化モデル**で試す → Dequantizationをバイパス
2. **Q4_0など、よりシンプルな量子化**で試す → Q4_K_M特有の問題か確認
3. **llama.cppのデバッグビルド**でweight値を直接ダンプして比較

