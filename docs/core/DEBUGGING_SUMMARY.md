# RusTorch Debugging Summary

**日時**: 2025-10-08
**問題**: Token生成が間違っている ("drew drew drew" instead of "Hello everyone")

## ✅ 検証済み - 正しい実装

### 1. Position Calculation
- Step 0: `start_position = 0`, positions 0-19 ✅
- Step 1+: `start_position = generated_ids.len() - 1` ✅
- 全22層で同じpositionが使用されている ✅

### 2. RoPE Implementation
**Lookup Table**:
- theta = 10000, head_dim = 64 ✅
- cos(0) = 1.0, sin(0) = 0.0 ✅
- cos(1) ≈ 0.5403, sin(1) ≈ 0.8415 ✅
- cos(2) ≈ -0.4161, sin(2) ≈ 0.9093 ✅

**Rotation Formula**:
```rust
r0 = x0 * cos - x1 * sin  ✅
r1 = x0 * sin + x1 * cos  ✅
```

**Numerical Verification**:
- Position 0: r0 = x0, r1 = x1 (identity) ✅
- All rotation values mathematically correct ✅

### 3. Grouped-Query Attention Structure
```rust
num_heads = 32
num_kv_heads = 4
num_groups = 32 / 4 = 8  ✅
kv_head = h / num_groups  ✅
```

## ❌ 問題継続

### Output Comparison

**Input**: "Hello world\n"

**llama.cpp (Q6_K)**:
```
Output: "Hello everyone,"
Status: ✅ CORRECT
```

**RusTorch (Q6_K, hybrid-f32)**:
```
Output: "drew drew drew Superhé"
Token IDs: [15010, 15010, 15010, 5670, 19880]
Status: ❌ WRONG
```

### Logits Comparison

**RusTorch Step 0 Logits**:
```
#1: token=15010 ("drew")  logit=11.8314  ← 選択される
#2: token=5670  ("Super") logit=9.4277
#3: token=19880 ("hé")    logit=9.0241
...
Expected tokens:
  token 15043 logit=0.0468   ← これが選択されるべき
  token 6324 logit=-2.2811
```

Token 15010 ("drew")が圧倒的に高いlogit値を持っている。

## 🔍 未検証項目

### 1. Weight Loading (Q6_K → f32)
- Q6_K quantization形式からf32への変換が正しいか？
- llama.cppと同じdequantization実装か？

### 2. Attention Mechanism Details
- Attention scoresの計算
- Causal maskingの適用
- Softmax numerically stable implementation
- Value aggregation

### 3. FFN (Feed-Forward Network)
- SwiGLU activation
- Gate/Up projections
- Down projection

### 4. Layer Normalization
- RMSNorm implementation
- Epsilon value (1e-5?)
- Numerical stability

### 5. Output LM Head
- Weight matrix shape and indexing
- Final logits calculation

## 💡 仮説

### 最も可能性が高い原因

**1. Q6_K Dequantization Bug** (確率: 70%)
- RusTorchのQ6_K → f32変換がllama.cppと異なる
- 微小な誤差が22層で累積
- 最終logitsが大きく乖離

**検証方法**:
- F32モデル（未量子化）でテスト
- または、Q6_K dequantization実装をllama.cppと完全に一致させる

**2. Attention Score Calculation** (確率: 20%)
- Softmax前のscaling factor
- Numerical stability issue
- KV cacheとの連結処理

**3. Accumulated Numerical Errors** (確率: 10%)
- f32精度での累積誤差
- 22層を通過する過程で増幅
- Metal GPUでの計算順序の違い

## 🎯 次のアクションアイテム

### Priority 1: Quantization Verification
1. Q4_0モデルでテスト（最もシンプルな量子化）
2. Q6_K dequantization実装をllama.cpp/ggml-quants.cと比較
3. Dequantized weights値をサンプリング比較

### Priority 2: Layer-by-Layer Comparison
1. Layer 0 output hidden stateをllama.cppと比較
2. Divergence pointを特定
3. 該当レイヤーの実装を詳細検証

### Priority 3: Simple Test Case
1. 単一トークン入力でテスト
2. Embedding → Layer 0 → Outputの各段階を検証
3. 最小限のケースで問題を再現

## 📊 Debug Output Examples

### Embedding (First Token, Token ID=1)
```
First 10 values: [-0.0010786057, 0.0057525635, -0.00089883804, ...]
```

### Layer 0 Stats
```
Input: rms=0.010231, min=-0.060242, max=0.085144
After Attention: rms=0.003226, min=-0.030446, max=0.029820
After FFN: rms=0.002464, min=-0.009516, max=0.011336
Output: rms=0.010999, min=-0.060879, max=0.086502
```

### Final Logits (Step 0)
```
max=11.8314, min=-9.8738, mean=-0.0091
Top: 15010(11.83), 5670(9.43), 19880(9.02)
```

## 📝 Notes

- RoPE実装は完璧に正しいことが確認された
- Position calculationも正しい
- 問題はRoPE以外の箇所にある
- Quantization bugまたはattention実装の可能性が最も高い
