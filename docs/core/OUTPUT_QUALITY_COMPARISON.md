# Output Quality Comparison: RusTorch vs llama.cpp

生成日時: 2025-10-08
モデル: TinyLlama-1.1B-Chat-v1.0 Q4_K_M
バックエンド: Metal GPU

## テスト条件

**共通設定:**
- モデル: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (638MB)
- 生成トークン数: 10
- Temperature: 0 (greedy sampling)
- プロンプト: "Hello"

## 出力比較

### llama.cpp (Reference Implementation)

```
Hello<|assistant|>
Hi there<|user|>
How are you?<|assistant|>
```

**分析:**
- ✅ 意味のある対話が生成されている
- ✅ 適切なフォーマットトークン (`<|assistant|>`, `<|user|>`)
- ✅ 文脈に沿った応答 ("Hi there", "How are you?")
- ✅ トークン生成速度: 244.64 tokens/sec (eval)

### RusTorch (Current Implementation)

```
Hello more wo ags ags O
```

**生成トークン詳細:**
```
Token 0: 901 -> 'more'
Token 1: 827 -> 'wo'
Token 2: 810 -> 'ags'
Token 3: 810 -> 'ags'
Token 4: 82  -> 'O'
```

**分析:**
- ❌ Gibberish（意味不明な出力）
- ❌ 繰り返しパターン ("ags ags")
- ❌ 文脈に沿っていない
- ✅ Segfaultなし、安定動作
- ✅ RoPE + Causal Masking適用済み

## 実装状況

### ✅ 完了した実装

1. **RoPE (Rotary Position Embedding)**
   - 事前計算: rope_cos, rope_sin (10000.0 theta)
   - Q/K投影後に適用
   - 各レイヤーで動作確認済み

2. **Causal Masking**
   - Upper triangular mask (j > i → -inf)
   - Softmax前に適用
   - 各レイヤーで動作確認済み

3. **GQA (Grouped Query Attention)**
   - K/V projection: kv_dim=256 (4 heads × 64 dim)
   - Q projection: d_model=2048 (32 heads × 64 dim)
   - KV head expansion: 4→32

4. **Auto d_ff Calculation**
   - TinyLlama: d_ff=5632 (非標準)
   - Weight sizeから自動計算

### 🔍 疑わしい実装箇所

#### 1. 位置トラッキング (最有力候補)

**現在:**
```rust
let start_position = 0; // TODO: Track position for multi-token generation
let q_proj = self.apply_rope(&q_proj, seq_len, num_q_heads, head_dim, start_position);
let k_proj = self.apply_rope(&k_proj, seq_len, num_kv_heads, head_dim, start_position);
```

**問題点:**
- 全トークン生成で`start_position=0`固定
- マルチトークン生成時、位置が更新されない
- llama.cppはKV Cacheと連動して位置管理

**影響:**
- 過去のトークンと現在のトークンの位置関係が正しくない
- アテンション計算が不正確
- 出力品質に重大な影響

#### 2. Simplified GQA Implementation

**現在:**
```rust
// Simplified GQA: Repeat KV heads to match Q heads
let k_expanded = Self::repeat_kv_heads(&k_proj, seq_len, num_kv_heads, num_q_heads, head_dim);
let v_expanded = Self::repeat_kv_heads(&v_proj, seq_len, num_kv_heads, num_q_heads, head_dim);
```

**問題点:**
- 単純なKVヘッド繰り返しのみ
- Full 32-head attentionではない
- Head-wise計算なし

**影響:**
- アテンション表現力が低下
- モデル本来の能力を発揮できない

#### 3. Layer Normalization

**現在:**
```rust
executor.layer_norm_f32(&x_f32, &ln1_weight_f32, &mut x_ln1, seq_len, d_model,
                        ln1_bias_f32.as_deref(), 1e-5)?;
```

**確認事項:**
- TinyLlamaはRMS Norm使用の可能性
- 現在のLayer Normは通常のLayerNorm (mean + variance)
- llama.cppのRMS Normと異なる可能性

#### 4. Softmax Numerical Stability

**現在:**
```rust
// Find max for numerical stability
let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

// Subtract max and compute exp
for val in row.iter_mut() {
    *val = (*val - max_val).exp();
}

// Normalize
let sum: f32 = row.iter().sum();
for val in row.iter_mut() {
    *val /= sum;
}
```

**確認事項:**
- 実装自体は安定
- Causal mask後の-infの処理は正しい

## 出力品質ギャップの根本原因候補

### 🔴 最優先: 位置トラッキング

**仮説:**
マルチトークン生成時、各トークンの位置情報が正しく更新されていないため、RoPEが誤った位置エンコーディングを適用している。

**検証方法:**
1. 単一トークン生成（seq_len=1）で品質確認
2. 位置を手動追跡して再テスト
3. llama.cppのKV Cache実装を参照

### 🟡 重要: RMS Norm vs Layer Norm

**仮説:**
TinyLlamaはRMS Normを使用しているが、RusTorchは通常のLayer Normを使用しているため、活性化値の分布が異なる。

**検証方法:**
1. GGUFメタデータでnormalization typeを確認
2. RMS Norm実装に切り替え
3. llama.cppのnormalization実装を参照

### 🟢 要検討: Full Multi-Head Attention

**仮説:**
Simplified GQA（KVヘッド繰り返しのみ）では、本来のアテンション計算と異なる。

**検証方法:**
1. Full 32-head attention loopを実装
2. Head-wise計算を正確に実行
3. パフォーマンスとの tradeoff検討

## 次のステップ

### Priority 1: 位置トラッキング修正

```rust
// Generate関数内で位置を追跡
let mut position = 0;
for _ in 0..max_tokens {
    let output = self.forward_metal(&input, position, debug)?;
    // ...
    position += 1; // 位置を更新
}
```

### Priority 2: RMS Norm実装

```rust
fn rms_norm_f32(x: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = x.len();
    // Compute RMS (Root Mean Square)
    let rms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / (n as f32);
    let rms = (rms + eps).sqrt();

    // Normalize and scale
    for i in 0..n {
        output[i] = (x[i] / rms) * weight[i];
    }
}
```

### Priority 3: KV Cache実装

- 過去のK/V投影を保存
- 位置と連動したキャッシュ管理
- メモリ効率と速度向上

### Priority 4: Full Multi-Head Attention

- 32-head loopで各ヘッドを個別処理
- Head-wise reshape/attention/concat
- パフォーマンス最適化

## パフォーマンス比較

**llama.cpp:**
- Prompt eval: 662.40 tokens/sec
- Generation: 244.64 tokens/sec
- Load time: 77.46 ms

**RusTorch:**
- TBD (速度測定未実施)
- Metal forward pass: 動作確認済み

## 結論

**現状:**
- RoPE + Causal Maskingを実装したが、出力品質は依然としてgibberish
- 構造的には正しく動作しているが、細部の実装差異が大きな品質ギャップを生んでいる

**根本原因（推定）:**
1. **位置トラッキング不足** (最有力) - マルチトークン生成時の位置更新なし
2. **RMS Norm未実装** - Layer NormとRMS Normの違い
3. **Simplified GQA** - Full multi-head attentionではない

**次の優先タスク:**
1. 位置トラッキング実装 (即効性大)
2. RMS Norm実装 (TinyLlama仕様確認)
3. KV Cache実装 (速度+品質向上)
4. Full multi-head attention (表現力向上)

---

**Status:** 🔍 Output quality gap identified - Position tracking most likely cause
**Next Milestone:** Position tracking + RMS Norm implementation
