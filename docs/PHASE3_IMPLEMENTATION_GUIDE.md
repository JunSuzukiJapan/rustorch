# Phase 3 Implementation Guide: True Parallel Batch Processing

## Overview
Phase 2で実装したバッチ対応Metalカーネルを使用して、真の並列バッチ処理を実現します。

## 実装タスク概要

### Task 1: Rustカーネルラッパーの追加 ⏳

#### 1.1 RMS Norm Batch Wrapper
**ファイル**: `src/gpu/metal_kernels.rs`

```rust
pub fn metal_rms_norm_batch_f32(
    input: &[f32],          // [batch_size, seq_len, hidden_dim]
    weight: &[f32],         // [hidden_dim]
    output: &mut [f32],     // [batch_size, seq_len, hidden_dim]
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    eps: f32,
) -> RusTorchResult<()>
```

**実装ポイント**:
- 3Dスレッドグリッド設定: `(batch, seq, dim)`
- バッファサイズ: `batch_size * seq_len * hidden_dim * sizeof(f32)`
- カーネル名: `"rms_norm_f32"`

#### 1.2 RoPE Batch Wrapper

```rust
pub fn metal_rope_batch_f32(
    x: &mut [f32],          // [batch_size, seq_len, num_heads, head_dim] (in-place)
    batch_size: usize,
    start_pos: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) -> RusTorchResult<()>
```

**実装ポイント**:
- 4Dスレッドグリッド: `(batch, pos, head, dim_pair)`
- dim_pairは `head_dim / 2`（回転ペア数）
- カーネル名: `"apply_rope_f32"`

#### 1.3 Attention Score Batch Wrapper

```rust
pub fn metal_attention_scores_batch_f32(
    q: &[f32],              // [batch_size, q_len, num_heads, head_dim]
    k: &[f32],              // [batch_size, kv_len, num_heads, head_dim]
    scores: &mut [f32],     // [batch_size, num_heads, q_len, kv_len]
    batch_size: usize,
    q_len: usize,
    kv_len: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,             // 1.0 / sqrt(head_dim)
) -> RusTorchResult<()>
```

**実装ポイント**:
- 4Dスレッドグリッド: `(batch, q_pos, kv_pos, head)`
- 大規模グリッド対策: スレッドグループサイズ調整
- カーネル名: `"compute_attention_scores_batch_f32"`

#### 1.4 Softmax Batch Wrappers

3段階パイプライン:

```rust
// Step 1: Find max
pub fn metal_softmax_max_batch_f32(...) -> RusTorchResult<()>

// Step 2: Compute exp and sum (既存関数使用可能)
pub fn metal_softmax_exp_sum_f32(...) -> RusTorchResult<()>

// Step 3: Normalize (既存関数使用可能)
pub fn metal_softmax_normalize_f32(...) -> RusTorchResult<()>
```

#### 1.5 Apply Attention to Values Batch Wrapper

```rust
pub fn metal_apply_attention_batch_f32(
    scores: &[f32],         // [batch_size, num_heads, q_len, kv_len]
    v: &[f32],              // [batch_size, kv_len, num_heads, head_dim]
    output: &mut [f32],     // [batch_size, q_len, num_heads, head_dim]
    batch_size: usize,
    q_len: usize,
    kv_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> RusTorchResult<()>
```

### Task 2: process_layer_batch実装 ⏳

**ファイル**: `src/models/llama.rs`

```rust
impl LlamaModel {
    fn process_layer_batch(
        &mut self,
        hidden_states: Vec<f32>,  // [batch_size, seq_len, hidden_dim]
        layer_idx: usize,
    ) -> RusTorchResult<Vec<f32>> {
        let batch_size = self.config.batch_size;
        let seq_len = hidden_states.len() / (batch_size * self.config.hidden_size);

        // 1. RMS Norm (Pre-attention)
        let mut normed = vec![0.0f32; hidden_states.len()];
        metal_rms_norm_batch_f32(
            &hidden_states,
            &self.get_weight("attn_norm", layer_idx),
            &mut normed,
            batch_size,
            seq_len,
            self.config.hidden_size,
            self.config.norm_eps as f32,
        )?;

        // 2. Q/K/V Projections (Batch Matmul)
        let q = self.batch_matmul(&normed, &self.get_weight("attn_q", layer_idx), ...)?;
        let k = self.batch_matmul(&normed, &self.get_weight("attn_k", layer_idx), ...)?;
        let v = self.batch_matmul(&normed, &self.get_weight("attn_v", layer_idx), ...)?;

        // 3. RoPE Application
        metal_rope_batch_f32(&mut q, batch_size, start_pos, seq_len, ...)?;
        metal_rope_batch_f32(&mut k, batch_size, start_pos, seq_len, ...)?;

        // 4. Attention Computation
        let mut scores = vec![0.0f32; batch_size * num_heads * seq_len * total_seq_len];
        metal_attention_scores_batch_f32(&q, &k, &mut scores, ...)?;

        // 5. Softmax
        metal_softmax_batch(&mut scores, ...)?;

        // 6. Apply to Values
        let mut attn_output = vec![0.0f32; batch_size * seq_len * hidden_dim];
        metal_apply_attention_batch_f32(&scores, &v, &mut attn_output, ...)?;

        // 7. Output Projection
        let attn_out = self.batch_matmul(&attn_output, &self.get_weight("attn_out", layer_idx), ...)?;

        // 8. Residual Connection
        let hidden_after_attn = element_wise_add(&hidden_states, &attn_out)?;

        // 9. RMS Norm (Pre-FFN)
        let mut normed_ffn = vec![0.0f32; hidden_after_attn.len()];
        metal_rms_norm_batch_f32(&hidden_after_attn, ..., &mut normed_ffn, ...)?;

        // 10. FFN (Gate, Up, Down projections with SwiGLU)
        let gate = self.batch_matmul(&normed_ffn, &self.get_weight("ffn_gate", layer_idx), ...)?;
        let up = self.batch_matmul(&normed_ffn, &self.get_weight("ffn_up", layer_idx), ...)?;
        let gate_up = swiglu_activation(&gate, &up)?;
        let ffn_out = self.batch_matmul(&gate_up, &self.get_weight("ffn_down", layer_idx), ...)?;

        // 11. Final Residual Connection
        Ok(element_wise_add(&hidden_after_attn, &ffn_out)?)
    }
}
```

### Task 3: forward_batch_metal最適化 ⏳

**現在の実装**（順次処理）:
```rust
fn forward_batch_metal(&mut self, input_ids_batch: &[&[usize]], ...) -> RusTorchResult<Vec<Tensor<f64>>> {
    // 各シーケンスを個別処理
    for input_ids in input_ids_batch {
        results.push(self.forward_metal(input_ids, ...)?);
    }
    Ok(results)
}
```

**最適化後**（並列処理）:
```rust
fn forward_batch_metal(&mut self, input_ids_batch: &[&[usize]], ...) -> RusTorchResult<Vec<Tensor<f64>>> {
    let batch_size = input_ids_batch.len();

    // バリデーション
    if batch_size > self.config.batch_size {
        return Err(...);
    }

    // 1. 入力の結合とembedding lookup
    let batch_embeddings = self.combine_and_embed_batch(input_ids_batch)?;
    // batch_embeddings: [batch_size, max_seq_len, hidden_dim]

    // 2. 全レイヤーをバッチで処理
    let mut hidden_states = batch_embeddings;
    for layer_idx in 0..self.config.num_layers {
        hidden_states = self.process_layer_batch(hidden_states, layer_idx)?;
    }

    // 3. Final RMS Norm
    let mut normed_output = vec![0.0f32; hidden_states.len()];
    metal_rms_norm_batch_f32(&hidden_states, ..., &mut normed_output, ...)?;

    // 4. Output projection (LM head)
    let logits = self.batch_matmul(&normed_output, &self.get_weight("output"), ...)?;

    // 5. 出力を個別テンソルに分割
    self.split_batch_output(logits, input_ids_batch)
}
```

### Task 4: ヘルパー関数実装 ⏳

#### 4.1 combine_and_embed_batch

```rust
fn combine_and_embed_batch(&self, input_ids_batch: &[&[usize]]) -> RusTorchResult<Vec<f32>> {
    let batch_size = input_ids_batch.len();
    let max_seq_len = input_ids_batch.iter().map(|ids| ids.len()).max().unwrap_or(0);
    let hidden_dim = self.config.hidden_size;

    let mut embeddings = vec![0.0f32; batch_size * max_seq_len * hidden_dim];

    for (batch_idx, input_ids) in input_ids_batch.iter().enumerate() {
        let seq_len = input_ids.len();

        for (pos, &token_id) in input_ids.iter().enumerate() {
            let emb_offset = (batch_idx * max_seq_len + pos) * hidden_dim;
            let token_offset = token_id * hidden_dim;

            embeddings[emb_offset..emb_offset + hidden_dim]
                .copy_from_slice(&self.weights["token_embd"][token_offset..token_offset + hidden_dim]);
        }

        // Padding for shorter sequences (zero-filled)
        // パディング部分は既にゼロ初期化済み
    }

    Ok(embeddings)
}
```

#### 4.2 split_batch_output

```rust
fn split_batch_output(
    &self,
    batch_logits: Vec<f32>,     // [batch_size, seq_len, vocab_size]
    input_ids_batch: &[&[usize]],
) -> RusTorchResult<Vec<Tensor<f64>>> {
    let batch_size = input_ids_batch.len();
    let vocab_size = self.config.vocab_size;
    let mut outputs = Vec::with_capacity(batch_size);

    for (batch_idx, input_ids) in input_ids_batch.iter().enumerate() {
        let seq_len = input_ids.len();

        // Extract logits for this sequence (last token only for generation)
        let last_token_offset = (batch_idx * max_seq_len + seq_len - 1) * vocab_size;
        let logits_f64: Vec<f64> = batch_logits[last_token_offset..last_token_offset + vocab_size]
            .iter()
            .map(|&x| x as f64)
            .collect();

        outputs.push(Tensor::from_vec(logits_f64, vec![vocab_size]));
    }

    Ok(outputs)
}
```

#### 4.3 batch_matmul

```rust
fn batch_matmul(
    &self,
    input: &[f32],          // [batch_size, seq_len, in_dim]
    weight: &[f32],         // [out_dim, in_dim]
    batch_size: usize,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> RusTorchResult<Vec<f32>> {
    let mut output = vec![0.0f32; batch_size * seq_len * out_dim];

    // バッチ内の各シーケンス位置に対してmatmul実行
    for batch_idx in 0..batch_size {
        for seq_idx in 0..seq_len {
            let input_offset = (batch_idx * seq_len + seq_idx) * in_dim;
            let output_offset = (batch_idx * seq_len + seq_idx) * out_dim;

            // 既存のmatmul関数を使用（または専用バッチmatmulカーネル）
            metal_matmul_transposed_f32(
                &input[input_offset..input_offset + in_dim],
                weight,
                &mut output[output_offset..output_offset + out_dim],
                1,  // m
                out_dim,  // n
                in_dim,  // k
            )?;
        }
    }

    Ok(output)
}
```

## 実装順序

1. **Week 1**: RMS Norm + RoPEラッパー実装
2. **Week 2**: Attentionラッパー実装（scores + softmax + values）
3. **Week 3**: ヘルパー関数実装（combine, split, batch_matmul）
4. **Week 4**: process_layer_batch実装
5. **Week 5**: forward_batch_metal最適化
6. **Week 6**: 統合テスト・デバッグ

## テスト戦略

### Unit Tests
各カーネルラッパーに対して：
```rust
#[test]
#[cfg(feature = "metal")]
fn test_rms_norm_batch() {
    let batch_size = 2;
    let seq_len = 4;
    let hidden_dim = 8;

    // Test input
    let input = vec![1.0f32; batch_size * seq_len * hidden_dim];
    let weight = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; batch_size * seq_len * hidden_dim];

    // Execute
    let result = metal_rms_norm_batch_f32(
        &input, &weight, &mut output,
        batch_size, seq_len, hidden_dim, 1e-5,
    );

    assert!(result.is_ok());
    // Verify output values
}
```

### Integration Tests
```rust
#[test]
#[cfg(feature = "metal")]
fn test_batch_inference_integration() {
    let model_path = "...";
    let mut model = LlamaModel::from_gguf_with_backend(model_path, DeviceType::Metal)?;

    // Test with batch_size=2
    let input_batch = vec![
        &[1_usize][..],
        &[1_usize, 2][..],
    ];

    let outputs = model.forward_batch(&input_batch)?;
    assert_eq!(outputs.len(), 2);

    // Verify output shapes and values
}
```

## パフォーマンス最適化

### メモリ最適化
- KVCache pre-allocation
- Buffer reuse across layers
- Zero-copy operations where possible

### 計算最適化
- Kernel fusion opportunities（RMS Norm + Add）
- Optimized thread group sizes
- Memory coalescing patterns

### プロファイリング
```bash
# Metal GPU profiler
instruments -t "Metal System Trace" -w ./target/release/rustorch-cli

# Performance metrics
cargo bench --features metal batch_inference
```

## トラブルシューティング

### よくある問題

1. **Metal buffer size mismatch**
   - 症状: Crash or incorrect results
   - 解決: デバッグ出力でバッファサイズを確認

2. **Thread group size too large**
   - 症状: Kernel launch failure
   - 解決: デバイスの最大スレッド数を確認（通常1024）

3. **Numerical instability**
   - 症状: NaN or Inf in output
   - 解決: Epsilon値を調整、入力範囲を確認

## 関連ドキュメント

- [BATCH_PROCESSING_STATUS.md](BATCH_PROCESSING_STATUS.md) - 全体進捗
- [LEGACY_KERNEL_REMOVAL_PLAN.md](LEGACY_KERNEL_REMOVAL_PLAN.md) - レガシーカーネル削除計画
- [METAL_GPU_DEBUGGING_STATUS.md](../METAL_GPU_DEBUGGING_STATUS.md) - Metal GPUデバッグガイド

---

**Status**: Implementation Guide Complete
**Target Completion**: 6 weeks
**Last Updated**: 2025-10-10
