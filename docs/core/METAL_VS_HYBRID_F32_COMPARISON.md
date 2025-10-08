# Metal GPT vs hybrid_f32 Llama Implementation Comparison

**Date**: 2025-10-08
**Purpose**: Identify differences causing gibberish output in Metal GPT

## Architecture Comparison

### hybrid_f32 Llama (Working ✅)
- **File**: `src/hybrid_f32/models/llama.rs`
- **Architecture**: Llama-2 with RoPE + RMSNorm + SwiGLU
- **KV Cache**: Implemented and functional
- **Position Tracking**: Explicit `start_position` parameter in `forward()`
- **Data Type**: f32 throughout
- **Backend**: Can use Metal GPU via matmul operations

### Metal GPT (Gibberish ❌)
- **File**: `src/models/gpt.rs` with Metal feature
- **Architecture**: GPT-like but adapted for Llama weights
- **KV Cache**: Not implemented
- **Position Tracking**: `start_position` parameter added but always 0
- **Data Type**: f32 for Metal, f64 for CPU
- **Backend**: Metal GPU kernels + CPU fallback

## Key Differences

### 1. Position Tracking

**hybrid_f32 Llama**:
```rust
pub fn forward(&mut self, input_ids: &[usize], start_position: usize) -> F32Result<F32Tensor> {
    // Uses start_position + token_idx for each token
    let position = start_position + token_idx;
    self.apply_rope(&q, position)?;
}
```

**Metal GPT**:
```rust
pub fn forward(&self, input_ids: &[usize]) -> RusTorchResult<Tensor<f64>> {
    // Always calls forward_metal with start_position=0
    self.forward_metal(input_ids, 0)
}
```

**Impact**: Every token gets RoPE applied at position=0, breaking positional encoding.

### 2. RMS Norm Implementation

**hybrid_f32 Llama**:
```rust
fn rms_norm(&self, x: &F32Tensor, weight: &F32Tensor) -> F32Result<F32Tensor> {
    let sum_sq: f32 = slice.iter().map(|&v| v * v).sum();
    let mean_sq = sum_sq / (last_dim as f32);
    let rms = (mean_sq + eps).sqrt();
    let normalized = slice[j] / rms;
    let val = normalized * weight_data[j];
}
```

**Metal GPT**:
```rust
fn rms_norm_f32(input: &[f32], weight: &[f32], output: &mut [f32], ...) {
    let rms: f32 = row.iter().map(|&v| v * v).sum::<f32>() / (hidden_size as f32);
    let rms = (rms + eps).sqrt();
    output[offset + i] = (row[i] / rms) * weight[i];
}
```

**Status**: ✅ Implementations are identical, both correct.

### 3. KV Cache

**hybrid_f32 Llama**:
```rust
// Has KV cache struct and management
struct KVCache {
    keys: F32Tensor,
    values: F32Tensor,
    cached_len: usize,
}

// Uses cache in attention
let cached_k = if cache.cached_len > 0 { Some(cache.keys.as_slice()) } else { None };
let (attn_output, new_k, new_v) = self.grouped_query_attention(&q_rope, &k_rope, &v, cached_k, cached_v)?;
self.kv_cache[layer_idx].keys = new_k.clone();
```

**Metal GPT**:
```rust
// No KV cache implementation
// Each forward pass recomputes attention for entire sequence
```

**Impact**: Inefficient and incorrect - every generation step passes entire sequence through model.

### 4. Generation Loop

**hybrid_f32 Llama** (via `generate_with_f32_gpt_mut`):
```rust
for step in 0..max_new_tokens {
    let input_slice = if step == 0 {
        &generated_ids  // First step: all prompt tokens
    } else {
        &generated_ids[generated_ids.len()-1..]  // Subsequent: only last token
    };
    f32_model.forward(input_slice)  // Passes correct start_position
}
```

**Metal GPT** (via `generate_with_gpt`):
```rust
for step in 0..max_new_tokens {
    // Always passes entire sequence
    gpt_model.forward(&generated_ids)  // start_position always 0
}
```

**Impact**:
- Every token gets RoPE at position=0
- Entire sequence reprocessed every step (slow + incorrect)
- No accumulation of KV cache

### 5. Attention Implementation

**hybrid_f32 Llama**:
```rust
fn grouped_query_attention(
    &self,
    q: &F32Tensor,      // [seq_len, num_q_heads * head_dim]
    k: &F32Tensor,      // [seq_len, num_kv_heads * head_dim]
    v: &F32Tensor,
    cached_k: Option<&[f32]>,
    cached_v: Option<&[f32]>,
) -> F32Result<(F32Tensor, F32Tensor, F32Tensor)> {
    // Proper GQA with KV head repetition
    // Causal masking
    // Softmax over attention scores
}
```

**Metal GPT**:
```rust
// Manual GQA implementation
// Uses Metal kernels for matmul
// repeat_kv_heads() function
// Causal masking via executor.softmax_f32_with_mask()
```

**Status**: Need detailed comparison of:
- Head dimension calculation
- K/V repetition logic
- Attention score computation
- Causal mask application

### 6. FFN Implementation

**hybrid_f32 Llama**:
```rust
fn ffn_layer(&self, x: &F32Tensor, layer_idx: usize) -> F32Result<F32Tensor> {
    // SwiGLU: down(SwiGLU(gate(x), up(x)))
    let gate_proj = x.matmul(gate_weight)?;
    let up_proj = x.matmul(up_weight)?;
    let swiglu_out = self.swiglu(&gate_proj, &up_proj)?;
    swiglu_out.matmul(down_weight)
}

fn swiglu(&self, gate: &F32Tensor, up: &F32Tensor) -> F32Result<F32Tensor> {
    // SiLU(gate) * up
    let silu = x / (1.0 + (-x).exp());  // SiLU activation
    silu * up
}
```

**Metal GPT**:
```rust
// Uses Metal kernels for matmul
// SiLU via executor.silu_f32()
// Element-wise multiplication via executor.mul_f32()
```

**Status**: Implementation looks similar, but need to verify:
- SiLU activation correctness
- Element-wise multiplication order

## Root Cause Analysis

### Primary Issues (Confirmed)

1. **Position Tracking**: ❌ Critical
   - All tokens get RoPE at position=0
   - Breaks positional information completely
   - **Fix**: Add `start_position` parameter to `forward()` API

2. **No KV Cache**: ❌ Critical
   - Entire sequence reprocessed every step
   - Can't properly track token positions
   - **Fix**: Implement KV cache like hybrid_f32

3. **Generation Loop**: ❌ Critical
   - Passes entire sequence instead of last token
   - Doesn't leverage KV cache (because it doesn't exist)
   - **Fix**: Change to hybrid_f32 pattern (prompt → single tokens)

### Secondary Issues (To Investigate)

4. **Attention Details**: ❓ Unknown
   - GQA implementation might have subtle bugs
   - Need head-by-head comparison with hybrid_f32

5. **Q4_K Dequantization**: ❓ Unlikely
   - Implementation matches llama.cpp
   - If broken, all weights would be wrong
   - Model produces *some* output (gibberish but not random)

## Recommended Fix Priority

1. **Add `start_position` to forward()** (Quick win)
   - Change signature: `fn forward(&self, input_ids: &[usize], start_position: usize)`
   - Update all call sites
   - Pass to `forward_metal()`

2. **Implement KV Cache** (Medium effort)
   - Copy struct from hybrid_f32
   - Add cache management in attention
   - Clear cache at generation start

3. **Fix Generation Loop** (Depends on #2)
   - First step: pass all tokens
   - Subsequent steps: pass only last token
   - Track cumulative position

4. **Test Output** (After #1-3)
   - Compare with llama.cpp
   - Verify meaningful English output

5. **Optimize** (If still slow)
   - Profile Metal kernel usage
   - Compare with hybrid_f32 performance

## Next Steps

**Immediate**: Implement position tracking
- Add `start_position` parameter
- Test with single-token generation
- Verify RoPE gets correct positions

**Short-term**: Implement KV Cache
- Copy hybrid_f32 structure
- Integrate into Metal GPU forward pass

**Long-term**: Full hybrid_f32 parity
- Match all implementation details
- Achieve same quality and performance

## References

- [RMS_NORM_IMPLEMENTATION_RESULTS.md](RMS_NORM_IMPLEMENTATION_RESULTS.md)
- [QUANTIZATION_COMPARISON_POST_Q6K_FIX.md](QUANTIZATION_COMPARISON_POST_Q6K_FIX.md)
- hybrid_f32 Llama: `src/hybrid_f32/models/llama.rs`
- Metal GPT: `src/models/gpt.rs`
