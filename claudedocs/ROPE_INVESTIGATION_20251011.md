# RoPE Investigation Results

**Date**: 2025-10-11
**Status**: 🎯 RoPE precomputation is CORRECT, but application may have issues

## Executive Summary

**Finding**: RoPE frequency precomputation is mathematically correct and matches llama.cpp implementation exactly. However, debug output shows Q/K values don't change after RoPE application, suggesting a potential issue in the `apply_rope` function or its usage.

## Test Results

### RoPE Precomputation Test

Created standalone test program (`/tmp/test_rope_precompute.rs`) that verifies:

✅ **Position 0** (no rotation):
```
cos=1.000, sin=0.000  (correct - angle = 0 * freq = 0)
```

✅ **Position 1** (rotation applied):
```
i=0: cos=0.540302, sin=0.841471  (correct - angle = 1 * 1.0)
i=1: cos=0.731761, sin=0.681561  (correct - angle = 1 * 0.7499)
i=2: cos=0.846009, sin=0.533168  (correct - angle = 1 * 0.5623)
```

✅ **Position 2** (more rotation):
```
i=0: cos=-0.416147, sin=0.909297  (correct - angle = 2 * 1.0)
i=1: cos=0.070948, sin=0.997480   (correct - angle = 2 * 0.7499)
i=2: cos=0.431463, sin=0.902131   (correct - angle = 2 * 0.5623)
```

### Formula Verification

**RusTorch implementation**:
```rust
let freq = 1.0 / theta.powf(2.0 * (i as f32) / (head_dim as f32));
let angle = (pos as f32) * freq;
let cos_val = angle.cos();
let sin_val = angle.sin();
```

**llama.cpp implementation** (from `ggml_rope_ext`):
```c
freq = 1.0 / pow(theta, 2*i/n_dims);
angle = pos * freq;
cos(angle), sin(angle)
```

✅ **Implementations match exactly**

### Index Calculation

**RusTorch**:
```rust
let rope_idx = position * (head_dim / 2) + i;
```

**Expected layout**:
- Position 0, i=0: index=0
- Position 1, i=0: index=32 (for head_dim=64)
- Position 1, i=1: index=33

✅ **Index calculation is correct**

## Debug Output Analysis

### From RusTorch CLI

**RoPE PRECOMPUTE** (at model initialization):
```
head_dim=64, max_seq_len=2048, theta=10000
Index 0-9:   cos=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Index 32-41: cos=[0.5403023, 0.731761, 0.84600914, ...]
```

✅ Precomputed values are correct

**RoPE CALLED** (during forward pass):
```
🌀 [RoPE CALLED] seq_len=14, total_dim=2048, head_dim=64, num_heads=32, start_position=0
🌀 [RoPE INPUT] First 10 values: [-0.00330, 0.03684, -0.05323, ...]
🌀 [RoPE FREQS] rope_cos.len=65536, rope_sin.len=65536
🌀 [RoPE FREQS] First 10 cos values: [1.0, 1.0, 1.0, 1.0, 1.0, ...]
🌀 [RoPE FREQS] First 10 sin values: [0.0, 0.0, 0.0, 0.0, 0.0, ...]
```

**Note**: First 10 values are all cos=1, sin=0 because they correspond to position=0. This is **correct**.

**RoPE DETAIL** (for specific tokens):
```
token=0, head=0, pair=0, pos=0, rope_idx=0
  cos=1.000000000, sin=0.000000000
  input:  x0=-0.003299128, x1=0.036841877

token=1, head=0, pair=0, pos=1, rope_idx=32
  (cos/sin values not shown in output)
```

### Problem Observed

**Q before RoPE**:
```
rms=0.088615, first_10=[-0.00330, 0.03684, -0.05323, ...]
```

**Q after RoPE**:
```
rms=0.088615, first_10=[-0.00330, 0.03684, -0.05323, ...]  ← SAME!
```

⚠️ **Issue**: Q values appear unchanged after RoPE application

## Possible Explanations

### 1. Debug Output Shows Token 0 Only ⭐⭐⭐⭐

**Hypothesis**: The "first_10" debug output only shows the first token (position=0), which correctly has no rotation (cos=1, sin=0).

**Evidence**:
- Position 0 should not change (cos=1, sin=0 means identity transformation)
- Other tokens (positions 1-13) should have rotation applied
- Debug output may only be printing first token's values

**Test needed**: Check if Q/K values for tokens at position >0 are actually being rotated.

### 2. Output Tensor Construction Issue ⭐⭐

**Hypothesis**: The `output` vector is constructed correctly but not returned properly.

**Evidence**:
```rust
let mut output = Vec::with_capacity(x_data.len());
// ... fill output ...
F32Tensor::from_vec(output, shape)  // Returns new tensor
```

This looks correct, but worth verifying the tensor is actually used.

### 3. apply_rope Not Being Used ⭐

**Hypothesis**: The result of `apply_rope` is computed but not actually used in attention.

**Evidence needed**: Check call site at line 739-740:
```rust
let q_rope = self.apply_rope(&q, position)?;
let k_rope = self.apply_rope(&k, position)?;
```

Are `q_rope` and `k_rope` actually passed to GQA?

## Next Steps

### 1. Verify RoPE Result Usage ⭐⭐⭐⭐⭐ (CRITICAL)

Check that `q_rope` and `k_rope` are actually used in the attention calculation:

```rust
// In attention_layer function
let q_rope = self.apply_rope(&q, position)?;
let k_rope = self.apply_rope(&k, position)?;

// Are these passed to grouped_query_attention?
let (output, new_k_cache, new_v_cache) = self.grouped_query_attention(
    &q_rope,  // ← Should be q_rope, not q!
    &k_rope,  // ← Should be k_rope, not k!
    &v,
    cached_k,
    cached_v,
)?;
```

**Action**: Inspect lines 739-755 to verify correct tensor usage.

### 2. Add Debug Output for All Tokens

Modify debug output to show values for multiple tokens, not just first:

```rust
eprintln!("🌀 [RoPE OUTPUT] Token 0 first 10: {:?}", &output[0..10]);
eprintln!("🌀 [RoPE OUTPUT] Token 1 first 10: {:?}", &output[2048..2058]);
eprintln!("🌀 [RoPE OUTPUT] Token 13 first 10: {:?}", &output[13*2048..13*2048+10]);
```

### 3. Compare Single Token Processing

Test with a single input token to simplify:
- Input: just BOS token (ID=1)
- Expected: No rotation (position=0, cos=1, sin=0)
- Verify: Output should equal input

Then test with 2 tokens:
- Token 0: position=0, no rotation
- Token 1: position=1, rotation applied
- Verify: Token 1 output should differ from input

## Conclusion

**RoPE precomputation is mathematically correct** and matches llama.cpp implementation exactly.

**Potential issue**: Either:
1. `apply_rope` result is not being used in attention (most likely)
2. Debug output only shows token 0, masking the fact that other tokens are being rotated correctly
3. Tensor construction/return has a subtle bug

**Immediate action**: Verify that `q_rope` and `k_rope` are actually passed to `grouped_query_attention` and not the original `q` and `k`.

## Files

- Test program: `/tmp/test_rope_precompute.rs`
- RoPE implementation: `llama.rs:411-474` (`apply_rope` function)
- RoPE usage: `llama.rs:739-740` (in `attention_layer`)
- Precomputation: `llama.rs:138-173` (`precompute_rope_frequencies`)
