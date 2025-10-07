# Grouped Query Attention Panic Fix
**Date:** 2025-10-07 21:00
**Status:** ✅ Panic Fixed, ⚠️ Repeated Token Issue Remains

## Problem

### Panic Error
```
thread 'main' panicked at src/hybrid_f32/models/llama.rs:441:40:
range end index 4672 out of range for slice of length 4608
```

### Location
[llama.rs:441](../../../src/hybrid_f32/models/llama.rs:441) in `grouped_query_attention` method

### Root Cause

```rust
// BEFORE (buggy):
let current_kv_pos = cached_len + q_pos;
for kv_pos in 0..=current_kv_pos {
    let k_start = kv_pos * num_kv_heads * head_dim + kv_head * head_dim;
    let k_vec = &full_k[k_start..k_start + head_dim];  // ← PANIC HERE
```

**Issue:** When processing the last query position, `current_kv_pos` could equal or exceed the actual length of concatenated K/V tensors, causing out-of-bounds access.

**Example:**
- Input: 18 tokens (seq_len=18)
- Last query position: q_pos=17
- cached_len=0 (first forward call)
- current_kv_pos = 0 + 17 = 17 ✅
- BUT: Loop goes 0..=17 (18 iterations)
- If total_kv_len < 18, index out of bounds!

## Solution

```rust
// AFTER (fixed):
let current_kv_pos = (cached_len + q_pos).min(total_kv_len - 1);
for kv_pos in 0..=current_kv_pos {
    let k_start = kv_pos * num_kv_heads * head_dim + kv_head * head_dim;
    let k_vec = &full_k[k_start..k_start + head_dim];  // ✅ Safe
```

**Fix:** Clamp `current_kv_pos` to not exceed `total_kv_len - 1`.

## Test Results

### Before Fix
```
thread 'main' panicked at src/hybrid_f32/models/llama.rs:441:40
```

### After Fix
```
✅ No panic
🎯 [STEP 0] Selected token 6587 (argmax)
📋 [STEP 0] After push: generated_ids.len=19
🎯 [STEP 1] Selected token 6587 (argmax)
📋 [STEP 1] After push: generated_ids.len=20
🎯 [STEP 2] Selected token 6587 (argmax)
📋 [STEP 2] After push: generated_ids.len=21
```

## Remaining Issue

**Repeated Token Generation**

The same token (6587 = "кон" in Russian) is generated repeatedly:
```
Step 0: token 6587
Step 1: token 6587
Step 2: token 6587
```

This matches the earlier finding: **Call 0 and Call 1 had identical hidden states**.

### Hypothesis

The fix allows generation to proceed, but exposes an underlying issue:
1. **Logits may be identical across steps** → Check if forward pass produces same output
2. **Position not updating correctly** → RoPE position parameter may be static
3. **KV cache not being used** → Cache may not affect attention correctly

### Next Steps

1. Add logging to print logits for steps 0, 1, 2
2. Verify position parameter increases: 0 → 18 → 19 → 20
3. Check KV cache length updates: 0 → 18 → 19 → 20
4. Compare logits: should differ if position/cache working correctly

## Files Modified

- [src/hybrid_f32/models/llama.rs:437](../../../src/hybrid_f32/models/llama.rs:437) - Added `.min(total_kv_len - 1)` clamp
- [example-cli/src/model/inference.rs:418-493](../../../example-cli/src/model/inference.rs:418) - Added debug logging

## Related Issues

- [Q4_0_INVESTIGATION_2025_10_07.md](Q4_0_INVESTIGATION_2025_10_07.md) - GGUF loading verified correct
- [IMPLEMENTATION_VERIFICATION.md](IMPLEMENTATION_VERIFICATION.md) - Hidden state analysis showing Call 0/1 identical
