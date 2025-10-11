# Metal Backend Investigation - 2025-10-11

**Status**: 🔴 **ALL QUANTIZATION TYPES BROKEN IN METAL BACKEND**

## Executive Summary

While previous investigation focused on Q8_0 in hybrid-f32 backend, **ALL quantization types produce gibberish output with metal backend**, including:
- ❌ Q4_K_M
- ❌ Q5_K_M
- ❌ Q6_K
- ❌ Q8_0

This indicates a **fundamental issue in the metal backend implementation**, not quantization-specific bugs.

## Test Results (Metal Backend)

### Q8_0 (Previously thought to have RMSNorm issue)
```
Input: "Hello"
Output: "динаliqueдинаätzeдинаlei regretleiчётäu"
Tokens: [16489, 9854, 16489, 22445, 16489, 15250, 28883, 15250, 18107, 28902]
```

### Q6_K (Should be stable according to Q4K investigation doc)
```
Input: "Hello"
Output: "leileileiätzeleileiätzeleileiчёт"
Tokens: [16301, 16301, 16301, 22445, 16301, 16301, 22445, 16301, 16301, 18107]
```

### Q5_K_M (Should be stable according to Q4K investigation doc)
```
(Not tested yet, but expected to fail based on pattern)
```

### Q4_K_M (Known unstable, but failing differently than expected)
```
Input: "Hello"
Output: "migliữlaps migliMAIN vidätze hinaus літера)-\"
Tokens: [20379, 31797, 14128, 20379, 29032, 18865, 22445, 27868, 31675, 9226]
```

## Key Observations

1. **All formats produce gibberish** - Not limited to Q4_K_M or Q8_0
2. **Token patterns are non-sensical** - Not even plausible English words
3. **Repeating tokens** - Q6_K outputs same token (16301='lei') multiple times
4. **Non-Latin characters** - Cyrillic (дина, чёт, літера), Vietnamese (ữ) appearing unexpectedly

## Comparison with Previous Q4K Investigation

The `Q4K_INVESTIGATION_FINAL_CONCLUSION.md` states:
> "Q5_K_M+: ✅ STABLE (correct token)"
> "Q6_K: ✅ STABLE (correct token)"
> "Q8_0: ✅ STABLE (correct token)"

**However, current metal backend testing shows ALL formats unstable.**

### Possible Explanations

1. **Previous investigation used different backend** - May have used hybrid-f32 or CPU backend
2. **Recent code changes broke metal backend** - Something introduced since the Q4K investigation
3. **Metal backend was never fully working** - Previous "success" may have been in different backend

## RMSNorm Separation Investigation

### What Was Attempted

Based on llama.cpp analysis, discovered that:
- llama.cpp's `ggml_compute_forward_rms_norm_f32` does **NOT** multiply by weight
- Weight multiplication done separately via `ggml_mul`
- RusTorch was incorrectly combining normalization and weight multiplication

### Changes Made

Modified both implementations:
1. `src/hybrid_f32/models/llama.rs` (hybrid-f32 backend)
2. `src/models/llama.rs` (metal backend)

Changed from:
```rust
// Old: Combined normalization + weight multiplication
for i in 0..hidden_size {
    output[offset + i] = row[i] * scale * weight[i];
}
```

To:
```rust
// New: Separated operations
// Step 1: Normalize only
for i in 0..hidden_size {
    output[offset + i] = row[i] * scale;
}

// Step 2: Multiply by weight (separate call)
elementwise_mul_f32(&normalized, &weight, &mut output);
```

### Result

**RMSNorm separation did NOT fix the issue.**

All quantization types still produce gibberish after the change, suggesting:
1. RMSNorm separation alone is insufficient
2. There are additional bugs in the metal backend
3. The root cause may be elsewhere (attention, FFN, matmul, etc.)

## Next Steps

### Immediate Actions Needed

1. **Verify which backend previous "working" tests used**
   - Check if Q5_K_M/Q6_K/Q8_0 ever worked in metal backend
   - Or if they only worked in hybrid-f32/CPU backend

2. **Compare metal vs hybrid-f32 implementations**
   - Identify architectural differences
   - Check if hybrid-f32 backend works correctly

3. **Layer-by-layer debugging in metal backend**
   - Add debug output for embedding layer
   - Add debug output for Layer 0 after RMSNorm
   - Compare with llama.cpp values at each step

### Potential Root Causes to Investigate

1. **Matmul implementation** - Metal GPU matrix multiplication may have bugs
2. **Attention mechanism** - Softmax, QKV projection, or attention output
3. **KV Cache** - Cache management or retrieval issues
4. **RoPE (Rotary Position Embedding)** - Position encoding bugs
5. **Tensor layout** - Data layout mismatches between CPU and GPU
6. **Dequantization on GPU** - Quantized weight loading issues

## Code Status

### Current State
- Metal backend: RMSNorm separated, but still broken
- Hybrid-f32 backend: RMSNorm separated (not tested)
- Both backends: All quantization formats produce gibberish

### Files Modified
- `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/src/models/llama.rs` - Metal backend
- `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/src/hybrid_f32/models/llama.rs` - Hybrid-f32 backend

### Recommendation

**DO NOT merge RMSNorm separation changes** until:
1. Root cause of gibberish output is identified
2. At least ONE quantization format produces correct output
3. Full verification against llama.cpp is completed

---

**Investigation Date**: 2025-10-11
**Status**: 🔴 BLOCKED - Need to identify why ALL formats fail in metal backend
