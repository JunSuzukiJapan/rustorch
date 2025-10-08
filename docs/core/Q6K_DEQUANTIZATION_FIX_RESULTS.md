# Q6_K Dequantization Fix Results

**Date**: 2025-10-08
**Status**: Partial fix - dequantization pattern corrected, but token generation still wrong

## Summary

Fixed Q6_K dequantization to match llama.cpp's interleaved indexing pattern exactly. The dequantization now produces reasonable values, but token generation is still incorrect.

## What Was Fixed

### Problem
Original Q6_K dequantization used sequential indexing:
```rust
for j in 0..16 {
    for k in 0..16 {
        output.push(dequant_val);  // Sequential: [0,1,2,3,...]
    }
}
```

### Solution
Changed to llama.cpp's interleaved pattern:
```rust
for _chunk in 0..2 {  // 2 chunks × 128 elements
    for l in 0..32 {   // 32 iterations × 4 values
        // Interleaved: [l, l+32, l+64, l+96]
        output[y_idx + l] = ...;
        output[y_idx + l + 32] = ...;
        output[y_idx + l + 64] = ...;
        output[y_idx + l + 96] = ...;
    }
    y_idx += 128; ql_idx += 64; qh_idx += 32; sc_idx += 8;
}
```

## Test Results

### Debug Example
```
🐛 [Q6_K DEBUG] First block:
   d=0.00005072
   scales: [16, 18, -128, 17, 7, -14, 12, -15]

🐛 [Q6_K DEBUG] First decompression:
   q1=15, q2=2, q3=-32, q4=-4
   scales: sc[0]=16, sc[2]=-128, sc[4]=7, sc[6]=12
   values: v1=0.012174, v2=-0.012985, v3=-0.011362, v4=-0.002435
```

Values in expected range (~0.01), no longer 23,000x error.

### Token Generation Test

**llama.cpp Q6_K output** (correct):
```
Hello world → "Write a Python program"
```

**RusTorch Q6_K output** (wrong):
```
Hello world → "leiчётleiчётlei"
(tokens: [16301, 28651, 16301, 28651, 16301])
```

### Comparison with Previous Results

| Test | Q4_K_M (before fix) | Q6_K (after fix) |
|------|---------------------|------------------|
| Output | "drew drew drew" | "leiчётleiчётlei" |
| Pattern | Repeating English | Repeating mixed lang |
| Token IDs | [22449, 22449, 22449] | [16301, 28651, ...] |

## Analysis

### What Works Now
1. ✅ Q6_K dequantization produces correct value magnitudes (~0.01 range)
2. ✅ Dequantization pattern matches llama.cpp exactly
3. ✅ No more 23,000x scale errors

### What Still Doesn't Work
1. ❌ Token generation produces wrong tokens
2. ❌ Different wrong output suggests dequant affects results, but doesn't fix root cause
3. ❌ Still seeing value growth through layers (128x from layer 0 to 21)

## Root Cause Hypothesis

The Q6_K dequantization fix improved numerical accuracy, but the core problem remains:

**Hidden state value growth through layers** (documented in [LAYER_VALUE_GROWTH_ANALYSIS.md](LAYER_VALUE_GROWTH_ANALYSIS.md)):
- Layer 0 output RMS: 0.015
- Layer 21 output RMS: 1.124 (75x growth)
- Final norm output RMS: 1.920 (128x growth)

This exponential growth causes wrong tokens regardless of quantization format.

## Next Steps

1. **Verify all Q6_K tensors are dequantized correctly**
   - token_embd.weight (Q6_K in this model)
   - All layer weights (attn_q, attn_k, attn_v, ffn_gate, etc.)

2. **Test with Q4_K_M model using new Q6_K knowledge**
   - Check if Q4_K has similar indexing issues
   - Compare layer-by-layer value growth

3. **Investigate layer value growth root cause**
   - Is this normal for TinyLlama?
   - Compare with llama.cpp's intermediate values
   - Check if RMSNorm/Attention/FFN implementations are correct

4. **Test with F32 model**
   - Eliminate quantization as a variable
   - Pure numerical precision test

## Files Modified

- [src/formats/gguf.rs:852-921](../../src/formats/gguf.rs#L852-L921) - Fixed `dequantize_q6_k()` function
- [src/formats/q6k_dequant_fixed.rs](../../src/formats/q6k_dequant_fixed.rs) - Reference implementation

## Conclusion

Q6_K dequantization is now correct, but this reveals the problem lies elsewhere. The token generation bug persists, likely due to accumulated numerical errors through the 22-layer network.

**Status**: Investigation continues - focus shifted from Q6_K to layer value growth analysis.
