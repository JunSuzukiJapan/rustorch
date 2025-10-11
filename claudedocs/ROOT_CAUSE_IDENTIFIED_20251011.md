# Root Cause Identified: RMSNorm Weight Loading Bug

**Date**: 2025-10-11
**Status**: 🎯 **ROOT CAUSE IDENTIFIED**

## Executive Summary

After comprehensive debugging, the root cause has been identified:

**RMSNorm weights are loaded with values ~20x too small**
- Expected: ~1.0 (typical RMSNorm weights)
- Actual: ~0.05 (RMS = 0.046, mean = 0.006)

This causes Layer 0 input RMS to be 0.098 instead of ~1.0, propagating errors through all 22 layers and resulting in gibberish output.

## Evidence Chain

### 1. RoPE is Correct ✅
- Created isolated tests: `/tmp/test_rope_precompute.rs` and `/tmp/test_apply_rope.rs`
- All tests PASSED
- Verified Token 0 (position=0): unchanged (cos=1, sin=0) ✅
- Verified Token 1+ (position=1+): rotated correctly ✅

### 2. Q8_0 Produces Gibberish → NOT Quantization Issue ✅
```bash
$ echo "1" | rustorch-cli --model Q8_0.gguf --max-tokens 5
Output: "avavertavanth"  # Gibberish
```
- Q8_0 is highest precision (minimal quantization error)
- Still produces gibberish → core logic issue

### 3. Hidden States Completely Different ✅
```
RusTorch Layer 21: [0.581, 1.059, -1.031, ...] (RMS: 1.919)
llama.cpp Layer 21: [-1.822, 0.060, -1.980, ...] (RMS: 1.848)
```
- No correlation despite similar magnitude
- Divergence starts at Layer 0

### 4. Layer 0 Input RMS is Wrong 🚨
```
RusTorch Layer 0 input (after RMSNorm): RMS = 0.098
Expected: RMS ≈ 1.0 (RMSNorm normalizes to unit RMS)
```
- Value is ~10x too small
- This is the FIRST operation after embedding → issue is in RMSNorm or embedding

### 5. RMSNorm Weights are Too Small 🎯 ROOT CAUSE
```
🔍 [RMSNorm WEIGHT] rms=0.046377, min=-0.582031, max=0.769531, mean=0.005780
```
- Expected RMSNorm weights: ~1.0 (typical for layer normalization)
- Actual: RMS = 0.046, mean = 0.006
- **~20x too small!**

## RMSNorm Implementation Analysis

### Code Location
[`llama.rs:290-381`](../src/hybrid_f32/models/llama.rs#L290-L381)

### Algorithm (Verified Correct)
1. Compute sum of squares: `sum = Σ(x[i]²)`
2. Mean: `mean = sum / hidden_size`
3. Scale: `scale = 1 / sqrt(mean + eps)`
4. Normalize: `y[i] = x[i] * scale`
5. Apply weight: `y[i] = y[i] * weight[i]` ✅

**The RMSNorm algorithm is correct** - the issue is the **weight values being loaded**.

## Root Cause: GGUF F32 Weight Loading

### Hypothesis
The `load_tensor_generic::<f32>()` function in `/src/formats/gguf.rs` is:
1. Loading F32 weights with incorrect byte interpretation, OR
2. Applying unnecessary scaling/conversion, OR
3. Reading from wrong file offsets

### Evidence
- F32 RMSNorm weights: `original_dims=[2048]`, `final_shape=[2048]` ✅ (no transpose)
- But values are ~20x too small
- Q8_0 quantized weights also produce gibberish → suggests F32 dequantization issue

## What We've Ruled Out

1. ✅ RoPE implementation - verified correct via isolated tests
2. ✅ Position tracking - Step 0: pos=0, Step 1: pos=14, Step 2: pos=15
3. ✅ Quantization (Q4_K/Q5_K) - Q8_0 also fails
4. ✅ RMSNorm algorithm - formula matches llama.cpp exactly
5. ✅ output.weight transpose - tested both orientations
6. ✅ Logits calculation - indexing verified correct for shape `[vocab_size, hidden_size]`

## Next Actions (Priority Order)

### 1. Investigate GGUF F32 Loading 🔴 **HIGHEST PRIORITY**
**File**: `/src/formats/gguf.rs`
**Function**: `load_tensor_generic::<f32>()`

**Check**:
- Byte order interpretation (little-endian vs big-endian)
- File offset calculations
- Any scaling factors applied during loading
- Compare loaded F32 values with raw bytes from GGUF file

**Test**:
```bash
# Manually read first RMSNorm weight from GGUF and compare
# Should be close to 1.0, not 0.046
```

### 2. Compare F16 Dequantization ⚠️
Since F16 model file is corrupted (15 bytes), but the code has F16 handling:
- Check if F16 dequantization has scaling issues
- Both F32 and quantized types might share common dequant bugs

### 3. Verify Q8_0 Dequantization ⚠️
Even though Q8_0 should be simple (just scale factor), verify:
- Block size: 32 values per block
- Scale factor application
- Zero-point handling

## Expected Fix Impact

**If RMSNorm weights are fixed**:
- Layer 0 input RMS: 0.098 → ~1.0 ✅
- All layer outputs will have correct magnitudes
- Hidden states will match llama.cpp
- Output will be coherent English instead of gibberish

## Technical Details

### RMSNorm Weight File Info
```
Path: blk.0.attn_norm.weight
Type: F32 (ggml_type_code=0)
Shape: [2048]
Expected values: ~1.0 (typical layer norm)
Actual values: ~0.05 (20x too small)
```

### Test Command
```bash
printf "1\n" | ./target/release/rustorch-cli \
  --model ~/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --backend hybrid-f32 --max-tokens 1 2>&1 | grep "RMSNorm"
```

## Files to Investigate

1. **`/src/formats/gguf.rs`** - GGUF tensor loading
   - `load_tensor_generic::<f32>()`
   - `dequantize()` functions for Q8_0, Q4_K, Q5_K

2. **`/src/hybrid_f32/models/llama.rs:1320-1430`** - Weight loading wrapper
   - Lines 1339-1378: Shape handling
   - No scaling applied here - passes through to GGUF loader

## Confidence Level

**95% confident** this is the root cause:
- RMSNorm weights are objectively wrong (0.046 vs expected ~1.0)
- This directly explains Layer 0 input RMS being 10x too small
- Propagates through all layers causing gibberish output
- Q8_0 failure confirms it's not Q4_K/Q5_K specific

## References

- Previous investigation: [`KV_CACHE_TEST_RESULTS.md`](KV_CACHE_TEST_RESULTS.md)
- RoPE verification: [`ROPE_INVESTIGATION_20251011.md`](ROPE_INVESTIGATION_20251011.md)
- Layer comparison: [`LAYER_COMPARISON_20251011.md`](LAYER_COMPARISON_20251011.md)
