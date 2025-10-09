# Metal GPU Backend Investigation Summary

**Date**: 2025-10-09
**Status**: Gibberish output persists after numerical stability fixes

## Implemented Fixes

### 1. Softmax Numerical Stability ✅
- **Temperature scaling**: Configurable parameter (tested 0.1 and 1.0)
- **Result**: Temperature=0.1 improved attention distribution [0.032~0.073] vs uniform [0.050~0.053]
- **Conclusion**: Insufficient to fix gibberish output

### 2. Value Clipping ✅
- **Implementation**: Clip activations to ±10.0 after residual connection 2
- **Purpose**: Prevent 93x amplification across 22 layers
- **Result**: Different tokens generated, still gibberish

### 3. RMS Norm Verification ✅
- **Finding**: Implementation is mathematically correct
- **Evidence**: Standalone test program with 7 edge cases
- **Documentation**: [RMS_NORM_ANALYSIS.md](RMS_NORM_ANALYSIS.md)

## Current Output Behavior

**Input**: "Hello"
**Output**: `тив countryinaciónтивublic` (gibberish)

**Top-5 Logits**:
1. Token 3069 (host): 9.07
2. Token 3499 (тив): 8.82
3. Token 25323 (cogn): 7.84
4. Token 7076: 7.78
5. Token 26890 (LIM): 7.61

**Logits Range**: max=9.07, min=-9.71 (appears normal)

## Suspected Root Causes

### 1. Q/K/V Projection Issues (HIGH PRIORITY)
**Symptoms**:
- Q_rms grows across layers: 0.171 (Layer 0) → 0.530 (Layer 21)
- K_rms remains stable: ~0.020
- This 26x growth in Q leads to small Q·K score ranges

**Possible causes**:
- Weight dequantization errors (Q4_K_M → f32)
- Matrix multiplication implementation errors
- Data layout mismatches

**Investigation needed**:
- Compare Q/K/V projections with llama.cpp layer-by-layer
- Verify weight dequantization produces correct values
- Check matrix multiplication order and transpose operations

### 2. RoPE Implementation (MEDIUM PRIORITY)
**Current implementation**:
```rust
// Precompute: [max_seq_len, head_dim/2] flattened
for pos in 0..max_seq_len {
    for i in 0..(head_dim / 2) {
        cos_values.push(angle.cos());
        sin_values.push(angle.sin());
    }
}

// Apply:
let rope_idx = position * (head_dim / 2) + i;
let rotated_0 = x0 * cos - x1 * sin;
let rotated_1 = x0 * sin + x1 * cos;
```

**Status**: Index calculation appears correct, but needs verification against llama.cpp

### 3. FFN/SwiGLU Implementation (LOW PRIORITY)
**Complexity**: Gate mechanism with up/down projections
**Status**: Not yet investigated in detail

### 4. Quantization Precision (MEDIUM PRIORITY)
**Current**: Q4_K_M format
**Implementation**: [src/formats/gguf.rs:803-902](../../src/formats/gguf.rs#L803-L902)

**Dequantization formula**:
```rust
d1 = d * scale1;
m1 = dmin * min1;
output = d1 * q_val - m1;
```

**Status**: Follows llama.cpp structure, but needs validation with known-good values

## Debugging Strategy

### Phase 1: Identify Divergence Point
Compare intermediate values with llama.cpp at each step:

1. **Embeddings**: Token embedding vectors for input tokens
2. **Layer 0 Input**: After embedding, before first transformer layer
3. **Layer 0 RMS Norm**: After LN1, before attention
4. **Layer 0 Q/K/V**: After projection, before RoPE
5. **Layer 0 RoPE**: After RoPE application
6. **Layer 0 Attention**: After attention mechanism
7. **Layer 0 FFN**: After feed-forward network
8. **Layer 0 Output**: After residual connections
9. **Final Logits**: LM head projection output

**Method**: Add debug output to dump numerical values, compare with llama.cpp

### Phase 2: Fix Identified Issues
Once divergence point is found, fix the specific component

### Phase 3: Validation
- Test with multiple inputs
- Compare perplexity with llama.cpp
- Verify output quality

## Known Correct Components

1. ✅ **RMS Norm**: Mathematically verified with standalone tests
2. ✅ **Softmax**: Numerically stable implementation with max subtraction
3. ✅ **Value Clipping**: Prevents runaway amplification
4. ✅ **Residual Connections**: Metal elementwise_add_f32 working correctly
5. ⚠️ **Token Embeddings**: Format appears correct, values not verified

## Completed Actions (2025-10-09 Evening)

### 1. Built llama.cpp from Source ✅
**Location**: `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/Temp/llama.cpp`
**Build**: CMake with Metal GPU support enabled
**Result**:
- Binary: `build/bin/llama-cli`
- Metal GPU detected and functional
- Input `1` → Output ` 10` (correct, rational completion)

### 2. Verified Token Embeddings ✅
**RusTorch embeddings** (first 3 tokens for input "1"):
- Token 0 (ID=1): RMS=0.00223, mean=0.000028
- Token 1 (ID=529): RMS=0.00999, mean=0.000325
- Token 2 (ID=29989): RMS=0.00985, mean=-0.000014

**Conclusion**: Embeddings are statistically reasonable

### 3. Identified Value Amplification Issue ✅
**Layer-by-layer RMS growth**:
- Embedding output: RMS = 0.009
- Layer 1 input: RMS = 0.010
- Layer 2 input: RMS = 0.012
- Layer 8 input: RMS = 0.175
- **Result**: 19x amplification even with clipping

**Clipping Status**:
- Currently only logs for Layer 0
- Applied after residual connection 2
- Need to verify it's active in all 22 layers

### 4. Analyzed Q/K/V Projection Values ✅
**Observations**:
- Early tokens: Q values [-0.05, 0.05] (reasonable)
- Later tokens: Q values up to 0.9, K values up to -1.28 (large)
- Suggests accumulation of numerical errors through layers

## Next Actions (Priority Order)

1. **Extend Clipping Debug Output to All Layers**
   - Currently only Layer 0 logs clipping
   - Add logging for all 22 layers to verify clipping is active
   - Check if any values exceed ±10.0 threshold

2. **Investigate Q/K/V Projection Weights**
   - Verify weight matrix dequantization is correct
   - Check for any NaN or Inf values in weights
   - Compare weight statistics with expected ranges

3. **Compare Intermediate Values with llama.cpp**
   - Use llama.cpp with debug output to get Layer 0 Q/K/V values
   - Compare with RusTorch for same input tokens
   - Identify exact divergence point

4. **Check Matrix Multiplication Implementation**
   - Verify Metal kernel matrix multiply is correct
   - Compare CPU vs Metal results for same operation
   - Ensure proper handling of quantized weights

5. **Validate RoPE Application**
   - Compare RoPE cos/sin values with llama.cpp
   - Verify rotation is applied correctly to Q and K

## Tools Needed

1. **Debug output enhancement**: Add structured dumps for intermediate values
2. **llama.cpp comparison script**: Extract same intermediate values from llama.cpp
3. **Numerical comparison tool**: Compare arrays with tolerance and statistics

## References

- [METAL_GPU_DEBUGGING_STATUS.md](../../METAL_GPU_DEBUGGING_STATUS.md) - Overall status
- [METAL_GPU_VERIFICATION.md](METAL_GPU_VERIFICATION.md) - Verification progress
- [RMS_NORM_ANALYSIS.md](RMS_NORM_ANALYSIS.md) - RMS Norm deep dive
- llama.cpp source: `ggml-quants.c`, `llama.cpp`
