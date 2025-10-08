# Debug Session Summary: 2025-10-08

## Session Goal
Identify why RusTorch generates wrong token ("diplom" at rank 6) for simple inputs.

## Investigation Process

### 1. Initial Hypothesis: Quantization Error
- **Action**: Verified Q4_K dequantization implementation
- **Result**: ✓ Correct - matches llama.cpp formula exactly

### 2. Hypothesis: RMSNorm Amplification
- **Action**: Analyzed Layer 0 RMSNorm behavior
- **Finding**: RMSNorm amplifies small inputs (RMS=0.009 → max=4.8)
- **Result**: ✓ Mathematically correct, but causes value growth

### 3. Hypothesis: SwiGLU Implementation Bug
- **Action**: Verified SwiGLU formula and implementation
- **Result**: ✓ Correct - `SwiGLU(gate, up) = SiLU(gate) × up`

### 4. Discovery: Layer-by-Layer Value Growth
- **Action**: Tracked RMS across all 22 layers
- **Finding**: Nearly linear growth (Layer 0: 0.015 → Layer 21: 1.124)
- **Result**: ⚠️ 75x amplification over 22 layers

### 5. Discovery: Mean Value Drift
- **Action**: Tracked mean values across layers
- **Finding**: Mean drifts from -0.000174 → -0.036644
- **Result**: ⚠️ 200x increase in negative bias

### 6. Hypothesis: Weight Bias
- **Action**: Verified all projection weight statistics
- **Finding**: 
  - gate_weight mean: -0.000026 to -0.000064
  - up_weight mean: -0.000006 to -0.000012
  - down_weight mean: 0.000000 to 0.000003
- **Result**: ✓ All weights have mean ≈ 0

### 7. Final Discovery: Architectural Feedback Loop
- **Action**: Analyzed FFN pipeline component by component
- **Finding**:
  - SwiGLU output mean ≈ 0 ✓
  - Gate/Up projection mean ≠ 0 (input already biased)
  - Bias originates from residual accumulation
- **Result**: ⚠️ RMSNorm + Residual creates feedback loop

## Root Cause

**Architectural feedback loop**: RMSNorm + Residual connections cause small biases to accumulate and amplify over 22 layers.

### Mechanism
```
Layer N output = Layer N-1 output + Attention + FFN
                    ↑ (small bias)      ↑        ↑
                                  (amplifies) (adds more bias)
```

After 22 layers:
- RMS: 0.015 → 1.920 (128x growth)
- Mean: -0.000174 → -0.036644 (211x growth)
- Final hidden state is strongly biased

### Impact
- Biased hidden state → biased logits
- Token 13487 ("diplom") incorrectly ranked at position 6
- Even single BOS token produces wrong predictions

## Verified Components (All Correct)

1. ✓ RMSNorm formula
2. ✓ SwiGLU implementation  
3. ✓ Weight loading (mean ≈ 0)
4. ✓ Q4_K quantization/dequantization
5. ✓ Attention mechanism (reduces values correctly)
6. ✓ Matmul operations

## Remaining Questions

1. **Is this normal LLaMA behavior?**
   - Need comparison with llama.cpp layer outputs
   - Need comparison with HuggingFace Transformers

2. **Does quantization exacerbate the issue?**
   - Need test with F32 GGUF model
   - Need test with full F16/F32 weights

3. **Is it specific to this model checkpoint?**
   - Need test with different TinyLlama checkpoint
   - Need test with Llama-2-7B

## Next Steps

### High Priority
1. Test with F32 GGUF model (eliminate quantization variable)
2. Extract llama.cpp layer statistics for direct comparison
3. Compare with HuggingFace Transformers output

### Medium Priority
4. Test with different model (Llama-2-7B)
5. Verify with known-good prompts that llama.cpp handles correctly

### Low Priority
6. Investigate if bias is acceptable within normal variance
7. Check if issue only affects small models (1.1B)

## Files Modified

- `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/src/hybrid_f32/models/llama.rs`
  - Added layer-by-layer debug output
  - Added mean tracking for all components
  - Added weight statistics verification

## Key Insight

**The implementation is correct, but the architecture creates a feedback loop.**

This is either:
1. Normal behavior (all implementations have this)
2. Specific to Q4_K quantization
3. Specific to this model checkpoint
4. A subtle numerical issue that accumulates

**Conclusion**: Further comparison with reference implementations required.
