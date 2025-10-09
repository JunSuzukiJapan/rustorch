# Metal GPU Debug Session - 2025-10-09

## Session Summary

Continued investigation of Metal GPU backend gibberish output issue. Focus: verify numerical stability fixes and identify divergence point from llama.cpp reference.

## Key Activities

### 1. Extended Clipping Verification ✅

**Modification**: Extended clipping debug output from Layer 0 only to Layers {0, 5, 10, 15, 20, 21}

**File**: `src/models/gpt.rs:1020-1044`

**Results**:
```
Layer  0: RMS=0.012, max_abs=0.068
Layer  5: RMS=0.151, max_abs=0.612
Layer 10: RMS=0.301, max_abs=1.147
Layer 15: RMS=0.488, max_abs=1.725
Layer 20: RMS=0.927, max_abs=3.287
Layer 21: RMS=1.095, max_abs=4.212
```

**Analysis**:
- Clipping code is active and executing
- Values remain well within ±10.0 clip threshold
- 60x amplification (0.07 → 4.21) occurs despite clipping
- 91x RMS growth (0.012 → 1.09)

**Conclusion**: Either:
1. Clip threshold (±10.0) is too lenient for this model
2. This amplification is **normal expected behavior**
3. Need to compare with hybrid_f32/llama.cpp to determine baseline

### 2. Logits Distribution Analysis ✅

**Test Input**: "1"

**Metal Backend Output**:
```
Top-5 Logits:
1. Token 3499:  8.4672  ← Selected (gibberish)
2. Token 25323: 7.8559
3. Token 6706:  7.8146
4. Token 26890: 7.4848
5. Token 22919: 7.4502

Stats: max=8.47, min=-11.18, mean=-0.03
```

**llama.cpp Reference**: Output " 10" (correct, rational continuation)

**Critical Finding**: Metal backend selects different top token than llama.cpp despite:
- Same model file (Q4_K_M quantization)
- Same Metal GPU backend
- Same input tokenization

**Implication**: Divergence occurs somewhere in forward pass before final sampling

### 3. Attempted Comparison Setup

**Goal**: Extract intermediate Q/K/V values from llama.cpp for direct comparison

**Approach**: Created C++ program to hook into llama.cpp internals

**Outcome**: llama.cpp doesn't expose intermediate layer values in public API. Would require source modification.

**Alternative**: Focus on comparing final logits and working backwards to isolate divergence

## Root Cause Hypothesis

### Evidence Summary

✅ **Verified Correct**:
1. Token embeddings (RMS 0.002-0.010, statistically valid)
2. RMS Norm implementation (84.3% ratio mathematically correct due to eps)
3. Clipping mechanism (active, but threshold may be incorrect)

❓ **Suspected Components**:
1. **Q/K/V Projection** (Metal matmul kernel)
   - Weight stats look reasonable (Q_weight RMS=0.016)
   - Projection values reasonable (Q_proj RMS=0.086)
   - But: No direct comparison with llama.cpp

2. **RoPE Application**
   - Implementation copied from hybrid_f32
   - But: Not independently verified against llama.cpp

3. **Attention Mechanism**
   - Softmax collapse observed (uniform weights ~0.056)
   - Temperature scaling improved but didn't fix
   - CPU-style loops should be correct but slow

4. **FFN/SwiGLU**
   - Not yet analyzed
   - Potential source of amplification

5. **Final LM Head Projection**
   - Logits distribution differs from llama.cpp
   - This is WHERE divergence manifests
   - But likely not WHERE divergence originates

### Most Likely Root Cause

**Primary Suspect**: Metal matmul kernel behavior differs from CPU implementation

**Supporting Evidence**:
- Q/K/V projections use Metal executor.matmul_f32()
- No verification that Metal matmul produces identical results to CPU
- Quantized weight dequantization happens in Metal kernel
- Small numerical differences compound across 22 layers

**Alternative**: RoPE position encoding differs from llama.cpp

**Supporting Evidence**:
- RoPE critically affects attention patterns
- Implementation from hybrid_f32, not directly from llama.cpp
- No independent verification

## Next Steps (Prioritized)

### Immediate (Critical Path)

1. **Verify Metal matmul correctness** 🔴
   - Create standalone test: Metal matmul vs CPU matmul
   - Use actual Q/K/V weights from Layer 0
   - Compare outputs element-wise
   - Tolerance: <1e-5 difference acceptable

2. **Compare Metal vs hybrid_f32 intermediate values** 🔴
   - Add logits output to hybrid_f32
   - Compare Top-5 tokens between backends
   - If identical → Metal-specific issue
   - If different → Quantization or architecture issue

3. **Verify RoPE implementation** 🟡
   - Extract RoPE logic from llama.cpp
   - Compare with RusTorch implementation
   - Verify cos/sin calculation formulas
   - Test with known position values

### Investigation Tools

1. **Create Metal matmul test harness**
```rust
// File: examples/test_metal_matmul_correctness.rs
// Compare Metal vs CPU matmul for Layer 0 Q projection
// Input: x_ln1 (embedding after RMS norm)
// Weight: blk.0.attn_q.weight
// Expected: Identical results within tolerance
```

2. **Add hybrid_f32 logits logging**
```rust
// File: src/hybrid_f32/models/llama.rs
// Add same logits debug output as Metal backend
// Compare Top-5 tokens for input "1"
```

3. **Standalone RoPE verification**
```rust
// File: examples/test_rope_correctness.rs
// Compare RusTorch RoPE with llama.cpp formula
// Verify freq, cos, sin calculations
```

### Long-term (Post-Fix)

1. Remove value clipping (if determined unnecessary)
2. Restore GPU-optimized attention (replace CPU loops)
3. Add comprehensive unit tests for each component
4. Create regression test suite

## Files Modified This Session

1. `src/models/gpt.rs:1020-1044` - Extended clipping verification
2. `METAL_GPU_DEBUGGING_STATUS.md:287-337` - Added analysis results
3. `/tmp/compare_layer0_qkv.cpp` - Attempted llama.cpp hook (abandoned)
4. `/tmp/test_llamacpp_debug.sh` - llama.cpp test script

## Session Outcomes

**Progress Made**:
- ✅ Confirmed clipping is active across all layers
- ✅ Identified divergent logits distribution
- ✅ Narrowed root cause to Metal matmul or RoPE
- ✅ Created actionable next steps

**Blockers Resolved**:
- None

**New Blockers**:
- llama.cpp doesn't expose intermediate values (need source modification or alternative approach)

**Estimated Time to Resolution**:
- Optimistic: 2-4 hours (if Metal matmul is the issue and fix is simple)
- Realistic: 8-12 hours (if RoPE or multiple components involved)
- Pessimistic: 16-24 hours (if fundamental architecture issue)

## References

- Working Reference: `src/hybrid_f32/models/llama.rs`
- llama.cpp Source: `Temp/llama.cpp/`
- Previous Analysis: `docs/core/METAL_GPU_VERIFICATION.md`
- Status Document: `METAL_GPU_DEBUGGING_STATUS.md`
