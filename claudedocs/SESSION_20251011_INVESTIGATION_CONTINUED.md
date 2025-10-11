# Q8_0 Investigation Continued - 2025-10-11 (Session 2)

## 🔍 Key Discovery: Shape Reversal is NOT the Root Cause

### Investigation Summary

Attempted to fix Q8_0 gibberish output by modifying shape reversal logic, but discovered:

1. **Q4_K_M is ALSO broken**: Previously thought to be working, but currently produces gibberish
2. **Shape reversal logic was correct**: Original code only reversed F32/F16, which was intentional
3. **Problem exists elsewhere**: Hidden states diverge from llama.cpp at a fundamental level

### Findings

#### Test Results
```bash
# Q8_0
Input: "1"
Output: "anthanthertanthertrun ChallengeniASErinört..."  ❌

# Q4_K_M (previously thought working)
Input: "1"
Output: "anthASEanthniAIASE AnthriaseAIASErinsdderni..."  ❌
```

#### Shape Reversal Investigation

**Tested 3 approaches**:

1. **Reverse all tensors** (Lines 220-232):
   - Result: Dimension mismatch `14x2048 and 256x2048`
   - K/V weights incorrectly reversed

2. **Conditional reversal (square + embedding)**:
   - Result: Still gibberish output
   - Doesn't fix the fundamental issue

3. **Original logic (F32/F16 only)** (CURRENT):
   - Quantized tensors NOT reversed
   - Same as working code, but Q8_0 still broken

### Why Shape Reversal Doesn't Fix It

**GGUF Weight Layout**:
- F32/F16: Need reversal (PyTorch vs GGUF order mismatch)
- Q8_0/Q4_K: Already in correct order for RusTorch matmul
- K/V weights (GQA): `[2048, 256]` - correct as-is

**Evidence**:
- Original code didn't reverse quantized tensors
- Q4_K_M previously worked (according to METAL_GPU_DEBUGGING_STATUS.md)
- Reversal causes dimension mismatches

### True Problem Location

From previous session (SESSION_20251011_Q8_SHAPE_FIX.md):

> **Hidden State Comparison**:
> ```
> Match rate: 0/20 (0.0%)
> Average difference: 1.507
> Max difference: 4.133
> ```

**This means**:
- Problem exists BEFORE output projection
- Individual components verified correct (dequant, RMSNorm, RoPE)
- Issue is in how components are COMBINED

### Layer Output Statistics

```
Embedding: RMS=0.008, Max=0.069
Layer 0:   RMS=0.014, Max=0.064  (1.7x growth)
Layer 1:   RMS=0.023, Max=0.086  (2.9x growth)
Layer 2:   RMS=0.038, Max=0.125  (4.8x growth)
Layer 21:  RMS=1.062, Max=4.263  (134x growth!)
```

Exponential growth suggests **numerical instability** or **incorrect operation ordering**.

## 🎯 Next Investigation Steps

### Priority 1: Verify Previous "Working" State

1. **Check git history**: When did Q4_K_M actually work?
2. **Find commit**: Identify last known good state
3. **Compare code**: What changed between working and broken?

### Priority 2: Deep Dive into Embedding Layer

Q8_0 dequantization verified correct, but need to check:

1. **Embedding lookup**: Are tokens correctly indexed?
2. **Memory layout**: Is data read in correct order?
3. **Comparison with llama.cpp**: Direct embedding output comparison

### Priority 3: Systematic Component Testing

Test each component in isolation:

```rust
// Test 1: Embedding only
let emb = model.embed(tokens);
// Compare with llama.cpp embedding

// Test 2: Embedding + RMSNorm
let normed = rms_norm(emb);
// Compare with llama.cpp after first norm

// Test 3: Full attention (no residual)
let attn = attention(normed);
// Compare with llama.cpp attention output
```

### Priority 4: Check Data Types and Precision

- Verify f32 vs f16 conversions
- Check for integer overflow in Q8_0 operations
- Validate scale factors are applied correctly

## 📊 Code State

### Modified Files

`src/hybrid_f32/models/llama.rs`:
- Lines 220-232: `from_gguf_with_device()` - Reverted to original F32/F16 only reversal
- Lines 1394-1410: `from_gguf_with_config()` - Reverted to original F32/F16 only reversal

### Current Behavior

**Both Q8_0 and Q4_K_M produce gibberish**, which contradicts previous session notes claiming Q4_K_M works.

**Possible explanations**:
1. Code regression between sessions
2. Environment/build configuration differences
3. Previous "working" state was actually partially broken
4. Documentation error in previous session

## 🚨 Critical Questions

1. **When did Q4_K_M last produce correct output?**
   - Need to verify this claim with actual test
   - Check git history and test results

2. **What is fundamentally different about Q8_0?**
   - Both use block-based quantization
   - Q8_0: 32 elements per block, simpler structure
   - Q4_K: 256 elements per super-block, complex sub-blocks

3. **Why does layer output grow exponentially?**
   - RMSNorm should stabilize values
   - Something is amplifying instead of normalizing
   - Possible residual connection issue?

## 📁 Related Documentation

- `METAL_GPU_DEBUGGING_STATUS.md`: Full debugging history
- `claudedocs/SESSION_20251011_Q8_SHAPE_FIX.md`: First session findings
- `claudedocs/Q8_0_ROOT_CAUSE_ANALYSIS_20251011.md`: Previous root cause analysis

## 🎓 Lessons Learned

1. **Don't assume working state without verification**: Q4_K_M thought to work, actually broken
2. **Shape reversal was a red herring**: Original logic was correct
3. **Need systematic bisection**: Test each component independently
4. **Documentation vs Reality**: Always verify claims with actual tests

## 💡 Hypothesis for Next Session

**The problem is NOT in weight loading or shape handling.**

**It's likely in one of these areas**:
1. ✅ Q8_0 dequantization (verified correct by previous session)
2. ❓ Token to embedding lookup (index calculation?)
3. ❓ Residual connections (add operations causing drift?)
4. ❓ Attention mask or position encoding
5. ❓ Layer normalization weight application
6. ❓ Metal kernel implementation differences

**Next action**: Create minimal reproduction case comparing RusTorch vs llama.cpp for **single token embedding lookup** only.
