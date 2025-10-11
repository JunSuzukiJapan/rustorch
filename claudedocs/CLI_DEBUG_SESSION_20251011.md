# CLI Debugging Session - 2025-10-11

**Status**: ❌ UNRESOLVED - CLI produces gibberish output
**Goal**: Fix CLI to produce correct text generation like llama.cpp

## Problem Statement

Both Q4_K_M and Q5_K_M+ models produce meaningless output:
```
Input: "Hello, how are you?"
Expected: "Hello, I am doing well. How about you?"
Actual: "avavertavanthbarertvanASEvon"
```

llama.cpp with same model works correctly.

## Investigation Steps

### 1. Initial Verification ✅
- Position calculation: CORRECT (start_position = 0, 14, 15, ...)
- RoPE implementation: CORRECT (verified in previous sessions)
- Q4_K quantization: CORRECT (100% match with llama.cpp)
- KV cache: CORRECT (tested and documented)

### 2. Logits Analysis ❌
**CRITICAL FINDING**: Logits are completely wrong

**RusTorch logits (first token)**:
```
Top 10:
  #1: token=485 ("av") logit=8.5582
  #2: token=814 ("ert") logit=7.4887
  #3: token=9716 ("anth") logit=6.7506
  ...

Expected tokens:
  token 15043 ("Hello") logit=-2.3852  ← NEGATIVE!
  token 6324 ("I") logit=0.4831
```

**Problem**: Nonsense tokens have highest logits, correct English tokens have negative/low logits.

### 3. RMS Norm Investigation ⚠️
**Found**: Potential double application of weight in RMS Norm

**Original code**:
```rust
// Line 922-930 (attention norm)
let normed = self.rms_norm(x, &attn_norm_weight)?;  // Step 1
let normed = normed.mul(&attn_norm_weight)?;         // Step 2: DOUBLE APPLICATION

// Line 964-966 (FFN norm)
let normed = self.rms_norm(&x, &ffn_norm_weight)?;   // Step 1
let normed = normed.mul(&ffn_norm_weight)?;          // Step 2: DOUBLE APPLICATION

// Line 1098-1100 (output norm)
let normed = self.rms_norm(&x, output_norm_weight)?;  // Step 1
let normed = normed.mul(output_norm_weight)?;         // Step 2: DOUBLE APPLICATION
```

**Fix Applied**:
- Modified `rms_norm()` to apply weight internally (line 366)
- Removed second multiplication in all three locations

**Result**: ❌ NO EFFECT - Logits unchanged

### 4. Key Observations

1. **RMS Norm fix had no effect** → Problem is elsewhere
2. **Logits distribution is fundamentally wrong** → Not just magnitude, but ranking
3. **Q5_K_M also broken** → Not a quantization precision issue
4. **Position and RoPE verified correct** → Not an attention masking issue

## Hypotheses Remaining

### Hypothesis 1: output.weight Transpose Issue ⚠️
**Evidence**:
- Line 1342: `let needs_transpose = false;`
- Code comment assumes row-major: `data[h * vocab_size + v]`
- GGUF format may store weights in different layout

**Test Required**:
```rust
// Current: logits[v] = sum_h(hidden[h] * data[h * vocab_size + v])
// Alternative: logits[v] = sum_h(hidden[h] * data[v * hidden_size + h])
```

### Hypothesis 2: Attention Calculation Error ⚠️
**Reasoning**:
- All layers produce wrong hidden states
- Error compounds through 22 layers
- Final logits completely wrong

**Areas to Check**:
- Q/K/V projection matrices
- Attention score computation
- Softmax application
- Value aggregation

### Hypothesis 3: Weight Loading Error ⚠️
**Evidence**:
- Debug shows token_embd.weight exists
- output.weight may be same as token_embd.weight (weight tying)
- Need to verify weights are loaded correctly

**Test Required**:
Compare first 100 values of output.weight between RusTorch and llama.cpp

### Hypothesis 4: FFN Calculation Error ⚠️
**Reasoning**:
- SwiGLU activation may be wrong
- gate/up/down projections may be incorrect
- Amplifies errors from attention

## RMS Norm Implementation Details

### Current Implementation (llama.rs:290-379)
```rust
fn rms_norm(&self, x: &F32Tensor, weight: &F32Tensor) -> F32Result<F32Tensor> {
    // 1. Normalize: x / rms(x)
    let scale = 1.0 / (mean + eps).sqrt();
    output[i] *= scale;

    // 2. Apply weight (ADDED IN FIX)
    output[i] *= weight_data[i];

    return output;
}
```

**Note**: Original implementation only normalized, didn't apply weight.
Weight was applied separately afterward (double application).

## Recommended Next Steps

### Immediate (High Priority)
1. **Verify output.weight transpose**
   - Try both indexing patterns: `[h, v]` vs `[v, h]`
   - Compare first 100 values with llama.cpp

2. **Test with F16 model**
   - Use non-quantized model to eliminate quantization variables
   - Verify core logic is correct

3. **Layer-by-layer comparison**
   - Save Layer 0 output to file
   - Compare with llama.cpp Layer 0 output
   - Identify where divergence begins

### Medium Priority
4. **Attention computation audit**
   - Review Q/K/V projection
   - Verify attention score calculation
   - Check softmax implementation

5. **FFN computation audit**
   - Review SwiGLU implementation
   - Verify gate/up/down projections

### Low Priority (if above don't fix)
6. **Weight loading verification**
   - Dump all weight shapes and first 10 values
   - Compare with llama.cpp weight dump
   - Verify GGUF parsing is correct

## Code Changes Made

### Files Modified
- `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/src/hybrid_f32/models/llama.rs`

### Changes
1. **Line 366**: Added weight application in `rms_norm()`
   ```rust
   output[y_offset + i00] *= weight_data[i00];  // Apply weight here
   ```

2. **Line 922**: Removed double weight application in attention norm
   ```rust
   // BEFORE:
   let normed = self.rms_norm(x, &attn_norm_weight)?;
   let normed = normed.mul(&attn_norm_weight)?;  // REMOVED

   // AFTER:
   let normed = self.rms_norm(x, &attn_norm_weight)?;
   ```

3. **Line 956**: Removed double weight application in FFN norm
   ```rust
   // BEFORE:
   let normed = self.rms_norm(&x, &ffn_norm_weight)?;
   let normed = normed.mul(&ffn_norm_weight)?;  // REMOVED

   // AFTER:
   let normed = self.rms_norm(&x, &ffn_norm_weight)?;
   ```

4. **Line 1100**: Removed double weight application in output norm
   ```rust
   // BEFORE:
   let normed = self.rms_norm(&x, output_norm_weight)?;
   let normed = normed.mul(output_norm_weight)?;  // REMOVED

   // AFTER:
   let normed = self.rms_norm(&x, output_norm_weight)?;
   ```

## Test Results

### Before Fix
```
Input: "Hello, how are you?"
Output: "avavertavanthbarertvanASEvon"
Top logits: 485, 814, 9716, ...
```

### After Fix
```
Input: "Hello, how are you?"
Output: "avavertavanthbarertvanASEvon"  ← UNCHANGED
Top logits: 485, 814, 9716, ...  ← UNCHANGED
```

**Conclusion**: RMS Norm double application was NOT the root cause.

## Time Spent
- Investigation: ~3 hours
- RMS Norm fix: ~30 minutes
- Testing: ~30 minutes
- **Total**: ~4 hours

## Next Session Priorities
1. Test output.weight transpose (HIGHEST PRIORITY)
2. Layer 0 output comparison with llama.cpp
3. If still broken, systematic layer-by-layer audit

---
**Session End**: 2025-10-11
**Status**: UNRESOLVED - Root cause not yet identified
**Confidence**: Medium - Several promising leads to investigate
