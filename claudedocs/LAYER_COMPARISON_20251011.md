# Layer-by-Layer Comparison Analysis

**Date**: 2025-10-11
**Critical Finding**: Hidden states are completely different between RusTorch and llama.cpp

## Executive Summary

✅ **Position calculation**: CORRECT
✅ **Logits calculation logic**: CORRECT
❌ **Hidden state after all layers**: COMPLETELY WRONG
→ **Conclusion**: Error is in layers 0-21 (transformer blocks)

## Hidden State Comparison

### Input Tokens
Same for both implementations:
```
[1, 29966, 29989, 1792, 29989, 29958, 13, 10994, 29966, 29989, 465, 22137, 29989, 29958]
```

### Final Hidden State (after layer 21 + output_norm)

**RusTorch** (`/tmp/hidden_state_call_0.txt`):
```
[0] =  0.581001
[1] =  1.058685
[2] = -1.030624
[3] =  0.968927
[4] =  0.606570
```

**llama.cpp** (`/tmp/llama_cpp_hidden.txt`):
```
[0] = -1.822490
[1] =  0.059779
[2] = -1.979830
[3] = -2.777430
[4] = -0.154090
```

**Difference**: No correlation at all - values are completely different

**RMS Magnitude**:
- RusTorch: 1.919 (mean squared: 3.684)
- llama.cpp: 1.848 (mean squared: 3.413)

→ Overall magnitude is similar, but individual values are unrelated

## Impact on Logits

### RusTorch Logits (WRONG)
```
Top tokens:
#1: token=485 ("av")   logit=8.5582
#2: token=814 ("ert")  logit=7.4887
#3: token=9716 ("anth") logit=6.7506
```

### Expected Tokens (from llama.cpp behavior)
```
Should generate: "Hello, I am doing well. How about you?"
First token should be around: "Hello", " I", " am", etc.
```

## Root Cause Analysis

### What We Know

1. **Position tracking is correct**: Verified in previous session
   - Step 0: position=0, input=14 tokens ✓
   - Step 1: position=14, input=1 token ✓
   - Step 2: position=15, input=1 token ✓

2. **Logits calculation is correct**: Simple matrix multiplication
   - No change when testing transpose
   - Logic matches llama.cpp implementation

3. **RMS Norm fix had no effect**: Double weight application was fixed but didn't change output

4. **Hidden states diverge completely**:
   - No correlation between RusTorch and llama.cpp
   - Similar magnitude but different values
   - Error must be in layers 0-21

### Where the Error Must Be

Since hidden states are completely different, the error is in one of:

1. **Token Embedding Layer** ⭐⭐⭐
   - Weight loading from GGUF
   - Embedding lookup logic
   - Shape/indexing

2. **Layer 0 (First Transformer Block)** ⭐⭐⭐
   - Attention calculation
   - FFN calculation
   - Residual connections

3. **RoPE (Rotary Position Embedding)** ⭐⭐
   - Frequency calculation
   - Application logic
   - Position indexing

4. **Attention Mechanism** ⭐⭐
   - Q/K/V projections
   - Attention scores
   - Output projection

5. **FFN (Feed-Forward Network)** ⭐
   - SwiGLU activation
   - Gate/Up/Down projections

## Next Steps (Priority Order)

### 1. Debug Token Embedding Layer ⭐⭐⭐ HIGHEST PRIORITY

**Why**: If embeddings are wrong from the start, everything downstream will be wrong

**Test Method**:
```rust
// In llama.rs, after embedding lookup (line ~500)
if step == 0 && position == 0 {
    eprintln!("🔍 [EMBEDDING] First token (id=1) embedding:");
    for i in 0..10 {
        eprintln!("  [{}] = {}", i, x_data[i]);
    }
}
```

**Expected**: Should match token_embd.weight[1 * 2048 : 1 * 2048 + 10] from GGUF file

### 2. Debug Layer 0 Input/Output ⭐⭐⭐

**Why**: If Layer 0 input is correct but output is wrong, error is in Layer 0

**Test Method**:
```rust
// Before and after Layer 0 forward pass
eprintln!("🔍 [LAYER 0 INPUT] First 10 values: {:?}", &x[0..10]);
let x = self.layers[0].forward(...)?;
eprintln!("🔍 [LAYER 0 OUTPUT] First 10 values: {:?}", &x[0..10]);
```

### 3. Debug RoPE Application ⭐⭐

**Why**: RoPE affects all attention calculations

**Test Method**:
```rust
// In apply_rope function
if layer_idx == 0 && head_idx == 0 {
    eprintln!("🔍 [ROPE] position={}, freq[0]={}, cos[0]={}, sin[0]={}",
              position, self.rope_freqs[0], cos_val, sin_val);
}
```

### 4. Compare GGUF Weight Loading ⭐

**Why**: If weights are loaded incorrectly, all computation will be wrong

**Test Method**:
- Extract specific weights from GGUF file using gguf-dump
- Compare with RusTorch loaded weights
- Focus on token_embd.weight and blk.0.* weights

## Key Questions to Answer

1. **Are token embeddings correct?**
   - Does RusTorch embedding for token 1 match GGUF file?

2. **Does Layer 0 input match embedding output?**
   - Is the embedding passed correctly to Layer 0?

3. **Does Layer 0 output make sense?**
   - Are values in reasonable range?
   - Do they change with different inputs?

4. **Are RoPE frequencies correct?**
   - Do they match llama.cpp calculation?

5. **Are attention scores reasonable?**
   - Are they in (0, 1) range after softmax?

## Files to Modify

### `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/src/hybrid_f32/models/llama.rs`

**Add debug output at**:
1. After embedding lookup (~line 500)
2. Before/after Layer 0 forward (~line 550)
3. In RoPE application (~line 700)
4. In attention calculation (~line 800)

## Methodology

**Phase 1**: Add targeted debug output
**Phase 2**: Run with single input token
**Phase 3**: Compare each value with expected values
**Phase 4**: Identify exact location of divergence
**Phase 5**: Fix the issue
**Phase 6**: Verify with full generation

## Success Criteria

✅ Token embedding matches GGUF file
✅ Layer 0 input equals embedding output
✅ Layer 0 output changes reasonably with input
✅ Final hidden state matches llama.cpp
✅ Generated text is coherent English
