# Critical Findings - CLI Debug Session

**Date**: 2025-10-11
**Status**: 🚨 ROOT CAUSE IDENTIFIED - Hidden states completely different

## Executive Summary

**Problem**: RusTorch generates gibberish text ("avavertavanthbarertvanASEvon") instead of coherent English.

**Root Cause**: Hidden states after all 22 layers are **completely different** from llama.cpp, indicating a fundamental error in the transformer layer computation.

## Comparison Results

### Input
Same for both implementations:
```
Prompt: "1"
Tokens after template: [1, 29966, 29989, 1792, 29989, 29958, 13, 29896, 29966, 29989, 465, 22137, 29989, 29958]
```

### Final Hidden State (After Layer 21 + output_norm)

**RusTorch** (first 10 values):
```
[0.5916366, 1.057492, -1.0569482, 0.965778, 0.5951404, ...]
RMS: 1.919
```

**llama.cpp** (first 10 values):
```
[-1.82249, 0.059779, -1.97983, -2.77743, -0.15409, ...]
RMS: 1.848
```

**Analysis**:
- NO correlation between values
- Similar RMS magnitude but completely different individual values
- This confirms error is in layers 0-21, NOT in logits calculation

### Layer 0 Output

**RusTorch**:
```
rms=0.014175, first_10=[0.0047632228, -1.03051425e-5, -0.0019137387, ...]
```

Values are very small (0.001-0.01 range), which is reasonable for intermediate layers.

### Token Embeddings

**Token 1 (BOS)** from RusTorch:
```
First 20 values: [-0.001172125, 0.001876652, -0.001781881, 0.003705919, ...]
Stats: mean=0.000022528, rms=0.002226600, min=-0.007652879, max=0.006207168
```

These look reasonable - small values with near-zero mean.

## Confirmed Facts

✅ **Position tracking is correct**:
- Step 0: position=0, input=14 tokens
- Step 1: position=14, input=1 token
- Step 2: position=15, input=1 token

✅ **Logits calculation logic is correct**:
- Simple matrix multiplication: `sum += hidden[h] * weight[v][h]`
- Tested transpose - no effect
- Logic matches llama.cpp

❌ **RMS Norm fix had no effect**:
- Fixed double weight application
- Logits remained identical
- Not the root cause

❌ **Hidden states completely different**:
- No correlation with llama.cpp
- Error must be in transformer layers (0-21)

## Impact on Output

### RusTorch Logits (WRONG)
```
Top tokens from wrong hidden state:
#1: token=485 ("av")   logit=8.5582
#2: token=814 ("ert")  logit=7.4887
#3: token=9716 ("anth") logit=6.7506

Expected tokens have negative/low logits:
token 15043 ("Hello") logit=-2.3852
```

### Expected Behavior (llama.cpp)
```
Output: "Hello, I am doing well. How about you?"
First token should be: "Hello", " I", " am", etc.
```

## Hypothesis Tree

### 🔴 MOST LIKELY: Attention Calculation Error (80% probability)

**Evidence**:
- Hidden states diverge completely from start
- Layer 0 output exists but may be wrong
- Attention is the most complex operation

**Possible Issues**:
1. **RoPE (Rotary Position Embedding) application**
   - Frequency calculation incorrect
   - Cos/sin application wrong
   - Position indexing bug

2. **Q/K/V Projections**
   - Weight loading incorrect
   - Matrix multiplication bug
   - Shape handling wrong

3. **Attention scores**
   - Dot product calculation
   - Scaling factor
   - Softmax application

4. **GQA (Grouped Query Attention)**
   - Head grouping logic
   - KV cache integration
   - Causal masking

### 🟡 POSSIBLE: Weight Loading Error (15% probability)

**Evidence**:
- Token embeddings look reasonable
- But could still have subtle bugs in layer weights

**Possible Issues**:
1. Q5_K dequantization incorrect
2. Weight tensor shapes wrong
3. Weight offsets in GGUF file

### 🟢 LESS LIKELY: FFN Calculation Error (5% probability)

**Evidence**:
- FFN is simpler than attention
- Usually less error-prone

**Possible Issues**:
1. SwiGLU activation incorrect
2. Gate/Up/Down projections wrong

## Next Steps (Priority Order)

### 1. Create Layer-by-Layer Comparison Tool ⭐⭐⭐

**Goal**: Find exact layer where divergence starts

**Method**:
```cpp
// Extract intermediate outputs from llama.cpp
// Compare with RusTorch at each layer:
// - After embedding
// - After each of 22 layers
// - After output_norm
```

**Why**: This will pinpoint the exact operation causing the error

### 2. Deep Dive into Layer 0 ⭐⭐⭐

**Goal**: Understand why Layer 0 output may be wrong

**Method**:
- Add detailed debug output for:
  - Attention norm input/output
  - Q/K/V projections
  - RoPE application
  - Attention scores
  - Attention output
  - FFN input/output

**Why**: Layer 0 is the first place where error can occur

### 3. Verify RoPE Implementation ⭐⭐

**Goal**: Ensure position encoding is correct

**Method**:
```rust
// In RoPE application, log:
eprintln!("RoPE: position={}, head={}, dim={}",  position, head, dim);
eprintln!("  freq={}, cos={}, sin={}", freq, cos, sin);
eprintln!("  before: q0={}, q1={}", q[dim], q[dim+1]);
eprintln!("  after: q0={}, q1={}", result[dim], result[dim+1]);
```

**Why**: RoPE is applied to every attention calculation and could corrupt all layers

### 4. Test with F16/F32 Model ⭐

**Goal**: Eliminate quantization as a variable

**Method**:
- Download non-quantized model
- Run same test
- Compare results

**Why**: If F32 model works, issue is in quantization

## Files to Investigate

### Priority 1: Attention Implementation
- `llama.rs:670-950` - `attention_layer()` function
- `llama.rs:200-260` - RoPE precomputation
- `llama.rs:480-605` - GQA implementation

### Priority 2: Layer Processing
- `llama.rs:880-990` - `transformer_layer()` function
- `llama.rs:290-380` - RMS Norm implementation

### Priority 3: Weight Loading
- `gguf_loader.rs` - GGUF file parsing
- `llama.rs:130-190` - Weight loading and quantization

## Success Criteria

✅ Layer 0 output matches llama.cpp (within quantization error ~0.1%)
✅ Final hidden state matches llama.cpp (within quantization error ~1%)
✅ Generated logits produce correct English text
✅ Output is coherent: "Hello, I am doing well" instead of "avavertavanthbarert"

## Additional Notes

- Model: TinyLlama-1.1B-Chat Q5_K_M
- Backend: hybrid-f32 (CPU dequantization)
- Hidden size: 2048
- Layers: 22
- Heads: 32 (Q) / 4 (KV)
- RoPE theta: 10000

## Conclusion

The error is **NOT** in:
- Position calculation ✓
- Logits calculation ✓
- KV cache logic (tested with cache disabled) ✓

The error **IS** in:
- Transformer layers (0-21) - hidden states completely wrong
- Most likely: Attention calculation (RoPE, Q/K/V, scoring, GQA)
- Needs: Layer-by-layer comparison to pinpoint exact operation
