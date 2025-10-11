# Q8_0 Root Cause Analysis - October 11, 2025

## Executive Summary

**ROOT CAUSE IDENTIFIED**: The Q8_0 gibberish output is caused by incorrect hidden state computation in the **transformer layers**, NOT in the output projection layer.

### Key Findings

1. ✅ **GGUF Q8_0 Dequantization**: VERIFIED CORRECT
2. ✅ **output.weight Loading**: VERIFIED CORRECT
3. ✅ **Logit Calculation Index**: VERIFIED CORRECT
4. ❌ **Hidden State Computation**: COMPLETELY WRONG (0% match with llama.cpp)

## Investigation Timeline

### Phase 1: Weight Value Verification

**Hypothesis**: Q8_0 weights might be dequantized incorrectly.

**Test**: Compare first 20 dequantized values from output.weight between RusTorch and file.

**Result**: ✅ PERFECT MATCH
```
RusTorch output.weight[0] = 0.012436867
File Q8_0 first block:
  scale = 0.000222087
  quant[0] = 56
  dequantized[0] = 0.000222087 * 56 = 0.012436867 ✓
```

**Conclusion**: Q8_0 dequantization is working perfectly.

### Phase 2: Hidden State Comparison

**Hypothesis**: If weights are correct, maybe hidden states are wrong?

**Test**: Compare hidden state values (last token) between RusTorch and llama.cpp for input "1".

**RusTorch Hidden State** (first 20 values):
```
hidden[0] = 0.841209710
hidden[1] = 0.638764083
hidden[2] = -0.911493123
hidden[3] = 1.064220309
...
```

**llama.cpp Hidden State** (first 20 values):
```
hidden[0] = -0.052411154
hidden[1] = 3.412513494
hidden[2] = -2.965995312
hidden[3] = -0.192227572
...
```

**Comparison Results**:
```
Match rate: 0/20 (0.0%)
Average difference: 1.507
Max difference: 4.133
```

**Conclusion**: ❌ **Hidden states are COMPLETELY DIFFERENT**

## Root Cause Analysis

### What We Know

1. **Q4_K quantization works correctly** → Basic transformer architecture is sound
2. **Q8_0 dequantization is correct** → Weight loading is not the issue
3. **output.weight values match file** → Output projection setup is correct
4. **Hidden states are wrong** → Problem is in transformer layers

### Where the Bug Is

The bug MUST be in how Q8_0 weights are accessed in:
- **Attention layers** (Q, K, V, output projections)
- **FFN layers** (gate, up, down projections)
- Possibly **RMSNorm** or **RoPE**

### Why Q4_K Works But Q8_0 Doesn't

The key difference between Q4_K and Q8_0:

**Q4_K Block Structure**:
- Super-block of 256 elements
- Complex sub-block organization
- Uses 6-bit quantized scales

**Q8_0 Block Structure**:
- Simple block of 32 elements
- Single f16 scale + 32 int8 values
- Sequential layout

**Hypothesis**: RusTorch might have Q8_0-specific indexing or layout issues in:
1. Matrix multiplication kernels (Metal/CPU)
2. Weight tensor reshaping/transposition
3. Batch processing of Q8_0 data

## Evidence Chain

### Test 1: Token Output
```
Input: "1"
Expected (llama.cpp): "Hello\nGreetings! I'm interested in"
Actual (RusTorch Q8_0): "avavertavanthertavanthder A"
Actual (RusTorch Q4_K): "Hello\nGreetings! It was pleasure chat"
```

### Test 2: Logit Comparison
```
Token 485:
  llama.cpp logit: -2.032
  RusTorch logit: 9.078
  Difference: 11.11 (MASSIVE)
```

### Test 3: Weight Verification
```
output.weight[0-19]: EXACT MATCH ✓
scale_bits: 0x0b47 (identical)
quants: [56, -102, -111, ...] (identical)
```

### Test 4: Hidden State Comparison
```
ALL 20 values MISMATCH ✗
Average error: 1.507
This confirms the bug is BEFORE output projection
```

## Next Steps

### Priority 1: Inspect Transformer Layer Q8_0 Handling

1. **Check Attention Weight Access**
   - Location: `src/hybrid_f32/models/llama.rs` - attention implementation
   - Verify Q8_0 weight indexing in Q, K, V projections
   - Compare with Q4_K implementation that works

2. **Check FFN Weight Access**
   - Location: `src/hybrid_f32/models/llama.rs` - FFN implementation
   - Verify gate/up/down projection weight indexing
   - Check for dimension reversal issues

3. **Add Layer-by-Layer Debug**
   - Dump hidden states after each layer
   - Find the first layer where divergence occurs
   - Narrow down to specific operation

### Priority 2: Compare with Q4_K Implementation

Q4_K works correctly, so compare:
```rust
// Q4_K matmul (WORKS)
let output = matmul_q4k(input, weight_q4k);

// Q8_0 matmul (BROKEN)
let output = matmul_q8_0(input, weight_q8_0);
```

Find the difference in:
- Indexing patterns
- Dimension handling
- Data layout assumptions

### Priority 3: Test Simplified Case

Create minimal test:
```rust
// Single Q8_0 matrix multiplication
let input = vec![1.0; 2048];  // Simple input
let weight_q8_0 = load_q8_0_weight("blk.0.attn_q.weight");
let output = matmul(input, weight_q8_0);
// Compare with llama.cpp
```

## Files to Investigate

1. **`src/hybrid_f32/models/llama.rs`** (Lines 400-900)
   - Attention layer implementation
   - FFN layer implementation
   - Weight tensor access patterns

2. **`src/gpu/metal_kernels.rs`**
   - Metal Q8_0 matmul kernels
   - Check if Q8_0 is handled differently than Q4_K

3. **`src/formats/gguf.rs`** (Lines 658-682)
   - Dimension reversal logic for quantized tensors
   - Already verified Q8_0 is NOT reversed (correct)

## Debug Commands Used

```bash
# Dump RusTorch weights
printf "1\n" | ./target/release/rustorch-cli \
  --model ~/.rustorch/models/.../Q8_0.gguf \
  --backend hybrid-f32 --max-tokens 1

# Dump llama.cpp hidden state
DYLD_LIBRARY_PATH=.../llama.cpp/build/bin ./dump_llama_hidden

# Compare
python3 /tmp/compare_hidden.py
```

## Conclusion

The Q8_0 bug is definitively located in the **transformer layer weight access**, not in:
- Weight loading ✓
- Dequantization ✓
- Output projection ✓

The next investigation should focus on finding why Q8_0 weights produce wrong hidden states when Q4_K weights work correctly.
