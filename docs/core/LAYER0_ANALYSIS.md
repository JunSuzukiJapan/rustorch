# Layer 0 Analysis - Forward Pass Debugging

## Date: 2025-10-08

## Input
- Text: "Test"
- Token IDs: `[1, 29966, 29989, 1792, 29989, 29958, 13, 3057, 829, 29879, 29958, 13, 29966, 29989, 465, 22137, 29989, 29958, 13, 2]`
- Sequence length: 20 tokens

## Layer 0 Detailed Statistics

### 1. Input to Layer 0
```
rms=0.009120, min=-0.062244, max=0.087731, mean=-0.000013
```
- **Analysis**: Normal embedding values, RMS is small (0.009)

### 2. Attention RMSNorm
**Before RMSNorm:**
```
input rms=0.009120, max=0.087731
attn_norm.weight stats: rms=0.046377, max=0.769531
```

**After RMSNorm:**
```
rms=0.107617, min=-4.141421, max=4.804578, mean=0.000860
```

**Analysis:**
- Input RMS is very small (0.009120)
- RMSNorm formula: `output = (input / RMS) * weight`
- Division by small RMS causes **~12x amplification**: 0.087731 / 0.009120 ≈ 9.6
- Further multiplication by weight (max=0.769531) gives max=4.804578
- This is mathematically correct behavior for RMSNorm

### 3. Attention Mechanism
**Input to Attention:**
```
rms=0.107617, max=4.804578
```

**Attention Output (before output projection):**
```
rms=0.032576, max=0.101824
```

**Output Projection Weight:**
```
o_weight stats: rms=0.008275, max=0.369265
```

**Final Attention Output:**
```
rms=0.011568, max=0.052531
```

**Analysis:**
- Attention successfully reduces large RMSNorm outputs
- From max=4.8 → max=0.1 (before o_proj) → max=0.05 (after o_proj)
- This is **correct behavior** - attention is working properly

### 4. Residual Connection (After Attention)
```
rms=0.014746, min=-0.078807, max=0.097641, mean=-0.000153
```
- Residual connection adds back original input
- Values remain reasonable

### 5. FFN RMSNorm
**After FFN RMSNorm:**
```
rms=0.082010, min=-1.432456, max=1.915984, mean=-0.000004
```
- Another normalization step before FFN

### 6. FFN Processing
**Gate Projection:**
```
rms=0.061091, max=0.302931
```

**Up Projection:**
```
rms=0.062791, max=0.294535
```

**SwiGLU Output:**
```
rms=0.001938, max=0.020044
```

**Analysis:**
- SwiGLU activation produces very small values (RMS=0.002)
- This might be too small - potential issue?

### 7. FFN Output
**Final FFN Output:**
```
rms=0.002623, min=-0.012237, max=0.011921, mean=-0.000021
```

**Down Weight Stats:**
```
rms=0.017737, min=-0.490723, max=0.340576
```

**Analysis:**
- FFN produces very small outputs (RMS=0.003)
- Multiplying small SwiGLU output by down_weight gives tiny result

### 8. Final Layer 0 Output
```
rms=0.015003, min=-0.078455, max=0.094904, mean=-0.000174
```

**Analysis:**
- After residual connection, values are back to ~0.015 RMS
- Similar to input (0.009), slightly larger
- This seems reasonable for Layer 0 output

## Potential Issues Identified

### Issue 1: SwiGLU Output Too Small?
- SwiGLU output: RMS=0.001938
- This is very small compared to gate/up projections (RMS~0.06)
- Need to verify SwiGLU implementation: `SwiGLU(gate, up) = gate * SiLU(up)`
- SiLU(x) = x * sigmoid(x) can produce small values if inputs are negative

### Issue 2: FFN Down Projection
- Down weight has large values (max=0.340576)
- But SwiGLU input is small (RMS=0.002)
- Result is still small (RMS=0.003)
- Is this the correct behavior?

## Next Steps

1. **Verify SwiGLU Implementation**
   - Check if `gate * silu(up)` is correct
   - Should it be `silu(gate) * up` instead?
   - Compare with llama.cpp implementation

2. **Compare with llama.cpp Layer 0 Output**
   - Run llama.cpp with verbose logging
   - Extract Layer 0 statistics
   - Compare RMS values at each step

3. **Check Gate vs Up Usage**
   - Verify which projection uses gate vs up
   - Confirm SwiGLU formula matches reference implementation

4. **Test with F32 Model**
   - Eliminate quantization as variable
   - Run same test with fully F32 weights

## Observations

1. **RMSNorm is mathematically correct** but causes value amplification
2. **Attention mechanism correctly handles** the amplified values
3. **FFN produces very small values** - this might be the root cause
4. **Value propagation** through 22 layers with these small FFN outputs could accumulate errors

## Critical Question

**Why does single BOS token produce "diplom" at rank 6?**

If Layer 0-21 all have similar behavior:
- Small FFN outputs (RMS~0.003)
- Residual connections maintain signal (RMS~0.015)
- After 22 layers, what is the final hidden state RMS?

From previous logs:
```
🔍 [FINAL NORM] After output_norm (last token): rms=1.920277, max=7.224306
```

This is **very large** compared to Layer 0 output (RMS=0.015).
Where does this 100x amplification come from?

## Hypothesis

The value growth from Layer 0 (RMS=0.015) → Final (RMS=1.920) suggests:
- Either residual connections accumulate signal over 22 layers
- Or some layer(s) produce large outputs that dominate
- Or the final RMSNorm amplifies due to small RMS values

Need to track RMS growth across all 22 layers to identify where amplification occurs.
