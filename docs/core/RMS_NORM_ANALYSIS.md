# RMS Norm Analysis and Verification

**Date**: 2025-10-09
**Context**: Metal GPU backend debugging - investigating apparent RMS Norm "bug"

## Summary

**Conclusion**: RMS Norm implementation is **mathematically correct**. The observed behavior (output RMS ≠ weight RMS for small inputs) is expected due to eps term in the normalization formula.

## Investigation

### Initial Observation

During Metal GPU debugging, we observed:
- Input RMS: 0.007
- Weight RMS: 0.046
- Output RMS: 0.130 (expected ~0.046)
- Amplification: 18.6x (unexpected)

This raised concerns about RMS Norm implementation correctness.

### Test Program

Created comprehensive standalone test: [examples/test_rms_norm_standalone.rs](../../examples/test_rms_norm_standalone.rs)

**Key Test Case** (Test 7):
```rust
let hidden_size = 2048;
let input: Vec<f32> = (0..hidden_size)
    .map(|i| 0.007 * ((i as f32 / 100.0).sin()))
    .collect();
let weight = vec![0.046; hidden_size]; // Uniform

rms_norm_f32(&input, &weight, &mut output, 1, hidden_size, eps);

// Results:
// Input RMS:  0.004956
// Output RMS: 0.038778 (expected: 0.046)
// Ratio: 0.8430 (84.3%)
```

**Manual calculation** produced output RMS = 0.046, suggesting RMS Norm function was wrong.

### Root Cause Discovery

The discrepancy comes from **eps handling**:

**RMS Norm formula**:
```rust
let mean_sq = input.iter().map(|&v| v * v).sum::<f32>() / hidden_size as f32;
let rms = (mean_sq + eps).sqrt();  // ← eps added BEFORE sqrt
output[i] = (input[i] / rms) * weight[i];
```

**Manual calculation** (incorrect):
```rust
let input_rms = sqrt(mean_sq);  // ← eps NOT added
let normalized = input / input_rms;
let output = normalized * weight;
```

### Mathematical Analysis

When input RMS is small relative to eps:

```
input_rms = 0.004956
eps = 0.00001

RMS Norm divisor = sqrt(0.004956² + 0.00001)
                 = sqrt(0.00002456 + 0.00001)
                 = sqrt(0.00003456)
                 = 0.00588

Normalization ratio = 0.004956 / 0.00588 = 0.843
```

**This means**:
- Normalized input has RMS = 0.843 (NOT 1.0!)
- Output = 0.843 × 0.046 = 0.038778 ✓ **Matches observed**

### Why eps Matters

The eps term `1e-5` is added to prevent division by zero:
- For large inputs (RMS >> eps): eps has negligible effect, normalized RMS ≈ 1.0
- For small inputs (RMS ~ eps): eps significantly affects divisor, normalized RMS < 1.0

**Example**:
- Input RMS = 0.1: normalized RMS = 0.1 / sqrt(0.01 + 0.00001) ≈ 0.9999 ✓
- Input RMS = 0.005: normalized RMS = 0.005 / sqrt(0.000025 + 0.00001) ≈ 0.845 ✓

## Non-Uniform Weight Amplification

Created [examples/test_nonuniform_weights.rs](../../examples/test_nonuniform_weights.rs) to understand why Layer 0 showed 18.6x amplification.

### Key Findings

**Scenario 1: Uniform weights (rms=0.046)**
- Output RMS: 0.039
- Amplification: 5.5x

**Scenario 2: Non-uniform spiky weights**
- Output RMS: 0.041
- Amplification: 5.8x

**Scenario 3: Spiky weights aligned with large input values**
- Output RMS: 0.058
- Amplification: 8.3x

**Actual model (Layer 0)**:
- Weight: mean=0.006, rms=0.046 (extreme distribution)
- Output RMS: 0.130
- Amplification: **18.6x**

### Explanation

Output RMS depends on:
1. **Weight RMS**: Higher → more amplification
2. **Weight distribution**: Spiky → more amplification
3. **Correlation**: Large weights at large input positions → more amplification

**Mathematical insight**:
```
Output RMS ≠ input_rms/divisor × weight_rms  (general case)

Instead:
Output RMS = sqrt(mean((input[i]/divisor * weight[i])²))
           = sqrt(sum((input[i] * weight[i])²) / (divisor² * N))
```

When weights correlate with input magnitude, certain positions contribute disproportionately to output RMS.

## Impact on Model Inference

### Layer-wise Amplification

Observed across 22 layers:
- Layer 0 input: RMS = 0.01
- Layer 21 output: RMS = 0.93
- **Total amplification: 93x**

Each layer amplifies by ~1.23x on average (geometric mean: 93^(1/22) ≈ 1.23)

### Attention Softmax Collapse

Large Q values (from amplification) cause softmax issues:
```
Layer 21:
Q_rms = 0.530 (grown from 0.171 in Layer 0)
K_rms = 0.020 (relatively stable)

Q·K scores: [-0.044, -0.017, ..., +0.038] ✓ Normal variation
After softmax: [0.0556, 0.0548, ..., 0.0531] ❌ Nearly uniform (~1/18)
```

**Result**: Attention weights become uniform → model pays equal attention to all tokens → gibberish output

## Conclusion

1. **RMS Norm implementation is correct** ✅
2. **eps behavior is mathematically expected** ✅
3. **Amplification comes from**:
   - Small input values relative to eps
   - Extreme weight distributions (mean=0.006, rms=0.046)
   - Weight-input correlation patterns
4. **Fix needed**: Not in RMS Norm, but in:
   - Numerical stability (value clipping)
   - Softmax stabilization (temperature scaling)
   - Possibly weight preprocessing

## References

- Test program: [examples/test_rms_norm_standalone.rs](../../examples/test_rms_norm_standalone.rs)
- Weight analysis: [examples/test_nonuniform_weights.rs](../../examples/test_nonuniform_weights.rs) (removed after analysis)
- Implementation: [src/models/gpt.rs:1162-1193](../../src/models/gpt.rs#L1162-L1193)
- Debugging status: [METAL_GPU_DEBUGGING_STATUS.md](../../METAL_GPU_DEBUGGING_STATUS.md)
