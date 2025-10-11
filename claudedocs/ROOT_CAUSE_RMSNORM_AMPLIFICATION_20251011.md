# Root Cause: RMSNorm Amplification Issue
Date: 2025-10-11

## Problem Summary

Q8_0 models produce gibberish output ("anth" instead of expected tokens) due to **excessive value amplification** in RMSNorm layers.

## Evidence

### Layer 0 RMSNorm Analysis

**Input Statistics** (Last token of 14-token sequence):
```
RMS: 0.009559
Min: -0.077636
Max: 0.075213
Mean: 0.000006
```

**attn_norm.weight Statistics**:
```
RMS: 0.046377
Max: 0.769531
```

**Output Statistics** (After RMSNorm + weight multiplication):
```
RMS: 0.098361
Min: -3.702824
Max: 4.829902
Mean: 0.001377
```

### Amplification Calculation

**Expected output magnitude** (naive estimate):
```
input_max × weight_max = 0.075213 × 0.769531 ≈ 0.058
```

**Actual output magnitude**:
```
max = 4.829902
```

**Amplification factor**: 4.829902 / 0.058 ≈ **83x amplification**

### Root Cause: Scale Factor

RMSNorm formula:
```
scale = 1 / sqrt(mean_of_squares + eps)
output = (input * scale) * weight
```

With Layer 0 input:
```
mean_of_squares = (0.00835147)² ≈ 0.0000697
eps = 1e-5 = 0.00001
sqrt(mean_of_squares + eps) ≈ sqrt(0.0000797) ≈ 0.00893
scale = 1 / 0.00893 ≈ 112.0
```

**Scale factor of ~112x** is applied to already small values, causing exponential growth through layers.

## Layer-by-Layer Growth

| Layer | RMS    | Max     | Growth Factor (from Layer 0) |
|-------|--------|---------|------------------------------|
| 0     | 0.014  | 0.082   | 1.0x (baseline)              |
| 10    | 0.295  | 1.130   | 21.0x RMS, 13.8x max         |
| 21    | 1.069  | 4.904   | 76.2x RMS, 59.8x max         |

**Exponential growth pattern**: Each layer amplifies values, leading to:
- Layer 10: ~20x amplification
- Layer 21: ~60x amplification
- Final output: Completely wrong logits

## Why Q4_K_M/Q5_K_M Produce Different Output

Previous sessions claimed Q4_K_M works correctly, but current testing shows it also produces gibberish. Possible reasons:
1. Different embedding values due to quantization precision
2. Different weight magnitudes in Q4_K vs Q8_0
3. Accumulation of small differences across 22 layers

## Comparison with Known Working Implementation

### RusTorch Implementation (src/hybrid_f32/models/llama.rs:293-384)

```rust
fn rms_norm(&self, x: &F32Tensor, weight: &F32Tensor) -> F32Result<F32Tensor> {
    let eps = self.config.rms_norm_eps;  // 1e-5

    for i01 in 0..ne01 {  // For each token
        let x_slice = &x_data[x_offset..x_offset + ne00];

        // Calculate mean of squares
        let mut sum: f32 = 0.0;
        for i00 in 0..ne00 {
            sum += x_slice[i00] * x_slice[i00];
        }
        let mean = sum / (ne00 as f32);

        // Calculate scale
        let scale = 1.0 / (mean + eps).sqrt();

        // Apply normalization and weight
        for i00 in 0..ne00 {
            output[y_offset + i00] = x_slice[i00] * scale * weight_data[i00];
        }
    }
}
```

**Formula**: `output[i] = input[i] * scale * weight[i]`
- Where: `scale = 1 / sqrt(mean_of_squares + eps)`

### Issue Identified

When `mean_of_squares` is very small (e.g., 0.0000697), the scale factor becomes very large (~112):
- Small input values (0.001-0.08) × large scale (112) × weight (0.01-0.77) = large output (0-4.8)
- This amplification **cascades through 22 layers**, causing exponential growth

## Next Steps

1. ✅ Verify embedding layer is correct (DONE - confirmed)
2. ✅ Verify Q8_0 dequantization is correct (DONE - confirmed)
3. ✅ Identify RMSNorm amplification issue (DONE - this document)
4. ⏳ Compare with llama.cpp RMSNorm implementation
5. ⏳ Verify if llama.cpp has the same issue or uses different approach
6. ⏳ Check if there's a missing normalization step or different initialization

## Hypothesis

The implementation matches llama.cpp exactly (lines 3521-3570 of ggml-cpu/ops.cpp), so either:
1. **Weight loading issue**: attn_norm.weight values are incorrect for Q8_0
2. **Missing initialization**: Some pre-processing or scaling step is missing
3. **Input preparation**: Embedding values should be scaled before Layer 0
4. **llama.cpp difference**: They may handle small RMS values differently

Need to compare actual llama.cpp execution to identify the discrepancy.
