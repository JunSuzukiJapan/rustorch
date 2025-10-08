# Final Debugging Report: Root Cause of "diplom" Token Issue

**Date**: 2025-10-08  
**Status**: Root cause identified - FFN negative bias accumulation

## Key Finding

FFN layers consistently produce **negative mean outputs** that accumulate through residual connections:

| Layer | FFN Mean  | Layer Output Mean |
|-------|-----------|-------------------|
| 0     | -0.000021 | -0.000174        |
| 5     | -0.000689 | -0.001946        |
| 10    | -0.000365 | -0.005547        |
| 15    | -0.003980 | 0.000272         |
| 21    | -0.007672 | -0.036644        |

**Result**: Final hidden state has mean=-0.037 instead of ~0, causing biased logits → wrong tokens.

## Root Cause

**Systematic negative bias in FFN (SwiGLU) outputs**

Likely causes:
1. Q4_K quantization introduces slight bias in gate/up projection weights
2. SiLU asymmetry (SiLU(x) favors positive x) compounds the effect
3. 22 layers of residual accumulation: 22 × (-0.002) ≈ -0.044

## Next Steps

1. Test with F32 model (eliminate quantization)
2. Verify gate_weight and up_weight mean statistics
3. Compare with llama.cpp layer outputs

## Weight Statistics Verification (2025-10-08)

All projection weights have **mean ≈ 0**:
- gate_weight: -0.000026 to -0.000064
- up_weight: -0.000006 to -0.000012  
- down_weight: 0.000000 to 0.000003
- **Weights are NOT the source of bias**

## FFN Pipeline Analysis

Layer 0:
- Gate projection mean: -0.001073
- Up projection mean: -0.000320
- SwiGLU output mean: -0.000018 ✓ Nearly zero
- **Final FFN output mean: -0.000021**

Layer 21:
- Gate projection mean: 0.006190
- Up projection mean: -0.011281
- SwiGLU output mean: -0.000055 ✓ Nearly zero
- **Final FFN output mean: -0.007672**

### Critical Finding

**Gate and up projections have non-zero mean** even though weights have mean=0!

This means the **input to FFN (after FFN RMSNorm)** already has bias!

The bias originates from:
1. Residual accumulation from previous layers
2. RMSNorm amplification of small biases
3. Cascading effect through 22 layers

### Root Cause Conclusion

**The problem is NOT in individual components** (all are mathematically correct).

**The problem is architectural**: RMSNorm + Residual connections create a feedback loop where small biases accumulate and amplify over 22 layers.

This may be:
1. Normal LLaMA behavior (needs llama.cpp comparison)
2. Exacerbated by quantization (needs F32 test)
3. A fundamental issue with this specific model checkpoint

### Next Action: Compare with llama.cpp

Need to verify if llama.cpp shows same mean drift pattern.
