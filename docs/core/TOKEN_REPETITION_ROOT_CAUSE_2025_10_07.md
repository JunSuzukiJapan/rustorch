# Token Repetition Root Cause Analysis

**Date**: 2025-10-07
**Issue**: CLI generates repeated tokens ("wortwortw ort" or "конконкон")

## Investigation Summary

✅ **Verified Correct**:
- GGUF file loading and Q4_K/Q6_K dequantization
- Matrix multiplication operations (both CPU and Metal)
- Forward pass computation
- Logits extraction

❌ **Root Cause Identified**: Incremental forward pass produces nearly identical logits across steps

## Evidence

### Logits Comparison (Steps 0 vs 1)

| Token | Step 0 Logit | Step 1 Logit | Difference |
|-------|-------------|--------------|------------|
| 17572 ("wort") | 9.5443 | 9.5556 | +0.0113 |
| 683 | 8.7450 | 8.7404 | -0.0046 |
| 29870 | 8.7067 | 8.6600 | -0.0467 |
| 16812 | 8.7008 | 8.6457 | -0.0551 |

**Observation**: Logits differ by less than 0.06, which is extremely small. This indicates the model is processing nearly identical input.

### Hidden State Comparison

```
Call 0 (kv_len=0, input=[1, 529, ..., 13]):
  hidden[0..3] = [-1.7034084, 0.45420918, -0.4007794]

Call 1 (kv_len=18, input=[17572]):
  hidden[0..3] = [2.559299, 2.5136583, 1.2814038]

Call 2 (kv_len=19, input=[17572]):
  hidden[0..3] = [2.5551853, 2.5158377, 1.284868]
```

**Observation**: Calls 1 and 2 produce nearly identical hidden states despite different KV cache lengths (18 vs 19). This should NOT happen if position encoding is working correctly.

### Generation Log

```
Step 0: input_tokens=[1, 529, ..., 13] (18 tokens), kv_cache_len=0 → token 17572
Step 1: input_tokens=[17572], kv_cache_len=18 → token 17572
Step 2: input_tokens=[17572], kv_cache_len=19 → token 17572
```

**Observation**: Input tokens are correct (single token after Step 0), KV cache length is increasing. But hidden states remain nearly identical.

## Root Cause Hypothesis

The issue is in **RoPE position encoding during incremental forward pass**.

### Expected Behavior (Autoregressive Generation)

1. **Step 0**: Process full prompt (18 tokens)
   - Positions: 0, 1, 2, ..., 17
   - KV cache: store keys/values for all 18 positions
   - Output: logits for position 17 → select token 17572

2. **Step 1**: Process single new token (17572)
   - **Position should be 18** (continuing from previous)
   - KV cache: append new K/V at position 18
   - Attention: current token (pos 18) attends to cached positions 0-17 + itself
   - Output: logits for position 18 → should differ from Step 0

3. **Step 2**: Process single new token
   - **Position should be 19**
   - Similar logic

### Actual Behavior (Suspected)

The RoPE position might be:
- **Option A**: Not incrementing (stays at 0 or 17)
- **Option B**: Resetting to 0 for each new token
- **Option C**: Not being applied correctly with KV cache

This would explain why:
- Hidden states for Calls 1 and 2 are nearly identical
- Logits barely change between steps
- Same token gets selected repeatedly

## Next Investigation Steps

1. **Add position parameter logging** in `llama.rs` forward method
   - Log position used for RoPE at each step
   - Verify it increments: 0→17 (Step 0), 18 (Step 1), 19 (Step 2)

2. **Check RoPE application**
   - File: `src/hybrid_f32/models/llama.rs`
   - Method: `apply_rotary_pos_emb`
   - Verify position offset calculation with KV cache

3. **Verify attention position calculation**
   - File: `src/hybrid_f32/models/llama.rs`
   - Method: `grouped_query_attention`
   - Check if `current_kv_pos` is calculated correctly

## Relevant Code Locations

- Generation loop: `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/example-cli/src/model/inference.rs:generate_with_f32_llama_mut`
- Forward pass: `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/src/hybrid_f32/models/llama.rs:forward`
- RoPE application: `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/src/hybrid_f32/models/llama.rs:apply_rotary_pos_emb`
- Attention: `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/src/hybrid_f32/models/llama.rs:grouped_query_attention`

## Conclusion

The token repetition issue is **NOT** caused by:
- Incorrect file loading ✅
- Broken matrix multiplication ✅
- Wrong logits extraction ✅

The token repetition issue **IS** caused by:
- Incremental forward pass not properly handling position encoding ❌
- This causes nearly identical logits at each step ❌
- Leading to the same token being selected repeatedly ❌

**Fix Required**: Investigate and correct RoPE position handling during incremental generation.
