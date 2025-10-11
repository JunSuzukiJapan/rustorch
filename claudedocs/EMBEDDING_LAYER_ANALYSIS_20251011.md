# Embedding Layer Analysis - Q8_0 Investigation
Date: 2025-10-11

## RusTorch Embedding Output (Q8_0)

### Model Configuration
- Model: tinyllama-1.1b-chat-v1.0.Q8_0.gguf
- token_embd.weight shape: [2048, 32000]
- hidden_size: 2048
- vocab_size: 32000
- embed_data.len(): 65536000

### Token 1 (BOS) Embedding
```
First 10 values:
[-0.001099706, 0.001935482, -0.001671553, 0.003782988, 0.001055717,
 0.003782988, 0.000703812, -0.000175953, -0.001099706, -0.000879765]

First 20 values:
[-0.001099706, 0.001935482, -0.001671553, 0.003782988, 0.001055717,
 0.003782988, 0.000703812, -0.000175953, -0.001099706, -0.000879765,
 0.002199411, 0.005586505, -0.001143694, -0.001143694, -0.002243400,
 0.000000000, 0.000351906, -0.000307918, -0.000923753, 0.001187682]

Statistics:
- mean: 0.000025814
- RMS: 0.002229580
- min: -0.007630348
- max: 0.006326437
- non_zero: 2037/2048
```

### Token 13 Embedding
```
First 10 values:
[-0.00065255165, -0.0010254383, -0.011839151, -0.0027034283, -0.0015847683,
 0.0013051033, -0.0007457733, 0.0014915466, -0.0010254383, 0.0]

Statistics:
- RMS: 0.004657
- min: -0.062254
- max: 0.068855
- non_zero: 1999/2048
```

### Token 29896 ("1") Embedding
```
First 10 values:
[-0.005837917, -0.003361225, 0.000353813, 0.022467136, -0.004953384,
 -0.000707626, -0.000530720, 0.006014824, 0.000176907, 0.001238346]

Statistics:
- RMS: 0.008698902
- min: -0.077636
- max: 0.075213
- non_zero: 2007/2048
```

## Embedding Implementation Details

### Code Location
File: `src/hybrid_f32/models/llama.rs`
Function: `get_embedding()` (lines 625-675)

### Implementation
```rust
pub fn get_embedding(&self, token_id: u32) -> Result<Vec<f32>, String> {
    let embed_shape = self.token_embd.weight.shape();

    // Shape: [hidden_size, vocab_size] or [vocab_size, hidden_size]
    let (hidden_size, vocab_size) = if embed_shape.len() == 2 {
        let dim0 = embed_shape[0];
        let dim1 = embed_shape[1];
        if dim0 == self.config.hidden_size && dim1 == self.config.vocab_size {
            (dim0, dim1)  // [hidden_size, vocab_size]
        } else {
            (dim1, dim0)  // [vocab_size, hidden_size]
        }
    } else {
        return Err(format!("Unexpected embedding shape: {:?}", embed_shape));
    };

    if token_id >= vocab_size as u32 {
        return Err(format!("Token ID {} out of vocab range {}", token_id, vocab_size));
    }

    let embed_data = self.token_embd.weight.to_vec_f32()?;

    // Calculate start position for this token's embedding
    let start = (token_id as usize) * hidden_size;
    let end = start + hidden_size;

    if end > embed_data.len() {
        return Err(format!("Embedding slice out of bounds: start={}, end={}, len={}",
                          start, end, embed_data.len()));
    }

    let embedding = embed_data[start..end].to_vec();
    Ok(embedding)
}
```

### Key Formula
```
embedding_vector = embed_weights[token_id * hidden_size .. (token_id + 1) * hidden_size]
```

## Next Steps

1. ✅ Verified embedding layer extraction works
2. ⏳ Compare with llama.cpp embedding output (need to access raw embedding table)
3. ⏳ Test Layer 0 RMSNorm with embedding input
4. ⏳ Compare Layer 0 RMSNorm output with llama.cpp
5. ⏳ Continue layer-by-layer comparison until divergence point found

## Observations

1. **Embedding values look reasonable**:
   - Small magnitudes (~ 0.001 to 0.08 range)
   - RMS values are small but non-zero (0.002 to 0.009)
   - Most values are non-zero (97-98% non-zero rate)

2. **Shape handling**:
   - Current shape: [2048, 32000] = [hidden_size, vocab_size]
   - Indexing: `token_id * hidden_size` assumes row-major layout
   - This matches GGUF Q8_0 storage format

3. **Q8_0 Dequantization Verification** ✅:
   - Token 1's embedding starts at element 2048 (block 64)
   - Block 64 Q8_0 data from token_embd.weight:
     - scale_bits = 0x02e2
     - scale = 0.000043988
     - quants = [-25, 44, -38, 86, 24, 86, 16, -4, -25, -20, ...]
   - Dequantized values MATCH embedding output exactly:
     - Both produce: [-0.001099706, 0.001935482, -0.001671553, 0.003782988, ...]
   - Formula verification: scale * quants[0] = 0.000043988 * (-25) = -0.001099706 ✅
   - **Conclusion**: Q8_0 dequantization is CORRECT ✅
   - **Conclusion**: Embedding layer is CORRECT ✅

4. **Divergence point identified**:
   - Embedding layer produces correct values
   - Problem must occur in Layer 0 processing (RMSNorm, Attention, or FFN)
   - Next step: Verify Layer 0 RMSNorm input/output
