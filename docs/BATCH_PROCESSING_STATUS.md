# Batch Processing Implementation Status

## Overview
This document tracks the implementation of batch processing optimization for the RusTorch Llama model, enabling parallel inference of multiple sequences.

## Completed Work

### Phase 1: Infrastructure Setup ✅

#### 1. KVCache Batch Support
- **Location**: [src/models/llama.rs:72-113](../src/models/llama.rs#L72-L113)
- **Changes**:
  - Updated KVCache structure from 2D to 3D arrays
  - Structure: `[num_layers][batch_size][max_tokens, kv_dim]`
  - Added `batch_size` field to track capacity
  - Changed `cached_tokens` from single counter to per-batch vector
  - Added `clear_batch(batch_idx)` method for individual batch clearing

#### 2. Configurable Batch Size
- **Location**: [src/models/llama.rs:17-59](../src/models/llama.rs#L17-L59)
- **Changes**:
  - Added `batch_size: usize` field to `LlamaConfig`
  - Default value: 1 (maintains backward compatibility)
  - Updated `from_model_params()` to initialize with default batch_size
  - KVCache initialization now uses `config.batch_size`

#### 3. Batch API Implementation
- **Location**: [src/models/llama.rs:209-273](../src/models/llama.rs#L209-L273)
- **New Methods**:
  ```rust
  pub fn forward_batch(&mut self, input_ids_batch: &[&[usize]]) -> RusTorchResult<Vec<Tensor<f64>>>
  pub fn forward_batch_with_position(&mut self, input_ids_batch: &[&[usize]], start_position: usize) -> RusTorchResult<Vec<Tensor<f64>>>
  fn forward_batch_metal(&mut self, input_ids_batch: &[&[usize]], start_position: usize) -> RusTorchResult<Vec<Tensor<f64>>>
  ```

#### 4. Implementation Details
**forward_batch_metal** (current implementation):
- Validates batch size against KVCache capacity
- Returns error if batch_size exceeds allocated capacity
- Processes each sequence individually using existing forward_metal()
- Clears KVCache for each batch item before processing
- Maintains correctness while preparing for future optimization

**Error Handling**:
- Empty batch validation
- Batch size capacity checking with clear error messages
- Maintains existing error propagation patterns

#### 5. Example and Documentation
- **Location**: [examples/batch_inference_demo.rs](../examples/batch_inference_demo.rs)
- Demonstrates batch API usage
- Shows current limitations and performance characteristics
- Documents next steps for optimization

## Test Results

### Build Status
```
✅ Cargo build successful (warnings only)
✅ Example compiles successfully
✅ Single-sequence inference still working
```

### Runtime Testing
```
✅ Model loads with default batch_size=1
✅ Batch size validation works correctly
⚠️  Cannot change batch_size after model loading (as expected)
```

## Current Limitations

1. **Sequential Processing**
   - `forward_batch_metal` processes sequences one at a time
   - No performance benefit from batching yet
   - Correctness preserved, optimization deferred

2. **Fixed Batch Size**
   - batch_size set at model creation time
   - Cannot dynamically adjust after loading
   - Requires recreating model to change batch_size

3. **Metal Kernel Constraints**
   - All Metal kernels assume single sequence:
     - RMS Norm
     - Matrix multiplication
     - RoPE (Rotary Position Embedding)
     - Attention computation

4. **API Limitations**
   - `from_gguf_with_backend()` doesn't expose config customization
   - No way to set batch_size before loading weights
   - Need new API: `from_gguf_with_config(path, config)`

## Pending Work

### Phase 2: Metal Kernel Batch Support (未着手)

#### Required Kernel Updates
Each kernel needs batch dimension support:

1. **RMS Norm Kernel** - `src/gpu/metal_shaders.metal`
   - Current: `(seq_len, hidden_dim)`
   - Target: `(batch_size, seq_len, hidden_dim)`
   - Thread organization: 2D grid for batch parallelism

2. **Matrix Multiplication** - `src/gpu/metal_kernels.rs`
   - Current: Single 2D matmul
   - Target: Batched matmul
   - May already support batch dim (needs verification)

3. **RoPE Kernel** - `src/gpu/metal_shaders.metal`
   - Current: Single sequence position encoding
   - Target: Batch-aware position tracking
   - Handle different sequence lengths per batch

4. **Attention Kernel** - `src/gpu/metal_shaders.metal`
   - Current: Single Q/K/V attention
   - Target: Batched attention computation
   - KVCache indexing for batch dimension

#### Implementation Strategy
```rust
// Proposed kernel signature changes
fn rms_norm_metal(
    input: &[f32],          // [batch_size, seq_len, hidden_dim]
    output: &mut [f32],     // [batch_size, seq_len, hidden_dim]
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
) -> RusTorchResult<()>
```

### Phase 3: True Parallel Batch Processing (未着手)

#### forward_batch_metal Optimization
Once kernels support batching:

1. **Concatenate Inputs**
   ```rust
   // Combine all sequences into single batch tensor
   let combined_len: usize = input_ids_batch.iter().map(|s| s.len()).sum();
   let batch_embeddings = ...; // [batch_size, seq_len, hidden_dim]
   ```

2. **Batch Processing**
   ```rust
   // Single pass through all layers for entire batch
   for layer_idx in 0..self.config.num_layers {
       batch_hidden = self.process_layer_batch(batch_hidden, layer_idx)?;
   }
   ```

3. **Split Outputs**
   ```rust
   // Separate batch results into individual tensors
   let outputs: Vec<Tensor<f64>> = split_batch_output(batch_hidden, input_ids_batch);
   ```

### Phase 4: API Enhancements (未着手)

#### Config Customization
```rust
// Add new loading method
impl LlamaModel {
    pub fn from_gguf_with_config<P: AsRef<Path>>(
        path: P,
        config: LlamaConfig,
    ) -> RusTorchResult<Self> {
        // Allow full config customization before loading weights
    }
}
```

#### Dynamic Batch Resizing
```rust
impl LlamaModel {
    pub fn set_batch_size(&mut self, new_batch_size: usize) -> RusTorchResult<()> {
        // Reallocate KVCache with new batch_size
        // Preserve existing weights
    }
}
```

## Performance Goals

### Target Improvements
- **Throughput**: 3-4x improvement for batch_size=4
- **Latency**: Minimal increase per sequence
- **Memory**: Linear scaling with batch_size
- **GPU Utilization**: >80% for batch_size >= 4

### Benchmarking Plan
```rust
// Planned benchmark scenarios
- Batch sizes: [1, 2, 4, 8, 16]
- Sequence lengths: [1, 16, 128, 512]
- Quantization types: Q4_K_M, Q5_K_M, Q6_K, Q8_0
- Measure: tokens/second, memory usage, GPU utilization
```

## Architecture Decisions

### Why Incremental Approach?
1. **Risk Mitigation**: Maintain working single-sequence inference
2. **Testing**: Each phase can be tested independently
3. **Flexibility**: Can pause/resume based on priorities
4. **Learning**: Understand bottlenecks before optimization

### Why Not Immediate Parallel Processing?
1. Metal kernels are complex (RoPE, GQA attention)
2. Need careful thread organization for batch parallelism
3. Risk of introducing subtle correctness bugs
4. Sequential fallback provides reference implementation

## Related Files
- [src/models/llama.rs](../src/models/llama.rs) - Main model implementation
- [src/gpu/metal_kernels.rs](../src/gpu/metal_kernels.rs) - Metal kernel wrappers
- [src/gpu/metal_shaders.metal](../src/gpu/metal_shaders.metal) - GPU compute kernels
- [examples/batch_inference_demo.rs](../examples/batch_inference_demo.rs) - Usage example

## Commit History
- `07c8315b2` - feat: Add batch processing infrastructure for Llama model
- `5f016d116` - feat: Implement batch support for KVCache structure

## Next Immediate Steps

1. **Verify Metal Matmul Batch Support**
   - Check if current matmul kernel already handles batch dimension
   - Test with batched input tensors
   - Document findings

2. **Design Batch Kernel API**
   - Define kernel signatures for batch operations
   - Plan thread organization (2D/3D grids)
   - Consider memory coalescing patterns

3. **Implement RMS Norm Batch Kernel**
   - Start with simplest kernel
   - Add batch dimension to shader
   - Test correctness vs CPU reference

4. **Create Batch Processing Tests**
   - Unit tests for each batch kernel
   - Integration test for full forward pass
   - Performance benchmarks

---

**Status**: Phase 1 Complete ✅ | Phase 2-4 Pending ⏳
**Last Updated**: 2025-10-10
**Maintainer**: Batch Processing Working Group
