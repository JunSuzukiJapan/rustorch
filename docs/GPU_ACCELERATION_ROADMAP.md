# GPU Acceleration Roadmap for RusTorch GPT Models
# RusTorch GPTモデルのGPUアクセラレーション ロードマップ

## Current Status (現状) - v0.6.30

### ✅ Implemented (実装済み)
- **Backend Selection Infrastructure** (バックエンド選択基盤)
  - `DeviceType` enum with CPU/CUDA/Metal/OpenCL support
  - `GPTModel::with_backend()` for explicit device selection
  - `from_gguf_with_backend()` for loading models with specific backend
  - Command-line `--backend` argument properly routed to GPTModel

- **GPU Infrastructure** (GPU基盤)
  - Comprehensive GPU module with Metal/CUDA/OpenCL support
  - `GpuActivation` trait with ReLU, Sigmoid, Tanh, GELU, etc.
  - `GpuMatrixExecutor` for GPU matrix operations
  - Metal kernels for activation functions and matrix multiplication
  - CoreML Neural Engine integration
  - Hybrid execution system with automatic fallback

- **Model Loading** (モデル読み込み)
  - GGUF weight loading (201 tensors from TinyLlama)
  - Tokenizer integration from tokenizer.json
  - Weight management in HashMap

### ⚠️ Current Limitations (現在の制限)
- **All tensor operations execute on CPU** (全テンソル演算はCPUで実行)
  - GPTModel.forward() uses CPU-only operations
  - LayerNorm: Manual CPU implementation
  - MultiheadAttention: RusTorch nn module (CPU)
  - Linear layers: RusTorch nn module (CPU)
  - GELU activation: RusTorch nn module (CPU)

- **Performance** (パフォーマンス)
  - Even 2-layer inference is slow due to CPU execution
  - No GPU acceleration despite backend selection

## Implementation Plan (実装計画)

### Phase 1: Critical Path Optimization (重要経路の最適化)
**Priority: HIGH** | **Complexity: MEDIUM** | **Impact: HIGH**

#### 1.1 LayerNorm GPU Acceleration
**File**: `src/models/gpt.rs::apply_layer_norm()`

**Current Implementation**:
```rust
// Manual CPU implementation
for b in 0..batch_size {
    for s in 0..seq_len {
        // CPU-based mean/variance calculation
        // CPU-based normalization
    }
}
```

**Target Implementation**:
```rust
fn apply_layer_norm(&self, input: &Variable<f64>, weight_key: &str, d_model: usize) -> Variable<f64> {
    match self.device_type {
        #[cfg(feature = "metal")]
        DeviceType::Metal => {
            // Use Metal GPU kernel
            self.apply_layer_norm_metal(input, weight_key, d_model)
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda => {
            // Use CUDA kernel
            self.apply_layer_norm_cuda(input, weight_key, d_model)
        }
        _ => {
            // CPU fallback
            self.apply_layer_norm_cpu(input, weight_key, d_model)
        }
    }
}
```

**Required Components**:
- [ ] Metal Shading Language LayerNorm kernel
- [ ] CUDA LayerNorm kernel (optional)
- [ ] GPU buffer management for weights
- [ ] Data transfer optimization (host ↔ device)

**Estimated Performance Gain**: 2-5x for LayerNorm operations

#### 1.2 Matrix Multiplication GPU Acceleration
**Files**: `src/nn/linear.rs`, `src/nn/attention.rs`

**Integration Points**:
1. Linear layers (fc1, fc2 in FFN, query/key/value projections)
2. Attention weight computation (Q @ K^T)
3. Attention output (attn_weights @ V)

**Approach**: Integrate existing `GpuMatrixExecutor`

**Required Work**:
- [ ] Modify `Linear::forward()` to use `GpuMatrixExecutor` when device is GPU
- [ ] Modify `MultiheadAttention::forward()` to use GPU matmul
- [ ] Ensure thread-safe GPU context sharing
- [ ] Benchmark CPU vs GPU threshold (small matrices may be faster on CPU)

**Estimated Performance Gain**: 10-50x for matrix operations

#### 1.3 Activation Function GPU Acceleration
**Files**: `src/nn/activation/gelu.rs`

**Integration**: Use existing `GpuActivation` trait

**Required Work**:
- [ ] Modify `GELU::forward()` to call `tensor.gpu_gelu()` when device is GPU
- [ ] Similar changes for other activations if used

**Estimated Performance Gain**: 3-8x for activation operations

### Phase 2: End-to-End GPU Pipeline (完全なGPUパイプライン)
**Priority: MEDIUM** | **Complexity: HIGH** | **Impact: VERY HIGH**

#### 2.1 Device-Aware Tensor Operations
**Goal**: All tensor operations respect device_type

**Changes Required**:
- Tensor creation on GPU devices
- Automatic operation routing based on device
- Minimal host-device transfers
- Memory pooling for GPU allocations

#### 2.2 Unified nn Module Interface
**Goal**: Single `forward()` interface, automatic GPU dispatch

**Design**:
```rust
pub trait Module<T> {
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        match input.device() {
            DeviceType::Cuda(_) => self.forward_cuda(input),
            DeviceType::Metal(_) => self.forward_metal(input),
            _ => self.forward_cpu(input),
        }
    }
}
```

### Phase 3: Optimizations (最適化)
**Priority: LOW** | **Complexity: MEDIUM** | **Impact: MEDIUM**

#### 3.1 Kernel Fusion
- Fuse LayerNorm + Attention
- Fuse GELU + Linear
- Reduce kernel launch overhead

#### 3.2 Memory Optimization
- In-place operations where possible
- Reuse buffers across layers
- Optimize weight transfer (load once, keep on GPU)

#### 3.3 Multi-GPU Support
- Model parallelism for large models
- Data parallelism for batch processing

## Technical Challenges (技術的課題)

### 1. nn Module Architecture
**Challenge**: Current nn modules are device-agnostic
**Solution**: Add device awareness to Module trait

### 2. autograd::Variable Integration
**Challenge**: Variable wraps CPU tensors
**Solution**: Extend Variable to support GPU tensors

### 3. Weight Management
**Challenge**: 201 GGUF weights need GPU transfer
**Solution**: Lazy loading + caching on GPU

### 4. Hybrid Execution
**Challenge**: Mix of GPU and CPU operations
**Solution**: Use existing HybridExecution system

## Quick Wins (即効性のある改善)

For immediate user value, prioritize:

1. **LayerNorm GPU Implementation** (1-2 days)
   - Most straightforward
   - Clear performance metrics
   - Demonstrates GPU usage

2. **Matrix Multiplication Integration** (2-3 days)
   - Highest performance impact
   - Existing executor available
   - Critical for attention mechanism

3. **Documentation & Messaging** (1 day)
   - Update warning messages with roadmap
   - Add performance expectations
   - Document limitations

## Measuring Success (成功指標)

### Performance Targets
- **2-layer model inference**: < 100ms per token (currently ~500ms+)
- **Full 22-layer model**: < 500ms per token
- **GPU utilization**: > 80% during forward pass

### User Experience
- Clear feedback on GPU usage
- Automatic fallback when GPU unavailable
- Predictable performance characteristics

## Contributing (貢献方法)

Developers interested in GPU acceleration can focus on:

1. **Metal Developers**: Implement LayerNorm kernel in `src/gpu/metal_kernels.rs`
2. **CUDA Developers**: Optimize matrix operations in `src/gpu/matrix_ops.rs`
3. **Architecture**: Design device-aware Variable system
4. **Testing**: Create GPU benchmark suite

## References (参考資料)

- Existing GPU implementation: `src/gpu/`
- Metal kernels: `src/gpu/metal_kernels.rs`
- Matrix operations: `src/gpu/matrix_ops.rs`
- Activation ops: `src/gpu/activation_ops.rs`
- Hybrid execution: `src/gpu/hybrid_executor.rs`

---

**Last Updated**: 2025-10-05
**Status**: Phase 0 (Infrastructure Ready, Operations on CPU)
**Next Milestone**: Phase 1.1 (LayerNorm GPU Acceleration)
