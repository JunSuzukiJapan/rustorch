# RusTorch Refactoring Roadmap
# RusTorchリファクタリング・ロードマップ

## 📊 Current State / 現状
- **171 Rust files** / 171個のRustファイル
- **~72,709 lines of code** / 約72,709行のコード  
- **Major Issues** / 主要課題:
  - Duplicate functionality across modules / モジュール間の重複機能
  - Large, complex files / 大きく複雑なファイル
  - Inconsistent APIs / 一貫性のないAPI
  - Missing abstractions / 抽象化の不足

## 🎯 Refactoring Strategy / リファクタリング戦略

### Phase 1: Critical Infrastructure (v0.4.0) / フェーズ1: 重要インフラ
**Duration**: 3-4 weeks / 期間: 3-4週間
**Impact**: Foundation for all future improvements / 将来の改善の基盤

#### 1.1 Backend Abstraction Layer / バックエンド抽象化レイヤー
**Priority**: 🔴 Critical / 最重要

**Current Issue** / 現在の問題:
```
src/gpu/cuda_kernels.rs     (GPU CUDA implementation)
src/gpu/metal_kernels.rs    (GPU Metal implementation)  
src/gpu/opencl_kernels.rs   (GPU OpenCL implementation)
↓ Separate, duplicated implementations
```

**Target Architecture** / 目標アーキテクチャ:
```rust
// New unified backend system
pub trait ComputeBackend {
    fn device_info(&self) -> DeviceInfo;
    fn allocate_memory(&self, size: usize) -> Result<DeviceBuffer>;
    fn matrix_multiply(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn elementwise_add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn convolution(&self, input: &Tensor, kernel: &Tensor, params: ConvParams) -> Result<Tensor>;
}

// Backend implementations
struct CpuBackend { /* SIMD optimizations */ }
struct CudaBackend { /* CUDA kernels */ }
struct MetalBackend { /* Metal shaders */ }
struct OpenCLBackend { /* OpenCL kernels */ }
```

**Implementation Plan** / 実装計画:
- [ ] Define `ComputeBackend` trait with core operations
- [ ] Extract common GPU operations into trait methods  
- [ ] Implement CPU backend with SIMD optimizations
- [ ] Migrate CUDA/Metal/OpenCL to unified interface
- [ ] Add backend selection and fallback logic
- [ ] Update Tensor to use backend abstraction

#### 1.2 Tensor Operations Split / テンソル演算の分割
**Priority**: 🔴 Critical / 最重要

**Current Issue** / 現在の問題:
```
src/tensor/operations.rs (2,490 lines) - TOO LARGE
```

**Target Structure** / 目標構造:
```
src/tensor/
├── operations/
│   ├── mod.rs              (public API)
│   ├── arithmetic.rs       (element-wise ops: +, -, *, /)
│   ├── linear_algebra.rs   (matmul, svd, qr, lu, eig)
│   ├── reduction_ops.rs    (sum, mean, max, min)
│   ├── shape_ops.rs        (reshape, transpose, permute)
│   ├── statistical.rs      (var, std, median, quantile)
│   ├── broadcasting.rs     (broadcast logic)
│   └── fft.rs             (fourier transforms)
└── ...
```

**Implementation Plan** / 実装計画:
- [ ] Create operations module directory
- [ ] Split operations.rs by functionality categories
- [ ] Maintain backward compatibility through re-exports
- [ ] Update tests for new module structure
- [ ] Add comprehensive documentation

#### 1.3 GPU Kernel Consolidation / GPUカーネル統合
**Priority**: 🟠 High / 高優先度

**Current Issue** / 現在の問題:
- 3 separate kernel implementations with duplicated logic
- Inconsistent error handling across GPU backends
- No shared optimization strategies

**Implementation Plan** / 実装計画:
- [ ] Create `KernelExecutor` trait for common operations
- [ ] Extract shared GPU memory management
- [ ] Implement unified kernel compilation pipeline
- [ ] Add performance benchmarking across backends

### Phase 2: Module Organization (v0.5.0) / フェーズ2: モジュール組織化  
**Duration**: 4-5 weeks / 期間: 4-5週間
**Impact**: Clean, maintainable codebase / クリーンで保守可能なコードベース

#### 2.1 Neural Network Layer Traits / ニューラルネットワーク層トレイト
**Priority**: 🟠 High / 高優先度

**Current Issue** / 現在の問題:
```
src/nn/ (28 files) - Inconsistent layer implementations
- conv1d.rs, conv2d.rs, conv3d.rs (similar but separate)
- No shared parameter initialization
- Inconsistent forward() signatures
```

**Target Architecture** / 目標アーキテクチャ:
```rust
// Base layer traits
pub trait Layer {
    type Input;
    type Output;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn zero_grad(&mut self);
}

pub trait ParameterizedLayer: Layer {
    fn parameter_count(&self) -> usize;
    fn init_parameters(&mut self, init: InitStrategy);
}

// Convolution trait hierarchy  
pub trait ConvolutionLayer: ParameterizedLayer {
    fn kernel_size(&self) -> &[usize];
    fn stride(&self) -> &[usize];
    fn padding(&self) -> &[usize];
}
```

#### 2.2 Device Management Refactoring / デバイス管理リファクタリング
**Priority**: 🟠 High / 高優先度

**Current Issue** / 現在の問題:
```
src/gpu/device.rs (879 lines) - Multiple responsibilities
- Device detection
- Context management  
- Memory allocation
- Performance monitoring
```

**Target Structure** / 目標構造:
```
src/gpu/
├── device/
│   ├── mod.rs              (public API)
│   ├── detection.rs        (device enumeration)
│   ├── context.rs          (GPU context management)
│   ├── selection.rs        (optimal device selection)
│   └── capabilities.rs     (feature detection)
├── memory.rs
├── kernels.rs
└── validation.rs
```

#### 2.3 Model Import/Convert Unification / モデルインポート/変換統一
**Priority**: 🟡 Medium / 中優先度

**Current Issue** / 現在の問題:
```
src/convert/      (3,058 lines across 6 files)
src/model_import/ (1,882 lines across 4 files)
↓ Overlapping functionality
```

**Target Structure** / 目標構造:  
```
src/model_io/
├── mod.rs
├── pytorch/
│   ├── import.rs           (PyTorch model loading)
│   ├── export.rs           (PyTorch model saving)
│   └── state_dict.rs       (state dictionary handling)
├── onnx/
│   ├── import.rs           (ONNX model loading)
│   └── inference.rs        (ONNX inference)
├── safetensors/
│   ├── import.rs           (Safetensors loading)
│   └── export.rs           (Safetensors saving)
└── common/
    ├── validation.rs       (model validation)
    └── conversion.rs       (format conversion utilities)
```

### Phase 3: API Consistency (v0.6.0) / フェーズ3: API一貫性
**Duration**: 2-3 weeks / 期間: 2-3週間  
**Impact**: Developer experience improvements / 開発者エクスペリエンス向上

#### 3.1 Error Handling Unification / エラーハンドリング統一
**Current Issue** / 現在の問題:
```rust
// Inconsistent error types across codebase
Result<Tensor, String>           // String errors
Result<Tensor, TensorError>      // Custom tensor errors  
Result<Tensor, Box<dyn Error>>   // Boxed errors
```

**Target Architecture** / 目標アーキテクチャ:
```rust
// Unified error system
#[derive(Debug, thiserror::Error)]
pub enum RusTorchError {
    #[error("Tensor operation failed: {message}")]
    TensorOp { message: String, source: Option<Box<dyn Error + Send + Sync>> },
    
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Device error: {message}")]
    Device { message: String },
    
    #[error("Backend not available: {backend}")]
    BackendUnavailable { backend: String },
}

pub type Result<T> = std::result::Result<T, RusTorchError>;
```

#### 3.2 SIMD Operations Consolidation / SIMD演算統合
**Current Issue** / 現在の問題:
```
src/simd/ (4 files, ~1,175 lines)
src/tensor/simd_integration.rs
src/tensor/simd_aligned.rs
src/tensor/simd_avx512.rs
↓ Scattered SIMD implementations
```

**Target Architecture** / 目標アーキテクチャ:
```
src/compute/
├── simd/
│   ├── mod.rs              (SIMD trait definitions)
│   ├── backend.rs          (SIMD backend selection)
│   ├── ops/
│   │   ├── arithmetic.rs   (vectorized math ops)
│   │   ├── reduction.rs    (vectorized reductions)
│   │   └── comparison.rs   (vectorized comparisons)
│   └── arch/
│       ├── avx2.rs         (AVX2 implementations)
│       ├── avx512.rs       (AVX512 implementations)
│       ├── neon.rs         (ARM NEON implementations)
│       └── fallback.rs     (scalar fallback)
```

#### 3.3 Memory Management Strategy / メモリ管理戦略
**Priority**: 🟡 Medium / 中優先度

**Current Issue** / 現在の問題:
- Memory allocation scattered across modules
- No unified memory pooling strategy  
- Inconsistent SIMD alignment handling

**Target Architecture** / 目標アーキテクチャ:
```rust
pub trait MemoryAllocator {
    fn allocate(&self, size: usize, alignment: usize) -> Result<*mut u8>;
    fn deallocate(&self, ptr: *mut u8, size: usize, alignment: usize);
    fn realloc(&self, ptr: *mut u8, old_size: usize, new_size: usize, alignment: usize) -> Result<*mut u8>;
}

// Allocator implementations
struct SystemAllocator;           // Standard system allocator
struct PoolAllocator;             // Memory pool for frequent allocations
struct AlignedAllocator;          // SIMD-aligned allocations
struct GpuMemoryAllocator;        // GPU memory management
```

## 📈 Expected Outcomes / 期待される結果

### Code Quality Metrics / コード品質指標
- **Code Reduction**: 15-20% reduction in total lines / 総行数15-20%削減
- **Duplication**: 80% reduction in duplicated code / 重複コード80%削減  
- **Test Coverage**: Improved through shared trait implementations / 共有トレイト実装によるテストカバレッジ向上
- **Compilation Time**: 20-30% faster due to better modularity / 優れたモジュール性により20-30%高速化

### Developer Experience / 開発者エクスペリエンス  
- **Consistent APIs**: Unified error handling and method signatures / 統一されたエラーハンドリングとメソッドシグネチャ
- **Better Documentation**: Clear module boundaries and responsibilities / 明確なモジュール境界と責任
- **Easier Testing**: Trait-based architecture enables better mocking / トレイトベースアーキテクチャによる優れたモック化
- **Performance**: Better optimization opportunities through backend abstraction / バックエンド抽象化による優れた最適化機会

### Performance Improvements / パフォーマンス向上
- **Backend Optimization**: Unified interface enables cross-backend optimizations / 統一インターフェースによるクロスバックエンド最適化  
- **Memory Efficiency**: Centralized memory management reduces fragmentation / 集中メモリ管理によるフラグメンテーション削減
- **SIMD Utilization**: Better vectorization through consolidated SIMD operations / 統合されたSIMD演算による優れたベクトル化

## 🚀 Implementation Schedule / 実装スケジュール

### Phase 1: Weeks 1-4 / フェーズ1: 1-4週目
- Week 1: Backend trait design and CPU implementation  
- Week 2: GPU backend migration (CUDA/Metal/OpenCL)
- Week 3: Tensor operations splitting  
- Week 4: Integration testing and documentation

### Phase 2: Weeks 5-9 / フェーズ2: 5-9週目  
- Week 5-6: NN layer trait hierarchy implementation
- Week 7: Device management refactoring
- Week 8: Model I/O consolidation
- Week 9: Integration and performance testing

### Phase 3: Weeks 10-12 / フェーズ3: 10-12週目
- Week 10: Error handling unification  
- Week 11: SIMD operations consolidation
- Week 12: Memory management implementation and final testing

## 🎯 Success Metrics / 成功指標

### Technical Metrics / 技術指標
- [ ] All 647+ tests continue passing / 647+個の全テストが引き続き合格
- [ ] No performance regression in benchmarks / ベンチマークでパフォーマンス回帰なし
- [ ] 20% reduction in total lines of code / 総行数20%削減
- [ ] 100% consistent error handling across modules / モジュール間で100%一貫性のあるエラーハンドリング

### Maintainability Metrics / 保守性指標  
- [ ] Average file size < 500 lines / 平均ファイルサイズ500行未満
- [ ] Zero code duplication for core operations / コア演算での重複コードゼロ
- [ ] 95% documentation coverage for public APIs / パブリックAPIで95%のドキュメントカバレッジ
- [ ] Clear module dependencies (no circular references) / 明確なモジュール依存関係（循環参照なし）

## 🔄 Migration Strategy / 移行戦略

### Backward Compatibility / 後方互換性
- Maintain public API compatibility during refactoring / リファクタリング中にパブリックAPI互換性を維持
- Use feature flags for gradual migration / 段階的移行のためのフィーチャーフラグ使用  
- Comprehensive deprecation warnings for changed APIs / 変更されたAPIの包括的な非推奨警告

### Testing Strategy / テスト戦略
- Run full test suite after each refactoring step / 各リファクタリングステップ後に完全なテストスイート実行
- Add integration tests for new trait abstractions / 新しいトレイト抽象化のための統合テスト追加
- Performance regression testing at each phase / 各フェーズでのパフォーマンス回帰テスト

### Release Strategy / リリース戦略
- Phase 1 → v0.4.0 (Major infrastructure changes) / フェーズ1 → v0.4.0（主要インフラ変更）
- Phase 2 → v0.5.0 (Module reorganization) / フェーズ2 → v0.5.0（モジュール再編成）  
- Phase 3 → v0.6.0 (API consistency) / フェーズ3 → v0.6.0（API一貫性）

This roadmap transforms RusTorch from a large, somewhat scattered codebase into a well-architected, maintainable deep learning library with clear separation of concerns and consistent abstractions.