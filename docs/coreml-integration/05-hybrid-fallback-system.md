# CoreML + GPU ハイブリッド実行システム詳細設計

> **作成日**: 2025-09-19
> **目的**: CoreML非対応演算のGPUフォールバック機能実装計画
> **対象**: Feature flag によるデバイス優先順位制御システム

## 📋 概要

CoreMLで対応できない演算を自動的にGPU → CPUの順でフォールバックする動的実行システムの設計。

### 🎯 プラットフォーム別実行優先順位

#### **Apple Silicon Mac (M1/M2/M3)**
```
CoreML → Metal → OpenCL → CPU
   ↓       ↓       ↓      ↓
 最高速   高速化   普通   確実
```

#### **Intel/AMD PC with NVIDIA GPU**
```
CUDA → OpenCL → CPU
  ↓      ↓      ↓
高速化   普通   確実
```

#### **Intel/AMD PC without NVIDIA GPU**
```
OpenCL → CPU
   ↓      ↓
 普通   確実
```

**🔍 重要**: CUDA（NVIDIA）とMetal（Apple）は物理的に異なるハードウェアなので、プラットフォーム検出により適切なチェーンを動的構築します。

---

## 🏗️ 1. Feature Flag 拡張設計

### 1.1 新規 Feature Flags

```toml
[features]
# 既存 features...
coreml = ["dep:objc2-core-ml", "dep:metal"]
coreml-hybrid = ["coreml", "metal", "cuda"]  # CoreML + GPU ハイブリッド
coreml-fallback = ["coreml-hybrid"]          # フォールバック機能有効
gpu-priority = ["cuda", "metal", "opencl"]   # GPU優先実行
```

### 1.2 条件コンパイル戦略

```rust
// Cargo.toml での条件有効化
[dependencies]
objc2-core-ml = { version = "0.2", optional = true }

// 実行時の機能検出
#[cfg(feature = "coreml")]
use objc2_core_ml::*;

#[cfg(any(feature = "cuda", feature = "metal", feature = "opencl"))]
use crate::gpu::*;
```

---

## 🎛️ 2. DeviceType 拡張

### 2.1 新しいデバイスタイプ

```rust
/// 拡張デバイスタイプ - フォールバック対応
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu,
    Cuda(usize),
    Metal(usize),
    OpenCL(usize),

    // 🆕 CoreML追加
    CoreML(usize),

    // 🆕 ハイブリッドデバイス
    CoreMLHybrid {
        coreml_id: usize,
        fallback_gpu: Option<GpuDevice>,
    },

    // 🆕 自動選択
    Auto,  // 利用可能な最高性能デバイスを自動選択
}

/// GPU デバイス詳細
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuDevice {
    Cuda(usize),
    Metal(usize),
    OpenCL(usize),
}
```

### 2.2 デバイス能力の管理

```rust
/// デバイス能力マトリクス
#[derive(Debug)]
pub struct DeviceCapability {
    pub device_type: DeviceType,
    pub supports_f16: bool,
    pub supports_f32: bool,
    pub supports_f64: bool,
    pub supports_complex: bool,
    pub supports_distributed: bool,
    pub max_memory_gb: f32,
    pub supported_operations: HashSet<OpType>,
}

/// 演算タイプ分類
#[derive(Debug, Hash, PartialEq, Eq)]
pub enum OpType {
    // CoreML完全対応
    LinearAlgebra,
    Convolution,
    Activation,

    // CoreML部分対応
    Reduction,
    Normalization,

    // CoreML非対応
    ComplexMath,
    Distribution,
    CustomKernel,
    DistributedOps,
}
```

---

## ⚙️ 3. ハイブリッド実行エンジン

### 3.1 動的フォールバック機構

```rust
/// ハイブリッド実行管理
pub struct HybridExecutor {
    primary_device: DeviceType,
    fallback_devices: Vec<DeviceType>,
    capability_cache: HashMap<DeviceType, DeviceCapability>,
    operation_routing: HashMap<OpType, Vec<DeviceType>>,
}

impl HybridExecutor {
    /// 演算に最適なデバイスを選択
    pub fn select_device(&self, op_type: OpType, tensor_info: &TensorInfo) -> DeviceType {
        // 1. CoreMLで対応可能かチェック
        if self.is_coreml_supported(op_type, tensor_info) {
            return DeviceType::CoreML(0);
        }

        // 2. GPUフォールバック
        if let Some(gpu_device) = self.select_gpu_device(op_type, tensor_info) {
            return gpu_device;
        }

        // 3. CPUフォールバック
        DeviceType::Cpu
    }

    /// CoreML対応可否判定
    fn is_coreml_supported(&self, op_type: OpType, tensor_info: &TensorInfo) -> bool {
        match op_type {
            OpType::LinearAlgebra | OpType::Convolution | OpType::Activation => {
                // データ型チェック
                matches!(tensor_info.dtype, DType::F16 | DType::F32)
                    && tensor_info.shape.len() <= 5  // 5次元まで対応
            },
            OpType::Reduction | OpType::Normalization => {
                // 制限付き対応
                tensor_info.shape.len() <= 4
                    && !tensor_info.requires_custom_kernel
            },
            _ => false,  // その他は非対応
        }
    }
}
```

### 3.2 実行時デバイス切り替え

```rust
/// デバイス間の動的切り替え
impl<T> Tensor<T> {
    /// ハイブリッド演算実行
    pub fn hybrid_operation<F>(&self, op: F) -> RusTorchResult<Tensor<T>>
    where
        F: Fn(&Self, DeviceType) -> RusTorchResult<Tensor<T>>,
    {
        let executor = HybridExecutor::global();
        let op_type = self.infer_operation_type();

        // デバイス選択
        let selected_device = executor.select_device(op_type, &self.tensor_info());

        // 実行試行とフォールバック
        match self.try_operation(&op, selected_device) {
            Ok(result) => Ok(result),
            Err(e) if e.is_device_error() => {
                // フォールバックデバイスで再試行
                let fallback_device = executor.next_fallback_device(selected_device);
                self.try_operation(&op, fallback_device)
            },
            Err(e) => Err(e),
        }
    }
}
```

---

## 🔄 4. 演算別フォールバック戦略

### 4.1 線形代数演算

```rust
impl<T> GpuLinearAlgebra<T> for Tensor<T> {
    fn gpu_matmul(&self, other: &Self) -> RusTorchResult<Tensor<T>> {
        // 🎯 ハイブリッド実行
        self.hybrid_operation(|tensor, device| {
            match device {
                DeviceType::CoreML(_) => {
                    // CoreML MLComputeGraph使用
                    tensor.coreml_matmul(other)
                },
                DeviceType::Cuda(_) => {
                    // CUDA cuBLAS使用
                    tensor.cuda_matmul(other)
                },
                DeviceType::Metal(_) => {
                    // Metal Performance Shaders使用
                    tensor.metal_matmul(other)
                },
                DeviceType::Cpu => {
                    // BLAS CPU実装
                    tensor.cpu_matmul(other)
                },
                _ => Err(RusTorchError::UnsupportedDevice),
            }
        })
    }
}
```

### 4.2 複素数演算（CoreML非対応）

```rust
impl<T> ComplexOperations<T> for Tensor<Complex<T>> {
    fn complex_multiply(&self, other: &Self) -> RusTorchResult<Self> {
        // CoreMLスキップ → GPU直行
        let executor = HybridExecutor::global();

        if executor.has_gpu_support() {
            // GPU実装
            self.gpu_complex_multiply(other)
        } else {
            // CPU フォールバック
            self.cpu_complex_multiply(other)
        }
    }
}
```

### 4.3 分散演算（CoreML非対応）

```rust
impl<T> DistributedOperations<T> for Tensor<T> {
    fn all_reduce(&self, op: ReduceOp) -> RusTorchResult<Self> {
        // CoreML非対応 → NCCL/MPI直行
        #[cfg(feature = "nccl")]
        {
            self.nccl_all_reduce(op)
        }
        #[cfg(not(feature = "nccl"))]
        {
            // シングルプロセスの場合はno-op
            Ok(self.clone())
        }
    }
}
```

---

## 📊 5. パフォーマンス最適化

### 5.1 デバイス選択ヒューリスティック

```rust
/// 最適デバイス選択ルール
impl DeviceSelector {
    fn select_optimal_device(&self, tensor_size: usize, op_complexity: f32) -> DeviceType {
        // 小さなテンソル（< 1MB）→ CPU
        if tensor_size < 1_000_000 {
            return DeviceType::Cpu;
        }

        // 中サイズ（1MB - 100MB）→ CoreML
        if tensor_size < 100_000_000 && self.coreml_available() {
            return DeviceType::CoreML(0);
        }

        // 大サイズ（> 100MB）→ 最高性能GPU
        self.best_gpu_device().unwrap_or(DeviceType::Cpu)
    }
}
```

### 5.2 メモリ転送最適化

```rust
/// デバイス間メモリ転送管理
pub struct MemoryManager {
    device_pools: HashMap<DeviceType, MemoryPool>,
    transfer_cache: LruCache<(DeviceType, DeviceType), TransferMethod>,
}

impl MemoryManager {
    /// 最適な転送方法選択
    fn optimize_transfer(&self, from: DeviceType, to: DeviceType) -> TransferMethod {
        match (from, to) {
            (DeviceType::Metal(_), DeviceType::CoreML(_)) => {
                // Metal ↔ CoreML: ゼロコピー
                TransferMethod::ZeroCopy
            },
            (DeviceType::Cuda(_), DeviceType::CoreML(_)) => {
                // CUDA → CoreML: ホスト経由
                TransferMethod::HostStaging
            },
            _ => TransferMethod::Standard,
        }
    }
}
```

---

## 🧪 6. テスト戦略

### 6.1 機能テスト

```rust
#[cfg(test)]
mod hybrid_tests {
    use super::*;

    #[test]
    fn test_coreml_fallback_to_gpu() {
        // CoreML非対応演算のGPUフォールバック
        let tensor = Tensor::randn(&[1000, 1000], DType::Complex64);

        // 複素数演算（CoreML非対応）
        let result = tensor.complex_conjugate();

        // GPUまたはCPUで実行されることを確認
        assert!(matches!(result.device(), DeviceType::Cuda(_) | DeviceType::Cpu));
    }

    #[test]
    fn test_gpu_fallback_to_cpu() {
        // GPU非利用時のCPUフォールバック
        std::env::set_var("RUSTORCH_DISABLE_GPU", "1");

        let tensor = Tensor::randn(&[100, 100], DType::F32);
        let result = tensor.matmul(&tensor);

        assert_eq!(result.device(), DeviceType::Cpu);
    }
}
```

### 6.2 性能テスト

```rust
#[cfg(test)]
mod performance_tests {
    use criterion::{black_box, Criterion};

    fn benchmark_hybrid_execution(c: &mut Criterion) {
        let tensor = Tensor::randn(&[1000, 1000], DType::F32);

        c.bench_function("hybrid_matmul", |b| {
            b.iter(|| {
                black_box(tensor.matmul(&tensor))
            })
        });
    }
}
```

---

## 🚀 7. 実装フェーズ

### Phase 1: 基盤実装 (2-3週間)
- [ ] DeviceType拡張
- [ ] HybridExecutor基本機能
- [ ] Feature flag設定

### Phase 2: コア演算対応 (3-4週間)
- [ ] 線形代数のハイブリッド実行
- [ ] 畳み込みのフォールバック
- [ ] アクティベーション関数対応

### Phase 3: 高度な機能 (2-3週間)
- [ ] 複素数演算GPU専用実装
- [ ] 分散演算の条件分岐
- [ ] パフォーマンス最適化

---

## 🎯 期待される効果

### 性能向上
- **CoreML対応演算**: 50-80% 高速化
- **GPU フォールバック**: 20-40% 高速化
- **自動最適化**: ユーザー介入不要

### 開発者体験
- **Zero Configuration**: feature flag設定のみ
- **Graceful Degradation**: エラーではなく自動フォールバック
- **透明性**: どのデバイスで実行されたかログで確認可能