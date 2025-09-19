use crate::dtype::DType;
/// Hybrid execution engine for CoreML + GPU fallback system
/// CoreML + GPU フォールバック用ハイブリッド実行エンジン
use crate::error::{RusTorchError, RusTorchResult};
use crate::gpu::{DeviceCapability, DeviceType, GpuDevice, OpType};
use crate::gpu::smart_device_selector::{SmartDeviceSelector, OperationProfile, OperationType};
use crate::gpu::device_cache::{DeviceCache, CoreMLCache};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Tensor information for device selection
/// デバイス選択用テンソル情報
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub requires_custom_kernel: bool,
    pub memory_size_bytes: usize,
}

/// Transfer method between devices
/// デバイス間転送方法
#[derive(Debug, Clone, Copy)]
pub enum TransferMethod {
    ZeroCopy,    // Metal ↔ CoreML
    HostStaging, // CUDA → CoreML via host memory
    Standard,    // General case
}

/// Hybrid execution engine managing device selection and fallback
/// デバイス選択とフォールバックを管理するハイブリッド実行エンジン
pub struct HybridExecutor {
    primary_device: DeviceType,
    fallback_devices: Vec<DeviceType>,
    capability_cache: HashMap<DeviceType, DeviceCapability>,
    operation_routing: HashMap<OpType, Vec<DeviceType>>,

    // Performance thresholds
    small_tensor_threshold: usize, // < 1MB → CPU
    large_tensor_threshold: usize, // > 100MB → best GPU

    // Smart device selection
    smart_selector: SmartDeviceSelector,
    device_cache: DeviceCache,
}

impl HybridExecutor {
    /// Create new hybrid executor
    /// 新しいハイブリッド実行器を作成
    pub fn new() -> Self {
        // Initialize device cache and detect available devices
        let device_cache = DeviceCache::new();
        device_cache.warmup(); // Pre-populate cache

        let available_devices = Self::detect_available_devices(&device_cache);

        let mut executor = Self {
            primary_device: DeviceType::Auto,
            fallback_devices: Vec::new(),
            capability_cache: HashMap::new(),
            operation_routing: HashMap::new(),
            small_tensor_threshold: 1_000_000,   // 1MB
            large_tensor_threshold: 100_000_000, // 100MB
            smart_selector: SmartDeviceSelector::new(available_devices),
            device_cache,
        };

        executor.initialize_device_capabilities();
        executor.build_fallback_chain();
        executor.setup_operation_routing();

        executor
    }

    /// Get global singleton instance
    /// グローバルシングルトンインスタンスを取得
    pub fn global() -> &'static Self {
        use std::sync::OnceLock;
        static EXECUTOR: OnceLock<HybridExecutor> = OnceLock::new();
        EXECUTOR.get_or_init(|| Self::new())
    }

    /// Select optimal device for operation
    /// 演算に最適なデバイスを選択
    pub fn select_device(&self, op_type: OpType, tensor_info: &TensorInfo) -> DeviceType {
        // 1. Size-based heuristics
        if tensor_info.memory_size_bytes < self.small_tensor_threshold {
            return DeviceType::Cpu;
        }

        // 2. CoreML compatibility check
        if self.is_coreml_supported(&op_type, tensor_info) {
            #[cfg(feature = "coreml")]
            if self.is_device_available(DeviceType::CoreML(0)) {
                return DeviceType::CoreML(0);
            }
        }

        // 3. GPU fallback selection
        if let Some(gpu_device) = self.select_gpu_device(&op_type, tensor_info) {
            return gpu_device;
        }

        // 4. CPU fallback
        DeviceType::Cpu
    }

    /// Check if CoreML supports the operation
    /// CoreMLが演算をサポートするかチェック
    fn is_coreml_supported(&self, op_type: &OpType, tensor_info: &TensorInfo) -> bool {
        #[cfg(feature = "coreml")]
        {
            // Get CoreML capability
            if let Some(capability) = self.capability_cache.get(&DeviceType::CoreML(0)) {
                // Check operation support
                if !capability.supports_operation(op_type) {
                    return false;
                }

                // Check data type support
                match tensor_info.dtype {
                    DType::Float16 | DType::Float32 => {
                        // Check tensor dimension limits
                        tensor_info.shape.len() <= 5 && !tensor_info.requires_custom_kernel
                    }
                    _ => false,
                }
            } else {
                false
            }
        }
        #[cfg(not(feature = "coreml"))]
        {
            false
        }
    }

    /// Select best GPU device for operation
    /// 演算に最適なGPUデバイスを選択
    fn select_gpu_device(&self, op_type: &OpType, tensor_info: &TensorInfo) -> Option<DeviceType> {
        // Check operation routing table
        if let Some(devices) = self.operation_routing.get(op_type) {
            for &device in devices {
                if self.is_device_available(device)
                    && self.is_operation_efficient(device, tensor_info)
                {
                    return Some(device);
                }
            }
        }

        // Default GPU selection
        #[cfg(feature = "cuda")]
        if self.is_device_available(DeviceType::Cuda(0)) {
            return Some(DeviceType::Cuda(0));
        }

        #[cfg(feature = "metal")]
        if self.is_device_available(DeviceType::Metal(0)) {
            return Some(DeviceType::Metal(0));
        }

        #[cfg(feature = "opencl")]
        if self.is_device_available(DeviceType::OpenCL(0)) {
            return Some(DeviceType::OpenCL(0));
        }

        None
    }

    /// Check if device is available
    /// デバイスが利用可能かチェック
    fn is_device_available(&self, device: DeviceType) -> bool {
        match device {
            DeviceType::Cpu => true,
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => crate::backends::DeviceManager::is_cuda_available(),
            #[cfg(feature = "metal")]
            DeviceType::Metal(_) => crate::backends::DeviceManager::is_metal_available(),
            #[cfg(feature = "opencl")]
            DeviceType::OpenCL(_) => crate::backends::DeviceManager::is_opencl_available(),
            #[cfg(feature = "coreml")]
            DeviceType::CoreML(_) => crate::backends::DeviceManager::is_coreml_available(),
            _ => false,
        }
    }

    /// Check if operation is efficient on device
    /// デバイスでの演算が効率的かチェック
    fn is_operation_efficient(&self, device: DeviceType, tensor_info: &TensorInfo) -> bool {
        // Large tensors benefit more from GPU acceleration
        if tensor_info.memory_size_bytes > self.large_tensor_threshold {
            return matches!(
                device,
                DeviceType::Cuda(_) | DeviceType::Metal(_) | DeviceType::CoreML(_)
            );
        }

        true
    }

    /// Get next fallback device
    /// 次のフォールバックデバイスを取得
    pub fn next_fallback_device(&self, failed_device: DeviceType) -> DeviceType {
        if let Some(pos) = self
            .fallback_devices
            .iter()
            .position(|&d| d == failed_device)
        {
            if pos + 1 < self.fallback_devices.len() {
                return self.fallback_devices[pos + 1];
            }
        }
        DeviceType::Cpu
    }

    /// Initialize device capabilities
    /// デバイス能力を初期化
    fn initialize_device_capabilities(&mut self) {
        // CPU capability
        let mut cpu_ops = HashSet::new();
        cpu_ops.insert(OpType::LinearAlgebra);
        cpu_ops.insert(OpType::Convolution);
        cpu_ops.insert(OpType::Activation);
        cpu_ops.insert(OpType::Reduction);
        cpu_ops.insert(OpType::Normalization);
        cpu_ops.insert(OpType::ComplexMath);
        cpu_ops.insert(OpType::Distribution);
        cpu_ops.insert(OpType::DistributedOps);

        self.capability_cache.insert(
            DeviceType::Cpu,
            DeviceCapability {
                device_type: DeviceType::Cpu,
                supports_f16: false,
                supports_f32: true,
                supports_f64: true,
                supports_complex: true,
                supports_distributed: true,
                max_memory_gb: 32.0,
                supported_operations: cpu_ops,
            },
        );

        // CoreML capability
        #[cfg(feature = "coreml")]
        {
            self.capability_cache
                .insert(DeviceType::CoreML(0), DeviceCapability::coreml_capability());
        }

        // GPU capabilities would be detected at runtime
    }

    /// Detect available devices using device cache
    /// デバイスキャッシュを使用して利用可能デバイスを検出
    fn detect_available_devices(device_cache: &DeviceCache) -> Vec<DeviceType> {
        let mut available = Vec::new();

        // Always include CPU
        available.push(DeviceType::Cpu);

        // Check CoreML availability
        if device_cache.is_device_available(&DeviceType::CoreML(0)) {
            available.push(DeviceType::CoreML(0));
        }

        // Check Metal availability
        if device_cache.is_device_available(&DeviceType::Metal(0)) {
            available.push(DeviceType::Metal(0));
        }

        // Check CUDA availability
        if device_cache.is_device_available(&DeviceType::Cuda(0)) {
            available.push(DeviceType::Cuda(0));
        }

        available
    }

    /// Smart device selection for operation
    /// 操作用スマートデバイス選択
    pub fn select_optimal_device(&self, tensor_info: &TensorInfo, op_type: OpType) -> DeviceType {
        // Convert OpType to OperationType
        let operation_type = match op_type {
            OpType::LinearAlgebra => OperationType::MatrixMultiplication,
            OpType::Activation => OperationType::Activation,
            OpType::Convolution => OperationType::Convolution,
            OpType::Reduction | OpType::Normalization => OperationType::ElementWise,
            _ => OperationType::ElementWise,
        };

        // Create operation profile
        let profile = OperationProfile::new(
            operation_type,
            &tensor_info.shape,
            self.get_dtype_size(&tensor_info.dtype),
        );

        // Use smart selector for optimal device
        let selected = self.smart_selector.select_device(&profile);

        // Validate device is actually available
        if self.device_cache.is_device_available(&selected) {
            selected
        } else {
            // Fallback to first available device
            self.fallback_devices.first().cloned().unwrap_or(DeviceType::Cpu)
        }
    }

    /// Get fallback chain for specific operation
    /// 特定操作用のフォールバックチェーンを取得
    pub fn get_operation_fallback_chain(&self, tensor_info: &TensorInfo, op_type: OpType) -> Vec<DeviceType> {
        // Convert OpType to OperationType
        let operation_type = match op_type {
            OpType::LinearAlgebra => OperationType::MatrixMultiplication,
            OpType::Activation => OperationType::Activation,
            OpType::Convolution => OperationType::Convolution,
            OpType::Reduction | OpType::Normalization => OperationType::ElementWise,
            _ => OperationType::ElementWise,
        };

        // Create operation profile
        let profile = OperationProfile::new(
            operation_type,
            &tensor_info.shape,
            self.get_dtype_size(&tensor_info.dtype),
        );

        // Get smart fallback chain
        self.smart_selector.get_fallback_chain(&profile)
    }

    /// Get data type size in bytes
    /// データタイプのサイズをバイト単位で取得
    fn get_dtype_size(&self, dtype: &DType) -> usize {
        match dtype {
            DType::Float16 | DType::BFloat16 => 2,
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::UInt8 => 1,
            DType::UInt16 => 2,
            DType::UInt32 => 4,
            DType::UInt64 => 8,
            DType::Bool => 1,
            DType::Complex64 => 8,  // 2 * 32-bit floats
            DType::Complex128 => 16, // 2 * 64-bit floats
        }
    }

    /// Build fallback device chain based on actual hardware availability
    /// 実際のハードウェア利用可能性に基づいてフォールバックデバイスチェーンを構築
    fn build_fallback_chain(&mut self) {
        self.fallback_devices.clear();

        // Primary: CoreML if available (Apple Silicon only)
        #[cfg(feature = "coreml")]
        if self.is_device_available(DeviceType::CoreML(0)) {
            self.fallback_devices.push(DeviceType::CoreML(0));
        }

        // Platform-specific GPU selection
        if self.is_apple_silicon() {
            // Apple Silicon: CoreML → Metal → OpenCL → CPU
            #[cfg(feature = "metal")]
            if self.is_device_available(DeviceType::Metal(0)) {
                self.fallback_devices.push(DeviceType::Metal(0));
            }
        } else {
            // Intel/AMD: CUDA (if NVIDIA GPU) → OpenCL → CPU
            #[cfg(feature = "cuda")]
            if self.is_device_available(DeviceType::Cuda(0)) {
                self.fallback_devices.push(DeviceType::Cuda(0));
            }
        }

        // OpenCL as universal fallback (works on both platforms)
        #[cfg(feature = "opencl")]
        if self.is_device_available(DeviceType::OpenCL(0)) {
            self.fallback_devices.push(DeviceType::OpenCL(0));
        }

        // Final: CPU (always available)
        self.fallback_devices.push(DeviceType::Cpu);

        // Debug log the actual fallback chain
        #[cfg(debug_assertions)]
        {
            eprintln!("🔄 Fallback chain: {:?}", self.fallback_devices);
        }
    }

    /// Check if running on Apple Silicon
    /// Apple Siliconで実行しているかチェック
    fn is_apple_silicon(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            // Check if we're on Apple Silicon (ARM64)
            cfg!(target_arch = "aarch64")
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Setup operation routing table
    /// 演算ルーティングテーブルを設定
    fn setup_operation_routing(&mut self) {
        // Complex math: Skip CoreML, go to GPU
        self.operation_routing.insert(
            OpType::ComplexMath,
            vec![
                DeviceType::Cuda(0),
                DeviceType::Metal(0),
                DeviceType::OpenCL(0),
                DeviceType::Cpu,
            ],
        );

        // Distributed ops: Skip CoreML and single GPU
        self.operation_routing.insert(
            OpType::DistributedOps,
            vec![
                DeviceType::Cpu, // NCCL/MPI fallback to CPU coordination
            ],
        );

        // Custom kernels: GPU only
        self.operation_routing.insert(
            OpType::CustomKernel,
            vec![
                DeviceType::Cuda(0),
                DeviceType::Metal(0),
                DeviceType::OpenCL(0),
                DeviceType::Cpu,
            ],
        );
    }

    /// Check if any GPU support is available
    /// GPUサポートが利用可能かチェック
    pub fn has_gpu_support(&self) -> bool {
        #[cfg(any(feature = "cuda", feature = "metal", feature = "opencl"))]
        {
            self.is_device_available(DeviceType::Cuda(0))
                || self.is_device_available(DeviceType::Metal(0))
                || self.is_device_available(DeviceType::OpenCL(0))
        }
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "opencl")))]
        {
            false
        }
    }
}

impl Default for HybridExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridExecutor {
    /// Execute operation with hybrid device selection
    /// ハイブリッドデバイス選択で演算を実行
    pub fn hybrid_operation<F, R>(
        &self,
        op_type: OpType,
        tensor_info: TensorInfo,
        operation: F,
    ) -> RusTorchResult<R>
    where
        F: Fn(DeviceType) -> RusTorchResult<R>,
    {
        // Get optimal device for this operation
        let device = self.select_device(op_type, &tensor_info);

        // Try to execute on selected device
        match operation(device) {
            Ok(result) => Ok(result),
            Err(err) => {
                // If failed, try fallback devices
                for fallback_device in self.get_fallback_chain(device) {
                    if let Ok(result) = operation(fallback_device) {
                        return Ok(result);
                    }
                }
                // If all devices failed, return original error
                Err(err)
            }
        }
    }

    /// Get fallback device chain for a given device
    /// 指定デバイスのフォールバックチェーンを取得
    fn get_fallback_chain(&self, device: DeviceType) -> Vec<DeviceType> {
        match device {
            DeviceType::CoreML(_) => {
                vec![DeviceType::Metal(0), DeviceType::Cuda(0), DeviceType::Cpu]
            }
            DeviceType::Metal(_) => vec![DeviceType::Cuda(0), DeviceType::Cpu],
            DeviceType::Cuda(_) => vec![DeviceType::Metal(0), DeviceType::Cpu],
            DeviceType::OpenCL(_) => vec![DeviceType::Cpu],
            _ => vec![DeviceType::Cpu],
        }
    }
}

/// Trait for hybrid execution support
/// ハイブリッド実行サポート用トレイト
pub trait HybridExecution<T> {
    /// Execute operation with automatic device selection and fallback
    /// 自動デバイス選択とフォールバック付きで演算を実行
    fn hybrid_operation<F, R>(&self, op_type: OpType, operation: F) -> RusTorchResult<R>
    where
        F: Fn(DeviceType) -> RusTorchResult<R>;

    /// Get tensor information for device selection
    /// デバイス選択用テンソル情報を取得
    fn tensor_info(&self) -> TensorInfo;
}
