//! GPU実行エンジン - f32統一バックエンド
//! GPU execution engines - f32 unified backends

use crate::error::RusTorchResult;
use super::tensor::F32Tensor;

pub mod metal;
pub mod coreml;

pub use metal::F32MetalExecutor;
pub use coreml::F32CoreMLExecutor;

/// f32統一GPU実行トレイト
/// f32 unified GPU execution trait
pub trait F32GPUExecutor {
    /// デバイス初期化
    /// Initialize device
    fn initialize(&mut self, device_id: usize) -> RusTorchResult<()>;

    /// テンソルをGPUに転送（ゼロコピー）
    /// Transfer tensor to GPU (zero-copy)
    fn transfer_to_gpu(&self, tensor: &mut F32Tensor) -> RusTorchResult<()>;

    /// テンソルをCPUに転送
    /// Transfer tensor to CPU
    fn transfer_to_cpu(&self, tensor: &mut F32Tensor) -> RusTorchResult<()>;

    /// f32行列乗算直接実行
    /// Direct f32 matrix multiplication execution
    fn matmul_f32(&self, a: &F32Tensor, b: &F32Tensor) -> RusTorchResult<F32Tensor>;

    /// f32畳み込み直接実行
    /// Direct f32 convolution execution
    fn conv2d_f32(
        &self,
        input: &F32Tensor,
        kernel: &F32Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> RusTorchResult<F32Tensor>;

    /// デバイス性能情報を取得
    /// Get device performance information
    fn get_performance_info(&self) -> DevicePerformanceInfo;
}

/// デバイス性能情報
/// Device performance information
#[derive(Debug, Clone)]
pub struct DevicePerformanceInfo {
    pub device_name: String,
    pub memory_bandwidth: f64,  // GB/s
    pub compute_units: usize,
    pub supports_f32: bool,
    pub supports_float16: bool,
    pub estimated_tflops_f32: f64,
    pub estimated_tflops_f16: f64,
}

impl DevicePerformanceInfo {
    pub fn cpu_baseline() -> Self {
        Self {
            device_name: "CPU".to_string(),
            memory_bandwidth: 50.0,
            compute_units: 8,
            supports_f32: true,
            supports_float16: false,
            estimated_tflops_f32: 0.5,
            estimated_tflops_f16: 0.0,
        }
    }

    pub fn metal_gpu_m1() -> Self {
        Self {
            device_name: "Apple M1 GPU".to_string(),
            memory_bandwidth: 68.25,
            compute_units: 8,
            supports_f32: true,
            supports_float16: true,
            estimated_tflops_f32: 2.6,
            estimated_tflops_f16: 5.2,
        }
    }

    pub fn neural_engine_m1() -> Self {
        Self {
            device_name: "Apple M1 Neural Engine".to_string(),
            memory_bandwidth: 68.25,
            compute_units: 16,
            supports_f32: true,
            supports_float16: true,
            estimated_tflops_f32: 7.0,   // 推定値
            estimated_tflops_f16: 15.8,  // 公式値
        }
    }
}

/// 統一GPU実行コンテキスト
/// Unified GPU execution context
#[derive(Debug)]
pub struct F32UnifiedGPUContext {
    metal_executor: Option<F32MetalExecutor>,
    coreml_executor: Option<F32CoreMLExecutor>,
    current_device: GPUDevice,
}

#[derive(Debug, Clone)]
pub enum GPUDevice {
    CPU,
    Metal(usize),
    CoreML(usize),
}

impl F32UnifiedGPUContext {
    pub fn new() -> Self {
        Self {
            metal_executor: None,
            coreml_executor: None,
            current_device: GPUDevice::CPU,
        }
    }

    /// 利用可能なGPUデバイスを検出
    /// Detect available GPU devices
    pub fn detect_available_devices(&self) -> Vec<(GPUDevice, DevicePerformanceInfo)> {
        let mut devices = vec![(GPUDevice::CPU, DevicePerformanceInfo::cpu_baseline())];

        // Metal GPU検出
        #[cfg(target_os = "macos")]
        {
            devices.push((GPUDevice::Metal(0), DevicePerformanceInfo::metal_gpu_m1()));
        }

        // Neural Engine検出
        #[cfg(all(target_os = "macos", feature = "coreml"))]
        {
            devices.push((GPUDevice::CoreML(0), DevicePerformanceInfo::neural_engine_m1()));
        }

        devices
    }

    /// 最適デバイス選択
    /// Select optimal device
    pub fn select_optimal_device(&self, operation: &str, tensor_size: usize) -> GPUDevice {
        match (operation, tensor_size) {
            ("matmul", size) if size > 50000 => GPUDevice::Metal(0),
            ("conv2d", size) if size > 1000 => GPUDevice::CoreML(0),
            ("activation", _) => GPUDevice::CoreML(0),
            _ => GPUDevice::CPU,
        }
    }

    /// デバイス初期化
    /// Initialize device
    pub fn initialize_device(&mut self, device: GPUDevice) -> RusTorchResult<()> {
        match device {
            GPUDevice::Metal(device_id) => {
                let mut executor = F32MetalExecutor::new();
                executor.initialize(device_id)?;
                self.metal_executor = Some(executor);
                self.current_device = device;
            }
            GPUDevice::CoreML(device_id) => {
                let mut executor = F32CoreMLExecutor::new();
                executor.initialize(device_id)?;
                self.coreml_executor = Some(executor);
                self.current_device = device;
            }
            GPUDevice::CPU => {
                self.current_device = device;
            }
        }

        Ok(())
    }

    /// 統一実行
    /// Unified execution
    pub fn execute_matmul(&self, a: &F32Tensor, b: &F32Tensor) -> RusTorchResult<F32Tensor> {
        crate::hybrid_f32_experimental!();

        match &self.current_device {
            GPUDevice::Metal(_) => {
                if let Some(executor) = &self.metal_executor {
                    executor.matmul_f32(a, b)
                } else {
                    Err(crate::error::RusTorchError::BackendUnavailable {
                        backend: "Metal".to_string(),
                    })
                }
            }
            GPUDevice::CoreML(_) => {
                if let Some(executor) = &self.coreml_executor {
                    executor.matmul_f32(a, b)
                } else {
                    Err(crate::error::RusTorchError::BackendUnavailable {
                        backend: "CoreML".to_string(),
                    })
                }
            }
            GPUDevice::CPU => {
                // CPU実行（フォールバック）
                a.matmul(b)
            }
        }
    }
}