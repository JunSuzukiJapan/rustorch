/// GPU acceleration support for RusTorch
/// RusTorchのGPU加速サポート

pub mod device;
pub mod memory;
pub mod kernels;

use num_traits::Float;
use std::fmt;

/// GPU device types
/// GPU デバイスタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU device
    /// CPUデバイス
    Cpu,
    /// CUDA GPU device
    /// CUDA GPUデバイス
    Cuda(usize), // device_id
    /// Metal GPU device (for Apple Silicon)
    /// Metal GPUデバイス（Apple Silicon用）
    Metal(usize),
    /// OpenCL device
    /// OpenCLデバイス
    OpenCl(usize),
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Cpu
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Cuda(id) => write!(f, "cuda:{}", id),
            DeviceType::Metal(id) => write!(f, "metal:{}", id),
            DeviceType::OpenCl(id) => write!(f, "opencl:{}", id),
        }
    }
}

/// GPU memory layout
/// GPUメモリレイアウト
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// Row-major (C-style) layout
    /// 行優先（C形式）レイアウト
    RowMajor,
    /// Column-major (Fortran-style) layout
    /// 列優先（Fortran形式）レイアウト
    ColMajor,
}

/// GPU operation context
/// GPU演算コンテキスト
pub struct GpuContext {
    device: DeviceType,
    memory_pool_size: usize,
    stream_count: usize,
}

impl GpuContext {
    /// Create a new GPU context
    /// 新しいGPUコンテキストを作成
    pub fn new(device: DeviceType) -> Result<Self, GpuError> {
        match device {
            DeviceType::Cpu => Ok(GpuContext {
                device,
                memory_pool_size: 0,
                stream_count: 1,
            }),
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    // CUDA initialization would go here
                    Ok(GpuContext {
                        device,
                        memory_pool_size: 1024 * 1024 * 1024, // 1GB default
                        stream_count: 4,
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(GpuError::UnsupportedDevice("CUDA support not compiled".to_string()))
                }
            }
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    // Metal initialization would go here
                    Ok(GpuContext {
                        device,
                        memory_pool_size: 512 * 1024 * 1024, // 512MB default
                        stream_count: 2,
                    })
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(GpuError::UnsupportedDevice("Metal support not compiled".to_string()))
                }
            }
            DeviceType::OpenCl(_) => {
                #[cfg(feature = "opencl")]
                {
                    // OpenCL initialization would go here
                    Ok(GpuContext {
                        device,
                        memory_pool_size: 256 * 1024 * 1024, // 256MB default
                        stream_count: 2,
                    })
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err(GpuError::UnsupportedDevice("OpenCL support not compiled".to_string()))
                }
            }
        }
    }

    /// Get the device type
    /// デバイスタイプを取得
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Check if GPU is available
    /// GPUが利用可能かチェック
    pub fn is_gpu_available(&self) -> bool {
        !matches!(self.device, DeviceType::Cpu)
    }

    /// Get memory pool size
    /// メモリプールサイズを取得
    pub fn memory_pool_size(&self) -> usize {
        self.memory_pool_size
    }

    /// Get number of streams
    /// ストリーム数を取得
    pub fn stream_count(&self) -> usize {
        self.stream_count
    }
}

/// GPU error types
/// GPUエラータイプ
#[derive(Debug, Clone)]
pub enum GpuError {
    /// Device not available
    /// デバイスが利用不可
    DeviceNotAvailable(String),
    /// Unsupported device
    /// サポートされていないデバイス
    UnsupportedDevice(String),
    /// Memory allocation error
    /// メモリ割り当てエラー
    MemoryAllocationError(String),
    /// Kernel execution error
    /// カーネル実行エラー
    KernelExecutionError(String),
    /// Data transfer error
    /// データ転送エラー
    DataTransferError(String),
    /// Invalid operation
    /// 無効な操作
    InvalidOperation(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::DeviceNotAvailable(msg) => write!(f, "Device not available: {}", msg),
            GpuError::UnsupportedDevice(msg) => write!(f, "Unsupported device: {}", msg),
            GpuError::MemoryAllocationError(msg) => write!(f, "Memory allocation error: {}", msg),
            GpuError::KernelExecutionError(msg) => write!(f, "Kernel execution error: {}", msg),
            GpuError::DataTransferError(msg) => write!(f, "Data transfer error: {}", msg),
            GpuError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

/// GPU device manager
/// GPUデバイスマネージャー
pub struct DeviceManager {
    contexts: Vec<GpuContext>,
    current_device: usize,
}

impl DeviceManager {
    /// Create a new device manager
    /// 新しいデバイスマネージャーを作成
    pub fn new() -> Self {
        let mut contexts = Vec::new();
        
        // Always add CPU context
        if let Ok(cpu_context) = GpuContext::new(DeviceType::Cpu) {
            contexts.push(cpu_context);
        }

        // Try to add GPU contexts
        #[cfg(feature = "cuda")]
        {
            for device_id in 0..Self::get_cuda_device_count() {
                if let Ok(cuda_context) = GpuContext::new(DeviceType::Cuda(device_id)) {
                    contexts.push(cuda_context);
                }
            }
        }

        #[cfg(feature = "metal")]
        {
            if let Ok(metal_context) = GpuContext::new(DeviceType::Metal(0)) {
                contexts.push(metal_context);
            }
        }

        DeviceManager {
            contexts,
            current_device: 0,
        }
    }

    /// Get available devices
    /// 利用可能なデバイスを取得
    pub fn available_devices(&self) -> Vec<DeviceType> {
        self.contexts.iter().map(|ctx| ctx.device()).collect()
    }

    /// Set current device
    /// 現在のデバイスを設定
    pub fn set_device(&mut self, device: DeviceType) -> Result<(), GpuError> {
        if let Some(index) = self.contexts.iter().position(|ctx| ctx.device() == device) {
            self.current_device = index;
            Ok(())
        } else {
            Err(GpuError::DeviceNotAvailable(format!("Device {} not found", device)))
        }
    }

    /// Get current device
    /// 現在のデバイスを取得
    pub fn current_device(&self) -> DeviceType {
        self.contexts[self.current_device].device()
    }

    /// Get current context
    /// 現在のコンテキストを取得
    pub fn current_context(&self) -> &GpuContext {
        &self.contexts[self.current_device]
    }

    /// Check if CUDA is available
    /// CUDAが利用可能かチェック
    pub fn is_cuda_available() -> bool {
        #[cfg(feature = "cuda")]
        {
            Self::get_cuda_device_count() > 0
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Check if Metal is available
    /// Metalが利用可能かチェック
    pub fn is_metal_available() -> bool {
        #[cfg(feature = "metal")]
        {
            // Metal availability check would go here
            cfg!(target_os = "macos")
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }

    #[cfg(feature = "cuda")]
    fn get_cuda_device_count() -> usize {
        // CUDA device count detection would go here
        // For now, return 0 as placeholder
        0
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global device manager instance
/// グローバルデバイスマネージャーインスタンス
static mut DEVICE_MANAGER: Option<DeviceManager> = None;
static DEVICE_MANAGER_INIT: std::sync::Once = std::sync::Once::new();

/// Get the global device manager
/// グローバルデバイスマネージャーを取得
pub fn get_device_manager() -> &'static mut DeviceManager {
    unsafe {
        DEVICE_MANAGER_INIT.call_once(|| {
            DEVICE_MANAGER = Some(DeviceManager::new());
        });
        DEVICE_MANAGER.as_mut().unwrap()
    }
}

/// Set the current device globally
/// グローバルに現在のデバイスを設定
pub fn set_device(device: DeviceType) -> Result<(), GpuError> {
    get_device_manager().set_device(device)
}

/// Get the current device
/// 現在のデバイスを取得
pub fn current_device() -> DeviceType {
    get_device_manager().current_device()
}

/// Check if any GPU is available
/// 何らかのGPUが利用可能かチェック
pub fn is_gpu_available() -> bool {
    DeviceManager::is_cuda_available() || DeviceManager::is_metal_available()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::Cpu.to_string(), "cpu");
        assert_eq!(DeviceType::Cuda(0).to_string(), "cuda:0");
        assert_eq!(DeviceType::Metal(1).to_string(), "metal:1");
        assert_eq!(DeviceType::OpenCl(2).to_string(), "opencl:2");
    }

    #[test]
    fn test_device_manager_creation() {
        let manager = DeviceManager::new();
        assert!(!manager.available_devices().is_empty());
        assert_eq!(manager.current_device(), DeviceType::Cpu);
    }

    #[test]
    fn test_gpu_context_cpu() {
        let context = GpuContext::new(DeviceType::Cpu).unwrap();
        assert_eq!(context.device(), DeviceType::Cpu);
        assert!(!context.is_gpu_available());
        assert_eq!(context.stream_count(), 1);
    }

    #[test]
    fn test_global_device_manager() {
        let device = current_device();
        assert_eq!(device, DeviceType::Cpu);
        
        let available = get_device_manager().available_devices();
        assert!(!available.is_empty());
    }

    #[test]
    fn test_gpu_availability() {
        // This will depend on the compilation features
        let _cuda_available = DeviceManager::is_cuda_available();
        let _metal_available = DeviceManager::is_metal_available();
        let _any_gpu = is_gpu_available();
    }
}
