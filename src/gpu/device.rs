/// GPU device management and capabilities
/// GPUデバイス管理と機能

use super::{DeviceType, GpuError};
use std::collections::HashMap;

/// GPU device capabilities
/// GPUデバイス機能
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device name
    /// デバイス名
    pub name: String,
    /// Total memory in bytes
    /// 総メモリ（バイト）
    pub total_memory: usize,
    /// Available memory in bytes
    /// 利用可能メモリ（バイト）
    pub available_memory: usize,
    /// Compute capability major version
    /// 計算能力メジャーバージョン
    pub compute_major: u32,
    /// Compute capability minor version
    /// 計算能力マイナーバージョン
    pub compute_minor: u32,
    /// Maximum threads per block
    /// ブロック当たり最大スレッド数
    pub max_threads_per_block: u32,
    /// Maximum block dimensions
    /// 最大ブロック次元
    pub max_block_dims: [u32; 3],
    /// Maximum grid dimensions
    /// 最大グリッド次元
    pub max_grid_dims: [u32; 3],
    /// Shared memory per block
    /// ブロック当たり共有メモリ
    pub shared_memory_per_block: u32,
    /// Warp size
    /// ワープサイズ
    pub warp_size: u32,
    /// Supports double precision
    /// 倍精度サポート
    pub supports_double: bool,
    /// Supports half precision
    /// 半精度サポート
    pub supports_half: bool,
    /// Supports tensor cores
    /// テンサーコアサポート
    pub supports_tensor_cores: bool,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        DeviceCapabilities {
            name: "CPU".to_string(),
            total_memory: 0,
            available_memory: 0,
            compute_major: 0,
            compute_minor: 0,
            max_threads_per_block: 1,
            max_block_dims: [1, 1, 1],
            max_grid_dims: [1, 1, 1],
            shared_memory_per_block: 0,
            warp_size: 1,
            supports_double: true,
            supports_half: false,
            supports_tensor_cores: false,
        }
    }
}

/// Device information and management
/// デバイス情報と管理
pub struct DeviceInfo {
    device_type: DeviceType,
    capabilities: DeviceCapabilities,
    is_available: bool,
}

impl DeviceInfo {
    /// Create device info for CPU
    /// CPU用デバイス情報を作成
    pub fn cpu() -> Self {
        DeviceInfo {
            device_type: DeviceType::Cpu,
            capabilities: DeviceCapabilities::default(),
            is_available: true,
        }
    }

    /// Create device info for CUDA device
    /// CUDAデバイス用デバイス情報を作成
    pub fn cuda(device_id: usize) -> Result<Self, GpuError> {
        #[cfg(feature = "cuda")]
        {
            // CUDA device query would go here
            let capabilities = DeviceCapabilities {
                name: format!("CUDA Device {}", device_id),
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB placeholder
                available_memory: 7 * 1024 * 1024 * 1024, // 7GB placeholder
                compute_major: 7,
                compute_minor: 5,
                max_threads_per_block: 1024,
                max_block_dims: [1024, 1024, 64],
                max_grid_dims: [2147483647, 65535, 65535],
                shared_memory_per_block: 49152,
                warp_size: 32,
                supports_double: true,
                supports_half: true,
                supports_tensor_cores: true,
            };

            Ok(DeviceInfo {
                device_type: DeviceType::Cuda(device_id),
                capabilities,
                is_available: true,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::UnsupportedDevice("CUDA not supported".to_string()))
        }
    }

    /// Create device info for Metal device
    /// Metalデバイス用デバイス情報を作成
    pub fn metal(device_id: usize) -> Result<Self, GpuError> {
        #[cfg(feature = "metal")]
        {
            let capabilities = DeviceCapabilities {
                name: format!("Metal Device {}", device_id),
                total_memory: 16 * 1024 * 1024 * 1024, // 16GB placeholder for Apple Silicon
                available_memory: 14 * 1024 * 1024 * 1024,
                compute_major: 3,
                compute_minor: 0,
                max_threads_per_block: 1024,
                max_block_dims: [1024, 1024, 1024],
                max_grid_dims: [65535, 65535, 65535],
                shared_memory_per_block: 32768,
                warp_size: 32, // SIMD group size
                supports_double: false, // Metal typically uses float
                supports_half: true,
                supports_tensor_cores: false,
            };

            Ok(DeviceInfo {
                device_type: DeviceType::Metal(device_id),
                capabilities,
                is_available: true,
            })
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(GpuError::UnsupportedDevice("Metal not supported".to_string()))
        }
    }

    /// Get device type
    /// デバイスタイプを取得
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Get device capabilities
    /// デバイス機能を取得
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// Check if device is available
    /// デバイスが利用可能かチェック
    pub fn is_available(&self) -> bool {
        self.is_available
    }

    /// Get optimal block size for given problem size
    /// 指定された問題サイズに対する最適ブロックサイズを取得
    pub fn optimal_block_size(&self, problem_size: usize) -> (u32, u32, u32) {
        match self.device_type {
            DeviceType::Cpu => (1, 1, 1),
            DeviceType::Cuda(_) => {
                // Simple heuristic for CUDA block size
                let threads_per_block = if problem_size < 256 {
                    128
                } else if problem_size < 1024 {
                    256
                } else {
                    512
                };
                (threads_per_block, 1, 1)
            }
            DeviceType::Metal(_) => {
                // Metal threadgroup size
                let threads_per_group = if problem_size < 256 {
                    64
                } else if problem_size < 1024 {
                    128
                } else {
                    256
                };
                (threads_per_group, 1, 1)
            }
            DeviceType::OpenCl(_) => {
                // OpenCL work group size
                (64, 1, 1)
            }
        }
    }

    /// Get optimal grid size for given problem size and block size
    /// 指定された問題サイズとブロックサイズに対する最適グリッドサイズを取得
    pub fn optimal_grid_size(&self, problem_size: usize, block_size: (u32, u32, u32)) -> (u32, u32, u32) {
        let total_threads = block_size.0 * block_size.1 * block_size.2;
        let num_blocks = ((problem_size as u32 + total_threads - 1) / total_threads).max(1);
        
        match self.device_type {
            DeviceType::Cpu => (1, 1, 1),
            _ => {
                // Simple 1D grid for now
                (num_blocks, 1, 1)
            }
        }
    }

    /// Check if device supports operation
    /// デバイスが操作をサポートするかチェック
    pub fn supports_operation(&self, operation: &str) -> bool {
        match operation {
            "matmul" => true,
            "conv2d" => true,
            "batchnorm" => true,
            "activation" => true,
            "reduction" => true,
            "double_precision" => self.capabilities.supports_double,
            "half_precision" => self.capabilities.supports_half,
            "tensor_cores" => self.capabilities.supports_tensor_cores,
            _ => false,
        }
    }

    /// Get memory usage information
    /// メモリ使用量情報を取得
    pub fn memory_info(&self) -> (usize, usize, f32) {
        let total = self.capabilities.total_memory;
        let available = self.capabilities.available_memory;
        let usage_percent = if total > 0 {
            ((total - available) as f32 / total as f32) * 100.0
        } else {
            0.0
        };
        (total, available, usage_percent)
    }
}

/// Device registry for managing multiple devices
/// 複数デバイス管理用デバイスレジストリ
pub struct DeviceRegistry {
    devices: HashMap<DeviceType, DeviceInfo>,
}

impl DeviceRegistry {
    /// Create a new device registry
    /// 新しいデバイスレジストリを作成
    pub fn new() -> Self {
        let mut registry = DeviceRegistry {
            devices: HashMap::new(),
        };

        // Register CPU
        registry.devices.insert(DeviceType::Cpu, DeviceInfo::cpu());

        // Try to register CUDA devices
        #[cfg(feature = "cuda")]
        {
            for device_id in 0..Self::get_cuda_device_count() {
                if let Ok(device_info) = DeviceInfo::cuda(device_id) {
                    registry.devices.insert(DeviceType::Cuda(device_id), device_info);
                }
            }
        }

        // Try to register Metal devices
        #[cfg(feature = "metal")]
        {
            if let Ok(device_info) = DeviceInfo::metal(0) {
                registry.devices.insert(DeviceType::Metal(0), device_info);
            }
        }

        registry
    }

    /// Get device info
    /// デバイス情報を取得
    pub fn get_device(&self, device_type: DeviceType) -> Option<&DeviceInfo> {
        self.devices.get(&device_type)
    }

    /// List all available devices
    /// 利用可能な全デバイスをリスト
    pub fn list_devices(&self) -> Vec<DeviceType> {
        self.devices.keys().copied().collect()
    }

    /// Get best device for operation
    /// 操作に最適なデバイスを取得
    pub fn best_device_for_operation(&self, operation: &str, data_size: usize) -> DeviceType {
        let mut best_device = DeviceType::Cpu;
        let mut best_score = 0.0f32;

        for (device_type, device_info) in &self.devices {
            if !device_info.supports_operation(operation) {
                continue;
            }

            let mut score = match device_type {
                DeviceType::Cpu => 1.0,
                DeviceType::Cuda(_) => 10.0,
                DeviceType::Metal(_) => 8.0,
                DeviceType::OpenCl(_) => 6.0,
            };

            // Adjust score based on data size
            if data_size < 1000 {
                // Small data might be better on CPU due to transfer overhead
                if matches!(device_type, DeviceType::Cpu) {
                    score *= 2.0;
                } else {
                    score *= 0.5;
                }
            }

            // Adjust score based on memory availability
            let (_, available_memory, _) = device_info.memory_info();
            if available_memory > 0 {
                let memory_score = (available_memory as f32 / (1024.0 * 1024.0 * 1024.0)).min(10.0);
                score *= 1.0 + memory_score * 0.1;
            }

            if score > best_score {
                best_score = score;
                best_device = *device_type;
            }
        }

        best_device
    }

    #[cfg(feature = "cuda")]
    fn get_cuda_device_count() -> usize {
        // CUDA device detection would go here
        0 // Placeholder
    }
}

impl Default for DeviceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_capabilities_default() {
        let caps = DeviceCapabilities::default();
        assert_eq!(caps.name, "CPU");
        assert!(caps.supports_double);
        assert!(!caps.supports_half);
    }

    #[test]
    fn test_device_info_cpu() {
        let device = DeviceInfo::cpu();
        assert_eq!(device.device_type(), DeviceType::Cpu);
        assert!(device.is_available());
        assert!(device.supports_operation("matmul"));
    }

    #[test]
    fn test_optimal_block_size() {
        let device = DeviceInfo::cpu();
        let block_size = device.optimal_block_size(1000);
        assert_eq!(block_size, (1, 1, 1));
    }

    #[test]
    fn test_device_registry() {
        let registry = DeviceRegistry::new();
        assert!(registry.get_device(DeviceType::Cpu).is_some());
        
        let devices = registry.list_devices();
        assert!(!devices.is_empty());
        assert!(devices.contains(&DeviceType::Cpu));
    }

    #[test]
    fn test_best_device_selection() {
        let registry = DeviceRegistry::new();
        
        // Small data should prefer CPU
        let device = registry.best_device_for_operation("matmul", 100);
        assert_eq!(device, DeviceType::Cpu);
        
        // Large data should prefer GPU if available
        let device = registry.best_device_for_operation("matmul", 1000000);
        // Will be CPU since no GPU is actually available in test
        assert_eq!(device, DeviceType::Cpu);
    }

    #[test]
    fn test_memory_info() {
        let device = DeviceInfo::cpu();
        let (total, available, usage) = device.memory_info();
        assert_eq!(total, 0);
        assert_eq!(available, 0);
        assert_eq!(usage, 0.0);
    }
}
