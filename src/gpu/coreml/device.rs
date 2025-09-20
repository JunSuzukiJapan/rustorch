//! Unified CoreML device management
//! 統一CoreMLデバイス管理

use super::common::*;
use crate::gpu::coreml::common::coreml_feature;
use super::common::error_helpers;
use crate::gpu::{DeviceType, device_cache::DeviceCache};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Unified CoreML device manager
/// 統一CoreMLデバイスマネージャー
pub struct CoreMLDeviceManager {
    /// Device availability cache
    /// デバイス可用性キャッシュ
    cache: DeviceCache,

    /// CoreML capabilities
    /// CoreML機能
    capabilities: CoreMLCapabilities,

    /// Last capability check time
    /// 最後の機能チェック時間
    last_check: Arc<Mutex<Option<Instant>>>,
}

impl CoreMLDeviceManager {
    /// Get global CoreML device manager instance
    /// グローバルCoreMLデバイスマネージャーインスタンスを取得
    pub fn global() -> &'static CoreMLDeviceManager {
        static MANAGER: OnceLock<CoreMLDeviceManager> = OnceLock::new();
        MANAGER.get_or_init(|| CoreMLDeviceManager::new())
    }

    /// Create new CoreML device manager
    /// 新しいCoreMLデバイスマネージャーを作成
    pub fn new() -> Self {
        Self {
            cache: DeviceCache::new(),
            capabilities: CoreMLCapabilities::default(),
            last_check: Arc::new(Mutex::new(None)),
        }
    }

    /// Check if CoreML device is available
    /// CoreMLデバイスが利用可能かチェック
    pub fn is_available(&self) -> bool {
        self.cache.is_device_available(&DeviceType::CoreML(0))
    }

    /// Get CoreML capabilities with caching
    /// キャッシュ付きでCoreML機能を取得
    pub fn capabilities(&self) -> &CoreMLCapabilities {
        // Refresh capabilities if cache is stale (> 30 seconds)
        if let Ok(mut last_check) = self.last_check.lock() {
            let should_refresh = last_check
                .map(|t| t.elapsed() > Duration::from_secs(30))
                .unwrap_or(true);

            if should_refresh {
                // Update capabilities here if needed
                *last_check = Some(Instant::now());
            }
        }

        &self.capabilities
    }

    /// Initialize CoreML device with error handling
    /// エラーハンドリング付きでCoreMLデバイスを初期化
    pub fn initialize(&self, device_id: usize) -> CoreMLResult<CoreMLDevice> {
        if !is_coreml_available() {
            return Err(error_helpers::not_available());
        }

        if !self.is_available() {
            return Err(error_helpers::device_error("CoreML device not available"));
        }

        CoreMLDevice::new(device_id)
    }

    /// Warmup CoreML device for faster first operation
    /// 初回演算を高速化するためのCoreMLデバイスウォームアップ
    pub fn warmup(&self) -> CoreMLResult<()> {
        if !self.is_available() {
            return Err(error_helpers::not_available());
        }

        // Pre-initialize CoreML device
        let _device = self.initialize(0)?;
        Ok(())
    }

    /// Get optimal CoreML device for operation
    /// 演算に最適なCoreMLデバイスを取得
    pub fn optimal_device_for_op(&self, op_type: CoreMLOpType) -> Option<usize> {
        if !self.is_available() {
            return None;
        }

        match op_type {
            CoreMLOpType::MatrixMultiplication => Some(0), // Primary Neural Engine
            CoreMLOpType::Convolution => Some(0),          // Primary Neural Engine
            CoreMLOpType::Activation => Some(0),           // Primary Neural Engine
            CoreMLOpType::ElementWise => Some(0),          // Primary Neural Engine
        }
    }
}

impl Default for CoreMLDeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// CoreML device representation
/// CoreMLデバイス表現
pub struct CoreMLDevice {
    /// Device ID
    /// デバイスID
    device_id: usize,

    /// Device capabilities
    /// デバイス機能
    capabilities: CoreMLCapabilities,

    /// Initialization timestamp
    /// 初期化タイムスタンプ
    initialized_at: Instant,
}

coreml_feature! {
    impl CoreMLDevice {
        /// Create new CoreML device
        /// 新しいCoreMLデバイスを作成
        pub fn new(device_id: usize) -> CoreMLResult<Self> {
            if !is_coreml_available() {
                return Err(error_helpers::not_available());
            }

            // Initialize device-specific capabilities
            let capabilities = Self::detect_capabilities(device_id)?;

            Ok(Self {
                device_id,
                capabilities,
                initialized_at: Instant::now(),
            })
        }

        /// Get device ID
        /// デバイスIDを取得
        pub fn device_id(&self) -> usize {
            self.device_id
        }

        /// Get device capabilities
        /// デバイス機能を取得
        pub fn capabilities(&self) -> &CoreMLCapabilities {
            &self.capabilities
        }

        /// Check if device is available for operations
        /// デバイスが演算に利用可能かチェック
        pub fn is_ready(&self) -> bool {
            // Device is ready if it was initialized successfully
            self.initialized_at.elapsed() < Duration::from_secs(3600) // 1 hour validity
        }

        /// Detect device-specific capabilities
        /// デバイス固有機能を検出
        fn detect_capabilities(device_id: usize) -> CoreMLResult<CoreMLCapabilities> {
            // Platform-specific capability detection
            #[cfg(target_os = "macos")]
            {
                use crate::backends::DeviceManager;

                if !DeviceManager::is_coreml_available() {
                    return Err(error_helpers::not_available());
                }

                let neural_engine = true; // Assume Neural Engine available on macOS

                Ok(CoreMLCapabilities {
                    max_tensor_size: 100 * 1024 * 1024, // 100MB
                    supports_f32: true,
                    supports_f64: false,      // CoreML limitation
                    supports_complex: false,  // CoreML limitation
                    neural_engine_available: neural_engine,
                    gpu_acceleration_available: true, // Assume true on macOS
                })
            }

            #[cfg(not(target_os = "macos"))]
            Err(error_helpers::device_error(COREML_MACOS_ONLY))
        }

        /// Validate tensor for CoreML operations
        /// CoreML演算用のテンソルを検証
        pub fn validate_tensor<T>(&self, tensor_size: usize, dtype: &crate::dtype::DType) -> CoreMLResult<()> {
            // Check tensor size
            if tensor_size > self.capabilities.max_tensor_size {
                return Err(error_helpers::unsupported_operation(&format!(
                    "Tensor too large: {} bytes > {} bytes max",
                    tensor_size, self.capabilities.max_tensor_size
                )));
            }

            // Check data type support
            match dtype {
                crate::dtype::DType::Float32 => {
                    if !self.capabilities.supports_f32 {
                        return Err(error_helpers::unsupported_operation("Float32 not supported"));
                    }
                }
                crate::dtype::DType::Float64 => {
                    if !self.capabilities.supports_f64 {
                        return Err(error_helpers::unsupported_operation("Float64 not supported by CoreML"));
                    }
                }
                crate::dtype::DType::Complex64 | crate::dtype::DType::Complex128 => {
                    return Err(error_helpers::unsupported_operation("Complex numbers not supported by CoreML"));
                }
                _ => {} // Other types assumed supported
            }

            Ok(())
        }
    }
}

/// Device not available implementation for non-CoreML builds
/// 非CoreMLビルド用のデバイス非利用可能実装
#[cfg(not(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback")))]
impl CoreMLDevice {
    pub fn new(_device_id: usize) -> CoreMLResult<Self> {
        Err(error_helpers::feature_disabled())
    }

    pub fn device_id(&self) -> usize {
        unreachable!("CoreML device not available without features")
    }

    pub fn capabilities(&self) -> &CoreMLCapabilities {
        unreachable!("CoreML device not available without features")
    }

    pub fn is_ready(&self) -> bool {
        false
    }

    pub fn validate_tensor<T>(&self, _tensor_size: usize, _dtype: &crate::dtype::DType) -> CoreMLResult<()> {
        Err(error_helpers::feature_disabled())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager_singleton() {
        let manager1 = CoreMLDeviceManager::global();
        let manager2 = CoreMLDeviceManager::global();

        // Should be the same instance
        assert!(std::ptr::eq(manager1, manager2));
    }

    #[test]
    fn test_capabilities_default() {
        let caps = CoreMLCapabilities::default();
        assert!(caps.supports_f32);
        assert!(!caps.supports_f64);
        assert!(!caps.supports_complex);
    }

    #[test]
    fn test_error_conversion() {
        let rustorch_err = crate::error::RusTorchError::Device {
            device: "CoreML".to_string(),
            message: COREML_NOT_AVAILABLE.to_string(),
        };

        match rustorch_err {
            crate::error::RusTorchError::Device { device, message } => {
                assert_eq!(device, "CoreML");
                assert_eq!(message, COREML_NOT_AVAILABLE);
            }
            _ => panic!("Unexpected error type"),
        }
    }
}