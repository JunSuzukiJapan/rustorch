//! Smart Device Selection Logic
//! スマートデバイス選択ロジック
//!
//! This module provides intelligent device selection based on operation characteristics
//! 操作特性に基づいたインテリジェントなデバイス選択を提供

use crate::gpu::DeviceType;
use std::collections::HashMap;

/// Operation type for device selection
/// デバイス選択用の操作タイプ
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperationType {
    MatrixMultiplication,
    Activation,
    Convolution,
    ElementWise,
    // CoreML unsupported operations - bypass CoreML entirely
    // CoreML非対応演算 - CoreMLを完全にバイパス
    ComplexNumber,      // Complex64/Complex128 operations
    StatisticalDistribution, // Probability distributions
    CustomKernel,       // Custom GPU kernels
    DistributedOp,      // Multi-GPU distributed operations
}

/// Operation characteristics for smart selection
/// スマート選択用の操作特性
#[derive(Debug, Clone)]
pub struct OperationProfile {
    pub op_type: OperationType,
    pub tensor_size: usize,
    pub dimensions: Vec<usize>,
    pub data_type_size: usize, // bytes per element
}

impl OperationProfile {
    pub fn new(op_type: OperationType, dimensions: &[usize], data_type_size: usize) -> Self {
        let tensor_size = dimensions.iter().product::<usize>();
        Self {
            op_type,
            tensor_size,
            dimensions: dimensions.to_vec(),
            data_type_size,
        }
    }

    /// Calculate memory footprint in bytes
    /// メモリフットプリントをバイト単位で計算
    pub fn memory_footprint(&self) -> usize {
        self.tensor_size * self.data_type_size
    }
}

/// Device selection thresholds
/// デバイス選択閾値
#[derive(Debug, Clone)]
pub struct DeviceThresholds {
    /// Minimum tensor size for CoreML (elements)
    /// CoreML用最小テンソルサイズ（要素数）
    pub coreml_min_size: usize,
    /// Maximum tensor size for CoreML (elements)
    /// CoreML用最大テンソルサイズ（要素数）
    pub coreml_max_size: usize,
    /// Minimum tensor size for Metal GPU (elements)
    /// Metal GPU用最小テンソルサイズ（要素数）
    pub metal_min_size: usize,
    /// Minimum memory footprint for GPU operations (bytes)
    /// GPU操作用最小メモリフットプリント（バイト）
    pub gpu_min_memory: usize,
}

impl Default for DeviceThresholds {
    fn default() -> Self {
        Self {
            // CoreML thresholds based on empirical testing
            coreml_min_size: 1024,      // 32x32 matrices minimum
            coreml_max_size: 1_048_576, // 1024x1024 matrices maximum

            // Metal GPU thresholds
            metal_min_size: 256,        // 16x16 matrices minimum

            // Memory thresholds (4KB minimum for GPU operations)
            gpu_min_memory: 4096,
        }
    }
}

/// Smart device selector with operation-specific logic
/// 操作固有ロジックを持つスマートデバイスセレクター
pub struct SmartDeviceSelector {
    thresholds: DeviceThresholds,
    /// Available devices in order of preference
    /// 優先順位順の利用可能デバイス
    available_devices: Vec<DeviceType>,
}

impl SmartDeviceSelector {
    pub fn new(available_devices: Vec<DeviceType>) -> Self {
        Self {
            thresholds: DeviceThresholds::default(),
            available_devices,
        }
    }

    pub fn with_thresholds(mut self, thresholds: DeviceThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Select optimal device for the given operation
    /// 指定された操作に最適なデバイスを選択
    pub fn select_device(&self, profile: &OperationProfile) -> DeviceType {
        match profile.op_type {
            // CoreML-supported operations - go through normal selection logic
            OperationType::MatrixMultiplication => self.select_for_matrix_mul(profile),
            OperationType::Activation => self.select_for_activation(profile),
            OperationType::Convolution => self.select_for_convolution(profile),
            OperationType::ElementWise => self.select_for_elementwise(profile),
            
            // CoreML-unsupported operations - bypass CoreML entirely
            // CoreML非対応演算 - CoreMLを完全にバイパス
            OperationType::ComplexNumber | 
            OperationType::StatisticalDistribution | 
            OperationType::CustomKernel | 
            OperationType::DistributedOp => self.select_non_coreml_device(profile),
        }
    }

    /// Matrix multiplication device selection
    /// 行列乗算デバイス選択
    fn select_for_matrix_mul(&self, profile: &OperationProfile) -> DeviceType {
        let size = profile.tensor_size;

        // Very large matrices: prefer CoreML if available
        if size >= self.thresholds.coreml_min_size && size <= self.thresholds.coreml_max_size {
            if self.is_device_available(&DeviceType::CoreML(0)) {
                return DeviceType::CoreML(0);
            }
        }

        // Medium-large matrices: prefer Metal GPU
        if size >= self.thresholds.metal_min_size {
            if self.is_device_available(&DeviceType::Metal(0)) {
                return DeviceType::Metal(0);
            }
        }

        // Small matrices or fallback: use CPU
        DeviceType::Cpu
    }

    /// Activation function device selection
    /// 活性化関数デバイス選択
    fn select_for_activation(&self, profile: &OperationProfile) -> DeviceType {
        let memory = profile.memory_footprint();

        // Metal GPU is very efficient for activation functions
        if memory >= self.thresholds.gpu_min_memory {
            if self.is_device_available(&DeviceType::Metal(0)) {
                return DeviceType::Metal(0);
            }
        }

        // CoreML for medium-size tensors
        if profile.tensor_size >= self.thresholds.coreml_min_size {
            if self.is_device_available(&DeviceType::CoreML(0)) {
                return DeviceType::CoreML(0);
            }
        }

        // Small tensors: direct CPU
        DeviceType::Cpu
    }

    /// Convolution device selection
    /// 畳み込みデバイス選択
    fn select_for_convolution(&self, profile: &OperationProfile) -> DeviceType {
        // Convolution analysis based on dimensions
        if profile.dimensions.len() >= 4 {
            let batch_size = profile.dimensions[0];
            let channels = profile.dimensions[1];
            let spatial_size = profile.dimensions[2] * profile.dimensions[3];

            // Large convolutions with many channels: CoreML
            if channels >= 16 && spatial_size >= 1024 && batch_size >= 4 {
                if self.is_device_available(&DeviceType::CoreML(0)) {
                    return DeviceType::CoreML(0);
                }
            }

            // Medium convolutions: Metal GPU
            if channels >= 4 && spatial_size >= 256 {
                if self.is_device_available(&DeviceType::Metal(0)) {
                    return DeviceType::Metal(0);
                }
            }
        }

        // Small convolutions: direct CPU (avoid overhead)
        DeviceType::Cpu
    }

    /// Element-wise operation device selection
    /// 要素ごと操作デバイス選択
    fn select_for_elementwise(&self, profile: &OperationProfile) -> DeviceType {
        let memory = profile.memory_footprint();

        // Large tensors: GPU
        if memory >= self.thresholds.gpu_min_memory * 4 {
            if self.is_device_available(&DeviceType::Metal(0)) {
                return DeviceType::Metal(0);
            }
            if self.is_device_available(&DeviceType::CoreML(0)) {
                return DeviceType::CoreML(0);
            }
        }

        // Small tensors: direct CPU
        DeviceType::Cpu
    }

    /// Check if device is available
    /// デバイスが利用可能かチェック
    fn is_device_available(&self, device: &DeviceType) -> bool {
        self.available_devices.iter().any(|d| std::mem::discriminant(d) == std::mem::discriminant(device))
    }

    /// Select device for CoreML-unsupported operations (GPU/CPU only)
    /// CoreML非対応演算用デバイス選択（GPU/CPUのみ）
    fn select_non_coreml_device(&self, profile: &OperationProfile) -> DeviceType {
        // For unsupported operations, prefer GPU over CPU for better performance
        // 非対応演算では、パフォーマンス向上のためGPUをCPUより優先
        
        // Find first available GPU (Metal, CUDA, OpenCL)
        for device in &self.available_devices {
            match device {
                DeviceType::Metal(_) | DeviceType::Cuda(_) | DeviceType::OpenCL(_) => {
                    return device.clone();
                }
                _ => continue,
            }
        }
        
        // If no GPU available, fallback to CPU
        DeviceType::Cpu
    }

    /// Get fallback chain for operation
    /// 操作用のフォールバックチェーンを取得
    pub fn get_fallback_chain(&self, profile: &OperationProfile) -> Vec<DeviceType> {
        let primary = self.select_device(profile);
        let mut chain = vec![primary];

        // Add other available devices as fallbacks
        for device in &self.available_devices {
            if !chain.iter().any(|d| std::mem::discriminant(d) == std::mem::discriminant(device)) {
                chain.push(device.clone());
            }
        }

        // Always end with CPU as final fallback
        if !chain.iter().any(|d| matches!(d, DeviceType::Cpu)) {
            chain.push(DeviceType::Cpu);
        }

        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_matrix_selection() {
        let selector = SmartDeviceSelector::new(vec![
            DeviceType::CoreML(0),
            DeviceType::Metal(0),
            DeviceType::Cpu,
        ]);

        // Small matrix (16x16 = 256 elements) should use CPU
        let profile = OperationProfile::new(
            OperationType::MatrixMultiplication,
            &[16, 16],
            4
        );

        assert_eq!(selector.select_device(&profile), DeviceType::Cpu);
    }

    #[test]
    fn test_large_matrix_selection() {
        let selector = SmartDeviceSelector::new(vec![
            DeviceType::CoreML(0),
            DeviceType::Metal(0),
            DeviceType::Cpu,
        ]);

        // Large matrix (512x512 = 262,144 elements) should use CoreML
        let profile = OperationProfile::new(
            OperationType::MatrixMultiplication,
            &[512, 512],
            4
        );

        assert_eq!(selector.select_device(&profile), DeviceType::CoreML(0));
    }

    #[test]
    fn test_small_convolution_selection() {
        let selector = SmartDeviceSelector::new(vec![
            DeviceType::CoreML(0),
            DeviceType::Metal(0),
            DeviceType::Cpu,
        ]);

        // Small convolution should use CPU directly
        let profile = OperationProfile::new(
            OperationType::Convolution,
            &[1, 3, 32, 32], // Small batch, few channels
            4
        );

        assert_eq!(selector.select_device(&profile), DeviceType::Cpu);
    }

    #[test]
    fn test_activation_selection() {
        let selector = SmartDeviceSelector::new(vec![
            DeviceType::CoreML(0),
            DeviceType::Metal(0),
            DeviceType::Cpu,
        ]);

        // Medium activation should prefer Metal GPU
        let profile = OperationProfile::new(
            OperationType::Activation,
            &[32, 64, 128, 128], // Large enough for GPU
            4
        );

        assert_eq!(selector.select_device(&profile), DeviceType::Metal(0));
    }
}