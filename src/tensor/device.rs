//! Device management for tensor operations
//! テンソル操作用デバイス管理

use serde::{Deserialize, Serialize};
use std::fmt;

/// Device types for tensor storage and computation
/// テンソルストレージと計算用デバイスタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    /// CPU device
    /// CPUデバイス
    Cpu,
    /// GPU device with optional device index
    /// GPU デバイス（オプションのデバイスインデックス付き）
    Cuda(usize),
    /// Metal Performance Shaders (macOS)
    /// Metal Performance Shaders（macOS）
    Mps,
    /// WebAssembly target
    /// WebAssemblyターゲット
    Wasm,
}

impl Default for Device {
    fn default() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Device::Wasm
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Device::Cpu
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{}", idx),
            Device::Mps => write!(f, "mps"),
            Device::Wasm => write!(f, "wasm"),
        }
    }
}

impl Device {
    /// Check if device is CPU
    /// CPUデバイスかチェック
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Check if device is CUDA GPU
    /// CUDA GPUかチェック
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Check if device is MPS
    /// MPSデバイスかチェック
    pub fn is_mps(&self) -> bool {
        matches!(self, Device::Mps)
    }

    /// Check if device is WASM
    /// WASMデバイスかチェック
    pub fn is_wasm(&self) -> bool {
        matches!(self, Device::Wasm)
    }

    /// Get CUDA device index if applicable
    /// 該当する場合CUDAデバイスインデックスを取得
    pub fn cuda_index(&self) -> Option<usize> {
        match self {
            Device::Cuda(idx) => Some(*idx),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        let cpu = Device::Cpu;
        let cuda = Device::Cuda(0);
        let mps = Device::Mps;
        let wasm = Device::Wasm;

        assert!(cpu.is_cpu());
        assert!(cuda.is_cuda());
        assert!(mps.is_mps());
        assert!(wasm.is_wasm());
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Cuda(1).to_string(), "cuda:1");
        assert_eq!(Device::Mps.to_string(), "mps");
        assert_eq!(Device::Wasm.to_string(), "wasm");
    }

    #[test]
    fn test_cuda_index() {
        assert_eq!(Device::Cuda(0).cuda_index(), Some(0));
        assert_eq!(Device::Cuda(5).cuda_index(), Some(5));
        assert_eq!(Device::Cpu.cuda_index(), None);
        assert_eq!(Device::Mps.cuda_index(), None);
    }

    #[test]
    fn test_default_device() {
        let default_device = Device::default();

        #[cfg(target_arch = "wasm32")]
        assert!(default_device.is_wasm());

        #[cfg(not(target_arch = "wasm32"))]
        assert!(default_device.is_cpu());
    }
}
