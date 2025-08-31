//! Cross-platform optimization module
//! クロスプラットフォーム最適化モジュール

pub mod hardware;
pub mod platform;
pub mod simd;

pub use hardware::{AcceleratorType, HardwareCapabilities, HardwareOptimizer};
pub use platform::{OptimizationLevel, PlatformFeatures, PlatformOptimizer};
pub use simd::{SimdBackend, SimdOptimizer, VectorizedOperation};

/// Cross-platform optimization configuration
/// クロスプラットフォーム最適化設定
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable platform-specific optimizations
    pub enable_platform_opt: bool,
    /// Enable hardware acceleration
    pub enable_hardware_accel: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Target architecture
    pub target_arch: String,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig {
            enable_simd: true,
            enable_platform_opt: true,
            enable_hardware_accel: true,
            optimization_level: 2,
            target_arch: std::env::consts::ARCH.to_string(),
        }
    }
}
