//! Unified CoreML module for Apple Neural Engine integration
//! Apple Neural Engine統合用統一CoreMLモジュール

/// CoreML feature detection and conditional compilation helpers
/// CoreML機能検出と条件付きコンパイルヘルパー
pub mod common;

/// Unified CoreML device management
/// 統一CoreMLデバイス管理
pub mod device;

/// CoreML operation implementations
/// CoreML演算実装
pub mod operations;

/// CoreML backend integration
/// CoreMLバックエンド統合
pub mod backend;

/// Hybrid execution integration with other GPU backends
/// 他のGPUバックエンドとのハイブリッド実行統合
pub mod integration;

// Re-export main interfaces
pub use common::*;
pub use device::*;
pub use operations::*;
pub use backend::*;
pub use integration::*;