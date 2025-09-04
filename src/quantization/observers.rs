//! Statistical observers for quantization calibration
//! 量子化キャリブレーション用統計観測器

// Re-export observers from calibration module for backward compatibility
// 後方互換性のためにキャリブレーションモジュールから観測器を再エクスポート
pub use super::calibration::{Observer, MinMaxObserver, HistogramObserver};

// This module serves as a focused interface for observers
// while the actual implementations remain in the calibration module
// このモジュールは観測器の焦点を絞ったインターフェースとして機能し、
// 実際の実装はキャリブレーションモジュールに残す