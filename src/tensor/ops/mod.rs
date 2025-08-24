//! Tensor operations organized by category
//! カテゴリ別に整理されたテンソル操作
//!
//! This module contains all tensor operations split into logical categories
//! for better code organization and maintainability.
//! このモジュールはコードの組織化と保守性向上のため、論理的なカテゴリに分割された
//! 全テンソル操作を含みます。

/// Basic arithmetic operations (add, sub, mul, div, etc.)
/// 基本算術演算（加算、減算、乗算、除算など）
pub mod basic;

/// Linear algebra operations (matmul, SVD, QR, LU, etc.)
/// 線形代数演算（行列乗算、SVD、QR、LUなど）
pub mod linalg;

/// Reduction and statistical operations (sum, mean, max, min, etc.)
/// リダクションと統計演算（合計、平均、最大、最小など）
pub mod reduction;

/// Mathematical functions (exp, sin, cos, sqrt, etc.)
/// 数学関数（指数、正弦、余弦、平方根など）
pub mod math;

/// Signal processing operations (FFT, IFFT, windowing, etc.)
/// 信号処理演算（FFT、IFFT、窓関数など）
pub mod signal;

/// Tensor creation operations (randn, rand, arange, linspace, etc.)
/// テンソル作成演算（正規乱数、一様乱数、等差数列、線形等間隔など）
pub mod creation;

/// Utility operations (map, select, chunk, concat, etc.)
/// ユーティリティ演算（マップ、選択、チャンク、結合など）
pub mod utils;

// Re-export WindowType for convenience
// WindowTypeを便利のため再エクスポート
pub use signal::WindowType;

// Note: Re-exports temporarily disabled to avoid conflicts with legacy operations.rs
// レガシーoperations.rsとの競合を避けるため再エクスポートを一時無効化
// pub use basic::*;
// pub use linalg::*;
// pub use reduction::*;
// pub use math::*;
// pub use signal::*;
// pub use creation::*;
// pub use utils::*;