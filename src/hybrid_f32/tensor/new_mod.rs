//! F32Tensor - リファクタリング済みテンソル実装
//! F32Tensor - Refactored tensor implementation

pub mod core;
pub mod ops;
pub mod indexing;
pub mod math;
pub mod stats;

// Re-exports
pub use core::*;
pub use ops::*;
pub use indexing::*;

use crate::hybrid_f32_experimental;