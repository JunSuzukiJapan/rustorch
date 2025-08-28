//! Organized tensor operations module
//! 整理されたテンソル操作モジュール
//!
//! This module provides a clean, organized interface to tensor operations,
//! grouped by functionality and abstracted through traits for better code organization.

pub mod zero_copy;

// Re-export important traits and types
pub use zero_copy::{ZeroCopyOps, TensorIterOps};