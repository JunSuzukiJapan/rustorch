//! Complex number support for tensors - Modular Organization
//! テンソルの複素数サポート - モジュール構成
//!
//! This module provides comprehensive complex number support for tensor operations,
//! organized into logical submodules for better maintainability:
//! 
//! - `core`: Core Complex struct and basic operations
//! - `arithmetic`: Arithmetic operations (+, -, *, /, etc.)
//! - `math`: Mathematical functions (exp, log, sin, cos, etc.)
//! - `tensor_ops`: Tensor-specific operations and factory functions
//! - `matrix`: Matrix operations and linear algebra functions

pub mod arithmetic;
pub mod core;
pub mod math;
pub mod matrix;
pub mod tensor_ops;

// Re-export the main Complex type and all its implementations
pub use core::Complex;

// Re-export all traits and implementations for backward compatibility
pub use arithmetic::*;
pub use math::*;
pub use matrix::*;
pub use tensor_ops::*;