//! ジェネリック型テンソル統合実装
//! Generic tensor unified implementation

// Core tensor implementations
pub mod core;
pub mod ops;

// Extended precision tensors
pub mod f64_tensor;
pub mod complex_tensor;

// Type system
pub mod type_conversion;
pub mod generic_tensor;

// Core re-exports (most commonly used)
pub use core::{F32Tensor, DeviceState};

// Precision tensors
pub use f64_tensor::F64Tensor;
pub use complex_tensor::ComplexTensor;

// Type conversion system
pub use type_conversion::{TensorConversion, TypeConverter, TensorVariant, PrecisionInfo};

// Generic API traits
pub use generic_tensor::{TensorOps, MatrixOps, MathOps, ComplexOps, TensorUtils, TensorFactory};

// Note: Indexing operations are implemented in core.rs