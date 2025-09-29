//! ジェネリック型テンソル統合実装
//! Generic tensor unified implementation

// Core tensor implementations
pub mod core;
pub mod ops;

// Extended precision tensors
pub mod complex_tensor;
pub mod f64_tensor;

// Type system
pub mod generic_tensor;
pub mod type_conversion;

// Core re-exports (most commonly used)
pub use core::{DeviceState, F32Tensor};

// Precision tensors
pub use complex_tensor::ComplexTensor;
pub use f64_tensor::F64Tensor;

// Type conversion system
pub use type_conversion::{PrecisionInfo, TensorConversion, TensorVariant, TypeConverter};

// Generic API traits
pub use generic_tensor::{ComplexOps, MathOps, MatrixOps, TensorFactory, TensorOps, TensorUtils};

// Note: Indexing operations are implemented in core.rs
