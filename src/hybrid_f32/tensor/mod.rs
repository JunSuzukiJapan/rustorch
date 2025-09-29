//! ジェネリック型テンソル統合実装
//! Generic tensor unified implementation

pub mod core;
pub mod f64_tensor;
pub mod complex_tensor;
pub mod type_conversion;
pub mod generic_tensor;

// Re-exports
pub use core::*;
pub use f64_tensor::F64Tensor;
pub use complex_tensor::ComplexTensor;
pub use type_conversion::{TensorConversion, TypeConverter, TensorVariant, PrecisionInfo};
pub use generic_tensor::{TensorOps, MatrixOps, MathOps, ComplexOps, TensorUtils, TensorFactory};