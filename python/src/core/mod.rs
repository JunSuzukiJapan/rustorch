//! Core functionality for RusTorch Python bindings
//! RusTorch Pythonバインディング用コア機能

pub mod errors;
pub mod tensor;
pub mod tensor_working;
pub mod variable;

// Re-exports for convenience
// 便利な再エクスポート
pub use errors::{RusTorchError, RusTorchResult};
pub use tensor::PyTensor as PyTensorOriginal;
pub use tensor_working::{PyTensor, zeros, ones, tensor as tensor_func};
pub use variable::PyVariable;