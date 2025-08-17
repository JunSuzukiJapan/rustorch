//! # RusTorch
//! A PyTorch-compatible deep learning library in Rust.

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

/// Automatic differentiation module
/// 自動微分モジュール
pub mod autograd;
pub mod nn;
/// Tensor operations and data structures
/// テンソル操作とデータ構造
pub mod tensor;
pub mod utils;

/// Re-exports of commonly used items
pub mod prelude {
    pub use crate::tensor::Tensor;
    pub use crate::nn::Module;
    pub use crate::autograd::Variable;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = tensor::Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.size(), vec![3]);
    }
}
