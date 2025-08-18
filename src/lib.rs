//! # RusTorch
//! A PyTorch-compatible deep learning library in Rust.

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

/// Automatic differentiation module
/// 自動微分モジュール
pub mod autograd;
/// Data loading and processing utilities
/// データ読み込みと処理のユーティリティ
pub mod data;
pub mod nn;
/// Optimization algorithms
/// 最適化アルゴリズム
pub mod optim;
/// Tensor operations and data structures
/// テンソル操作とデータ構造
pub mod tensor;
pub mod utils;

/// Re-exports of commonly used items
pub mod prelude {
    pub use crate::tensor::Tensor;
    pub use crate::nn::{Module, Linear, Conv2d, MaxPool2d, AvgPool2d, BatchNorm1d, BatchNorm2d, Dropout, AlphaDropout, dropout};
    pub use crate::autograd::Variable;
    pub use crate::nn::activation::{relu, sigmoid, tanh, leaky_relu, softmax, gelu, swish, elu, selu, mish, hardswish};
    pub use crate::nn::loss::{mse_loss, binary_cross_entropy, cross_entropy, nll_loss, huber_loss};
    pub use crate::optim::{Optimizer, SGD, Adam};
    pub use crate::data::{Dataset, TensorDataset, DataLoader};
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
