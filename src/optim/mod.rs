//! Optimization algorithms module
//! 最適化アルゴリズムモジュール

pub mod sgd;
pub mod adam;

use crate::autograd::Variable;
use num_traits::Float;

/// Trait for optimization algorithms
/// 最適化アルゴリズムのトレイト
pub trait Optimizer<T: Float + Send + Sync + 'static> {
    /// Performs a single optimization step
    /// 最適化の一ステップを実行します
    fn step(&mut self);
    
    /// Clears all gradients of optimized parameters
    /// 最適化対象パラメータの勾配をクリアします
    fn zero_grad(&mut self);
    
    /// Adds a parameter group to the optimizer
    /// オプティマイザーにパラメータグループを追加します
    fn add_param_group(&mut self, params: Vec<Variable<T>>);
}

// Re-export optimizers
pub use sgd::SGD;
pub use adam::Adam;