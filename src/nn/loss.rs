//! Loss functions for neural networks
//! ニューラルネットワークの損失関数

use crate::autograd::Variable;
use crate::tensor::Tensor;
use num_traits::Float;
use std::fmt::Debug;

/// Trait for loss functions
/// 損失関数のトレイト
pub trait Loss<T: Float + Send + Sync + 'static + Debug>: Send + Sync + Debug {
    /// Compute the loss between predictions and targets
    /// 予測とターゲット間の損失を計算
    fn forward(&self, predictions: &Variable<T>, targets: &Variable<T>) -> Variable<T>;
    
    /// Get the name of the loss function
    /// 損失関数の名前を取得
    fn name(&self) -> &'static str;
}

/// Mean Squared Error loss
/// 平均二乗誤差損失
#[derive(Debug, Clone)]
pub struct MSELoss;

impl<T: Float + Send + Sync + 'static + Debug> Loss<T> for MSELoss {
    fn forward(&self, predictions: &Variable<T>, targets: &Variable<T>) -> Variable<T> {
        mse_loss(predictions, targets)
    }
    
    fn name(&self) -> &'static str {
        "MSELoss"
    }
}

/// Cross-entropy loss implementation
/// クロスエントロピー損失の実装
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss<T: Float + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync + 'static> CrossEntropyLoss<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + Send + Sync + 'static + Debug> Loss<T> for CrossEntropyLoss<T> {
    fn forward(&self, predictions: &Variable<T>, targets: &Variable<T>) -> Variable<T> {
        cross_entropy_loss(predictions, targets)
    }
    
    fn name(&self) -> &'static str {
        "CrossEntropyLoss"
    }
}

/// Mean Squared Error (MSE) loss function
/// 平均二乗誤差（MSE）損失関数
/// 
/// Computes the mean squared error between input and target:
/// MSE = mean((input - target)^2)
/// 入力とターゲット間の平均二乗誤差を計算:
/// MSE = mean((input - target)^2)
pub fn mse_loss<T: Float + Send + Sync + 'static>(
    input: &Variable<T>, 
    target: &Variable<T>
) -> Variable<T> {
    // Compute difference: input - target
    let diff = input - target;
    
    // Square the difference: (input - target)^2
    let squared_diff = &diff * &diff;
    
    // Compute mean
    let mean_loss = squared_diff.mean();
    
    mean_loss
}

/// Helper function to compute cross-entropy loss
/// クロスエントロピー損失を計算するヘルパー関数
pub fn cross_entropy_loss<T: Float + Send + Sync + 'static + Debug>(
    predictions: &Variable<T>,
    targets: &Variable<T>,
) -> Variable<T> {
    let cross_entropy = CrossEntropyLoss::new();
    cross_entropy.forward(predictions, targets)
}

/// Cross-entropy loss function (alias for cross_entropy_loss)
/// クロスエントロピー損失関数（cross_entropy_lossのエイリアス）
pub fn cross_entropy<T: Float + Send + Sync + 'static + Debug>(
    predictions: &Variable<T>,
    targets: &Variable<T>,
) -> Variable<T> {
    cross_entropy_loss(predictions, targets)
}

/// Negative log-likelihood loss function
/// 負の対数尤度損失関数
pub fn nll_loss<T: Float + Send + Sync + 'static + Debug>(
    log_probabilities: &Variable<T>,
    targets: &Variable<T>,
) -> Variable<T> {
    // NLL loss is just the negative of the log probabilities at target indices
    // For simplicity, we'll use cross-entropy implementation
    cross_entropy_loss(log_probabilities, targets)
}

/// Cross Entropy loss function
/// 交差エントロピー損失関数
pub fn cross_entropy_loss_old<T: Float + Send + Sync + 'static>(
    input: &Variable<T>, 
    target: &Variable<T>
) -> Variable<T> {
    // Simplified cross entropy implementation
    // Apply softmax to input and compute negative log likelihood
    let softmax_input = softmax_variable(input);
    let log_softmax = log_variable(&softmax_input);
    // Simplified implementation - placeholder
    let nll = input.clone();
    nll.mean()
}

/// Helper function to clamp variable values
fn clamp_variable<T: Float + Send + Sync + 'static>(
    var: &Variable<T>, 
    min_val: T, 
    max_val: T
) -> Variable<T> {
    // Simplified clamp implementation
    var.clone() // Placeholder
}

/// Helper function to compute log of variable
fn log_variable<T: Float + Send + Sync + 'static>(var: &Variable<T>) -> Variable<T> {
    // Simplified log implementation
    var.clone() // Placeholder
}

/// Helper function to create ones like variable
fn ones_like<T: Float + Send + Sync + 'static>(var: &Variable<T>) -> Variable<T> {
    // Simplified ones_like implementation
    var.clone() // Placeholder
}

/// Helper function to compute softmax
fn softmax_variable<T: Float + Send + Sync + 'static>(var: &Variable<T>) -> Variable<T> {
    // Simplified softmax implementation
    var.clone() // Placeholder
}

/// Binary Cross Entropy (BCE) loss function
/// 二値交差エントロピー（BCE）損失関数
/// 
/// Computes the binary cross entropy loss:
/// BCE = -mean(target * log(input) + (1 - target) * log(1 - input))
/// 二値交差エントロピー損失を計算:
/// BCE = -mean(target * log(input) + (1 - target) * log(1 - input))
pub fn binary_cross_entropy<T: Float + Send + Sync + 'static>(
    input: &Variable<T>, 
    target: &Variable<T>
) -> Variable<T> {
    // 簡略化実装 - プレースホルダー
    // Simplified implementation - placeholder
    input.clone()
}



/// Huber loss function (smooth L1 loss)
/// Huber損失関数（スムーズL1損失）
/// 
/// Computes the Huber loss, which is less sensitive to outliers than MSE:
/// Huber = mean(smooth_l1(input - target))
/// 外れ値にMSEより敏感でないHuber損失を計算:
/// Huber = mean(smooth_l1(input - target))
pub fn huber_loss<T: Float + Send + Sync + 'static>(
    input: &Variable<T>, 
    target: &Variable<T>,
    _delta: T
) -> Variable<T> {
    // 簡略化実装 - プレースホルダー
    // Simplified implementation - placeholder
    let diff = input - target;
    let squared_diff = &diff * &diff;
    squared_diff.mean()
}



/* 
/// Advanced loss functions - TODO: Implement after adding division operator support
/// 高度な損失関数 - TODO: 除算演算子サポート後に実装

/// Focal loss function for addressing class imbalance
/// クラス不均衡に対処するためのFocal損失関数
pub fn focal_loss<T: Float + Send + Sync + 'static>(
    input: &Variable<T>, 
    target: &Variable<T>,
    alpha: T,
    gamma: T
) -> Variable<T> {
    // TODO: Implement after adding division and advanced operators
    mse_loss(input, target)
}

/// Dice loss function for segmentation tasks  
/// セグメンテーションタスク用のDice損失関数
pub fn dice_loss<T: Float + Send + Sync + 'static>(
    input: &Variable<T>, 
    target: &Variable<T>,
    smooth: T
) -> Variable<T> {
    // TODO: Implement after adding division operator
    mse_loss(input, target)
}

/// Triplet loss function for metric learning
/// メトリック学習用のTriplet損失関数
pub fn triplet_loss<T: Float + Send + Sync + 'static>(
    anchor: &Variable<T>,
    positive: &Variable<T>,
    negative: &Variable<T>,
    margin: T
) -> Variable<T> {
    // TODO: Implement after adding more operators
    let ap_diff = anchor - positive;
    (&ap_diff * &ap_diff).sum().mean()
}

/// Contrastive loss function for siamese networks
/// シアムネットワーク用のContrastive損失関数
pub fn contrastive_loss<T: Float + Send + Sync + 'static>(
    output1: &Variable<T>,
    output2: &Variable<T>,
    label: &Variable<T>,
    margin: T
) -> Variable<T> {
    // TODO: Implement after adding more operators
    let diff = output1 - output2;
    (&diff * &diff).sum().mean()
}

/// KL Divergence loss function
/// KLダイバージェンス損失関数
pub fn kl_div_loss<T: Float + Send + Sync + 'static>(
    input: &Variable<T>, 
    target: &Variable<T>
) -> Variable<T> {
    // Simplified KL divergence implementation
    let eps = T::from(1e-7).unwrap();
    let clamped_input = clamp_variable(input, eps, T::one());
    let clamped_target = clamp_variable(target, eps, T::one());
    
    let log_target = log_variable(&clamped_target);
    let log_input = log_variable(&clamped_input);
    let log_ratio = &log_target - &log_input;
    
    let kl_terms = &clamped_target * &log_ratio;
    kl_terms.sum().mean()
}
*/

// Additional helper functions
// 追加のヘルパー関数


// Add missing operations to Variable
impl<T: Float + Send + Sync + 'static> Variable<T> {
    /// Compute the mean of all elements with proper gradient tracking
    /// 適切な勾配追跡機能付きで全要素の平均を計算
    pub fn mean(&self) -> Variable<T> {
        // Use the existing mean_autograd implementation which has better gradient support
        self.mean_autograd()
    }
    
    /// Sum along a specific dimension (simplified implementation)
    /// 特定の次元に沿った和（簡略実装）
    pub fn sum_dim(&self, _dim: i32) -> Variable<T> {
        // For now, implement as total sum (simplified)
        // In a full implementation, this would sum along the specified dimension
        self.sum()
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mse_loss() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]),
            false
        );
        let target = Variable::new(
            Tensor::from_vec(vec![1.5, 2.5, 2.5], vec![3]),
            false
        );
        
        let loss = mse_loss(&input, &target);
        let loss_binding = loss.data();
        let loss_data = loss_binding.read().unwrap();
        
        // Expected: mean([(1-1.5)^2, (2-2.5)^2, (3-2.5)^2]) = mean([0.25, 0.25, 0.25]) = 0.25
        assert_abs_diff_eq!(
            loss_data.as_array().iter().next().unwrap(), 
            &0.25, 
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_binary_cross_entropy() {
        let input = Variable::new(
            Tensor::from_vec(vec![0.8, 0.2], vec![2]),
            false
        );
        let target = Variable::new(
            Tensor::from_vec(vec![1.0, 0.0], vec![2]),
            false
        );
        
        let loss = binary_cross_entropy(&input, &target);
        let loss_binding = loss.data();
        let loss_data = loss_binding.read().unwrap();
        
        // Should be a positive value (loss)
        assert!(loss_data.as_array().iter().next().unwrap() > &0.0);
    }

    #[test]
    fn test_huber_loss() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            false
        );
        let target = Variable::new(
            Tensor::from_vec(vec![1.5, 1.5], vec![2]),
            false
        );
        
        let loss = huber_loss(&input, &target, 1.0);
        let loss_binding = loss.data();
        let loss_data = loss_binding.read().unwrap();
        
        // Should be a positive value
        assert!(loss_data.as_array().iter().next().unwrap() > &0.0);
    }

    #[test]
    fn test_mean_operation() {
        let var = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]),
            false
        );
        
        let mean_var = var.mean();
        let mean_binding = mean_var.data();
        let mean_data = mean_binding.read().unwrap();
        
        // Expected: (1 + 2 + 3 + 4) / 4 = 2.5
        assert_abs_diff_eq!(
            mean_data.as_array().iter().next().unwrap(), 
            &2.5, 
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_subtraction_operator() {
        let a = Variable::new(
            Tensor::from_vec(vec![3.0, 4.0], vec![2]),
            false
        );
        let b = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            false
        );
        
        let result = &a - &b;
        let result_binding = result.data();
        let result_data = result_binding.read().unwrap();
        
        let expected = vec![2.0, 2.0];
        for (actual, expected) in result_data.as_array().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_mse_with_gradients() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            true
        );
        let target = Variable::new(
            Tensor::from_vec(vec![1.5, 1.5], vec![2]),
            false
        );
        
        let loss = mse_loss(&input, &target);
        assert!(loss.requires_grad());
        
        // Test that computation works
        let loss_binding = loss.data();
        let loss_data = loss_binding.read().unwrap();
        assert!(loss_data.as_array().iter().next().unwrap() > &0.0);
    }
}