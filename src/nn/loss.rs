//! Loss functions for neural networks
//! ニューラルネットワークの損失関数

use crate::autograd::Variable;
use crate::tensor::Tensor;
use num_traits::Float;

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
    // Clamp input to avoid log(0)
    let eps = T::from(1e-7).unwrap();
    let clamped_input = clamp_variable(input, eps, T::one() - eps);
    
    // Compute log terms
    let log_input = log_variable(&clamped_input);
    let one_minus_input = ones_like(&clamped_input) - &clamped_input;
    let log_one_minus_input = log_variable(&one_minus_input);
    
    // Compute BCE: -mean(target * log(input) + (1 - target) * log(1 - input))
    let one_minus_target = ones_like(target) - target;
    let term1 = target * &log_input;
    let term2 = &one_minus_target * &log_one_minus_input;
    let bce = &term1 + &term2;
    let mean_bce = bce.mean();
    
    // Return negative mean
    zeros_like(&mean_bce) - &mean_bce
}

/// Cross Entropy loss function
/// 交差エントロピー損失関数
/// 
/// Computes the cross entropy loss for multi-class classification:
/// CE = -mean(sum(target * log(softmax(input))))
/// 多クラス分類の交差エントロピー損失を計算:
/// CE = -mean(sum(target * log(softmax(input))))
pub fn cross_entropy<T: Float + Send + Sync + 'static>(
    input: &Variable<T>, 
    target: &Variable<T>
) -> Variable<T> {
    // Apply softmax to input
    let softmax_input = softmax_variable(input);
    
    // Clamp to avoid log(0)
    let eps = T::from(1e-7).unwrap();
    let clamped_softmax = clamp_variable(&softmax_input, eps, T::one());
    
    // Compute log softmax
    let log_softmax = log_variable(&clamped_softmax);
    
    // Compute cross entropy: -mean(sum(target * log(softmax(input))))
    let ce_terms = target * &log_softmax;
    let ce_sum = ce_terms.sum();
    let mean_ce = ce_sum.mean();
    
    // Return negative mean
    zeros_like(&mean_ce) - &mean_ce
}

/// Negative Log Likelihood (NLL) loss function
/// 負の対数尤度（NLL）損失関数
/// 
/// Computes the negative log likelihood loss:
/// NLL = -mean(sum(target * log(input)))
/// 負の対数尤度損失を計算:
/// NLL = -mean(sum(target * log(input)))
pub fn nll_loss<T: Float + Send + Sync + 'static>(
    input: &Variable<T>, 
    target: &Variable<T>
) -> Variable<T> {
    // Clamp input to avoid log(0)
    let eps = T::from(1e-7).unwrap();
    let clamped_input = clamp_variable(input, eps, T::one());
    
    // Compute log
    let log_input = log_variable(&clamped_input);
    
    // Compute NLL: -mean(sum(target * log(input)))
    let nll_terms = target * &log_input;
    let nll_sum = nll_terms.sum();
    let mean_nll = nll_sum.mean();
    
    // Return negative mean
    zeros_like(&mean_nll) - &mean_nll
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
    delta: T
) -> Variable<T> {
    // Compute difference
    let diff = input - target;
    let abs_diff = abs_variable(&diff);
    
    // Apply Huber formula
    let huber_values = smooth_l1_variable(&abs_diff, delta);
    
    // Compute mean
    huber_values.mean()
}

// Helper functions for mathematical operations on Variables
// Variable上での数学演算のヘルパー関数

fn clamp_variable<T: Float + Send + Sync + 'static>(
    var: &Variable<T>, 
    min_val: T, 
    max_val: T
) -> Variable<T> {
    let input_data = var.data().read().unwrap().clone();
    let clamped_data = apply_clamp(&input_data, min_val, max_val);
    
    if var.requires_grad() {
        Variable::new(clamped_data, true)
    } else {
        Variable::new(clamped_data, false)
    }
}

fn log_variable<T: Float + Send + Sync + 'static>(var: &Variable<T>) -> Variable<T> {
    let input_data = var.data().read().unwrap().clone();
    let log_data = apply_log(&input_data);
    
    if var.requires_grad() {
        Variable::new(log_data, true)
    } else {
        Variable::new(log_data, false)
    }
}

fn abs_variable<T: Float + Send + Sync + 'static>(var: &Variable<T>) -> Variable<T> {
    let input_data = var.data().read().unwrap().clone();
    let abs_data = apply_abs(&input_data);
    
    if var.requires_grad() {
        Variable::new(abs_data, true)
    } else {
        Variable::new(abs_data, false)
    }
}

fn ones_like<T: Float + Send + Sync + 'static>(var: &Variable<T>) -> Variable<T> {
    let binding = var.data();
    let input_data = binding.read().unwrap();
    let ones_data = Tensor::ones(input_data.shape());
    
    Variable::new(ones_data, false)
}

fn zeros_like<T: Float + Send + Sync + 'static>(var: &Variable<T>) -> Variable<T> {
    let binding = var.data();
    let input_data = binding.read().unwrap();
    let zeros_data = Tensor::zeros(input_data.shape());
    
    Variable::new(zeros_data, false)
}

fn softmax_variable<T: Float + Send + Sync + 'static>(var: &Variable<T>) -> Variable<T> {
    use crate::nn::activation::softmax;
    softmax(var)
}

fn smooth_l1_variable<T: Float + Send + Sync + 'static>(
    var: &Variable<T>, 
    delta: T
) -> Variable<T> {
    let input_data = var.data().read().unwrap().clone();
    let smooth_l1_data = apply_smooth_l1(&input_data, delta);
    
    if var.requires_grad() {
        Variable::new(smooth_l1_data, true)
    } else {
        Variable::new(smooth_l1_data, false)
    }
}

// Tensor-level mathematical operations
// テンソルレベルの数学演算

fn apply_clamp<T: Float + 'static>(tensor: &Tensor<T>, min_val: T, max_val: T) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data.iter().map(|&x| {
        if x < min_val { min_val }
        else if x > max_val { max_val }
        else { x }
    }).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_log<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data.iter().map(|&x| x.ln()).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_abs<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data.iter().map(|&x| x.abs()).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_smooth_l1<T: Float + 'static>(tensor: &Tensor<T>, delta: T) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data.iter().map(|&x| {
        if x.abs() < delta {
            T::from(0.5).unwrap() * x * x / delta
        } else {
            x.abs() - T::from(0.5).unwrap() * delta
        }
    }).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
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

#[allow(dead_code)]
fn pow_variable<T: Float + Send + Sync + 'static>(
    var: &Variable<T>, 
    exponent: T
) -> Variable<T> {
    let input_data = var.data().read().unwrap().clone();
    let pow_data = apply_pow(&input_data, exponent);
    
    if var.requires_grad() {
        Variable::new(pow_data, true)
    } else {
        Variable::new(pow_data, false)
    }
}

#[allow(dead_code)]
fn sqrt_variable<T: Float + Send + Sync + 'static>(var: &Variable<T>) -> Variable<T> {
    let input_data = var.data().read().unwrap().clone();
    let sqrt_data = apply_sqrt(&input_data);
    
    if var.requires_grad() {
        Variable::new(sqrt_data, true)
    } else {
        Variable::new(sqrt_data, false)
    }
}

#[allow(dead_code)]
fn relu_variable<T: Float + Send + Sync + 'static>(
    var: &Variable<T>,
    _threshold: &Variable<T>
) -> Variable<T> {
    use crate::nn::activation::relu;
    relu(var)
}

#[allow(dead_code)]
fn scalar_variable<T: Float + Send + Sync + 'static>(value: T) -> Variable<T> {
    let scalar_tensor = Tensor::from_vec(vec![value], vec![]);
    Variable::new(scalar_tensor, false)
}

#[allow(dead_code)]
fn flatten_variable<T: Float + Send + Sync + 'static>(var: &Variable<T>) -> Variable<T> {
    let binding = var.data();
    let input_data = binding.read().unwrap();
    let total_elements = input_data.len();
    let flattened_tensor = Tensor::from_vec(
        input_data.as_array().iter().cloned().collect(),
        vec![total_elements]
    );
    
    if var.requires_grad() {
        Variable::new(flattened_tensor, true)
    } else {
        Variable::new(flattened_tensor, false)
    }
}

#[allow(dead_code)]
fn apply_pow<T: Float + 'static>(tensor: &Tensor<T>, exponent: T) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data.iter().map(|&x| x.powf(exponent)).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

#[allow(dead_code)]
fn apply_sqrt<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data.iter().map(|&x| x.sqrt()).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

// Add missing operations to Variable
impl<T: Float + Send + Sync + 'static> Variable<T> {
    /// Compute the mean of all elements
    /// 全要素の平均を計算
    pub fn mean(&self) -> Variable<T> {
        let binding = self.data();
        let input_data = binding.read().unwrap();
        let sum_val = input_data.sum();
        let count = T::from(input_data.len()).unwrap();
        
        // Create a scalar tensor with the mean value
        let mean_val = *sum_val.as_array().iter().next().unwrap() / count;
        let mean_tensor = Tensor::from_vec(vec![mean_val], vec![]);
        
        if self.requires_grad() {
            Variable::new(mean_tensor, true)
        } else {
            Variable::new(mean_tensor, false)
        }
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