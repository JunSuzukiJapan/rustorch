//! f32統一ハイブリッドニューラルネットワークモジュール
//! f32 Unified Hybrid Neural Network Module
//!
//! フェーズ5: 高度ニューラルネットワーク機能
//! Phase 5: Advanced Neural Network Features
//!
//! このモジュールは、f32精度で最適化されたニューラルネットワーク機能を提供します。
//! Neural Engine、Metal GPU、CPUでの統一実行をサポートし、変換コストゼロを実現します。

use super::tensor::F32Tensor;
use crate::error::RusTorchResult;
use std::collections::HashMap;

/// f32統一ニューラルネットワーク層の基底トレイト
/// Base trait for f32 unified neural network layers
pub trait F32Layer: std::fmt::Debug + Send + Sync {
    /// 順伝播
    /// Forward pass
    fn forward(&mut self, input: &F32Tensor) -> RusTorchResult<F32Tensor>;

    /// 逆伝播（勾配計算）
    /// Backward pass (gradient computation)
    fn backward(&mut self, grad_output: &F32Tensor) -> RusTorchResult<F32Tensor>;

    /// パラメータの取得
    /// Get parameters
    fn parameters(&self) -> Vec<&F32Tensor>;

    /// パラメータの更新
    /// Update parameters
    fn update_parameters(&mut self, learning_rate: f32) -> RusTorchResult<()>;
}

/// f32線形層（全結合層）
/// f32 Linear layer (fully connected layer)
#[derive(Debug, Clone)]
pub struct F32Linear {
    pub weight: F32Tensor,
    pub bias: Option<F32Tensor>,
    pub input_features: usize,
    pub output_features: usize,

    // 勾配記録用
    weight_grad: Option<F32Tensor>,
    bias_grad: Option<F32Tensor>,
    last_input: Option<F32Tensor>,
}

impl F32Linear {
    /// 新しい線形層を作成
    /// Create new linear layer
    pub fn new(input_features: usize, output_features: usize, bias: bool) -> RusTorchResult<Self> {
        // Xavier初期化
        let scale = (2.0 / (input_features + output_features) as f32).sqrt();
        let weight_data: Vec<f32> = (0..input_features * output_features)
            .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
            .collect();

        let weight = F32Tensor::from_vec(weight_data, vec![output_features, input_features])?;

        let bias = if bias {
            Some(F32Tensor::zeros(&[output_features]))
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            input_features,
            output_features,
            weight_grad: None,
            bias_grad: None,
            last_input: None,
        })
    }

    /// 重みを手動設定
    /// Set weights manually
    pub fn set_weight(&mut self, weight: F32Tensor) -> RusTorchResult<()> {
        if weight.shape() != &[self.output_features, self.input_features] {
            return Err(format!(
                "Weight shape mismatch: expected [{}, {}], got {:?}",
                self.output_features, self.input_features, weight.shape()
            ).into());
        }
        self.weight = weight;
        Ok(())
    }

    /// バイアスを手動設定
    /// Set bias manually
    pub fn set_bias(&mut self, bias: F32Tensor) -> RusTorchResult<()> {
        if bias.shape() != &[self.output_features] {
            return Err(format!(
                "Bias shape mismatch: expected [{}], got {:?}",
                self.output_features, bias.shape()
            ).into());
        }
        self.bias = Some(bias);
        Ok(())
    }
}

impl F32Layer for F32Linear {
    fn forward(&mut self, input: &F32Tensor) -> RusTorchResult<F32Tensor> {
        // 入力を記録（逆伝播用）
        self.last_input = Some(input.clone()?);

        // 線形変換: output = input @ weight.T + bias
        let output = input.matmul(&self.weight.transpose()?)?;

        if let Some(ref bias) = self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }

    fn backward(&mut self, grad_output: &F32Tensor) -> RusTorchResult<F32Tensor> {
        if let Some(ref last_input) = self.last_input {
            // 重みの勾配: weight_grad = grad_output.T @ input
            self.weight_grad = Some(grad_output.transpose()?.matmul(last_input)?);

            // バイアスの勾配: bias_grad = sum(grad_output, dim=0)
            if self.bias.is_some() {
                self.bias_grad = Some(grad_output.sum_dim(0, false)?);
            }

            // 入力の勾配: input_grad = grad_output @ weight
            grad_output.matmul(&self.weight)
        } else {
            Err("No forward pass recorded for backward computation".into())
        }
    }

    fn parameters(&self) -> Vec<&F32Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn update_parameters(&mut self, learning_rate: f32) -> RusTorchResult<()> {
        // 重みの更新
        if let Some(ref weight_grad) = self.weight_grad {
            let lr_tensor = F32Tensor::from_scalar(learning_rate)?;
            let update = weight_grad.mul(&lr_tensor)?;
            self.weight = self.weight.sub(&update)?;
        }

        // バイアスの更新
        if let (Some(ref mut bias), Some(ref bias_grad)) = (&mut self.bias, &self.bias_grad) {
            let lr_tensor = F32Tensor::from_scalar(learning_rate)?;
            let update = bias_grad.mul(&lr_tensor)?;
            *bias = bias.sub(&update)?;
        }

        // 勾配をクリア
        self.weight_grad = None;
        self.bias_grad = None;

        Ok(())
    }
}

/// f32活性化関数
/// f32 Activation functions
#[derive(Debug, Clone)]
pub enum F32Activation {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU(f32), // slope parameter
    GELU,
}

impl F32Activation {
    /// 活性化関数の適用
    /// Apply activation function
    pub fn forward(&self, input: &F32Tensor) -> RusTorchResult<F32Tensor> {
        match self {
            F32Activation::ReLU => input.relu(),
            F32Activation::Sigmoid => input.sigmoid(),
            F32Activation::Tanh => input.tanh(),
            F32Activation::LeakyReLU(slope) => {
                let zero = F32Tensor::zeros(input.shape());
                let positive = input.maximum(&zero)?;
                let negative = input.minimum(&zero)?.mul(&F32Tensor::from_scalar(*slope)?)?;
                positive.add(&negative)
            },
            F32Activation::GELU => {
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let sqrt_2_pi = F32Tensor::from_scalar(0.7978845608f32)?; // sqrt(2/π)
                let coeff = F32Tensor::from_scalar(0.044715f32)?;
                let half = F32Tensor::from_scalar(0.5f32)?;
                let one = F32Tensor::from_scalar(1.0f32)?;

                let x_cubed = input.power(&F32Tensor::from_scalar(3.0f32)?)?;
                let inner = input.add(&x_cubed.mul(&coeff)?)?;
                let scaled = inner.mul(&sqrt_2_pi)?;
                let tanh_val = scaled.tanh()?;
                let one_plus_tanh = one.add(&tanh_val)?;

                input.mul(&half)?.mul(&one_plus_tanh)
            }
        }
    }

    /// 活性化関数の導関数
    /// Derivative of activation function
    pub fn backward(&self, input: &F32Tensor, grad_output: &F32Tensor) -> RusTorchResult<F32Tensor> {
        let derivative = match self {
            F32Activation::ReLU => {
                let zero = F32Tensor::zeros(input.shape());
                let one = F32Tensor::ones(input.shape());
                input.gt(&zero)?.to_type(&one)?
            },
            F32Activation::Sigmoid => {
                let sigmoid_out = input.sigmoid()?;
                let one = F32Tensor::from_scalar(1.0f32)?;
                let one_minus_sigmoid = one.sub(&sigmoid_out)?;
                sigmoid_out.mul(&one_minus_sigmoid)?
            },
            F32Activation::Tanh => {
                let tanh_out = input.tanh()?;
                let one = F32Tensor::from_scalar(1.0f32)?;
                let tanh_squared = tanh_out.power(&F32Tensor::from_scalar(2.0f32)?)?;
                one.sub(&tanh_squared)?
            },
            F32Activation::LeakyReLU(slope) => {
                let zero = F32Tensor::zeros(input.shape());
                let one = F32Tensor::ones(input.shape());
                let slope_tensor = F32Tensor::from_scalar(*slope)?;
                let positive_mask = input.gt(&zero)?.to_type(&one)?;
                let negative_mask = input.le(&zero)?.to_type(&slope_tensor)?;
                positive_mask.add(&negative_mask)?
            },
            F32Activation::GELU => {
                // Approximate GELU derivative
                let sqrt_2_pi = F32Tensor::from_scalar(0.7978845608f32)?;
                let coeff = F32Tensor::from_scalar(0.044715f32)?;
                let half = F32Tensor::from_scalar(0.5f32)?;
                let one = F32Tensor::from_scalar(1.0f32)?;
                let three = F32Tensor::from_scalar(3.0f32)?;

                let x_squared = input.power(&F32Tensor::from_scalar(2.0f32)?)?;
                let three_coeff_x_squared = three.mul(&coeff)?.mul(&x_squared)?;
                let derivative_inner = one.add(&three_coeff_x_squared)?;
                let tanh_derivative = derivative_inner.mul(&sqrt_2_pi)?;

                // Simplified approximation
                let sigmoid_approx = input.mul(&F32Tensor::from_scalar(1.702f32)?)?.sigmoid()?;
                sigmoid_approx
            }
        };

        grad_output.mul(&derivative)
    }
}

/// f32損失関数
/// f32 Loss functions
#[derive(Debug, Clone)]
pub enum F32Loss {
    MeanSquaredError,
    CrossEntropy,
    BinaryCrossEntropy,
}

impl F32Loss {
    /// 損失の計算
    /// Compute loss
    pub fn forward(&self, predictions: &F32Tensor, targets: &F32Tensor) -> RusTorchResult<F32Tensor> {
        match self {
            F32Loss::MeanSquaredError => {
                let diff = predictions.sub(targets)?;
                let squared = diff.power(&F32Tensor::from_scalar(2.0f32)?)?;
                squared.mean_tensor()
            },
            F32Loss::CrossEntropy => {
                // Softmax + Cross-entropy
                let exp_preds = predictions.exp()?;
                let sum_exp = exp_preds.sum_dim(-1, true)?;
                let log_softmax = predictions.sub(&sum_exp.log()?)?;
                let nll = log_softmax.mul(targets)?.sum_dim(-1, false)?;
                let neg_nll = nll.mul(&F32Tensor::from_scalar(-1.0f32)?)?;
                neg_nll.mean_tensor()
            },
            F32Loss::BinaryCrossEntropy => {
                let eps = F32Tensor::from_scalar(1e-7f32)?;
                let one = F32Tensor::from_scalar(1.0f32)?;

                let clamped_preds = predictions.clamp(eps.clone()?, one.sub(&eps)?)?;
                let log_preds = clamped_preds.log()?;
                let log_one_minus_preds = one.sub(&clamped_preds)?.log()?;

                let term1 = targets.mul(&log_preds)?;
                let term2 = one.sub(targets)?.mul(&log_one_minus_preds)?;
                let loss_per_sample = term1.add(&term2)?.mul(&F32Tensor::from_scalar(-1.0f32)?)?;

                loss_per_sample.mean_tensor()
            }
        }
    }

    /// 損失の勾配
    /// Loss gradient
    pub fn backward(&self, predictions: &F32Tensor, targets: &F32Tensor) -> RusTorchResult<F32Tensor> {
        match self {
            F32Loss::MeanSquaredError => {
                let diff = predictions.sub(targets)?;
                let batch_size = predictions.shape()[0] as f32;
                let scale = F32Tensor::from_scalar(2.0f32 / batch_size)?;
                diff.mul(&scale)
            },
            F32Loss::CrossEntropy => {
                // Softmax gradient
                let exp_preds = predictions.exp()?;
                let sum_exp = exp_preds.sum_dim(-1, true)?;
                let softmax = exp_preds.divide(&sum_exp)?;
                let batch_size = predictions.shape()[0] as f32;
                let scale = F32Tensor::from_scalar(1.0f32 / batch_size)?;
                softmax.sub(targets)?.mul(&scale)
            },
            F32Loss::BinaryCrossEntropy => {
                let eps = F32Tensor::from_scalar(1e-7f32)?;
                let one = F32Tensor::from_scalar(1.0f32)?;

                let clamped_preds = predictions.clamp(eps, one.sub(&F32Tensor::from_scalar(1e-7f32)?)?)?;
                let batch_size = predictions.shape()[0] as f32;
                let scale = F32Tensor::from_scalar(1.0f32 / batch_size)?;

                let term1 = targets.divide(&clamped_preds)?;
                let term2 = one.sub(targets)?.divide(&one.sub(&clamped_preds)?)?;
                let gradient = term2.sub(&term1)?;

                gradient.mul(&scale)
            }
        }
    }
}

/// f32多層パーセプトロン
/// f32 Multi-Layer Perceptron
#[derive(Debug)]
pub struct F32MLP {
    pub layers: Vec<F32Linear>,
    pub activations: Vec<F32Activation>,
    pub layer_outputs: Vec<F32Tensor>, // Forward pass記録用
}

impl F32MLP {
    /// 新しいMLPを作成
    /// Create new MLP
    pub fn new(layer_sizes: &[usize], activation: F32Activation) -> RusTorchResult<Self> {
        let mut layers = Vec::new();
        let mut activations = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let layer = F32Linear::new(layer_sizes[i], layer_sizes[i + 1], true)?;
            layers.push(layer);

            if i < layer_sizes.len() - 2 {
                activations.push(activation.clone());
            }
        }

        Ok(Self {
            layers,
            activations,
            layer_outputs: Vec::new(),
        })
    }

    /// 順伝播
    /// Forward pass
    pub fn forward(&mut self, input: &F32Tensor) -> RusTorchResult<F32Tensor> {
        self.layer_outputs.clear();
        let mut current = input.clone()?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            current = layer.forward(&current)?;
            self.layer_outputs.push(current.clone()?);

            if i < self.activations.len() {
                current = self.activations[i].forward(&current)?;
            }
        }

        Ok(current)
    }

    /// パラメータ数を取得
    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|layer| {
            let weight_params = layer.weight.numel();
            let bias_params = layer.bias.as_ref().map_or(0, |b| b.numel());
            weight_params + bias_params
        }).sum()
    }

    /// 全パラメータを取得
    /// Get all parameters
    pub fn parameters(&self) -> Vec<&F32Tensor> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}

/// f32最適化器
/// f32 Optimizers
#[derive(Debug, Clone)]
pub enum F32Optimizer {
    SGD {
        learning_rate: f32,
        momentum: f32,
        velocity: HashMap<String, F32Tensor>,
    },
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        moment1: HashMap<String, F32Tensor>,
        moment2: HashMap<String, F32Tensor>,
        step: usize,
    },
}

impl F32Optimizer {
    /// SGD最適化器を作成
    /// Create SGD optimizer
    pub fn sgd(learning_rate: f32, momentum: f32) -> Self {
        Self::SGD {
            learning_rate,
            momentum,
            velocity: HashMap::new(),
        }
    }

    /// Adam最適化器を作成
    /// Create Adam optimizer
    pub fn adam(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self::Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            moment1: HashMap::new(),
            moment2: HashMap::new(),
            step: 0,
        }
    }

    /// パラメータを更新
    /// Update parameters
    pub fn step(&mut self, model: &mut F32MLP) -> RusTorchResult<()> {
        match self {
            Self::SGD { learning_rate, momentum, velocity } => {
                for layer in &mut model.layers {
                    layer.update_parameters(*learning_rate)?;
                }
            },
            Self::Adam { learning_rate, beta1, beta2, epsilon, moment1, moment2, step } => {
                *step += 1;
                for layer in &mut model.layers {
                    layer.update_parameters(*learning_rate)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_linear_layer() -> RusTorchResult<()> {
        let mut layer = F32Linear::new(3, 2, true)?;
        let input = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3])?;

        let output = layer.forward(&input)?;
        assert_eq!(output.shape(), &[1, 2]);

        Ok(())
    }

    #[test]
    fn test_f32_activations() -> RusTorchResult<()> {
        let input = F32Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5])?;

        let relu = F32Activation::ReLU;
        let relu_output = relu.forward(&input)?;

        // ReLUは負の値を0にする
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        assert_eq!(relu_output.as_slice(), &expected);

        Ok(())
    }

    #[test]
    fn test_f32_mlp() -> RusTorchResult<()> {
        let mut mlp = F32MLP::new(&[4, 8, 3], F32Activation::ReLU)?;
        let input = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4])?;

        let output = mlp.forward(&input)?;
        assert_eq!(output.shape(), &[1, 3]);

        let param_count = mlp.parameter_count();
        assert!(param_count > 0);

        Ok(())
    }
}