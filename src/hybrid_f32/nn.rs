//! f32統一ハイブリッドニューラルネットワークモジュール
//! f32 Unified Hybrid Neural Network Module
//!
//! フェーズ5: 高度ニューラルネットワーク機能
//! Phase 5: Advanced Neural Network Features
//!
//! このモジュールは、f32精度で最適化されたニューラルネットワーク機能を提供します。
//! Neural Engine、Metal GPU、CPUでの統一実行をサポートし、変換コストゼロを実現します。

use super::tensor::F32Tensor;
use crate::error::{RusTorchResult, RusTorchError};
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
        weight_decay: f32,
        velocity: HashMap<String, F32Tensor>,
    },
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        moment1: HashMap<String, F32Tensor>,
        moment2: HashMap<String, F32Tensor>,
        step: usize,
    },
    AdamW {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        moment1: HashMap<String, F32Tensor>,
        moment2: HashMap<String, F32Tensor>,
        step: usize,
    },
    RMSprop {
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
        weight_decay: f32,
        momentum: f32,
        squared_avg: HashMap<String, F32Tensor>,
        momentum_buffer: HashMap<String, F32Tensor>,
    },
}

impl F32Optimizer {
    /// SGD最適化器を作成
    /// Create SGD optimizer
    pub fn sgd(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        Self::SGD {
            learning_rate,
            momentum,
            weight_decay,
            velocity: HashMap::new(),
        }
    }

    /// Adam最適化器を作成
    /// Create Adam optimizer
    pub fn adam(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) -> Self {
        Self::Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            moment1: HashMap::new(),
            moment2: HashMap::new(),
            step: 0,
        }
    }

    /// AdamW最適化器を作成
    /// Create AdamW optimizer
    pub fn adamw(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) -> Self {
        Self::AdamW {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            moment1: HashMap::new(),
            moment2: HashMap::new(),
            step: 0,
        }
    }

    /// RMSprop最適化器を作成
    /// Create RMSprop optimizer
    pub fn rmsprop(learning_rate: f32, alpha: f32, epsilon: f32, weight_decay: f32, momentum: f32) -> Self {
        Self::RMSprop {
            learning_rate,
            alpha,
            epsilon,
            weight_decay,
            momentum,
            squared_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
        }
    }

    /// パラメータを更新
    /// Update parameters
    pub fn step(&mut self, model: &mut F32MLP) -> RusTorchResult<()> {
        match self {
            Self::SGD { learning_rate, momentum, weight_decay, velocity } => {
                for (layer_idx, layer) in model.layers.iter_mut().enumerate() {
                    // 重みの更新（SGD with momentum）
                    if let Some(ref weight_grad) = layer.weight_grad {
                        let weight_key = format!("layer_{}_weight", layer_idx);

                        // Weight decay (L2 regularization)
                        let mut grad_with_decay = weight_grad.clone()?;
                        if *weight_decay > 0.0 {
                            let weight_decay_term = layer.weight.mul(&F32Tensor::from_scalar(*weight_decay)?)?;
                            grad_with_decay = grad_with_decay.add(&weight_decay_term)?;
                        }

                        // velocity = momentum * velocity + learning_rate * gradient
                        let current_velocity = velocity.get(&weight_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(grad_with_decay.shape())))
                            .unwrap();

                        let momentum_term = current_velocity.mul(&F32Tensor::from_scalar(*momentum)?)?;
                        let lr_grad = grad_with_decay.mul(&F32Tensor::from_scalar(*learning_rate)?)?;
                        let new_velocity = momentum_term.add(&lr_grad)?;

                        // weight = weight - velocity
                        layer.weight = layer.weight.sub(&new_velocity)?;
                        velocity.insert(weight_key, new_velocity);
                    }

                    // バイアスの更新
                    if let (Some(ref mut bias), Some(ref bias_grad)) = (&mut layer.bias, &layer.bias_grad) {
                        let bias_key = format!("layer_{}_bias", layer_idx);

                        let current_velocity = velocity.get(&bias_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(bias_grad.shape())))
                            .unwrap();

                        let momentum_term = current_velocity.mul(&F32Tensor::from_scalar(*momentum)?)?;
                        let lr_grad = bias_grad.mul(&F32Tensor::from_scalar(*learning_rate)?)?;
                        let new_velocity = momentum_term.add(&lr_grad)?;

                        *bias = bias.sub(&new_velocity)?;
                        velocity.insert(bias_key, new_velocity);
                    }

                    // 勾配をクリア
                    layer.weight_grad = None;
                    layer.bias_grad = None;
                }
            },
            Self::Adam { learning_rate, beta1, beta2, epsilon, weight_decay, moment1, moment2, step } => {
                *step += 1;
                let step_f32 = *step as f32;

                // Bias correction terms
                let bias_correction1 = 1.0 - beta1.powf(step_f32);
                let bias_correction2 = 1.0 - beta2.powf(step_f32);

                for (layer_idx, layer) in model.layers.iter_mut().enumerate() {
                    // 重みの更新（Adam with bias correction）
                    if let Some(ref weight_grad) = layer.weight_grad {
                        let weight_key = format!("layer_{}_weight", layer_idx);

                        // moment1 = beta1 * moment1 + (1 - beta1) * gradient
                        let current_m1 = moment1.get(&weight_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(weight_grad.shape())))
                            .unwrap();

                        let beta1_tensor = F32Tensor::from_scalar(*beta1)?;
                        let one_minus_beta1 = F32Tensor::from_scalar(1.0 - *beta1)?;
                        let m1_term = current_m1.mul(&beta1_tensor)?;
                        let grad_term = weight_grad.mul(&one_minus_beta1)?;
                        let new_m1 = m1_term.add(&grad_term)?;

                        // moment2 = beta2 * moment2 + (1 - beta2) * gradient^2
                        let current_m2 = moment2.get(&weight_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(weight_grad.shape())))
                            .unwrap();

                        let beta2_tensor = F32Tensor::from_scalar(*beta2)?;
                        let one_minus_beta2 = F32Tensor::from_scalar(1.0 - *beta2)?;
                        let m2_term = current_m2.mul(&beta2_tensor)?;
                        let grad_squared = weight_grad.mul(weight_grad)?;
                        let grad_squared_term = grad_squared.mul(&one_minus_beta2)?;
                        let new_m2 = m2_term.add(&grad_squared_term)?;

                        // Bias-corrected moments
                        let m1_hat = new_m1.divide(&F32Tensor::from_scalar(bias_correction1)?)?;
                        let m2_hat = new_m2.divide(&F32Tensor::from_scalar(bias_correction2)?)?;

                        // Update: weight = weight - learning_rate * m1_hat / (sqrt(m2_hat) + epsilon)
                        let sqrt_m2_hat = m2_hat.power(&F32Tensor::from_scalar(0.5f32)?)?;
                        let denominator = sqrt_m2_hat.add(&F32Tensor::from_scalar(*epsilon)?)?;
                        let update = m1_hat.divide(&denominator)?;
                        let lr_update = update.mul(&F32Tensor::from_scalar(*learning_rate)?)?;

                        layer.weight = layer.weight.sub(&lr_update)?;
                        moment1.insert(weight_key.clone(), new_m1);
                        moment2.insert(weight_key, new_m2);
                    }

                    // バイアスの更新
                    if let (Some(ref mut bias), Some(ref bias_grad)) = (&mut layer.bias, &layer.bias_grad) {
                        let bias_key = format!("layer_{}_bias", layer_idx);

                        let current_m1 = moment1.get(&bias_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(bias_grad.shape())))
                            .unwrap();

                        let beta1_tensor = F32Tensor::from_scalar(*beta1)?;
                        let one_minus_beta1 = F32Tensor::from_scalar(1.0 - *beta1)?;
                        let m1_term = current_m1.mul(&beta1_tensor)?;
                        let grad_term = bias_grad.mul(&one_minus_beta1)?;
                        let new_m1 = m1_term.add(&grad_term)?;

                        let current_m2 = moment2.get(&bias_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(bias_grad.shape())))
                            .unwrap();

                        let beta2_tensor = F32Tensor::from_scalar(*beta2)?;
                        let one_minus_beta2 = F32Tensor::from_scalar(1.0 - *beta2)?;
                        let m2_term = current_m2.mul(&beta2_tensor)?;
                        let grad_squared = bias_grad.mul(bias_grad)?;
                        let grad_squared_term = grad_squared.mul(&one_minus_beta2)?;
                        let new_m2 = m2_term.add(&grad_squared_term)?;

                        let m1_hat = new_m1.divide(&F32Tensor::from_scalar(bias_correction1)?)?;
                        let m2_hat = new_m2.divide(&F32Tensor::from_scalar(bias_correction2)?)?;

                        let sqrt_m2_hat = m2_hat.power(&F32Tensor::from_scalar(0.5f32)?)?;
                        let denominator = sqrt_m2_hat.add(&F32Tensor::from_scalar(*epsilon)?)?;
                        let update = m1_hat.divide(&denominator)?;
                        let lr_update = update.mul(&F32Tensor::from_scalar(*learning_rate)?)?;

                        *bias = bias.sub(&lr_update)?;
                        moment1.insert(bias_key.clone(), new_m1);
                        moment2.insert(bias_key, new_m2);
                    }

                    // 勾配をクリア
                    layer.weight_grad = None;
                    layer.bias_grad = None;
                }
            },
            Self::AdamW { learning_rate, beta1, beta2, epsilon, weight_decay, moment1, moment2, step } => {
                *step += 1;
                let step_f32 = *step as f32;

                // Bias correction terms
                let bias_correction1 = 1.0 - beta1.powf(step_f32);
                let bias_correction2 = 1.0 - beta2.powf(step_f32);

                for (layer_idx, layer) in model.layers.iter_mut().enumerate() {
                    // 重みの更新（AdamW - decoupled weight decay）
                    if let Some(ref weight_grad) = layer.weight_grad {
                        let weight_key = format!("layer_{}_weight", layer_idx);

                        // moment1 = beta1 * moment1 + (1 - beta1) * gradient
                        let current_m1 = moment1.get(&weight_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(weight_grad.shape())))
                            .unwrap();

                        let beta1_tensor = F32Tensor::from_scalar(*beta1)?;
                        let one_minus_beta1 = F32Tensor::from_scalar(1.0 - *beta1)?;
                        let m1_term = current_m1.mul(&beta1_tensor)?;
                        let grad_term = weight_grad.mul(&one_minus_beta1)?;
                        let new_m1 = m1_term.add(&grad_term)?;

                        // moment2 = beta2 * moment2 + (1 - beta2) * gradient^2
                        let current_m2 = moment2.get(&weight_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(weight_grad.shape())))
                            .unwrap();

                        let beta2_tensor = F32Tensor::from_scalar(*beta2)?;
                        let one_minus_beta2 = F32Tensor::from_scalar(1.0 - *beta2)?;
                        let m2_term = current_m2.mul(&beta2_tensor)?;
                        let grad_squared = weight_grad.mul(weight_grad)?;
                        let grad_squared_term = grad_squared.mul(&one_minus_beta2)?;
                        let new_m2 = m2_term.add(&grad_squared_term)?;

                        // Bias-corrected moments
                        let m1_hat = new_m1.divide(&F32Tensor::from_scalar(bias_correction1)?)?;
                        let m2_hat = new_m2.divide(&F32Tensor::from_scalar(bias_correction2)?)?;

                        // AdamW update: weight = weight - learning_rate * m1_hat / (sqrt(m2_hat) + epsilon) - learning_rate * weight_decay * weight
                        let sqrt_m2_hat = m2_hat.power(&F32Tensor::from_scalar(0.5f32)?)?;
                        let denominator = sqrt_m2_hat.add(&F32Tensor::from_scalar(*epsilon)?)?;
                        let grad_update = m1_hat.divide(&denominator)?;
                        let lr_grad_update = grad_update.mul(&F32Tensor::from_scalar(*learning_rate)?)?;

                        // Decoupled weight decay
                        let weight_decay_update = layer.weight.mul(&F32Tensor::from_scalar(*learning_rate * *weight_decay)?)?;

                        layer.weight = layer.weight.sub(&lr_grad_update)?.sub(&weight_decay_update)?;
                        moment1.insert(weight_key.clone(), new_m1);
                        moment2.insert(weight_key, new_m2);
                    }

                    // バイアスの更新（weight decay適用しない）
                    if let (Some(ref mut bias), Some(ref bias_grad)) = (&mut layer.bias, &layer.bias_grad) {
                        let bias_key = format!("layer_{}_bias", layer_idx);

                        let current_m1 = moment1.get(&bias_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(bias_grad.shape())))
                            .unwrap();

                        let beta1_tensor = F32Tensor::from_scalar(*beta1)?;
                        let one_minus_beta1 = F32Tensor::from_scalar(1.0 - *beta1)?;
                        let m1_term = current_m1.mul(&beta1_tensor)?;
                        let grad_term = bias_grad.mul(&one_minus_beta1)?;
                        let new_m1 = m1_term.add(&grad_term)?;

                        let current_m2 = moment2.get(&bias_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(bias_grad.shape())))
                            .unwrap();

                        let beta2_tensor = F32Tensor::from_scalar(*beta2)?;
                        let one_minus_beta2 = F32Tensor::from_scalar(1.0 - *beta2)?;
                        let m2_term = current_m2.mul(&beta2_tensor)?;
                        let grad_squared = bias_grad.mul(bias_grad)?;
                        let grad_squared_term = grad_squared.mul(&one_minus_beta2)?;
                        let new_m2 = m2_term.add(&grad_squared_term)?;

                        let m1_hat = new_m1.divide(&F32Tensor::from_scalar(bias_correction1)?)?;
                        let m2_hat = new_m2.divide(&F32Tensor::from_scalar(bias_correction2)?)?;

                        let sqrt_m2_hat = m2_hat.power(&F32Tensor::from_scalar(0.5f32)?)?;
                        let denominator = sqrt_m2_hat.add(&F32Tensor::from_scalar(*epsilon)?)?;
                        let update = m1_hat.divide(&denominator)?;
                        let lr_update = update.mul(&F32Tensor::from_scalar(*learning_rate)?)?;

                        *bias = bias.sub(&lr_update)?;
                        moment1.insert(bias_key.clone(), new_m1);
                        moment2.insert(bias_key, new_m2);
                    }

                    // 勾配をクリア
                    layer.weight_grad = None;
                    layer.bias_grad = None;
                }
            },
            Self::RMSprop { learning_rate, alpha, epsilon, weight_decay, momentum, squared_avg, momentum_buffer } => {
                for (layer_idx, layer) in model.layers.iter_mut().enumerate() {
                    // 重みの更新（RMSprop）
                    if let Some(ref weight_grad) = layer.weight_grad {
                        let weight_key = format!("layer_{}_weight", layer_idx);

                        // Weight decay
                        let mut grad_with_decay = weight_grad.clone()?;
                        if *weight_decay > 0.0 {
                            let weight_decay_term = layer.weight.mul(&F32Tensor::from_scalar(*weight_decay)?)?;
                            grad_with_decay = grad_with_decay.add(&weight_decay_term)?;
                        }

                        // squared_avg = alpha * squared_avg + (1 - alpha) * gradient^2
                        let current_avg = squared_avg.get(&weight_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(grad_with_decay.shape())))
                            .unwrap();

                        let alpha_tensor = F32Tensor::from_scalar(*alpha)?;
                        let one_minus_alpha = F32Tensor::from_scalar(1.0 - *alpha)?;
                        let avg_term = current_avg.mul(&alpha_tensor)?;
                        let grad_squared = grad_with_decay.mul(&grad_with_decay)?;
                        let grad_squared_term = grad_squared.mul(&one_minus_alpha)?;
                        let new_avg = avg_term.add(&grad_squared_term)?;

                        if *momentum > 0.0 {
                            // With momentum
                            let buf_key = format!("layer_{}_weight_buf", layer_idx);
                            let current_buf = momentum_buffer.get(&buf_key)
                                .map(|v| v.clone())
                                .unwrap_or_else(|| Ok(F32Tensor::zeros(grad_with_decay.shape())))
                                .unwrap();

                            let sqrt_avg = new_avg.power(&F32Tensor::from_scalar(0.5f32)?)?;
                            let denominator = sqrt_avg.add(&F32Tensor::from_scalar(*epsilon)?)?;
                            let grad_normalized = grad_with_decay.divide(&denominator)?;

                            // buf = momentum * buf + grad_normalized
                            let momentum_term = current_buf.mul(&F32Tensor::from_scalar(*momentum)?)?;
                            let new_buf = momentum_term.add(&grad_normalized)?;

                            let lr_update = new_buf.mul(&F32Tensor::from_scalar(*learning_rate)?)?;
                            layer.weight = layer.weight.sub(&lr_update)?;
                            momentum_buffer.insert(buf_key, new_buf);
                        } else {
                            // Without momentum
                            let sqrt_avg = new_avg.power(&F32Tensor::from_scalar(0.5f32)?)?;
                            let denominator = sqrt_avg.add(&F32Tensor::from_scalar(*epsilon)?)?;
                            let update = grad_with_decay.divide(&denominator)?;
                            let lr_update = update.mul(&F32Tensor::from_scalar(*learning_rate)?)?;

                            layer.weight = layer.weight.sub(&lr_update)?;
                        }

                        squared_avg.insert(weight_key, new_avg);
                    }

                    // バイアスの更新（同様のロジック）
                    if let (Some(ref mut bias), Some(ref bias_grad)) = (&mut layer.bias, &layer.bias_grad) {
                        let bias_key = format!("layer_{}_bias", layer_idx);

                        let current_avg = squared_avg.get(&bias_key)
                            .map(|v| v.clone())
                            .unwrap_or_else(|| Ok(F32Tensor::zeros(bias_grad.shape())))
                            .unwrap();

                        let alpha_tensor = F32Tensor::from_scalar(*alpha)?;
                        let one_minus_alpha = F32Tensor::from_scalar(1.0 - *alpha)?;
                        let avg_term = current_avg.mul(&alpha_tensor)?;
                        let grad_squared = bias_grad.mul(bias_grad)?;
                        let grad_squared_term = grad_squared.mul(&one_minus_alpha)?;
                        let new_avg = avg_term.add(&grad_squared_term)?;

                        let sqrt_avg = new_avg.power(&F32Tensor::from_scalar(0.5f32)?)?;
                        let denominator = sqrt_avg.add(&F32Tensor::from_scalar(*epsilon)?)?;
                        let update = bias_grad.divide(&denominator)?;
                        let lr_update = update.mul(&F32Tensor::from_scalar(*learning_rate)?)?;

                        *bias = bias.sub(&lr_update)?;
                        squared_avg.insert(bias_key, new_avg);
                    }

                    // 勾配をクリア
                    layer.weight_grad = None;
                    layer.bias_grad = None;
                }
            }
        }
        Ok(())
    }

    /// 勾配をゼロクリア
    /// Zero gradients
    pub fn zero_grad(&mut self, model: &mut F32MLP) {
        for layer in &mut model.layers {
            layer.weight_grad = None;
            layer.bias_grad = None;
        }
    }
}

/// f32学習・検証フレームワーク
/// f32 Training & Validation Framework
#[derive(Debug)]
pub struct F32Trainer {
    pub model: F32MLP,
    pub optimizer: F32Optimizer,
    pub loss_fn: F32Loss,
    pub history: TrainingHistory,
}

/// 学習履歴
/// Training history
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TrainingHistory {
    pub train_losses: Vec<f32>,
    pub val_losses: Vec<f32>,
    pub train_accuracies: Vec<f32>,
    pub val_accuracies: Vec<f32>,
    pub epochs: usize,
}

/// メトリクス計算機
/// Metrics calculator
#[derive(Debug)]
pub struct F32Metrics;

impl F32Trainer {
    /// 新しいトレーナーを作成
    /// Create new trainer
    pub fn new(model: F32MLP, optimizer: F32Optimizer, loss_fn: F32Loss) -> Self {
        Self {
            model,
            optimizer,
            loss_fn,
            history: TrainingHistory::default(),
        }
    }

    /// モデルを学習
    /// Train model
    pub fn fit(
        &mut self,
        train_x: &F32Tensor,
        train_y: &F32Tensor,
        val_x: Option<&F32Tensor>,
        val_y: Option<&F32Tensor>,
        epochs: usize,
        batch_size: usize,
        verbose: bool,
    ) -> RusTorchResult<()> {
        let num_samples = train_x.shape()[0];
        let num_batches = (num_samples + batch_size - 1) / batch_size;

        for epoch in 0..epochs {
            let mut epoch_train_loss = 0.0f32;
            let mut epoch_train_acc = 0.0f32;

            // 学習フェーズ
            for batch_idx in 0..num_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = (start_idx + batch_size).min(num_samples);

                // バッチデータを取得
                let batch_x = train_x.slice(0, start_idx, end_idx, 1)?;
                let batch_y = train_y.slice(0, start_idx, end_idx, 1)?;

                // 勾配をクリア
                self.optimizer.zero_grad(&mut self.model);

                // フォワードパス
                let predictions = self.model.forward(&batch_x)?;

                // 損失計算
                let loss = self.loss_fn.forward(&predictions, &batch_y)?;
                epoch_train_loss += loss.as_slice()[0];

                // 精度計算
                let accuracy = F32Metrics::accuracy(&predictions, &batch_y)?;
                epoch_train_acc += accuracy;

                // バックワードパス（簡素化）
                let grad_output = self.loss_fn.backward(&predictions, &batch_y)?;
                self.backward_through_model(&grad_output)?;

                // パラメータ更新
                self.optimizer.step(&mut self.model)?;
            }

            // エポック平均
            epoch_train_loss /= num_batches as f32;
            epoch_train_acc /= num_batches as f32;

            self.history.train_losses.push(epoch_train_loss);
            self.history.train_accuracies.push(epoch_train_acc);

            // バリデーション
            if let (Some(val_x), Some(val_y)) = (val_x, val_y) {
                let (val_loss, val_acc) = self.validate(val_x, val_y)?;
                self.history.val_losses.push(val_loss);
                self.history.val_accuracies.push(val_acc);

                if verbose {
                    println!(
                        "Epoch {}/{} - train_loss: {:.4}, train_acc: {:.4}, val_loss: {:.4}, val_acc: {:.4}",
                        epoch + 1, epochs, epoch_train_loss, epoch_train_acc, val_loss, val_acc
                    );
                }
            } else if verbose {
                println!(
                    "Epoch {}/{} - train_loss: {:.4}, train_acc: {:.4}",
                    epoch + 1, epochs, epoch_train_loss, epoch_train_acc
                );
            }
        }

        self.history.epochs = epochs;
        Ok(())
    }

    /// モデルをバリデーション
    /// Validate model
    pub fn validate(&mut self, val_x: &F32Tensor, val_y: &F32Tensor) -> RusTorchResult<(f32, f32)> {
        let predictions = self.model.forward(val_x)?;
        let loss = self.loss_fn.forward(&predictions, val_y)?;
        let accuracy = F32Metrics::accuracy(&predictions, val_y)?;

        Ok((loss.as_slice()[0], accuracy))
    }

    /// 予測を実行
    /// Make predictions
    pub fn predict(&mut self, x: &F32Tensor) -> RusTorchResult<F32Tensor> {
        self.model.forward(x)
    }

    /// 評価メトリクスを計算
    /// Compute evaluation metrics
    pub fn evaluate(&mut self, x: &F32Tensor, y: &F32Tensor) -> RusTorchResult<EvaluationMetrics> {
        let predictions = self.model.forward(x)?;
        let loss = self.loss_fn.forward(&predictions, y)?;
        let accuracy = F32Metrics::accuracy(&predictions, y)?;
        let precision = F32Metrics::precision(&predictions, y)?;
        let recall = F32Metrics::recall(&predictions, y)?;
        let f1_score = F32Metrics::f1_score(&predictions, y)?;

        Ok(EvaluationMetrics {
            loss: loss.as_slice()[0],
            accuracy,
            precision,
            recall,
            f1_score,
        })
    }

    /// 学習履歴を取得
    /// Get training history
    pub fn get_history(&self) -> &TrainingHistory {
        &self.history
    }

    /// モデルの保存パラメータを取得
    /// Get model save parameters
    pub fn get_model_state(&self) -> RusTorchResult<ModelState> {
        let layers = self.model.layers.iter()
            .map(|layer| {
                let weight_data = layer.weight.as_slice().to_vec();
                let weight_shape = layer.weight.shape().to_vec();

                let (bias_data, bias_shape) = if let Some(ref bias_tensor) = layer.bias {
                    (Some(bias_tensor.as_slice().to_vec()), Some(bias_tensor.shape().to_vec()))
                } else {
                    (None, None)
                };

                Ok(LayerState {
                    weight_data,
                    weight_shape,
                    bias_data,
                    bias_shape,
                    input_features: layer.input_features,
                    output_features: layer.output_features,
                })
            })
            .collect::<RusTorchResult<Vec<_>>>()?;

        Ok(ModelState {
            layers,
            activations: self.model.activations.clone(),
        })
    }

    /// プライベート：バックワード伝播
    /// Private: Backward propagation
    fn backward_through_model(&mut self, grad_output: &F32Tensor) -> RusTorchResult<()> {
        // 簡素化されたバックワード実装
        // 実際は各層を逆順に処理
        for layer in self.model.layers.iter_mut().rev() {
            layer.backward(grad_output)?;
        }
        Ok(())
    }
}

/// 評価メトリクス
/// Evaluation metrics
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
}

/// モデル状態（保存用）
/// Model state (for saving)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelState {
    pub layers: Vec<LayerState>,
    pub activations: Vec<F32Activation>,
}

/// 層状態（シリアライゼーション用）
/// Layer state (for serialization)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerState {
    pub weight_data: Vec<f32>,
    pub weight_shape: Vec<usize>,
    pub bias_data: Option<Vec<f32>>,
    pub bias_shape: Option<Vec<usize>>,
    pub input_features: usize,
    pub output_features: usize,
}

impl F32Metrics {
    /// 分類精度を計算
    /// Calculate classification accuracy
    pub fn accuracy(predictions: &F32Tensor, targets: &F32Tensor) -> RusTorchResult<f32> {
        // 簡素化：回帰タスクの場合はMSE閾値、分類タスクの場合はargmax
        let pred_data = predictions.as_slice();
        let target_data = targets.as_slice();

        if predictions.shape()[1] > 1 {
            // 分類タスク：argmax
            let batch_size = predictions.shape()[0];
            let mut correct = 0;

            for i in 0..batch_size {
                let start_idx = i * predictions.shape()[1];
                let end_idx = start_idx + predictions.shape()[1];

                let pred_class = Self::argmax(&pred_data[start_idx..end_idx]);
                let target_class = Self::argmax(&target_data[start_idx..end_idx]);

                if pred_class == target_class {
                    correct += 1;
                }
            }

            Ok(correct as f32 / batch_size as f32)
        } else {
            // 回帰タスク：閾値精度
            let threshold = 0.1;
            let mut correct = 0;

            for (pred, target) in pred_data.iter().zip(target_data.iter()) {
                if (pred - target).abs() < threshold {
                    correct += 1;
                }
            }

            Ok(correct as f32 / pred_data.len() as f32)
        }
    }

    /// 精密度を計算
    /// Calculate precision
    pub fn precision(predictions: &F32Tensor, targets: &F32Tensor) -> RusTorchResult<f32> {
        // 簡素化された二値分類精密度
        let pred_data = predictions.as_slice();
        let target_data = targets.as_slice();

        let mut true_positive = 0;
        let mut false_positive = 0;

        for (pred, target) in pred_data.iter().zip(target_data.iter()) {
            let pred_class = if *pred > 0.5 { 1 } else { 0 };
            let target_class = if *target > 0.5 { 1 } else { 0 };

            if pred_class == 1 && target_class == 1 {
                true_positive += 1;
            } else if pred_class == 1 && target_class == 0 {
                false_positive += 1;
            }
        }

        if true_positive + false_positive == 0 {
            Ok(0.0)
        } else {
            Ok(true_positive as f32 / (true_positive + false_positive) as f32)
        }
    }

    /// 再現率を計算
    /// Calculate recall
    pub fn recall(predictions: &F32Tensor, targets: &F32Tensor) -> RusTorchResult<f32> {
        let pred_data = predictions.as_slice();
        let target_data = targets.as_slice();

        let mut true_positive = 0;
        let mut false_negative = 0;

        for (pred, target) in pred_data.iter().zip(target_data.iter()) {
            let pred_class = if *pred > 0.5 { 1 } else { 0 };
            let target_class = if *target > 0.5 { 1 } else { 0 };

            if pred_class == 1 && target_class == 1 {
                true_positive += 1;
            } else if pred_class == 0 && target_class == 1 {
                false_negative += 1;
            }
        }

        if true_positive + false_negative == 0 {
            Ok(0.0)
        } else {
            Ok(true_positive as f32 / (true_positive + false_negative) as f32)
        }
    }

    /// F1スコアを計算
    /// Calculate F1 score
    pub fn f1_score(predictions: &F32Tensor, targets: &F32Tensor) -> RusTorchResult<f32> {
        let precision = Self::precision(predictions, targets)?;
        let recall = Self::recall(predictions, targets)?;

        if precision + recall == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * precision * recall / (precision + recall))
        }
    }

    /// ヘルパー：argmax
    /// Helper: argmax
    fn argmax(slice: &[f32]) -> usize {
        slice.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
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

// ===== フェーズ5: Data Loading機能 / Phase 5: Data Loading Features =====

/// PyTorch風データセット抽象化
/// PyTorch-style dataset abstraction
pub trait F32Dataset {
    fn len(&self) -> usize;
    fn get_item(&self, index: usize) -> RusTorchResult<(F32Tensor, F32Tensor)>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// メモリ内データセット実装
/// In-memory dataset implementation
#[derive(Debug)]
pub struct F32MemoryDataset {
    pub data: Vec<F32Tensor>,
    pub targets: Vec<F32Tensor>,
}

impl F32MemoryDataset {
    /// 新しいメモリデータセットを作成
    /// Create new memory dataset
    pub fn new(data: Vec<F32Tensor>, targets: Vec<F32Tensor>) -> RusTorchResult<Self> {
        if data.len() != targets.len() {
            return Err(RusTorchError::tensor_op(
                format!("Data and targets length mismatch: {} vs {}", data.len(), targets.len())
            ));
        }

        Ok(Self { data, targets })
    }

    /// テンソルからデータセットを作成
    /// Create dataset from tensors
    pub fn from_tensors(x: F32Tensor, y: F32Tensor) -> RusTorchResult<Self> {
        let batch_size = x.shape()[0];
        let y_batch_size = y.shape()[0];

        if batch_size != y_batch_size {
            return Err(RusTorchError::tensor_op(
                format!("Batch size mismatch: {} vs {}", batch_size, y_batch_size)
            ));
        }

        let mut data = Vec::new();
        let mut targets = Vec::new();

        for i in 0..batch_size {
            let sample = x.slice(0, i, i + 1, 1)?;
            let target = y.slice(0, i, i + 1, 1)?;
            data.push(sample);
            targets.push(target);
        }

        Ok(Self { data, targets })
    }
}

impl F32Dataset for F32MemoryDataset {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get_item(&self, index: usize) -> RusTorchResult<(F32Tensor, F32Tensor)> {
        if index >= self.len() {
            return Err(RusTorchError::tensor_op(
                format!("Index {} out of bounds for dataset of size {}", index, self.len())
            ));
        }

        Ok((self.data[index].clone()?, self.targets[index].clone()?))
    }
}

/// PyTorch風データローダー
/// PyTorch-style data loader
#[derive(Debug)]
pub struct F32DataLoader<T: F32Dataset> {
    dataset: T,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_batch: usize,
    total_batches: usize,
}

impl<T: F32Dataset> F32DataLoader<T> {
    /// 新しいデータローダーを作成
    /// Create new data loader
    pub fn new(dataset: T, batch_size: usize, shuffle: bool) -> Self {
        let dataset_size = dataset.len();
        let total_batches = (dataset_size + batch_size - 1) / batch_size;
        let mut indices: Vec<usize> = (0..dataset_size).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_batch: 0,
            total_batches,
        }
    }

    /// 次のバッチを取得
    /// Get next batch
    pub fn next_batch(&mut self) -> RusTorchResult<Option<(F32Tensor, F32Tensor)>> {
        if self.current_batch >= self.total_batches {
            return Ok(None);
        }

        let start_idx = self.current_batch * self.batch_size;
        let end_idx = std::cmp::min(start_idx + self.batch_size, self.dataset.len());

        let mut batch_data = Vec::new();
        let mut batch_targets = Vec::new();

        for i in start_idx..end_idx {
            let dataset_idx = self.indices[i];
            let (data, target) = self.dataset.get_item(dataset_idx)?;
            batch_data.push(data);
            batch_targets.push(target);
        }

        // バッチテンソルを結合
        let batch_x_refs: Vec<&F32Tensor> = batch_data.iter().collect();
        let batch_y_refs: Vec<&F32Tensor> = batch_targets.iter().collect();
        let batch_x = F32Tensor::stack(&batch_x_refs, 0)?;
        let batch_y = F32Tensor::stack(&batch_y_refs, 0)?;

        self.current_batch += 1;
        Ok(Some((batch_x, batch_y)))
    }

    /// エポック開始時にリセット
    /// Reset for new epoch
    pub fn reset(&mut self) {
        self.current_batch = 0;

        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// 残りバッチ数を取得
    /// Get remaining batches
    pub fn remaining_batches(&self) -> usize {
        self.total_batches.saturating_sub(self.current_batch)
    }

    /// 総バッチ数を取得
    /// Get total number of batches
    pub fn total_batches(&self) -> usize {
        self.total_batches
    }
}

/// データローダーのイテレータ実装
/// Iterator implementation for data loader
impl<T: F32Dataset> Iterator for F32DataLoader<T> {
    type Item = RusTorchResult<(F32Tensor, F32Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

// ===== フェーズ5: Model Serialization機能 / Phase 5: Model Serialization Features =====

/// モデル保存・ロード機能
/// Model save/load functionality
impl F32Trainer {
    /// モデルを保存
    /// Save model to file
    pub fn save_model(&self, path: &str) -> RusTorchResult<()> {
        let model_state = self.get_model_state()?;
        let serialized = serde_json::to_string_pretty(&model_state)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to serialize model: {}", e)))?;

        std::fs::write(path, serialized)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to write model file: {}", e)))?;

        Ok(())
    }

    /// モデルをロード
    /// Load model from file
    pub fn load_model(&mut self, path: &str) -> RusTorchResult<()> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to read model file: {}", e)))?;

        let model_state: ModelState = serde_json::from_str(&contents)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to deserialize model: {}", e)))?;

        self.set_model_state(model_state)?;
        Ok(())
    }

    /// 学習履歴を保存
    /// Save training history
    pub fn save_history(&self, path: &str) -> RusTorchResult<()> {
        let serialized = serde_json::to_string_pretty(&self.history)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to serialize history: {}", e)))?;

        std::fs::write(path, serialized)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to write history file: {}", e)))?;

        Ok(())
    }

    /// 学習履歴をロード
    /// Load training history
    pub fn load_history(&mut self, path: &str) -> RusTorchResult<()> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to read history file: {}", e)))?;

        self.history = serde_json::from_str(&contents)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to deserialize history: {}", e)))?;

        Ok(())
    }

    /// モデル状態を設定
    /// Set model state
    pub fn set_model_state(&mut self, state: ModelState) -> RusTorchResult<()> {
        if state.layers.len() != self.model.layers.len() {
            return Err(RusTorchError::tensor_op(
                format!("Layer count mismatch: expected {}, got {}",
                    self.model.layers.len(), state.layers.len())
            ));
        }

        for (i, layer_state) in state.layers.iter().enumerate() {
            let layer = &mut self.model.layers[i];

            // 重みテンソルを復元
            layer.weight = F32Tensor::from_vec(
                layer_state.weight_data.clone(),
                layer_state.weight_shape.clone()
            )?;

            // バイアステンソルを復元
            layer.bias = if let (Some(ref bias_data), Some(ref bias_shape)) =
                (&layer_state.bias_data, &layer_state.bias_shape) {
                Some(F32Tensor::from_vec(bias_data.clone(), bias_shape.clone())?)
            } else {
                None
            };
        }

        self.model.activations = state.activations;
        Ok(())
    }

    /// チェックポイントを保存（モデル+履歴）
    /// Save checkpoint (model + history)
    pub fn save_checkpoint(&self, base_path: &str) -> RusTorchResult<()> {
        let model_path = format!("{}_model.json", base_path);
        let history_path = format!("{}_history.json", base_path);

        self.save_model(&model_path)?;
        self.save_history(&history_path)?;

        println!("✅ Checkpoint saved: {} (model + history)", base_path);
        Ok(())
    }

    /// チェックポイントをロード（モデル+履歴）
    /// Load checkpoint (model + history)
    pub fn load_checkpoint(&mut self, base_path: &str) -> RusTorchResult<()> {
        let model_path = format!("{}_model.json", base_path);
        let history_path = format!("{}_history.json", base_path);

        self.load_model(&model_path)?;

        // 履歴ファイルが存在する場合のみロード
        if std::path::Path::new(&history_path).exists() {
            self.load_history(&history_path)?;
        }

        println!("✅ Checkpoint loaded: {} (model + history)", base_path);
        Ok(())
    }
}

/// F32MLPの独立した保存・ロード機能
/// Standalone save/load functionality for F32MLP
impl F32MLP {
    /// MLPモデルを保存
    /// Save MLP model
    pub fn save(&self, path: &str) -> RusTorchResult<()> {
        let model_state = ModelState {
            layers: self.layers.iter()
                .map(|layer| {
                    let weight_data = layer.weight.as_slice().to_vec();
                    let weight_shape = layer.weight.shape().to_vec();

                    let (bias_data, bias_shape) = if let Some(ref bias_tensor) = layer.bias {
                        (Some(bias_tensor.as_slice().to_vec()), Some(bias_tensor.shape().to_vec()))
                    } else {
                        (None, None)
                    };

                    Ok(LayerState {
                        weight_data,
                        weight_shape,
                        bias_data,
                        bias_shape,
                        input_features: layer.input_features,
                        output_features: layer.output_features,
                    })
                })
                .collect::<RusTorchResult<Vec<_>>>()?,
            activations: self.activations.clone(),
        };

        let serialized = serde_json::to_string_pretty(&model_state)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to serialize model: {}", e)))?;

        std::fs::write(path, serialized)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to write model file: {}", e)))?;

        Ok(())
    }

    /// MLPモデルをロード
    /// Load MLP model
    pub fn load(path: &str) -> RusTorchResult<Self> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to read model file: {}", e)))?;

        let model_state: ModelState = serde_json::from_str(&contents)
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to deserialize model: {}", e)))?;

        let mut layers = Vec::new();
        for layer_state in model_state.layers {
            // 重みテンソルを復元
            let weight = F32Tensor::from_vec(
                layer_state.weight_data,
                layer_state.weight_shape
            )?;

            // バイアステンソルを復元
            let bias = if let (Some(bias_data), Some(bias_shape)) =
                (layer_state.bias_data, layer_state.bias_shape) {
                Some(F32Tensor::from_vec(bias_data, bias_shape)?)
            } else {
                None
            };

            layers.push(F32Linear {
                weight,
                bias,
                weight_grad: None,
                bias_grad: None,
                last_input: None,
                input_features: layer_state.input_features,
                output_features: layer_state.output_features,
            });
        }

        Ok(Self {
            layers,
            activations: model_state.activations,
            layer_outputs: Vec::new(),
        })
    }
}

// ===== フェーズ5: Learning Rate Scheduler機能 / Phase 5: Learning Rate Scheduler Features =====

/// 学習率スケジューリング戦略
/// Learning rate scheduling strategy
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum F32LRSchedulerType {
    /// 固定学習率
    /// Constant learning rate
    Constant,
    /// ステップ減衰
    /// Step decay
    StepLR { step_size: usize, gamma: f32 },
    /// 指数減衰
    /// Exponential decay
    ExponentialLR { gamma: f32 },
    /// Cosine Annealing
    CosineLR { t_max: usize, eta_min: f32 },
    /// Reduce on Plateau
    ReduceOnPlateau { factor: f32, patience: usize, threshold: f32 },
    /// Linear Warm-up + Cosine Decay
    WarmupCosine { warmup_steps: usize, total_steps: usize },
}

/// f32学習率スケジューラー
/// f32 Learning Rate Scheduler
#[derive(Debug)]
pub struct F32LRScheduler {
    scheduler_type: F32LRSchedulerType,
    initial_lr: f32,
    current_lr: f32,
    current_step: usize,
    last_metric: f32,
    no_improvement_count: usize,
    best_metric: f32,
}

impl F32LRScheduler {
    /// 新しいスケジューラーを作成
    /// Create new scheduler
    pub fn new(scheduler_type: F32LRSchedulerType, initial_lr: f32) -> Self {
        Self {
            scheduler_type,
            initial_lr,
            current_lr: initial_lr,
            current_step: 0,
            last_metric: f32::INFINITY,
            no_improvement_count: 0,
            best_metric: f32::INFINITY,
        }
    }

    /// 学習率をステップ更新
    /// Step update learning rate
    pub fn step(&mut self, metric: Option<f32>) -> f32 {
        self.current_step += 1;

        let new_lr = match &self.scheduler_type {
            F32LRSchedulerType::Constant => {
                self.initial_lr
            }

            F32LRSchedulerType::StepLR { step_size, gamma } => {
                if self.current_step % step_size == 0 {
                    self.current_lr * gamma
                } else {
                    self.current_lr
                }
            }

            F32LRSchedulerType::ExponentialLR { gamma } => {
                self.initial_lr * gamma.powf(self.current_step as f32)
            }

            F32LRSchedulerType::CosineLR { t_max, eta_min } => {
                let progress = (self.current_step as f32) / (*t_max as f32);
                let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                eta_min + (self.initial_lr - eta_min) * cosine_factor
            }

            F32LRSchedulerType::ReduceOnPlateau { factor, patience, threshold } => {
                if let Some(current_metric) = metric {
                    if current_metric < self.best_metric - threshold {
                        self.best_metric = current_metric;
                        self.no_improvement_count = 0;
                    } else {
                        self.no_improvement_count += 1;
                        if self.no_improvement_count >= *patience {
                            self.no_improvement_count = 0;
                            return self.current_lr * factor;
                        }
                    }
                }
                self.current_lr
            }

            F32LRSchedulerType::WarmupCosine { warmup_steps, total_steps } => {
                if self.current_step <= *warmup_steps {
                    // Linear warmup
                    self.initial_lr * (self.current_step as f32) / (*warmup_steps as f32)
                } else {
                    // Cosine decay
                    let decay_steps = *total_steps - *warmup_steps;
                    let decay_progress = ((self.current_step - *warmup_steps) as f32) / (decay_steps as f32);
                    let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * decay_progress).cos());
                    self.initial_lr * cosine_factor
                }
            }
        };

        self.current_lr = new_lr;
        self.current_lr
    }

    /// 現在の学習率を取得
    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// スケジューラーをリセット
    /// Reset scheduler
    pub fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.current_step = 0;
        self.no_improvement_count = 0;
        self.best_metric = f32::INFINITY;
    }

    /// スケジューラー状態を取得
    /// Get scheduler state
    pub fn state(&self) -> (usize, f32, f32) {
        (self.current_step, self.current_lr, self.best_metric)
    }
}

/// F32Optimizerに学習率スケジューラーサポートを追加
/// Add learning rate scheduler support to F32Optimizer
impl F32Optimizer {
    /// 学習率を更新
    /// Update learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        match self {
            F32Optimizer::SGD { learning_rate, .. } => *learning_rate = lr,
            F32Optimizer::Adam { learning_rate, .. } => *learning_rate = lr,
            F32Optimizer::AdamW { learning_rate, .. } => *learning_rate = lr,
            F32Optimizer::RMSprop { learning_rate, .. } => *learning_rate = lr,
        }
    }

    /// 現在の学習率を取得
    /// Get current learning rate
    pub fn get_learning_rate(&self) -> f32 {
        match self {
            F32Optimizer::SGD { learning_rate, .. } => *learning_rate,
            F32Optimizer::Adam { learning_rate, .. } => *learning_rate,
            F32Optimizer::AdamW { learning_rate, .. } => *learning_rate,
            F32Optimizer::RMSprop { learning_rate, .. } => *learning_rate,
        }
    }
}

/// F32Trainerに学習率スケジューラーサポートを追加
/// Add learning rate scheduler support to F32Trainer
impl F32Trainer {
    /// 学習率スケジューラーを設定
    /// Set learning rate scheduler
    pub fn set_lr_scheduler(&mut self, scheduler: F32LRScheduler) -> &mut Self {
        // まず現在のオプティマイザーの学習率をスケジューラーに合わせる
        let initial_lr = scheduler.initial_lr;
        self.optimizer.set_learning_rate(initial_lr);

        // スケジューラーをメンバーとして保存したい場合は、構造体に追加する必要があります
        // ここでは関数パラメータとして使用する方式を採用
        self
    }

    /// 学習率スケジューラーと組み合わせた学習メソッド
    /// Training method with learning rate scheduler
    pub fn fit_with_scheduler(
        &mut self,
        train_x: &F32Tensor,
        train_y: &F32Tensor,
        val_x: Option<&F32Tensor>,
        val_y: Option<&F32Tensor>,
        epochs: usize,
        batch_size: usize,
        mut lr_scheduler: F32LRScheduler,
        verbose: bool,
    ) -> RusTorchResult<()> {
        let num_batches = (train_x.shape()[0] + batch_size - 1) / batch_size;

        for epoch in 0..epochs {
            let mut epoch_train_loss = 0.0;
            let mut epoch_train_acc = 0.0;

            // バッチ処理
            for batch_idx in 0..num_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = std::cmp::min(start_idx + batch_size, train_x.shape()[0]);

                // バッチデータを取得
                let batch_x = train_x.slice(0, start_idx, end_idx, 1)?;
                let batch_y = train_y.slice(0, start_idx, end_idx, 1)?;

                // 勾配をクリア
                self.optimizer.zero_grad(&mut self.model);

                // フォワードパス
                let predictions = self.model.forward(&batch_x)?;

                // 損失計算
                let loss = self.loss_fn.forward(&predictions, &batch_y)?;
                epoch_train_loss += loss.as_slice()[0];

                // 精度計算
                let accuracy = F32Metrics::accuracy(&predictions, &batch_y)?;
                epoch_train_acc += accuracy;

                // バックワードパス（簡素化）
                let grad_output = self.loss_fn.backward(&predictions, &batch_y)?;
                self.backward_through_model(&grad_output)?;

                // パラメータ更新
                self.optimizer.step(&mut self.model)?;
            }

            // エポック平均
            epoch_train_loss /= num_batches as f32;
            epoch_train_acc /= num_batches as f32;

            self.history.train_losses.push(epoch_train_loss);
            self.history.train_accuracies.push(epoch_train_acc);

            // バリデーション
            let mut val_loss_for_scheduler = None;
            if let (Some(val_x), Some(val_y)) = (val_x, val_y) {
                let (val_loss, val_acc) = self.validate(val_x, val_y)?;
                self.history.val_losses.push(val_loss);
                self.history.val_accuracies.push(val_acc);
                val_loss_for_scheduler = Some(val_loss);

                if verbose {
                    println!(
                        "Epoch {}/{} - train_loss: {:.4}, train_acc: {:.4}, val_loss: {:.4}, val_acc: {:.4}, lr: {:.6}",
                        epoch + 1, epochs, epoch_train_loss, epoch_train_acc, val_loss, val_acc, lr_scheduler.get_lr()
                    );
                }
            } else if verbose {
                println!(
                    "Epoch {}/{} - train_loss: {:.4}, train_acc: {:.4}, lr: {:.6}",
                    epoch + 1, epochs, epoch_train_loss, epoch_train_acc, lr_scheduler.get_lr()
                );
            }

            // 学習率スケジューラーをステップ更新
            let new_lr = lr_scheduler.step(val_loss_for_scheduler);
            self.optimizer.set_learning_rate(new_lr);
        }

        self.history.epochs = epochs;
        Ok(())
    }
}

// ============================================================================
// Computer Vision Models / コンピュータビジョンモデル
// ============================================================================

/// 2D畳み込み層
/// 2D Convolution layer
#[derive(Debug)]
pub struct F32Conv2d {
    pub weight: F32Tensor,     // (out_channels, in_channels, kernel_h, kernel_w)
    pub bias: Option<F32Tensor>, // (out_channels,)
    pub weight_grad: Option<F32Tensor>,
    pub bias_grad: Option<F32Tensor>,
    pub last_input: Option<F32Tensor>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl F32Conv2d {
    /// 新しい2D畳み込み層を作成
    /// Create a new 2D convolution layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Result<Self, RusTorchError> {
        let (kernel_h, kernel_w) = kernel_size;

        // Heの初期化
        let fan_in = in_channels * kernel_h * kernel_w;
        let std = (2.0 / fan_in as f32).sqrt();

        let weight_shape = vec![out_channels, in_channels, kernel_h, kernel_w];
        let weight = F32Tensor::randn(&weight_shape);
        let std_tensor = F32Tensor::from_vec(vec![std], vec![1])?;
        let weight = weight.mul(&std_tensor)?;

        let bias_tensor = if bias {
            Some(F32Tensor::zeros(&[out_channels]))
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_tensor,
            weight_grad: None,
            bias_grad: None,
            last_input: None,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    /// フォワードパス（簡素化された畳み込み）
    /// Forward pass (simplified convolution)
    pub fn forward(&mut self, input: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        // 入力を保存（バックワードパス用）
        self.last_input = Some(input.clone()?);

        // 入力形状: (batch_size, in_channels, height, width)
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(RusTorchError::tensor_op("Conv2d input must be 4D (batch, channels, height, width)"));
        }

        let batch_size = input_shape[0];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        // 出力サイズ計算
        let output_height = (input_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let output_width = (input_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        // 簡素化された畳み込み（im2colを使わない直接実装）
        let mut output_data = vec![0.0; batch_size * self.out_channels * output_height * output_width];
        let input_data = input.as_slice();
        let weight_data = self.weight.as_slice();

        for b in 0..batch_size {
            for out_c in 0..self.out_channels {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum = 0.0;

                        // カーネル内の畳み込み
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = oh * self.stride.0 + kh;
                                let iw = ow * self.stride.1 + kw;

                                // パディングチェック
                                if ih >= self.padding.0 && ih < input_height + self.padding.0 &&
                                   iw >= self.padding.1 && iw < input_width + self.padding.1 {
                                    let ih_actual = ih - self.padding.0;
                                    let iw_actual = iw - self.padding.1;

                                    for in_c in 0..self.in_channels {
                                        let input_idx = b * self.in_channels * input_height * input_width +
                                                      in_c * input_height * input_width +
                                                      ih_actual * input_width + iw_actual;
                                        let weight_idx = out_c * self.in_channels * self.kernel_size.0 * self.kernel_size.1 +
                                                       in_c * self.kernel_size.0 * self.kernel_size.1 +
                                                       kh * self.kernel_size.1 + kw;

                                        sum += input_data[input_idx] * weight_data[weight_idx];
                                    }
                                }
                            }
                        }

                        // バイアス追加
                        if let Some(ref bias) = self.bias {
                            sum += bias.as_slice()[out_c];
                        }

                        let output_idx = b * self.out_channels * output_height * output_width +
                                       out_c * output_height * output_width +
                                       oh * output_width + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }

        let output_shape = vec![batch_size, self.out_channels, output_height, output_width];
        F32Tensor::from_vec(output_data, output_shape)
    }
}

impl F32Layer for F32Conv2d {
    fn forward(&mut self, input: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        self.forward(input)
    }

    fn backward(&mut self, grad_output: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        // 簡素化されたバックワード実装
        // TODO: 実際の畳み込みの逆伝播を実装
        Ok(grad_output.clone()?)
    }

    fn parameters(&self) -> Vec<&F32Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn update_parameters(&mut self, learning_rate: f32) -> Result<(), RusTorchError> {
        // 簡素化された勾配更新
        if let Some(ref weight_grad) = self.weight_grad {
            let lr_tensor = F32Tensor::from_vec(vec![learning_rate], vec![1])?;
            let update = weight_grad.mul(&lr_tensor)?;
            self.weight = self.weight.sub(&update)?;
        }

        if let (Some(ref mut bias), Some(ref bias_grad)) = (&mut self.bias, &self.bias_grad) {
            let lr_tensor = F32Tensor::from_vec(vec![learning_rate], vec![1])?;
            let update = bias_grad.mul(&lr_tensor)?;
            *bias = bias.sub(&update)?;
        }

        Ok(())
    }
}

/// バッチ正規化層
/// Batch Normalization layer
#[derive(Debug)]
pub struct F32BatchNorm2d {
    pub num_features: usize,
    pub weight: F32Tensor,     // (num_features,)
    pub bias: F32Tensor,       // (num_features,)
    pub running_mean: F32Tensor, // (num_features,)
    pub running_var: F32Tensor,  // (num_features,)
    pub momentum: f32,
    pub eps: f32,
    pub training: bool,
    pub weight_grad: Option<F32Tensor>,
    pub bias_grad: Option<F32Tensor>,
}

impl F32BatchNorm2d {
    /// 新しいバッチ正規化層を作成
    /// Create a new batch normalization layer
    pub fn new(num_features: usize, momentum: f32, eps: f32) -> Result<Self, RusTorchError> {
        Ok(Self {
            num_features,
            weight: F32Tensor::ones(&[num_features]),
            bias: F32Tensor::zeros(&[num_features]),
            running_mean: F32Tensor::zeros(&[num_features]),
            running_var: F32Tensor::ones(&[num_features]),
            momentum,
            eps,
            training: true,
            weight_grad: None,
            bias_grad: None,
        })
    }

    /// 訓練モードを設定
    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    /// フォワードパス
    /// Forward pass
    pub fn forward(&mut self, input: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        // 入力形状: (batch_size, num_features, height, width)
        let input_shape = input.shape();
        if input_shape.len() != 4 || input_shape[1] != self.num_features {
            return Err(RusTorchError::tensor_op("BatchNorm2d input shape mismatch"));
        }

        let batch_size = input_shape[0];
        let height = input_shape[2];
        let width = input_shape[3];
        let spatial_size = height * width;

        let input_data = input.as_slice();
        let mut output_data = vec![0.0; input_data.len()];

        if self.training {
            // 訓練モード：バッチ統計を計算
            let mut batch_mean = vec![0.0; self.num_features];
            let mut batch_var = vec![0.0; self.num_features];

            // 平均計算
            for c in 0..self.num_features {
                let mut sum = 0.0;
                for b in 0..batch_size {
                    for spatial in 0..spatial_size {
                        let idx = b * self.num_features * spatial_size + c * spatial_size + spatial;
                        sum += input_data[idx];
                    }
                }
                batch_mean[c] = sum / (batch_size * spatial_size) as f32;
            }

            // 分散計算
            for c in 0..self.num_features {
                let mut sum_sq = 0.0;
                for b in 0..batch_size {
                    for spatial in 0..spatial_size {
                        let idx = b * self.num_features * spatial_size + c * spatial_size + spatial;
                        let diff = input_data[idx] - batch_mean[c];
                        sum_sq += diff * diff;
                    }
                }
                batch_var[c] = sum_sq / (batch_size * spatial_size) as f32;
            }

            // 移動平均更新
            let running_mean_data = self.running_mean.as_slice();
            let running_var_data = self.running_var.as_slice();
            let mut new_running_mean = vec![0.0; self.num_features];
            let mut new_running_var = vec![0.0; self.num_features];

            for c in 0..self.num_features {
                new_running_mean[c] = (1.0 - self.momentum) * running_mean_data[c] + self.momentum * batch_mean[c];
                new_running_var[c] = (1.0 - self.momentum) * running_var_data[c] + self.momentum * batch_var[c];
            }

            self.running_mean = F32Tensor::from_vec(new_running_mean, vec![self.num_features])?;
            self.running_var = F32Tensor::from_vec(new_running_var, vec![self.num_features])?;

            // 正規化と変換
            let weight_data = self.weight.as_slice();
            let bias_data = self.bias.as_slice();

            for c in 0..self.num_features {
                let std = (batch_var[c] + self.eps).sqrt();
                for b in 0..batch_size {
                    for spatial in 0..spatial_size {
                        let idx = b * self.num_features * spatial_size + c * spatial_size + spatial;
                        let normalized = (input_data[idx] - batch_mean[c]) / std;
                        output_data[idx] = weight_data[c] * normalized + bias_data[c];
                    }
                }
            }
        } else {
            // 推論モード：保存された統計を使用
            let running_mean_data = self.running_mean.as_slice();
            let running_var_data = self.running_var.as_slice();
            let weight_data = self.weight.as_slice();
            let bias_data = self.bias.as_slice();

            for c in 0..self.num_features {
                let std = (running_var_data[c] + self.eps).sqrt();
                for b in 0..batch_size {
                    for spatial in 0..spatial_size {
                        let idx = b * self.num_features * spatial_size + c * spatial_size + spatial;
                        let normalized = (input_data[idx] - running_mean_data[c]) / std;
                        output_data[idx] = weight_data[c] * normalized + bias_data[c];
                    }
                }
            }
        }

        F32Tensor::from_vec(output_data, input_shape.to_vec())
    }
}

impl F32Layer for F32BatchNorm2d {
    fn forward(&mut self, input: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        self.forward(input)
    }

    fn backward(&mut self, grad_output: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        // 簡素化されたバックワード実装
        // TODO: 実際のバッチ正規化の逆伝播を実装
        Ok(grad_output.clone()?)
    }

    fn parameters(&self) -> Vec<&F32Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn update_parameters(&mut self, learning_rate: f32) -> Result<(), RusTorchError> {
        if let Some(ref weight_grad) = self.weight_grad {
            let lr_tensor = F32Tensor::from_vec(vec![learning_rate], vec![1])?;
            let update = weight_grad.mul(&lr_tensor)?;
            self.weight = self.weight.sub(&update)?;
        }

        if let Some(ref bias_grad) = self.bias_grad {
            let lr_tensor = F32Tensor::from_vec(vec![learning_rate], vec![1])?;
            let update = bias_grad.mul(&lr_tensor)?;
            self.bias = self.bias.sub(&update)?;
        }

        Ok(())
    }
}

/// 簡単なCNNモデル
/// Simple CNN model
#[derive(Debug)]
pub struct F32SimpleCNN {
    pub conv1: F32Conv2d,
    pub bn1: F32BatchNorm2d,
    pub conv2: F32Conv2d,
    pub bn2: F32BatchNorm2d,
    pub fc: F32Linear,
    pub num_classes: usize,
}

impl F32SimpleCNN {
    /// 新しいCNNモデルを作成
    /// Create a new CNN model
    pub fn new(
        input_channels: usize,
        num_classes: usize,
        hidden_channels: usize,
    ) -> Result<Self, RusTorchError> {
        let conv1 = F32Conv2d::new(input_channels, hidden_channels, (3, 3), (1, 1), (1, 1), true)?;
        let bn1 = F32BatchNorm2d::new(hidden_channels, 0.1, 1e-5)?;
        let conv2 = F32Conv2d::new(hidden_channels, hidden_channels * 2, (3, 3), (2, 2), (1, 1), true)?;
        let bn2 = F32BatchNorm2d::new(hidden_channels * 2, 0.1, 1e-5)?;

        // 仮定：28x28の入力画像（MNIST風）でのFC層の入力サイズ
        // 28x28 -> conv1 -> 28x28 -> conv2 (stride=2) -> 14x14
        let fc_input_size = hidden_channels * 2 * 14 * 14;
        let fc = F32Linear::new(fc_input_size, num_classes, true)?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            fc,
            num_classes,
        })
    }

    /// 訓練モードを設定
    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.bn1.train(mode);
        self.bn2.train(mode);
    }

    /// フォワードパス
    /// Forward pass
    pub fn forward(&mut self, input: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        // Conv1 -> BN1 -> ReLU
        let x = self.conv1.forward(input)?;
        let x = self.bn1.forward(&x)?;
        let x = F32Activation::ReLU.forward(&x)?;

        // Conv2 -> BN2 -> ReLU
        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward(&x)?;
        let x = F32Activation::ReLU.forward(&x)?;

        // Flatten
        let batch_size = x.shape()[0];
        let flattened_size = x.shape().iter().skip(1).product();
        let x = x.reshape(&[batch_size, flattened_size])?;

        // FC layer
        self.fc.forward(&x)
    }
}

impl F32Layer for F32SimpleCNN {
    fn forward(&mut self, input: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        self.forward(input)
    }

    fn backward(&mut self, grad_output: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        // 簡素化されたバックワード実装
        // TODO: 実際のCNNの逆伝播を実装
        Ok(grad_output.clone()?)
    }

    fn parameters(&self) -> Vec<&F32Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.fc.parameters());
        params
    }

    fn update_parameters(&mut self, learning_rate: f32) -> Result<(), RusTorchError> {
        self.conv1.update_parameters(learning_rate)?;
        self.bn1.update_parameters(learning_rate)?;
        self.conv2.update_parameters(learning_rate)?;
        self.bn2.update_parameters(learning_rate)?;
        self.fc.update_parameters(learning_rate)?;
        Ok(())
    }
}

// ============================================================================
// Pre-trained Models & Image Processing / 事前学習モデル＆画像処理
// ============================================================================

/// 事前学習モデルのメタデータ
/// Pre-trained model metadata
#[derive(Debug, Clone)]
pub struct F32PretrainedModelInfo {
    pub name: String,
    pub architecture: String,
    pub input_size: (usize, usize, usize), // (channels, height, width)
    pub num_classes: usize,
    pub mean: Vec<f32>,     // 正規化の平均値
    pub std: Vec<f32>,      // 正規化の標準偏差
    pub file_path: String,
}

/// 事前学習モデルローダー
/// Pre-trained model loader
#[derive(Debug)]
pub struct F32PretrainedLoader {
    pub available_models: Vec<F32PretrainedModelInfo>,
}

impl F32PretrainedLoader {
    /// 新しいローダーを作成
    /// Create a new loader
    pub fn new() -> Self {
        let mut available_models = Vec::new();

        // ResNet18風のモデル情報（例）
        available_models.push(F32PretrainedModelInfo {
            name: "resnet18".to_string(),
            architecture: "ResNet".to_string(),
            input_size: (3, 224, 224),
            num_classes: 1000,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            file_path: "models/resnet18.safetensors".to_string(),
        });

        // MobileNetV2風のモデル情報（例）
        available_models.push(F32PretrainedModelInfo {
            name: "mobilenet_v2".to_string(),
            architecture: "MobileNet".to_string(),
            input_size: (3, 224, 224),
            num_classes: 1000,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            file_path: "models/mobilenet_v2.safetensors".to_string(),
        });

        Self { available_models }
    }

    /// 利用可能なモデル一覧を取得
    /// Get list of available models
    pub fn list_models(&self) -> &[F32PretrainedModelInfo] {
        &self.available_models
    }

    /// モデル情報を取得
    /// Get model info
    pub fn get_model_info(&self, name: &str) -> Option<&F32PretrainedModelInfo> {
        self.available_models.iter().find(|model| model.name == name)
    }

    /// 事前学習モデルをロード（簡素化された実装）
    /// Load pre-trained model (simplified implementation)
    pub fn load_model(&self, name: &str) -> Result<F32SimpleCNN, RusTorchError> {
        let model_info = self.get_model_info(name)
            .ok_or_else(|| RusTorchError::tensor_op(&format!("Model '{}' not found", name)))?;

        // 簡素化：実際のファイルからロードする代わりに、新しいモデルを作成
        // TODO: 実際の事前学習重みをロードする実装
        println!("Warning: Loading architecture only, not pre-trained weights for {}", name);

        match model_info.architecture.as_str() {
            "ResNet" | "MobileNet" => {
                let input_channels = model_info.input_size.0;
                let num_classes = model_info.num_classes;
                F32SimpleCNN::new(input_channels, num_classes, 64)
            }
            _ => Err(RusTorchError::tensor_op(&format!("Unsupported architecture: {}", model_info.architecture)))
        }
    }
}

impl Default for F32PretrainedLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// 画像前処理ユーティリティ
/// Image preprocessing utilities
#[derive(Debug)]
pub struct F32ImagePreprocessor {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub resize_size: Option<(usize, usize)>,
}

impl F32ImagePreprocessor {
    /// 新しい前処理器を作成
    /// Create a new preprocessor
    pub fn new(mean: Vec<f32>, std: Vec<f32>, resize_size: Option<(usize, usize)>) -> Self {
        Self { mean, std, resize_size }
    }

    /// ImageNet用の標準前処理器
    /// Standard preprocessor for ImageNet
    pub fn imagenet() -> Self {
        Self::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
            Some((224, 224)),
        )
    }

    /// 正規化を適用（簡素化された実装）
    /// Apply normalization (simplified implementation)
    pub fn normalize(&self, input: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(RusTorchError::tensor_op("Image input must be 4D (batch, channels, height, width)"));
        }

        let channels = input_shape[1];
        if channels != self.mean.len() || channels != self.std.len() {
            return Err(RusTorchError::tensor_op("Channel count mismatch with mean/std"));
        }

        let input_data = input.as_slice();
        let mut output_data = vec![0.0; input_data.len()];

        let batch_size = input_shape[0];
        let height = input_shape[2];
        let width = input_shape[3];
        let spatial_size = height * width;

        for b in 0..batch_size {
            for c in 0..channels {
                for spatial in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + spatial;
                    output_data[idx] = (input_data[idx] - self.mean[c]) / self.std[c];
                }
            }
        }

        F32Tensor::from_vec(output_data, input_shape.to_vec())
    }

    /// 前処理パイプライン（正規化のみ）
    /// Preprocessing pipeline (normalization only)
    pub fn preprocess(&self, input: &F32Tensor) -> Result<F32Tensor, RusTorchError> {
        // TODO: リサイズ機能の実装
        self.normalize(input)
    }
}

impl Default for F32ImagePreprocessor {
    fn default() -> Self {
        Self::imagenet()
    }
}