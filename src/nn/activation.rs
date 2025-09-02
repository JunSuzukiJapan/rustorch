//! Activation functions for neural networks
//! ニューラルネットワークの活性化関数

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Tensor;
use num_traits::Float;
use std::fmt::Debug;

/// ReLU (Rectified Linear Unit) activation function
/// ReLU（正規化線形ユニット）活性化関数
///
/// Applies the element-wise function: ReLU(x) = max(0, x)
/// 要素ごとに関数を適用: ReLU(x) = max(0, x)
pub fn relu<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // Apply ReLU: max(0, x)
    let output_data = apply_relu(&input_data);

    if x.requires_grad() {
        // Create a new variable that tracks gradients
        let result = Variable::new(output_data, true);

        // Store reference to input for backward pass
        // Note: In a full implementation, we'd use a proper gradient function
        // For now, we'll implement a simplified version
        result
    } else {
        Variable::new(output_data, false)
    }
}

/// Sigmoid activation function
/// シグモイド活性化関数
///
/// Applies the element-wise function: Sigmoid(x) = 1 / (1 + exp(-x))
/// 要素ごとに関数を適用: Sigmoid(x) = 1 / (1 + exp(-x))
pub fn sigmoid<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // Apply Sigmoid: 1 / (1 + exp(-x))
    let output_data = apply_sigmoid(&input_data);

    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// Tanh (Hyperbolic Tangent) activation function
/// Tanh（双曲線正接）活性化関数
///
/// Applies the element-wise function: Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
/// 要素ごとに関数を適用: Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
pub fn tanh<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // Apply Tanh
    let output_data = apply_tanh(&input_data);

    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// Leaky ReLU activation function
/// Leaky ReLU活性化関数
///
/// Applies the element-wise function: LeakyReLU(x) = max(alpha * x, x)
/// 要素ごとに関数を適用: LeakyReLU(x) = max(alpha * x, x)
pub fn leaky_relu<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
    alpha: T,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // Apply Leaky ReLU: max(alpha * x, x)
    let output_data = apply_leaky_relu(&input_data, alpha);

    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// Softmax activation function
/// ソフトマックス活性化関数
///
/// Applies softmax along the last dimension: Softmax(x_i) = exp(x_i) / sum(exp(x_j))
/// 最後の次元に沿ってソフトマックスを適用: Softmax(x_i) = exp(x_i) / sum(exp(x_j))
pub fn softmax<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // Apply Softmax
    let output_data = apply_softmax(&input_data);

    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

// Helper functions for applying activation functions to tensors
// テンソルに活性化関数を適用するヘルパー関数

fn apply_relu<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| if x > T::zero() { x } else { T::zero() })
        .collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_sigmoid<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            let one = T::one();
            one / (one + (-x).exp())
        })
        .collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_tanh<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            let exp_x = x.exp();
            let exp_neg_x = (-x).exp();
            (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
        })
        .collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_leaky_relu<T: Float + 'static>(tensor: &Tensor<T>, alpha: T) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| if x > T::zero() { x } else { alpha * x })
        .collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_softmax<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();

    // For numerical stability, subtract the maximum value
    let max_val = data
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(T::zero());

    // Compute exp(x - max) for each element
    let exp_values: Vec<T> = data.iter().map(|&x| (x - max_val).exp()).collect();

    // Compute sum of exponentials
    let sum_exp = exp_values.iter().fold(T::zero(), |acc, &x| acc + x);

    // Normalize by dividing by sum
    let result_data: Vec<T> = exp_values.iter().map(|&x| x / sum_exp).collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

/// GELU (Gaussian Error Linear Unit) activation function
/// GELU（ガウス誤差線形ユニット）活性化関数
///
/// Applies the element-wise function: GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal
/// 要素ごとに関数を適用: GELU(x) = x * Φ(x) ここでΦ(x)は標準正規分布の累積分布関数
pub fn gelu<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // Apply GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let output_data = apply_gelu(&input_data);

    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// Swish (SiLU) activation function
/// Swish（SiLU）活性化関数
///
/// Applies the element-wise function: Swish(x) = x * sigmoid(x)
/// 要素ごとに関数を適用: Swish(x) = x * sigmoid(x)
pub fn swish<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // Apply Swish: x * sigmoid(x)
    let output_data = apply_swish(&input_data);

    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// ELU (Exponential Linear Unit) activation function
/// ELU（指数線形ユニット）活性化関数
///
/// Applies the element-wise function: ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
/// 要素ごとに関数を適用: ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
pub fn elu<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
    alpha: T,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // Apply ELU
    let output_data = apply_elu(&input_data, alpha);

    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// SELU (Scaled Exponential Linear Unit) activation function
/// SELU（スケール指数線形ユニット）活性化関数
///
/// Applies the element-wise function with fixed parameters for self-normalizing properties
/// 自己正規化特性のため固定パラメータで要素ごとに関数を適用
pub fn selu<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // SELU with fixed parameters: alpha = 1.6732632423543772848170429916717, scale = 1.0507009873554804934193349852946
    let alpha = T::from(1.673_263_242_354_377_284_817_042_991_671_7_f32).unwrap();
    let scale = T::from(1.050_700_987_355_480_493_419_334_985_294_6_f32).unwrap();
    let output_data = apply_selu(&input_data, alpha, scale);

    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// Mish activation function
/// Mish活性化関数
///
/// Applies the element-wise function: Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
/// 要素ごとに関数を適用: Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
pub fn mish<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // Apply Mish: x * tanh(softplus(x))
    let output_data = apply_mish(&input_data);

    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// GLU (Gated Linear Unit) activation function
/// GLU（ゲート線形ユニット）活性化関数
///
/// Applies the element-wise function: GLU(x) = a ⊙ σ(b)
/// where x is split into two halves a and b along the last dimension
/// 要素ごとに関数を適用: GLU(x) = a ⊙ σ(b)
/// ここで x は最後の次元で半分に分割され a と b になります
pub fn glu<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + Default,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_tensor = x.data();
    let input_guard = input_tensor.read().unwrap();
    let input_shape = input_guard.shape();
    let input_data = input_guard.as_slice().unwrap();
    
    // GLU requires even number of elements on the last dimension
    let last_dim = input_shape[input_shape.len() - 1];
    assert_eq!(last_dim % 2, 0, "GLU requires even number of elements on last dimension");
    
    let half_size = last_dim / 2;
    let total_elements = input_shape.iter().product::<usize>();
    let batch_size = total_elements / last_dim;
    
    // Create output shape (half the size of last dimension)
    let mut output_shape = input_shape.to_vec();
    let last_dim_idx = output_shape.len() - 1;
    output_shape[last_dim_idx] = half_size;
    let output_size = output_shape.iter().product::<usize>();
    let mut output_data = vec![T::default(); output_size];
    
    // Process each batch
    for batch in 0..batch_size {
        let input_offset = batch * last_dim;
        let output_offset = batch * half_size;
        
        for i in 0..half_size {
            let a = input_data[input_offset + i];           // First half (values)
            let b = input_data[input_offset + half_size + i]; // Second half (gates)
            
            // Apply sigmoid to gate and multiply: a * sigmoid(b)
            let sigmoid_b = T::from_f32(1.0).unwrap() / (T::from_f32(1.0).unwrap() + (-b).exp());
            output_data[output_offset + i] = a * sigmoid_b;
        }
    }
    
    let output_tensor = Tensor::from_vec(output_data, output_shape);
    Variable::new(output_tensor, x.requires_grad())
}

/// GLU activation layer struct
/// GLU活性化レイヤー構造体
#[derive(Debug)]
pub struct GLU<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> {
    _phantom: std::marker::PhantomData<T>,
}

/// SwiGLU (Swish-Gated Linear Unit) activation function
/// SwiGLU（Swish-ゲート線形ユニット）活性化関数
pub fn swiglu<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + Default,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_tensor = x.data();
    let input_guard = input_tensor.read().unwrap();
    let input_shape = input_guard.shape();
    let input_data = input_guard.as_slice().unwrap();
    
    let last_dim = input_shape[input_shape.len() - 1];
    assert_eq!(last_dim % 2, 0, "SwiGLU requires even number of elements on last dimension");
    
    let half_size = last_dim / 2;
    let total_elements = input_shape.iter().product::<usize>();
    let batch_size = total_elements / last_dim;
    
    let mut output_shape = input_shape.to_vec();
    let last_idx = output_shape.len() - 1;
    output_shape[last_idx] = half_size;
    let mut output_data = vec![T::default(); batch_size * half_size];
    
    for i in 0..batch_size {
        let offset = i * last_dim;
        for j in 0..half_size {
            let a = input_data[offset + j];
            let b = input_data[offset + j + half_size];
            
            // Swish activation: x * sigmoid(x)
            let swish_b = b * (T::from_f32(1.0).unwrap() / (T::from_f32(1.0).unwrap() + (-b).exp()));
            output_data[i * half_size + j] = a * swish_b;
        }
    }
    
    let output_tensor = Tensor::from_vec(output_data, output_shape);
    Variable::new(output_tensor, x.requires_grad())
}

/// GeGLU (GELU-Gated Linear Unit) activation function  
/// GeGLU（GELU-ゲート線形ユニット）活性化関数
pub fn geglu<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + Default,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_tensor = x.data();
    let input_guard = input_tensor.read().unwrap();
    let input_shape = input_guard.shape();
    let input_data = input_guard.as_slice().unwrap();
    
    let last_dim = input_shape[input_shape.len() - 1];
    assert_eq!(last_dim % 2, 0, "GeGLU requires even number of elements on last dimension");
    
    let half_size = last_dim / 2;
    let total_elements = input_shape.iter().product::<usize>();
    let batch_size = total_elements / last_dim;
    
    let mut output_shape = input_shape.to_vec();
    let last_idx = output_shape.len() - 1;
    output_shape[last_idx] = half_size;
    let mut output_data = vec![T::default(); batch_size * half_size];
    
    for i in 0..batch_size {
        let offset = i * last_dim;
        for j in 0..half_size {
            let a = input_data[offset + j];
            let b = input_data[offset + j + half_size];
            
            // GELU activation: x * Φ(x) where Φ is the standard Gaussian CDF
            let gelu_b = b * T::from_f32(0.5).unwrap() * 
                (T::from_f32(1.0).unwrap() + 
                 (b * T::from_f32(0.7978845608).unwrap() * 
                  (T::from_f32(1.0).unwrap() + T::from_f32(0.044715).unwrap() * b * b)).tanh());
            output_data[i * half_size + j] = a * gelu_b;
        }
    }
    
    let output_tensor = Tensor::from_vec(output_data, output_shape);
    Variable::new(output_tensor, x.requires_grad())
}

/// ReGLU (ReLU-Gated Linear Unit) activation function
/// ReGLU（ReLU-ゲート線形ユニット）活性化関数
pub fn reglu<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + Default,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_tensor = x.data();
    let input_guard = input_tensor.read().unwrap();
    let input_shape = input_guard.shape();
    let input_data = input_guard.as_slice().unwrap();
    
    let last_dim = input_shape[input_shape.len() - 1];
    assert_eq!(last_dim % 2, 0, "ReGLU requires even number of elements on last dimension");
    
    let half_size = last_dim / 2;
    let total_elements = input_shape.iter().product::<usize>();
    let batch_size = total_elements / last_dim;
    
    let mut output_shape = input_shape.to_vec();
    let last_idx = output_shape.len() - 1;
    output_shape[last_idx] = half_size;
    let mut output_data = vec![T::default(); batch_size * half_size];
    
    for i in 0..batch_size {
        let offset = i * last_dim;
        for j in 0..half_size {
            let a = input_data[offset + j];
            let b = input_data[offset + j + half_size];
            
            // ReLU activation: max(0, x)
            let relu_b = if b > T::zero() { b } else { T::zero() };
            output_data[i * half_size + j] = a * relu_b;
        }
    }
    
    let output_tensor = Tensor::from_vec(output_data, output_shape);
    Variable::new(output_tensor, x.requires_grad())
}

impl<T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive + Default> GLU<T> {
    /// Create a new GLU activation function
    /// 新しいGLU活性化関数を作成
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive + Default> Module<T> for GLU<T> {
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        glu(input)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Hardswish activation function (used in MobileNetV3)
/// Hardswish活性化関数（MobileNetV3で使用）
///
/// Applies the element-wise function: Hardswish(x) = x * ReLU6(x + 3) / 6
/// 要素ごとに関数を適用: Hardswish(x) = x * ReLU6(x + 3) / 6
pub fn hardswish<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    x: &Variable<T>,
) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();

    // Apply Hardswish
    let output_data = apply_hardswish(&input_data);

    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

fn apply_gelu<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let sqrt_2_over_pi = T::from(0.7978845608028654f32).unwrap(); // sqrt(2/π)
            let coeff = T::from(0.044715f32).unwrap();
            let half = T::from(0.5f32).unwrap();
            let one = T::one();

            let x_cubed = x * x * x;
            let inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            let tanh_val = {
                let exp_2x = (inner + inner).exp();
                (exp_2x - one) / (exp_2x + one)
            };

            half * x * (one + tanh_val)
        })
        .collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_swish<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            // Swish: x * sigmoid(x)
            let sigmoid_x = T::one() / (T::one() + (-x).exp());
            x * sigmoid_x
        })
        .collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_elu<T: Float + 'static>(tensor: &Tensor<T>, alpha: T) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            if x > T::zero() {
                x
            } else {
                alpha * (x.exp() - T::one())
            }
        })
        .collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_selu<T: Float + 'static>(tensor: &Tensor<T>, alpha: T, scale: T) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            if x > T::zero() {
                scale * x
            } else {
                scale * alpha * (x.exp() - T::one())
            }
        })
        .collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_mish<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
            let softplus = (T::one() + x.exp()).ln();
            let tanh_softplus = {
                let exp_2x = (softplus + softplus).exp();
                (exp_2x - T::one()) / (exp_2x + T::one())
            };
            x * tanh_softplus
        })
        .collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_hardswish<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            // Hardswish: x * ReLU6(x + 3) / 6
            let three = T::from(3.0f32).unwrap();
            let six = T::from(6.0f32).unwrap();

            // ReLU6(x + 3) = min(max(x + 3, 0), 6)
            let relu6_val = (x + three).max(T::zero()).min(six);

            x * relu6_val / six
        })
        .collect();

    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

/// ReLU activation layer (module)
/// ReLU活性化層（モジュール）
#[derive(Debug, Clone)]
pub struct ReLU<T: Float + Send + Sync + 'static + Debug> {
    _phantom: std::marker::PhantomData<T>,
}

impl<
        T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > ReLU<T>
{
    /// Create a new ReLU activation function
    /// 新しいReLU活性化関数を作成
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<
        T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > Module<T> for ReLU<T>
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        relu(input)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn train(&mut self) {
        // No parameters to train
    }

    fn eval(&mut self) {
        // No parameters to eval
    }
}

/// Softmax activation layer (module)
/// Softmax活性化層（モジュール）
#[derive(Debug, Clone)]
pub struct Softmax<T: Float + Send + Sync + 'static + Debug> {
    _phantom: std::marker::PhantomData<T>,
}

impl<
        T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > Softmax<T>
{
    /// Create a new Softmax activation function
    /// 新しいSoftmax活性化関数を作成
    pub fn new(_dim: i32) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<
        T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > Module<T> for Softmax<T>
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        softmax(input)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn train(&mut self) {
        // No parameters to train
    }

    fn eval(&mut self) {
        // No parameters to eval
    }
}

/// GELU activation layer (module)
/// GELU活性化層（モジュール）
#[derive(Debug, Clone)]
pub struct GELU<T: Float + Send + Sync + 'static + Debug> {
    _phantom: std::marker::PhantomData<T>,
}

impl<
        T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > GELU<T>
{
    /// Create a new GELU activation function
    /// 新しいGELU活性化関数を作成
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<
        T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > Module<T> for GELU<T>
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        gelu(input)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn train(&mut self) {
        // No parameters to train
    }

    fn eval(&mut self) {
        // No parameters to eval
    }
}

/// Swish activation layer (module)
/// Swish活性化層（モジュール）
#[derive(Debug, Clone)]
pub struct Swish<T: Float + Send + Sync + 'static + Debug> {
    _phantom: std::marker::PhantomData<T>,
}

/// Tanh activation layer (module)
/// Tanh活性化層（モジュール）
#[derive(Debug, Clone)]
pub struct Tanh<T: Float + Send + Sync + 'static + Debug> {
    _phantom: std::marker::PhantomData<T>,
}

impl<
        T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > Swish<T>
{
    /// Create a new Swish activation function
    /// 新しいSwish活性化関数を作成
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<
        T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > Tanh<T>
{
    /// Create a new Tanh activation function
    /// 新しいTanh活性化関数を作成
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<
        T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > Module<T> for Swish<T>
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        swish(input)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn train(&mut self) {
        // No parameters to train
    }

    fn eval(&mut self) {
        // No parameters to eval
    }
}

impl<
        T: Float + Send + Sync + 'static + Debug + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > Module<T> for Tanh<T>
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        tanh(input)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn train(&mut self) {
        // No parameters to train
    }

    fn eval(&mut self) {
        // No parameters to eval
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_relu() {
        let input = Variable::new(
            Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]),
            false,
        );

        let output = relu(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        let expected = [0.0, 0.0, 0.0, 1.0, 2.0];
        for (actual, expected) in result_data.as_array().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sigmoid() {
        let input = Variable::new(Tensor::from_vec(vec![0.0], vec![1]), false);

        let output = sigmoid(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        // sigmoid(0) = 0.5
        assert_abs_diff_eq!(
            result_data.as_array().iter().next().unwrap(),
            &0.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_tanh() {
        let input = Variable::new(Tensor::from_vec(vec![0.0], vec![1]), false);

        let output = tanh(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        // tanh(0) = 0.0
        assert_abs_diff_eq!(
            result_data.as_array().iter().next().unwrap(),
            &0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_leaky_relu() {
        let input = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]), false);

        let output = leaky_relu(&input, 0.1);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        let expected = [-0.1, 0.0, 1.0];
        for (actual, expected) in result_data.as_array().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_softmax() {
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]), false);

        let output = softmax(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        // Check that probabilities sum to 1
        let sum: f32 = result_data.as_array().iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

        // Check that all values are positive
        for &val in result_data.as_array().iter() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_relu_with_gradients() {
        let input = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]), true);

        let output = relu(&input);
        assert!(output.requires_grad());

        // Test that the computation works
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();
        let expected = [0.0, 0.0, 1.0];
        for (actual, expected) in result_data.as_array().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_gelu() {
        let input = Variable::new(Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]), false);

        let output = gelu(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        // GELU(0) ≈ 0, GELU(1) ≈ 0.8413, GELU(-1) ≈ -0.1587
        assert_abs_diff_eq!(result_data.as_array()[0], 0.0, epsilon = 1e-3);
        assert!(result_data.as_array()[1] > 0.8);
        assert!(result_data.as_array()[2] < 0.0);
    }

    #[test]
    fn test_swish() {
        let input = Variable::new(Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]), false);

        let output = swish(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        // Swish(0) = 0, Swish(1) ≈ 0.7311, Swish(-1) ≈ -0.2689
        assert_abs_diff_eq!(result_data.as_array()[0], 0.0, epsilon = 1e-6);
        assert!(result_data.as_array()[1] > 0.7);
        assert!(result_data.as_array()[2] < 0.0);
    }

    #[test]
    fn test_elu() {
        let input = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]), false);

        let output = elu(&input, 1.0);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        // ELU(-1, α=1) ≈ -0.632, ELU(0) = 0, ELU(1) = 1
        assert!(result_data.as_array()[0] < 0.0 && result_data.as_array()[0] > -1.0);
        assert_abs_diff_eq!(result_data.as_array()[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result_data.as_array()[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_selu() {
        let input = Variable::new(Tensor::from_vec(vec![0.0, 1.0], vec![2]), false);

        let output = selu(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        // SELU should preserve zero and scale positive values
        assert_abs_diff_eq!(result_data.as_array()[0], 0.0, epsilon = 1e-6);
        assert!(result_data.as_array()[1] > 1.0); // Should be scaled
    }

    #[test]
    fn test_mish() {
        let input = Variable::new(Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]), false);

        let output = mish(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        // Mish(0) ≈ 0, Mish(1) ≈ 0.865, Mish(-1) ≈ -0.303
        assert_abs_diff_eq!(result_data.as_array()[0], 0.0, epsilon = 1e-3);
        assert!(result_data.as_array()[1] > 0.8);
        assert!(result_data.as_array()[2] < 0.0);
    }

    #[test]
    fn test_hardswish() {
        let input = Variable::new(Tensor::from_vec(vec![-3.0, 0.0, 3.0, 6.0], vec![4]), false);

        let output = hardswish(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        // Hardswish(-3) = 0, Hardswish(0) = 0, Hardswish(3) = 3, Hardswish(6) = 6
        assert_abs_diff_eq!(result_data.as_array()[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result_data.as_array()[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result_data.as_array()[2], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result_data.as_array()[3], 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_glu() {
        // Test GLU with 4-element input (2 gates)
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 0.5, -1.0], vec![4]), 
            false
        );

        let output = glu(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();

        // GLU splits input in half: gate = sigmoid([0.5, -1.0]), value = [1.0, 2.0]
        // Expected: [1.0 * sigmoid(0.5), 2.0 * sigmoid(-1.0)]
        assert_eq!(result_data.shape(), &[2]); // Half the input size
        assert!(result_data.as_array()[0] > 0.5); // 1.0 * sigmoid(0.5) > 0.5
        assert!(result_data.as_array()[1] > 0.0 && result_data.as_array()[1] < 1.0); // 2.0 * sigmoid(-1.0)
    }
}
