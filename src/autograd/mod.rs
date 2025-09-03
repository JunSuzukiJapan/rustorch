use crate::autograd::grad_fn::{
    AddBackward, MatMulBackward, MulBackward, SubBackward, SumBackward,
};
use crate::serialization::core::{Loadable, Saveable, SerializationError, SerializationResult};
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

// Re-export Phase 4 gradient utilities
pub use context::{
    detect_anomaly, enable_grad, is_anomaly_detection_enabled, is_grad_enabled, no_grad,
    set_anomaly_detection, set_grad_enabled, AnomalyDetectionGuard, EnableGradGuard,
    GradientContext, NoGradGuard,
};
pub use grad_utils::{grad, gradient, is_variable_in_graph, validate_grad_setup, GradError};
pub use gradcheck::{gradcheck, gradcheck_simple, GradCheckConfig, GradCheckResult};
pub use higher_order::{hessian, hvp, jacobian};

pub mod context;
pub mod function;
pub mod grad_fn;
pub mod grad_utils;
pub mod gradcheck;
pub mod graph;
pub mod higher_order;
pub mod linear_grad_fn;
pub mod visualization;

#[cfg(test)]
mod tests;

/// Gradient function trait for backward computation
/// 逆伝播計算のための勾配関数トレイト
pub trait GradFn<T: Float + Send + Sync + 'static>: Send + Sync {
    /// Apply the gradient function to compute input gradients
    /// 勾配関数を適用して入力勾配を計算
    fn apply(&self, grad_outputs: &[Tensor<T>]) -> Vec<Option<Tensor<T>>>;
}

// Global counter for unique Variable IDs
static VARIABLE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A variable that supports automatic differentiation.
/// 自動微分をサポートする変数
pub struct Variable<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> {
    data: Arc<RwLock<Tensor<T>>>,
    grad: Arc<RwLock<Option<Tensor<T>>>>,
    requires_grad: bool,
    unique_id: usize,
    grad_fn: Option<Arc<dyn GradFn<T>>>,
    _marker: PhantomData<T>,
}

impl<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> std::fmt::Debug
    for Variable<T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Variable")
            .field("requires_grad", &self.requires_grad)
            .finish()
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Clone
    for Variable<T>
{
    fn clone(&self) -> Self {
        Variable {
            data: self.data.clone(),
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            unique_id: self.unique_id,
            grad_fn: self.grad_fn.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    Variable<T>
{
    /// Creates a new variable with the given tensor.
    /// 与えられたテンソルで新しい変数を作成します。
    pub fn new(data: Tensor<T>, requires_grad: bool) -> Self {
        Variable {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(None)),
            requires_grad,
            unique_id: VARIABLE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            grad_fn: None,
            _marker: PhantomData,
        }
    }

    /// Returns the unique identifier for this Variable
    /// この変数の一意識別子を返します
    pub fn id(&self) -> usize {
        self.unique_id
    }

    /// Creates a new variable with gradient function
    /// 勾配関数付きの新しい変数を作成します
    pub fn new_with_grad_fn(
        data: Tensor<T>,
        requires_grad: bool,
        grad_fn: Option<Arc<dyn GradFn<T>>>,
    ) -> Self {
        Variable {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(None)),
            requires_grad,
            unique_id: VARIABLE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            grad_fn,
            _marker: PhantomData,
        }
    }

    /// Returns the data tensor.
    /// データテンソルを返します。
    pub fn data(&self) -> Arc<RwLock<Tensor<T>>> {
        self.data.clone()
    }

    /// Returns whether this variable requires gradients.
    /// この変数が勾配を必要とするかどうかを返します。
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Returns the gradient tensor.
    /// 勾配テンソルを返します。
    pub fn grad(&self) -> Arc<RwLock<Option<Tensor<T>>>> {
        self.grad.clone()
    }

    /// Returns the gradient function if any.
    /// 勾配関数があれば返します。
    pub fn grad_fn(&self) -> &Option<Arc<dyn GradFn<T>>> {
        &self.grad_fn
    }

    /// Zeros out the gradient.
    /// 勾配をゼロクリアします。
    pub fn zero_grad(&self) {
        if let Ok(mut grad) = self.grad.write() {
            *grad = None;
        }
    }

    /// Performs backward pass to compute gradients.
    /// 逆伝播を実行して勾配を計算します。
    pub fn backward(&self) {
        self.backward_with_grad(None);
    }

    /// Performs backward pass with a specific gradient.
    /// 特定の勾配で逆伝播を実行します。
    pub fn backward_with_grad(&self, grad_output: Option<Tensor<T>>) {
        use crate::autograd::context::is_grad_enabled;

        if !self.requires_grad || !is_grad_enabled() {
            return;
        }

        // Initialize gradient if not provided
        let initial_grad = grad_output.unwrap_or_else(|| {
            let data = self.data.read().unwrap();
            if data.numel() == 1 && data.shape().is_empty() {
                // Scalar case - gradient is 1 with scalar shape
                Tensor::ones(&[])
            } else {
                // Vector/matrix case - gradient is ones with same shape
                Tensor::ones(data.shape())
            }
        });

        // Set the gradient for this variable
        if let Ok(mut grad) = self.grad.write() {
            match grad.as_mut() {
                Some(existing_grad) => {
                    // Accumulate gradients
                    *existing_grad = &*existing_grad + &initial_grad;
                }
                None => {
                    *grad = Some(initial_grad.clone());
                }
            }
        }

        // Call the gradient function if it exists (for non-leaf nodes)
        if let Some(grad_fn) = &self.grad_fn {
            let _grad_inputs = grad_fn.apply(&[initial_grad]);
        }
    }

    /// Matrix multiplication with automatic differentiation support
    /// 自動微分をサポートする行列乗算
    pub fn matmul(&self, other: &Variable<T>) -> Variable<T> {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = other.data.read().unwrap().clone();

        // Try batch matmul for higher dimensions, fall back to regular matmul
        let result_data = if lhs_data.shape().len() > 2 || rhs_data.shape().len() > 2 {
            use crate::tensor::parallel_traits::MatrixParallelOp;
            lhs_data
                .batch_matmul(&rhs_data)
                .or_else(|_| lhs_data.matmul(&rhs_data))
                .expect("MatMul failed")
        } else {
            lhs_data.matmul(&rhs_data).expect("MatMul failed")
        };

        if self.requires_grad || other.requires_grad {
            let grad_fn = Arc::new(MatMulBackward {
                input0_data: lhs_data,
                input1_data: rhs_data,
                input0_var: Some(self.clone()),
                input1_var: Some(other.clone()),
            });
            Variable::new_with_grad_fn(result_data, true, Some(grad_fn))
        } else {
            Variable::new(result_data, false)
        }
    }

    /// Transpose the last two dimensions
    /// 最後の2次元を転置
    pub fn transpose_last_two(&self) -> Variable<T> {
        let input_data = self.data.read().unwrap();
        let result_data = input_data
            .transpose_last_two()
            .expect("Transpose last two failed");

        // For now, no gradient support for transpose
        // 現在のところ、転置の勾配サポートはなし
        Variable::new(result_data, false)
    }

    /// Batch matrix multiplication for 4D tensors (batch_size, num_heads, seq_len, d_k)
    /// 4Dテンソル用のバッチ行列乗算（batch_size, num_heads, seq_len, d_k）
    pub fn attention_matmul(&self, other: &Variable<T>) -> Variable<T> {
        let lhs_data = self.data.read().unwrap();
        let rhs_data = other.data.read().unwrap();
        let lhs_shape = lhs_data.shape();
        let rhs_shape = rhs_data.shape();

        if lhs_shape.len() != 4 || rhs_shape.len() != 4 {
            return self.matmul(other); // Fallback to regular matmul
        }

        let (batch_size, num_heads, seq_len_lhs, d_k_lhs) =
            (lhs_shape[0], lhs_shape[1], lhs_shape[2], lhs_shape[3]);
        let (batch_size_rhs, num_heads_rhs, seq_len_rhs, d_k_rhs) =
            (rhs_shape[0], rhs_shape[1], rhs_shape[2], rhs_shape[3]);

        if batch_size != batch_size_rhs || num_heads != num_heads_rhs || d_k_lhs != seq_len_rhs {
            panic!(
                "Attention matmul dimension mismatch: {:?} vs {:?}",
                lhs_shape, rhs_shape
            );
        }

        // Result shape: (batch_size, num_heads, seq_len_lhs, d_k_rhs)
        let output_shape = vec![batch_size, num_heads, seq_len_lhs, d_k_rhs];
        let mut result_data = vec![T::zero(); batch_size * num_heads * seq_len_lhs * d_k_rhs];

        // Perform batch matrix multiplication
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_lhs {
                    for j in 0..d_k_rhs {
                        let mut sum = T::zero();
                        for k in 0..d_k_lhs {
                            // Use ndarray's get method with proper multi-dimensional indexing
                            let lhs_val = lhs_data
                                .data
                                .get(ndarray::IxDyn(&[b, h, i, k]))
                                .copied()
                                .unwrap_or(T::zero());
                            let rhs_val = rhs_data
                                .data
                                .get(ndarray::IxDyn(&[b, h, k, j]))
                                .copied()
                                .unwrap_or(T::zero());
                            sum = sum + lhs_val * rhs_val;
                        }
                        let result_idx = b * (num_heads * seq_len_lhs * d_k_rhs)
                            + h * (seq_len_lhs * d_k_rhs)
                            + i * d_k_rhs
                            + j;
                        result_data[result_idx] = sum;
                    }
                }
            }
        }

        let result_tensor = Tensor::from_vec(result_data, output_shape);

        // For now, no gradient support for attention matmul
        Variable::new(result_tensor, false)
    }

    /// Sum all elements with automatic differentiation support
    /// 自動微分をサポートする全要素の和
    pub fn sum(&self) -> Variable<T> {
        let input_data = self.data.read().unwrap();
        let input_shape = input_data.shape().to_vec();
        let sum_value = input_data.sum();
        let result_data = Tensor::from_vec(vec![sum_value], vec![1]);

        if self.requires_grad {
            let grad_fn = Arc::new(SumBackward {
                input_shape,
                input_var: self.clone(),
                _phantom: PhantomData,
            });
            Variable::new_with_grad_fn(result_data, true, Some(grad_fn))
        } else {
            Variable::new(result_data, false)
        }
    }

    /// Power function with automatic differentiation support
    /// 自動微分をサポートするべき乗関数
    pub fn pow(&self, exponent: T) -> Variable<T> {
        let input_data = self.data.read().unwrap().clone();
        let mut result_data = input_data.clone();
        result_data
            .as_array_mut()
            .mapv_inplace(|x| x.powf(exponent));

        if self.requires_grad {
            // For now, return without proper gradient function
            Variable::new(result_data, true)
        } else {
            Variable::new(result_data, false)
        }
    }

    /// Mean of all elements with automatic differentiation support
    /// 自動微分をサポートする全要素の平均
    pub fn mean_autograd(&self) -> Variable<T> {
        let sum_var = self.sum();
        let input_data = self.data.read().unwrap();
        let numel = T::from(input_data.numel()).unwrap();

        let sum_data = sum_var.data.read().unwrap().clone();
        let mut mean_data = sum_data;
        mean_data.as_array_mut().mapv_inplace(|x| x / numel);

        if self.requires_grad {
            let grad_fn = std::sync::Arc::new(crate::autograd::grad_fn::MeanBackward {
                input_var: Some(self.clone()),
                numel,
            });
            Variable::new_with_grad_fn(mean_data, true, Some(grad_fn))
        } else {
            Variable::new(mean_data, false)
        }
    }
}

// Implement arithmetic operators for Variables
impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add
    for &Variable<T>
{
    type Output = Variable<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = rhs.data.read().unwrap().clone();
        let result_data = &lhs_data + &rhs_data;

        if self.requires_grad || rhs.requires_grad {
            let grad_fn = Arc::new(AddBackward {
                input0_data: lhs_data,
                input1_data: rhs_data,
                input0_var: self.clone(),
                input1_var: rhs.clone(),
            });
            Variable::new_with_grad_fn(result_data, true, Some(grad_fn))
        } else {
            Variable::new(result_data, false)
        }
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul
    for &Variable<T>
{
    type Output = Variable<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = rhs.data.read().unwrap().clone();
        let result_data = &lhs_data * &rhs_data;

        if self.requires_grad || rhs.requires_grad {
            let grad_fn = Arc::new(MulBackward {
                input0_data: lhs_data,
                input1_data: rhs_data,
                input0_var: self.clone(),
                input1_var: rhs.clone(),
            });
            Variable::new_with_grad_fn(result_data, true, Some(grad_fn))
        } else {
            Variable::new(result_data, false)
        }
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub
    for &Variable<T>
{
    type Output = Variable<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = rhs.data.read().unwrap().clone();
        let result_data = &lhs_data - &rhs_data;

        if self.requires_grad || rhs.requires_grad {
            let grad_fn = Arc::new(SubBackward {
                input0_data: lhs_data,
                input1_data: rhs_data,
                input0_var: self.clone(),
                input1_var: rhs.clone(),
            });
            Variable::new_with_grad_fn(result_data, true, Some(grad_fn))
        } else {
            Variable::new(result_data, false)
        }
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    ops::Sub<&Variable<T>> for Variable<T>
{
    type Output = Variable<T>;

    fn sub(self, rhs: &Variable<T>) -> Self::Output {
        &self - rhs
    }
}

// Macro for implementing binary operations to reduce code duplication
macro_rules! impl_binary_op_owned {
    ($trait:ident, $method:ident) => {
        impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
            ops::$trait for Variable<T>
        {
            type Output = Variable<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! impl_binary_op_mixed {
    ($trait:ident, $method:ident) => {
        impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
            ops::$trait<&Variable<T>> for Variable<T>
        {
            type Output = Variable<T>;

            fn $method(self, rhs: &Variable<T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

// Apply macros for all binary operations
impl_binary_op_owned!(Add, add);
impl_binary_op_owned!(Mul, mul);
impl_binary_op_owned!(Sub, sub);

impl_binary_op_mixed!(Add, add);
impl_binary_op_mixed!(Mul, mul);

// Add division operator
impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div
    for &Variable<T>
{
    type Output = Variable<T>;

    fn div(self, rhs: Self) -> Self::Output {
        let lhs_data = self.data.read().unwrap().clone();
        let rhs_data = rhs.data.read().unwrap().clone();
        let result_data = &lhs_data / &rhs_data;

        if self.requires_grad || rhs.requires_grad {
            // For now, create division without gradient function
            Variable::new(result_data, true)
        } else {
            Variable::new(result_data, false)
        }
    }
}

// Serialization support for Variable
impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Saveable
    for Variable<T>
{
    fn save_binary(&self) -> SerializationResult<Vec<u8>> {
        let mut buffer = Vec::new();

        // Save requires_grad flag
        buffer.push(self.requires_grad as u8);

        // Save tensor data
        let data_guard = self.data.read().map_err(|_| {
            SerializationError::FormatError("Failed to read tensor data".to_string())
        })?;
        let tensor_data = data_guard.save_binary()?;
        let tensor_size = tensor_data.len() as u64;
        buffer.extend_from_slice(&tensor_size.to_le_bytes());
        buffer.extend_from_slice(&tensor_data);

        // Save gradient (if present)
        let grad_guard = self.grad.read().map_err(|_| {
            SerializationError::FormatError("Failed to read gradient data".to_string())
        })?;
        let has_grad = grad_guard.is_some();
        buffer.push(has_grad as u8);

        if let Some(ref grad_tensor) = *grad_guard {
            let grad_data = grad_tensor.save_binary()?;
            let grad_size = grad_data.len() as u64;
            buffer.extend_from_slice(&grad_size.to_le_bytes());
            buffer.extend_from_slice(&grad_data);
        }

        Ok(buffer)
    }

    fn type_id(&self) -> &'static str {
        "autograd.Variable"
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert("requires_grad".to_string(), self.requires_grad.to_string());
        meta.insert("unique_id".to_string(), self.unique_id.to_string());

        if let Ok(data_guard) = self.data.read() {
            meta.insert("shape".to_string(), format!("{:?}", data_guard.shape()));
            meta.insert("numel".to_string(), data_guard.numel().to_string());
        }

        meta
    }
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Loadable
    for Variable<T>
{
    fn load_binary(data: &[u8]) -> SerializationResult<Self> {
        let mut offset = 0;

        // Load requires_grad flag
        if data.len() < offset + 1 {
            return Err(SerializationError::FormatError(
                "Insufficient data for requires_grad".to_string(),
            ));
        }
        let requires_grad = data[offset] != 0;
        offset += 1;

        // Load tensor data
        if data.len() < offset + 8 {
            return Err(SerializationError::FormatError(
                "Insufficient data for tensor size".to_string(),
            ));
        }

        let tensor_size = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;
        offset += 8;

        if data.len() < offset + tensor_size {
            return Err(SerializationError::FormatError(
                "Insufficient data for tensor".to_string(),
            ));
        }

        let tensor_data = &data[offset..offset + tensor_size];
        let tensor = Tensor::load_binary(tensor_data)?;
        offset += tensor_size;

        // Create Variable with basic data first
        let variable = Variable::new(tensor, requires_grad);

        // Load gradient (if present)
        if data.len() < offset + 1 {
            return Err(SerializationError::FormatError(
                "Insufficient data for gradient flag".to_string(),
            ));
        }

        let has_grad = data[offset] != 0;
        offset += 1;

        if has_grad {
            if data.len() < offset + 8 {
                return Err(SerializationError::FormatError(
                    "Insufficient data for gradient size".to_string(),
                ));
            }

            let grad_size = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]) as usize;
            offset += 8;

            if data.len() < offset + grad_size {
                return Err(SerializationError::FormatError(
                    "Insufficient data for gradient".to_string(),
                ));
            }

            let grad_data = &data[offset..offset + grad_size];
            let grad_tensor = Tensor::load_binary(grad_data)?;

            // Set gradient
            if let Ok(mut grad_guard) = variable.grad.write() {
                *grad_guard = Some(grad_tensor);
            }
        }

        Ok(variable)
    }

    fn expected_type_id() -> &'static str {
        "autograd.Variable"
    }
}
