// F64Tensor実装 - 高精度数値計算用
// F64Tensor implementation - for high-precision numerical computation

use ndarray::{Array, IxDyn};
use std::sync::Arc;
use std::ops::{Add, Sub, Mul, Div, Neg, Index, IndexMut};
use crate::common::RusTorchResult;
use super::core::{Index2D, Index3D, DeviceState, MetalBuffer, CoreMLBuffer};

/// f64専用テンソル（高精度計算特化）
/// f64-specific tensor (high-precision computation optimized)
#[derive(Debug)]
pub struct F64Tensor {
    /// CPU側データ
    /// CPU-side data
    pub data: Array<f64, IxDyn>,

    /// GPU共有バッファ（Metal用）
    /// GPU shared buffer (for Metal)
    pub metal_buffer: Option<Arc<MetalBuffer>>,

    /// Neural Engine共有バッファ（CoreML用）
    /// Neural Engine shared buffer (for CoreML)
    pub coreml_buffer: Option<Arc<CoreMLBuffer>>,

    /// デバイス最適化状態
    /// Device optimization state
    pub device_state: DeviceState,

    /// 勾配追跡
    /// Gradient tracking
    pub requires_grad: bool,

    /// テンソル形状
    /// Tensor shape
    shape: Vec<usize>,
}

impl Clone for F64Tensor {
    fn clone(&self) -> Self {
        F64Tensor {
            data: self.data.clone(),
            metal_buffer: self.metal_buffer.clone(),
            coreml_buffer: self.coreml_buffer.clone(),
            device_state: self.device_state.clone(),
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// 演算子実装 - Addition
impl Add<F64Tensor> for F64Tensor {
    type Output = F64Tensor;
    fn add(self, other: F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data + &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Add<&F64Tensor> for F64Tensor {
    type Output = F64Tensor;
    fn add(self, other: &F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data + &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Add for &F64Tensor {
    type Output = F64Tensor;
    fn add(self, other: &F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data + &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Add<f64> for F64Tensor {
    type Output = F64Tensor;
    fn add(self, scalar: f64) -> F64Tensor {
        F64Tensor {
            data: &self.data + scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Add<f64> for &F64Tensor {
    type Output = F64Tensor;
    fn add(self, scalar: f64) -> F64Tensor {
        F64Tensor {
            data: &self.data + scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// 演算子実装 - Subtraction
impl Sub<F64Tensor> for F64Tensor {
    type Output = F64Tensor;
    fn sub(self, other: F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data - &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Sub<&F64Tensor> for F64Tensor {
    type Output = F64Tensor;
    fn sub(self, other: &F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data - &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Sub for &F64Tensor {
    type Output = F64Tensor;
    fn sub(self, other: &F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data - &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Sub<f64> for F64Tensor {
    type Output = F64Tensor;
    fn sub(self, scalar: f64) -> F64Tensor {
        F64Tensor {
            data: &self.data - scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Sub<f64> for &F64Tensor {
    type Output = F64Tensor;
    fn sub(self, scalar: f64) -> F64Tensor {
        F64Tensor {
            data: &self.data - scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// 演算子実装 - Multiplication
impl Mul<F64Tensor> for F64Tensor {
    type Output = F64Tensor;
    fn mul(self, other: F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data * &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Mul<&F64Tensor> for F64Tensor {
    type Output = F64Tensor;
    fn mul(self, other: &F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data * &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Mul for &F64Tensor {
    type Output = F64Tensor;
    fn mul(self, other: &F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data * &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Mul<f64> for F64Tensor {
    type Output = F64Tensor;
    fn mul(self, scalar: f64) -> F64Tensor {
        F64Tensor {
            data: &self.data * scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Mul<f64> for &F64Tensor {
    type Output = F64Tensor;
    fn mul(self, scalar: f64) -> F64Tensor {
        F64Tensor {
            data: &self.data * scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// 演算子実装 - Division
impl Div<F64Tensor> for F64Tensor {
    type Output = F64Tensor;
    fn div(self, other: F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data / &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Div<&F64Tensor> for F64Tensor {
    type Output = F64Tensor;
    fn div(self, other: &F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data / &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Div for &F64Tensor {
    type Output = F64Tensor;
    fn div(self, other: &F64Tensor) -> F64Tensor {
        F64Tensor {
            data: &self.data / &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Div<f64> for F64Tensor {
    type Output = F64Tensor;
    fn div(self, scalar: f64) -> F64Tensor {
        F64Tensor {
            data: &self.data / scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Div<f64> for &F64Tensor {
    type Output = F64Tensor;
    fn div(self, scalar: f64) -> F64Tensor {
        F64Tensor {
            data: &self.data / scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// 演算子実装 - Negation
impl Neg for F64Tensor {
    type Output = F64Tensor;
    fn neg(self) -> F64Tensor {
        F64Tensor {
            data: -&self.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Neg for &F64Tensor {
    type Output = F64Tensor;
    fn neg(self) -> F64Tensor {
        F64Tensor {
            data: -&self.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// F64Tensor主要実装
impl F64Tensor {
    /// 新しいテンソルを作成
    /// Create a new tensor
    pub fn new(data: Array<f64, IxDyn>) -> Self {
        let shape = data.shape().to_vec();
        F64Tensor {
            data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: false,
            shape,
        }
    }

    /// ゼロテンソルを作成
    /// Create a zero tensor
    pub fn zeros(shape: &[usize]) -> RusTorchResult<Self> {
        let data = Array::zeros(shape);
        Ok(F64Tensor::new(data))
    }

    /// ワンテンソルを作成
    /// Create a ones tensor
    pub fn ones(shape: &[usize]) -> RusTorchResult<Self> {
        let data = Array::ones(shape);
        Ok(F64Tensor::new(data))
    }

    /// ランダムテンソルを作成
    /// Create a random tensor
    pub fn randn(shape: &[usize]) -> RusTorchResult<Self> {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::StandardNormal;
        let data = Array::random(shape, StandardNormal);
        Ok(F64Tensor::new(data))
    }

    /// 形状を取得
    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 次元数を取得
    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// 要素数を取得
    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// データ型を取得
    /// Get data type
    pub fn dtype(&self) -> &'static str {
        "f64"
    }

    /// 勾配を必要とするかを設定
    /// Set requires gradient
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// リシェイプ
    /// Reshape
    pub fn reshape(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        let new_data = self.data.clone().into_shape_with_order(new_shape)?;
        let mut result = F64Tensor::new(new_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// 転置
    /// Transpose
    pub fn transpose(&self) -> RusTorchResult<Self> {
        let transposed = self.data.t().to_owned();
        let mut shape = self.shape.clone();
        shape.reverse();
        let mut result = F64Tensor::new(transposed);
        result.shape = shape;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// 行列積
    /// Matrix multiplication
    pub fn matmul(&self, other: &F64Tensor) -> RusTorchResult<Self> {
        use ndarray::linalg::general_mat_mul;

        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k != k2 {
            return Err(crate::error::RusTorchError::tensor_op(
                format!("Cannot multiply matrices with shapes {:?} and {:?}", self.shape, other.shape)
            ));
        }

        let mut result_data = Array::zeros((m, n));
        general_mat_mul(1.0, &self.data.view().into_dimensionality()?, &other.data.view().into_dimensionality()?, 0.0, &mut result_data.view_mut());

        let result_dyn = result_data.into_dyn();
        let mut result = F64Tensor::new(result_dyn);
        result.requires_grad = self.requires_grad || other.requires_grad;
        Ok(result)
    }

    /// 合計
    /// Sum
    pub fn sum(&self) -> f64 {
        self.data.sum()
    }

    /// 平均
    /// Mean
    pub fn mean(&self) -> f64 {
        self.data.mean().unwrap_or(0.0)
    }

    /// 標準偏差
    /// Standard deviation
    pub fn std(&self) -> f64 {
        let mean = self.mean();
        let variance = self.data.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        variance.sqrt()
    }

    /// 最大値
    /// Maximum value
    pub fn max(&self) -> f64 {
        self.data.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
    }

    /// 最小値
    /// Minimum value
    pub fn min(&self) -> f64 {
        self.data.fold(f64::INFINITY, |acc, &x| acc.min(x))
    }

    /// 次元を追加
    /// Add dimension
    pub fn unsqueeze(&self, dim: usize) -> RusTorchResult<Self> {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape)
    }

    /// 形状を拡張
    /// Expand shape
    pub fn expand(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        let expanded_data = self.data.broadcast(new_shape)
            .ok_or_else(|| crate::error::RusTorchError::tensor_op("Cannot broadcast to new shape"))?
            .to_owned();
        let mut result = F64Tensor::new(expanded_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// 次元を交換
    /// Transpose dimensions
    pub fn transpose_dims(&self, dim1: usize, dim2: usize) -> RusTorchResult<Self> {
        let mut permutation: Vec<usize> = (0..self.ndim()).collect();
        permutation.swap(dim1, dim2);
        let transposed = self.data.clone().permuted_axes(permutation);
        let mut result = F64Tensor::new(transposed);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// ソフトマックス
    /// Softmax
    pub fn softmax(&self, dim: Option<usize>) -> RusTorchResult<Self> {
        let axis = dim.unwrap_or(self.ndim() - 1);
        let max_vals = self.data.map_axis(ndarray::Axis(axis), |lane| {
            lane.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
        });

        let shifted = &self.data - &max_vals.insert_axis(ndarray::Axis(axis));
        let exp_vals = shifted.mapv(|x| x.exp());
        let sum_exp = exp_vals.sum_axis(ndarray::Axis(axis));
        let result_data = exp_vals / sum_exp.insert_axis(ndarray::Axis(axis));

        let mut result = F64Tensor::new(result_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }
}

// インデックス実装
impl Index<usize> for F64Tensor {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for F64Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Index<Index2D> for F64Tensor {
    type Output = f64;
    fn index(&self, index: Index2D) -> &Self::Output {
        &self.data[[index.0, index.1]]
    }
}

impl IndexMut<Index2D> for F64Tensor {
    fn index_mut(&mut self, index: Index2D) -> &mut Self::Output {
        &mut self.data[[index.0, index.1]]
    }
}

impl Index<Index3D> for F64Tensor {
    type Output = f64;
    fn index(&self, index: Index3D) -> &Self::Output {
        &self.data[[index.0, index.1, index.2]]
    }
}

impl IndexMut<Index3D> for F64Tensor {
    fn index_mut(&mut self, index: Index3D) -> &mut Self::Output {
        &mut self.data[[index.0, index.1, index.2]]
    }
}