use ndarray::{ArrayD, Ix1, Ix2, IxDyn, ArrayViewD, Axis};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use std::ops;
use std::fmt;
use crate::memory::{get_f32_pool, get_f64_pool};

mod pool_integration;
mod simd_integration;
mod parallel_ops;

/// A multi-dimensional array that supports automatic differentiation.
/// 自動微分をサポートする多次元配列
#[derive(Debug, Clone)]
pub struct Tensor<T: Float> {
    data: ArrayD<T>,
}

impl<T: Float + 'static> Tensor<T> {
    /// Creates a new tensor from an array.
    /// 配列から新しいテンソルを作成します。
    pub fn new(data: ArrayD<T>) -> Self {
        Tensor { data }
    }

    /// Creates a tensor from a vector and shape.
    /// ベクトルと形状からテンソルを作成します。
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let array = ArrayD::from_shape_vec(shape, data).expect("Invalid shape for data");
        Tensor::new(array)
    }

    /// Returns the size (shape) of the tensor.
    /// テンソルのサイズ（形状）を返します。
    pub fn size(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Creates a tensor filled with zeros using memory pool.
    /// メモリプールを使用してゼロで埋められたテンソルを作成します。
    pub fn zeros(shape: &[usize]) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            if let Ok(mut pool) = get_f32_pool().lock() {
                let data = unsafe { std::mem::transmute(pool.allocate(shape)) };
                return Tensor { data };
            }
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            if let Ok(mut pool) = get_f64_pool().lock() {
                let data = unsafe { std::mem::transmute(pool.allocate(shape)) };
                return Tensor { data };
            }
        }
        
        // Fallback to regular allocation
        Tensor {
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    /// Creates a tensor filled with ones using memory pool.
    /// メモリプールを使用して1で埋められたテンソルを作成します。
    pub fn ones(shape: &[usize]) -> Self {
        let mut tensor = Self::zeros(shape); // Use pool allocation
        tensor.data.fill(T::one());
        tensor
    }

    /// Returns a reference to the underlying array.
    /// 内部の配列への参照を返します。
    pub fn as_array(&self) -> &ArrayD<T> {
        &self.data
    }

    /// Returns a view of the underlying array.
    /// 内部の配列のビューを返します。
    pub fn view(&self) -> ArrayViewD<T> {
        self.data.view()
    }

    /// Returns a mutable reference to the underlying array.
    /// 内部の配列への可変参照を返します。
    pub fn as_array_mut(&mut self) -> &mut ArrayD<T> {
        &mut self.data
    }

    /// Returns the data as a slice if possible
    /// 可能であればデータをスライスとして返します
    pub fn as_slice(&self) -> Option<&[T]> {
        self.data.as_slice()
    }
    
    /// Returns the data as a mutable slice if possible
    /// 可能であればデータを可変スライスとして返します
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.data.as_slice_mut()
    }

    /// Returns the shape of the tensor.
    /// テンソルの形状を返します。
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns the number of elements in the tensor.
    /// テンソルの要素数を返します。
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the tensor contains no elements.
    /// テンソルが空の場合は`true`を返します。
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reshapes the tensor to the given shape.
    /// テンソルを指定された形状に変形します。
    pub fn reshape(self, shape: &[usize]) -> Self {
        Tensor {
            data: self.data.into_shape(shape).unwrap(),
        }
    }

    /// Transposes the tensor by reversing its dimensions.
    /// テンソルの次元を反転させて転置します。
    pub fn transpose(&self) -> Self {
        let ndim = self.data.ndim();
        let axes: Vec<usize> = (0..ndim).rev().collect();
        if ndim < 2 {
            return self.clone();
        }
        Tensor {
            data: self.data.view().permuted_axes(axes.as_slice()).to_owned(),
        }
    }

    /// Performs matrix multiplication with another tensor.
    /// 別のテンソルとの行列乗算を実行します。
    pub fn matmul(&self, rhs: &Tensor<T>) -> Tensor<T> {
        let lhs = &self.data;
        let rhs = &rhs.data;
        
        match (lhs.ndim(), rhs.ndim()) {
            (1, 1) => {
                // Dot product
                let sum = lhs.iter().zip(rhs.iter()).fold(T::zero(), |acc, (&a, &b)| acc + a * b);
                Tensor::new(ArrayD::from_elem(IxDyn(&[]), sum))
            },
            (2, 1) => {
                // Matrix-vector multiplication
                let rhs = rhs.view().into_dimensionality::<Ix1>().unwrap();
                let lhs = lhs.view().into_dimensionality::<Ix2>().unwrap();
                let result = lhs.dot(&rhs);
                Tensor::new(result.into_dyn())
            },
            (2, 2) => {
                // Matrix-matrix multiplication - optimized with BLAS
                let lhs = lhs.view().into_dimensionality::<Ix2>().unwrap();
                let rhs = rhs.view().into_dimensionality::<Ix2>().unwrap();
                let result = lhs.dot(&rhs);
                Tensor::new(result.into_dyn())
            },
            _ => panic!("Unsupported dimensions for matmul: {:?} and {:?}", lhs.shape(), rhs.shape()),
        }
    }

    /// Computes the sum of the tensor along the specified axis.
    /// 指定された軸に沿ってテンソルの和を計算します。
    pub fn sum_axis(&self, axis: usize) -> Tensor<T> {
        let sum = self.data.sum_axis(ndarray::Axis(axis));
        let dim = sum.raw_dim();
        Tensor {
            data: sum.into_shape(dim).unwrap(),
        }
    }

    /// In-place addition with another tensor.
    /// 別のテンソルとのin-place加算を実行します。
    pub fn add_inplace(&mut self, rhs: &Tensor<T>) {
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a + *b;
        }
    }

    /// In-place multiplication with another tensor.
    /// 別のテンソルとのin-place乗算を実行します。
    pub fn mul_inplace(&mut self, rhs: &Tensor<T>) {
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a * *b;
        }
    }

    /// In-place subtraction with another tensor.
    /// 別のテンソルとのin-place減算を実行します。
    pub fn sub_inplace(&mut self, rhs: &Tensor<T>) {
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a - *b;
        }
    }

    /// In-place scalar multiplication.
    /// スカラーとのin-place乗算を実行します。
    pub fn mul_scalar_inplace(&mut self, scalar: T) {
        self.data.mapv_inplace(|x| x * scalar);
    }

    /// Parallel matrix multiplication for large tensors.
    /// 大規模テンソル用の並列行列乗算を実行します。
    pub fn matmul_parallel(&self, rhs: &Tensor<T>) -> Tensor<T> {
        // For now, use regular matmul with ndarray's built-in parallelization
        self.matmul(rhs)
    }

    /// Element-wise operations.
    /// 要素ごと演算を実行します。
    pub fn apply<F>(&self, f: F) -> Tensor<T>
    where
        F: Fn(T) -> T,
    {
        let new_data: Vec<T> = self.data.iter().map(|&x| f(x)).collect();
        Tensor::from_vec(new_data, self.size())
    }

    /// Computes the sum of all elements in the tensor.
    /// テンソルの全要素の和を計算します。
    pub fn sum(&self) -> Tensor<T> {
        let sum_value = self.data.iter().fold(T::zero(), |acc, &x| acc + x);
        Tensor::from_vec(vec![sum_value], vec![])
    }
    
    /// Creates a batch tensor by stacking tensors along the first dimension.
    /// テンソルを第一次元に沿って積み重ねてバッチテンソルを作成します。
    pub fn stack(tensors: &[&Tensor<T>]) -> Result<Tensor<T>, String> {
        if tensors.is_empty() {
            return Err("Cannot stack empty list of tensors".to_string());
        }
        
        // Check that all tensors have the same shape
        let first_shape = tensors[0].shape();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != first_shape {
                return Err(format!(
                    "All tensors must have the same shape. Tensor 0 has shape {:?}, tensor {} has shape {:?}",
                    first_shape, i, tensor.shape()
                ));
            }
        }
        
        // Create new shape with batch dimension
        let mut new_shape = vec![tensors.len()];
        new_shape.extend_from_slice(first_shape);
        
        // Collect all data
        let mut all_data = Vec::new();
        for tensor in tensors {
            all_data.extend(tensor.data.iter().cloned());
        }
        
        Ok(Tensor::from_vec(all_data, new_shape))
    }
    
    /// Gets a slice of the tensor along the first dimension (batch dimension).
    /// 第一次元（バッチ次元）に沿ってテンソルのスライスを取得します。
    pub fn batch_get(&self, index: usize) -> Result<Tensor<T>, String> {
        if self.data.ndim() == 0 {
            return Err("Cannot index into scalar tensor".to_string());
        }
        
        let batch_size = self.data.shape()[0];
        if index >= batch_size {
            return Err(format!(
                "Index {} out of bounds for batch size {}",
                index, batch_size
            ));
        }
        
        let sliced = self.data.index_axis(ndarray::Axis(0), index);
        Ok(Tensor::new(sliced.to_owned()))
    }
    
    /// Returns the batch size (size of the first dimension).
    /// バッチサイズ（第一次元のサイズ）を返します。
    pub fn batch_size(&self) -> usize {
        if self.data.ndim() == 0 {
            1 // Scalar tensor has batch size 1
        } else {
            self.data.shape()[0]
        }
    }
    
    /// Computes mean along the specified axis.
    /// 指定された軸に沿った平均を計算します。
    pub fn mean_axis(&self, axis: usize) -> Tensor<T> {
        let sum = self.data.sum_axis(ndarray::Axis(axis));
        let axis_size = T::from(self.data.shape()[axis]).unwrap();
        let mean_data = sum.mapv(|x| x / axis_size);
        Tensor::new(mean_data.into_dyn())
    }
    
    /// Broadcasts this tensor to the given shape.
    /// このテンソルを指定された形状にブロードキャストします。
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor<T>, String> {
        match self.data.broadcast(shape) {
            Some(broadcasted) => Ok(Tensor::new(broadcasted.to_owned())),
            None => Err(format!(
                "Cannot broadcast tensor with shape {:?} to shape {:?}",
                self.shape(), shape
            )),
        }
    }
}

impl<T: Float + fmt::Display> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<T: Float + 'static> From<ArrayD<T>> for Tensor<T> {
    fn from(data: ArrayD<T>) -> Self {
        Tensor::new(data)
    }
}

impl<T: Float + 'static> From<ndarray::Array1<T>> for Tensor<T> {
    fn from(array: ndarray::Array1<T>) -> Self {
        Tensor::new(array.into_dyn())
    }
}

impl<T: Float + 'static> From<ndarray::Array2<T>> for Tensor<T> {
    fn from(array: ndarray::Array2<T>) -> Self {
        Tensor::new(array.into_dyn())
    }
}

impl<T: Float + 'static> ops::Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        // Use memory pool for result allocation
        let mut result = Tensor::zeros(self.data.shape());
        result.data = &self.data + &rhs.data;
        result
    }
}

impl<T: Float + 'static> ops::Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        // Use memory pool for result allocation
        let mut result = Tensor::zeros(self.data.shape());
        result.data = &self.data - &rhs.data;
        result
    }
}

impl<T: Float + 'static> ops::Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        // Use memory pool for result allocation
        let mut result = Tensor::zeros(self.data.shape());
        result.data = &self.data * &rhs.data;
        result
    }
}

impl<T: Float + 'static> ops::Div for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        // Use memory pool for result allocation
        let mut result = Tensor::zeros(self.data.shape());
        result.data = &self.data / &rhs.data;
        result
    }
}

impl<T: Float + 'static> ops::Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        // Use memory pool for result allocation
        let mut result = Tensor::zeros(self.data.shape());
        result.data.assign(&self.data.mapv(|x| -x));
        result
    }
}

impl<T: Float> ops::AddAssign<&Tensor<T>> for Tensor<T> {
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        let rhs_data = &rhs.data;
        self.data.zip_mut_with(rhs_data, |a, &b| *a = *a + b);
    }
}

impl<T: Float> ops::SubAssign<&Tensor<T>> for Tensor<T> {
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        let rhs_data = &rhs.data;
        self.data.zip_mut_with(rhs_data, |a, &b| *a = *a - b);
    }
}

impl<T: Float> ops::MulAssign<&Tensor<T>> for Tensor<T> {
    fn mul_assign(&mut self, rhs: &Tensor<T>) {
        let rhs_data = &rhs.data;
        self.data.zip_mut_with(rhs_data, |a, &b| *a = *a * b);
    }
}

impl<T: Float> ops::DivAssign<&Tensor<T>> for Tensor<T> {
    fn div_assign(&mut self, rhs: &Tensor<T>) {
        let rhs_data = &rhs.data;
        self.data.zip_mut_with(rhs_data, |a, &b| *a = *a / b);
    }
}
