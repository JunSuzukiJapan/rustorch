//! # Tensor Operations and Data Structures
//! テンソル操作とデータ構造
//! 
//! This module provides the core tensor functionality for RusTorch, including
//! basic tensor operations, advanced parallel processing, GPU acceleration,
//! and memory optimization features.
//! 
//! ## Core Components
//! 
//! - [`Tensor`]: The main tensor data structure with n-dimensional array support
//! - [`parallel_traits`]: Unified parallel tensor operations system
//! - [`gpu_parallel`]: GPU-accelerated tensor operations with device management
//! - [`memory_optimized`]: Advanced memory management strategies
//! - [`zero_copy`]: Zero-copy tensor views and shared ownership
//! - [`simd_aligned`]: SIMD-aligned tensor operations for vectorization
//! 
//! ## Key Features
//! 
//! ### High-Performance Computing
//! - **Parallel Processing**: Automatic parallelization for large tensor operations
//! - **SIMD Acceleration**: AVX2/SSE4.1 vectorized operations for f32 tensors
//! - **GPU Integration**: CUDA/Metal/OpenCL support with intelligent fallback
//! - **Memory Optimization**: Pool allocation, zero-copy views, and cache-friendly operations
//! 
//! ### Mathematical Operations
//! - **Element-wise Operations**: Addition, multiplication, trigonometric functions
//! - **Linear Algebra**: Matrix multiplication, decompositions, eigenvalues
//! - **Broadcasting**: NumPy-style broadcasting for operations on different shapes
//! - **Reduction Operations**: Sum, mean, variance, and statistical functions
//! 
//! ### Memory Management
//! - **Zero-Copy Views**: Efficient tensor slicing without data duplication
//! - **Memory Pooling**: Reduced allocation overhead for frequent operations
//! - **SIMD Alignment**: 32-byte aligned allocation for optimal vectorization
//! - **Shared Ownership**: Thread-safe reference counting for tensor sharing
//! 
//! ## Usage Examples
//! 
//! ### Basic Tensor Operations
//! 
//! ```rust
//! use rustorch::tensor::Tensor;
//! 
//! // Create tensors
//! let a = Tensor::<f32>::ones(&[3, 3]);
//! let b = Tensor::<f32>::zeros(&[3, 3]);
//! 
//! // Basic arithmetic
//! let c = &a + &b;
//! let d = a.matmul(&b);
//! 
//! // Mathematical functions
//! let e = a.sin();
//! let f = a.exp();
//! ```
//! 
//! ### Parallel Operations
//! 
//! RusTorch provides efficient parallel tensor operations for high-performance computing.
//! 
//! ### GPU Acceleration
//! 
//! RusTorch supports GPU acceleration with automatic fallback to CPU when GPU is unavailable.
//! 
//! ### Memory Optimization
//! 
//! Advanced memory management strategies for optimal performance and memory usage.

use ndarray::{ArrayD, ArrayViewD, IxDyn, Ix1, Ix2};
// use rayon::prelude::*;
use num_traits::Float;
#[cfg(not(target_arch = "wasm32"))]
use crate::memory::{get_f32_pool, get_f64_pool};
use std::{fmt, ops};

#[cfg(not(target_arch = "wasm32"))]
mod pool_integration;
pub mod parallel_errors;
pub mod parallel_traits;
pub mod parallel_impl;
/// Parallel tensor operations module
/// 並列テンソル演算モジュール
pub mod parallel_ops;

/// Parallel tensor operations for batch processing and SIMD acceleration
/// バッチ処理とSIMD加速のための並列テンソル操作
#[cfg(not(target_arch = "wasm32"))]
pub mod simd_integration;
#[cfg(not(target_arch = "wasm32"))]
pub mod simd_aligned;
#[cfg(not(target_arch = "wasm32"))]
pub mod simd_avx512;
#[cfg(not(target_arch = "wasm32"))]
pub mod advanced_memory;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_parallel;
// Enable modules step by step
mod broadcasting;

// Re-export important types and functions
pub use broadcasting::BroadcastError;
pub use parallel_errors::{ParallelError, ParallelResult};

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
    
    /// Get pointer address for unique identification
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    /// Copy data from another tensor (unsafe internal implementation)
    pub fn copy_from(&self, other: &Tensor<T>) {
        unsafe {
            let self_ptr = self.data.as_ptr() as *mut T;
            let other_ptr = other.data.as_ptr();
            let len = self.data.len().min(other.data.len());
            std::ptr::copy_nonoverlapping(other_ptr, self_ptr, len);
        }
    }
    
    /// Convert tensor to different device (mock implementation)
    pub fn to_device(&self, _device: std::sync::Arc<dyn crate::gpu::device::GpuDevice>) -> Self {
        self.clone()
    }
    
    /// Convert tensor to CPU
    pub fn to_cpu(&self) -> Self {
        self.clone()
    }
    
    /// Convert to half precision (mock)
    pub fn to_half(&self) -> Self {
        self.clone()
    }
    
    /// Convert to float precision (mock)
    pub fn to_float(&self) -> Self {
        self.clone()
    }
    
    /// Get tensor data type
    pub fn dtype(&self) -> crate::dtype::DType {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            crate::dtype::DType::Float32
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            crate::dtype::DType::Float64
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
            crate::dtype::DType::Int32
        } else {
            crate::dtype::DType::Float32 // fallback
        }
    }
    
    /// Get element count
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    
    /// Get single item (for scalar tensors)
    pub fn item(&self) -> T {
        if self.data.len() == 1 {
            self.data[[0]]
        } else {
            panic!("item() called on non-scalar tensor")
        }
    }
    
    /// Element-wise maximum with another tensor
    pub fn max_elementwise(&self, _other: &Tensor<T>) -> Tensor<T> {
        // Mock implementation
        self.clone()
    }
    
    /// Element-wise square root
    pub fn sqrt(&self) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x.sqrt());
        result
    }
    
    /// Element-wise exponential
    pub fn exp(&self) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x.exp());
        result
    }
    
    /// Element-wise logarithm
    pub fn log(&self) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x.ln());
        result
    }
    
    /// Element-wise sine
    pub fn sin(&self) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x.sin());
        result
    }
    
    /// Element-wise cosine
    pub fn cos(&self) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x.cos());
        result
    }
    
    /// Element-wise tangent
    pub fn tan(&self) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x.tan());
        result
    }
    
    /// Create a view of the tensor with new shape
    pub fn view(&self, shape: &[usize]) -> Self {
        Tensor {
            data: self.data.view().into_shape(shape).unwrap().to_owned()
        }
    }
    
    /// Concatenate tensors along specified dimension
    pub fn cat(tensors: &[&Tensor<T>], _dim: usize) -> Tensor<T> {
        if tensors.is_empty() {
            panic!("Cannot concatenate empty tensor list");
        }
        
        // Simple implementation - just return first tensor for now
        tensors[0].clone()
    }
    
    /// Split tensor into chunks
    pub fn chunk(&self, chunks: usize, _dim: usize) -> Vec<Tensor<T>> {
        // Simple implementation - just return copies for now
        vec![self.clone(); chunks]
    }
    
    /// Absolute value
    pub fn abs(&self) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x.abs());
        result
    }
    
    /// Element-wise map operation
    pub fn map<F>(&self, f: F) -> Tensor<T>
    where
        F: Fn(T) -> T,
    {
        let mut result = self.clone();
        result.data.mapv_inplace(f);
        result
    }

    /// Returns the size (shape) of the tensor.
    /// テンソルのサイズ（形状）を返します。
    pub fn size(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Returns the shape of the tensor.
    /// テンソルの形状を返します。
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns a reference to the underlying data as a slice.
    /// 基底データへのスライス参照を返します。
    pub fn as_slice(&self) -> Option<&[T]> {
        self.data.as_slice()
    }

    /// Get a single element at the given index.
    /// 指定されたインデックスの単一要素を取得します。
    pub fn get(&self, index: &[usize]) -> T {
        self.data[ndarray::IxDyn(index)]
    }
    
    /// Index with optional ranges (mock implementation)
    pub fn index(&self, _indices: &[Option<std::ops::Range<usize>>]) -> Tensor<T> {
        self.clone()
    }

    /// Creates a tensor filled with zeros using memory pool.
    /// メモリプールを使用してゼロで埋められたテンソルを作成します。
    pub fn zeros(shape: &[usize]) -> Self {
#[cfg(not(target_arch = "wasm32"))]
        {
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
        }
        
        let total_size = shape.iter().product();
        let data = vec![T::zero(); total_size];
        Tensor::from_vec(data, shape.to_vec())
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
    pub fn array_view(&self) -> ArrayViewD<T> {
        self.data.view()
    }

    /// Creates a tensor from a scalar value.
    /// スカラー値からテンソルを作成します。
    pub fn from_scalar(value: T) -> Self {
        let data = ArrayD::from_elem(ndarray::IxDyn(&[]), value);
        Tensor::new(data)
    }

    /// Returns a mutable reference to the underlying array.
    /// 内部の配列への可変参照を返します。
    pub fn as_array_mut(&mut self) -> &mut ArrayD<T> {
        &mut self.data
    }

    /// Returns the data as a mutable slice if possible
    /// 可能であればデータを可変スライスとして返します
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.data.as_slice_mut()
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
            (3, 2) => {
                // Batch matrix-vector multiplication: (B, M, N) x (N, K) -> (B, M, K)
                let lhs_shape = lhs.shape();
                let rhs_shape = rhs.shape();
                let batch_size = lhs_shape[0];
                let m = lhs_shape[1];
                let n = lhs_shape[2];
                let k = rhs_shape[1];
                
                if n != rhs_shape[0] {
                    panic!("Incompatible dimensions for matmul: {} != {}", n, rhs_shape[0]);
                }
                
                let mut result_data = Vec::with_capacity(batch_size * m * k);
                let rhs_2d = rhs.view().into_dimensionality::<Ix2>().unwrap();
                
                for b in 0..batch_size {
                    for i in 0..m {
                        for j in 0..k {
                            let mut sum = T::zero();
                            for l in 0..n {
                                sum = sum + lhs[[b, i, l]] * rhs_2d[[l, j]];
                            }
                            result_data.push(sum);
                        }
                    }
                }
                
                Tensor::from_vec(result_data, vec![batch_size, m, k])
            },
            (3, 3) => {
                // Batch matrix-matrix multiplication: (B, M, N) x (B, N, K) -> (B, M, K)
                let lhs_shape = lhs.shape();
                let rhs_shape = rhs.shape();
                let batch_size = lhs_shape[0];
                let m = lhs_shape[1];
                let n = lhs_shape[2];
                let k = rhs_shape[2];
                
                if batch_size != rhs_shape[0] || n != rhs_shape[1] {
                    panic!("Incompatible dimensions for batch matmul: {:?} and {:?}", lhs_shape, rhs_shape);
                }
                
                let mut result_data = Vec::with_capacity(batch_size * m * k);
                
                for b in 0..batch_size {
                    for i in 0..m {
                        for j in 0..k {
                            let mut sum = T::zero();
                            for l in 0..n {
                                sum = sum + lhs[[b, i, l]] * rhs[[b, l, j]];
                            }
                            result_data.push(sum);
                        }
                    }
                }
                
                Tensor::from_vec(result_data, vec![batch_size, m, k])
            },
            (4, 4) => {
                // 4D tensor multiplication for Multi-Head Attention
                // (B, H, S, D) x (B, H, D, S') -> (B, H, S, S')
                let lhs_shape = lhs.shape();
                let rhs_shape = rhs.shape();
                let batch_size = lhs_shape[0];
                let num_heads = lhs_shape[1];
                let seq_len = lhs_shape[2];
                let d_k = lhs_shape[3];
                let seq_len_rhs = rhs_shape[3];
                
                if batch_size != rhs_shape[0] || num_heads != rhs_shape[1] || d_k != rhs_shape[2] {
                    panic!("Incompatible dimensions for 4D matmul: {:?} and {:?}", lhs_shape, rhs_shape);
                }
                
                let mut result_data = Vec::with_capacity(batch_size * num_heads * seq_len * seq_len_rhs);
                
                for b in 0..batch_size {
                    for h in 0..num_heads {
                        for i in 0..seq_len {
                            for j in 0..seq_len_rhs {
                                let mut sum = T::zero();
                                for k in 0..d_k {
                                    sum = sum + lhs[[b, h, i, k]] * rhs[[b, h, k, j]];
                                }
                                result_data.push(sum);
                            }
                        }
                    }
                }
                
                Tensor::from_vec(result_data, vec![batch_size, num_heads, seq_len, seq_len_rhs])
            },
            _ => panic!("Unsupported dimensions for matmul: {:?} and {:?}", lhs.shape(), rhs.shape()),
        }
    }

    /// Transposes the last two dimensions of a tensor
    /// テンソルの最後の2次元を転置します
    pub fn transpose_last_two(&self) -> Tensor<T> {
        let shape = self.shape();
        let ndim = shape.len();
        
        if ndim < 2 {
            panic!("Cannot transpose last two dimensions: tensor has fewer than 2 dimensions");
        }
        
        match ndim {
            2 => {
                // Simple matrix transpose
                let transposed = self.data.t().to_owned();
                Tensor { data: transposed }
            },
            3 => {
                // Batch matrix transpose: (B, M, N) -> (B, N, M)
                let batch_size = shape[0];
                let m = shape[1];
                let n = shape[2];
                
                let mut result_data = Vec::with_capacity(batch_size * n * m);
                
                for b in 0..batch_size {
                    for j in 0..n {
                        for i in 0..m {
                            result_data.push(self.data[[b, i, j]]);
                        }
                    }
                }
                
                Tensor::from_vec(result_data, vec![batch_size, n, m])
            },
            4 => {
                // 4D tensor transpose: (B, H, M, N) -> (B, H, N, M)
                let batch_size = shape[0];
                let num_heads = shape[1];
                let m = shape[2];
                let n = shape[3];
                
                let mut result_data = Vec::with_capacity(batch_size * num_heads * n * m);
                
                for b in 0..batch_size {
                    for h in 0..num_heads {
                        for j in 0..n {
                            for i in 0..m {
                                result_data.push(self.data[[b, h, i, j]]);
                            }
                        }
                    }
                }
                
                Tensor::from_vec(result_data, vec![batch_size, num_heads, n, m])
            },
            _ => panic!("Transpose not implemented for tensors with {} dimensions", ndim)
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
    pub fn stack(tensors: &[&Tensor<T>]) -> ParallelResult<Tensor<T>> {
        if tensors.is_empty() {
            return Err(ParallelError::empty_tensor_list("stack"));
        }
        
        // Check that all tensors have the same shape
        let first_shape = tensors[0].shape();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != first_shape {
                return Err(ParallelError::shape_mismatch(
                    first_shape,
                    tensor.shape(),
                    &format!("stack operation (tensor {})", i)
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
    pub fn batch_get(&self, index: usize) -> ParallelResult<Tensor<T>> {
        if self.data.ndim() == 0 {
            return Err(ParallelError::ScalarIndexing);
        }
        
        let batch_size = self.data.shape()[0];
        if index >= batch_size {
            return Err(ParallelError::dimension_error(
                index,
                batch_size - 1,
                "batch indexing"
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
    
    /// Compute mean along all dimensions
    pub fn mean(&self) -> Tensor<T> {
        let sum = self.data.iter().fold(T::zero(), |acc, &x| acc + x);
        let count = T::from(self.data.len()).unwrap();
        Tensor::new(ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), sum / count))
    }
    
    /// Compute mean along specified dimensions
    pub fn mean_dim(&self, _dims: &[usize], _keepdim: bool) -> Tensor<T> {
        // Mock implementation
        self.mean()
    }
    
    /// Compute variance along all dimensions
    pub fn var(&self) -> Tensor<T> {
        let mean_val = self.mean().item();
        let variance = self.data.iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x) / T::from(self.data.len()).unwrap();
        Tensor::new(ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), variance))
    }
    
    /// Compute variance along specified dimensions
    pub fn var_dim(&self, _dims: &[usize], _keepdim: bool) -> Tensor<T> {
        // Mock implementation
        self.var()
    }
    
    /// Find maximum value
    pub fn max(&self) -> Tensor<T> {
        let max_val = self.data.iter().fold(T::neg_infinity(), |acc, &x| {
            if x > acc { x } else { acc }
        });
        Tensor::new(ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), max_val))
    }
    
    /// Element-wise maximum between two tensors
    pub fn maximum(&self, other: &Tensor<T>) -> Tensor<T> {
        let mut result = self.clone();
        ndarray::Zip::from(&mut result.data)
            .and(&other.data)
            .for_each(|a, &b| {
                if b > *a {
                    *a = b;
                }
            });
        result
    }
    
    /// Find minimum value  
    pub fn min(&self) -> Tensor<T> {
        let min_val = self.data.iter().fold(T::infinity(), |acc, &x| {
            if x < acc { x } else { acc }
        });
        Tensor::new(ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), min_val))
    }
    
    
    /// Sum along specified dimension
    pub fn sum_dim(&self, _dims: &[usize], _keepdim: bool) -> Tensor<T> {
        // Mock implementation
        self.sum()
    }
    
    /// Create random tensor with normal distribution
    pub fn randn(shape: &[usize]) -> Tensor<T> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let total_size = shape.iter().product();
        let data: Vec<T> = (0..total_size)
            .map(|_| T::from(rng.gen::<f32>() - 0.5).unwrap())
            .collect();
        Tensor::from_vec(data, shape.to_vec())
    }
    
    /// Create range tensor
    pub fn arange(start: T, end: T, step: T) -> Tensor<T> {
        let mut values = Vec::new();
        let mut current = start;
        while current < end {
            values.push(current);
            current = current + step;
        }
        let length = values.len();
        Tensor::from_vec(values, vec![length])
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

    fn add(self, rhs: Self) -> Tensor<T> {
        // Use memory pool for result allocation
        let mut result = Tensor::zeros(self.data.shape());
        result.data = &self.data + &rhs.data;
        result
    }
}

impl<T: Float + 'static> ops::Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Tensor<T> {
        // Use memory pool for result allocation
        let mut result = Tensor::zeros(self.data.shape());
        result.data = &self.data - &rhs.data;
        result
    }
}

impl<T: Float + 'static> ops::Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Tensor<T> {
        // Use memory pool for result allocation
        let mut result = Tensor::zeros(self.data.shape());
        result.data = &self.data * &rhs.data;
        result
    }
}

impl<T: Float + 'static> ops::Div for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Tensor<T> {
        // Use memory pool for result allocation
        let mut result = Tensor::zeros(self.data.shape());
        result.data = &self.data / &rhs.data;
        result
    }
}

impl<T: Float + 'static> ops::Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Tensor<T> {
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

// Add owned tensor operations
impl<T: Float + 'static> ops::Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Tensor<T> {
        &self + &rhs
    }
}

impl<T: Float + 'static> ops::Mul<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, scalar: T) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x * scalar);
        result
    }
}

impl<T: Float + 'static> ops::Mul<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, scalar: T) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x * scalar);
        result
    }
}

impl<T: Float + 'static> ops::Div<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, scalar: T) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x / scalar);
        result
    }
}

impl<T: Float + 'static> ops::Div<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, scalar: T) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x / scalar);
        result
    }
}


impl<T: Float + 'static> ops::Add<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, scalar: T) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x + scalar);
        result
    }
}

impl<T: Float + 'static> ops::Add<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, scalar: T) -> Tensor<T> {
        let mut result = self.clone();
        result.data.mapv_inplace(|x| x + scalar);
        result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_4d_matmul() {
        // Test 4D tensor multiplication for Multi-Head Attention
        // Q: [1, 2, 3, 4] (batch=1, heads=2, seq_len=3, d_k=4)
        let q_data = (0..24).map(|i| i as f32).collect::<Vec<f32>>();
        let q = Tensor::from_vec(q_data, vec![1, 2, 3, 4]);
        
        // K^T: [1, 2, 4, 3] (batch=1, heads=2, d_k=4, seq_len=3)  
        let k_data = (0..24).map(|i| (i * 2) as f32).collect::<Vec<f32>>();
        let k_t = Tensor::from_vec(k_data, vec![1, 2, 4, 3]);
        
        // QK^T should be [1, 2, 3, 3]
        let attention_scores = q.matmul(&k_t);
        assert_eq!(attention_scores.shape(), &[1, 2, 3, 3]);
        
        // Test with values: [1,2,3,4] × [1,3; 5,7; 9,11; 13,15] = [1×1+2×5+3×9+4×13, 1×3+2×7+3×11+4×15]
        let simple_q = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 4]);
        let simple_k = Tensor::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0], vec![1, 1, 4, 2]);
        
        let result = simple_q.matmul(&simple_k);
        assert_eq!(result.shape(), &[1, 1, 1, 2]);
        
        let expected = 1.0*1.0 + 2.0*5.0 + 3.0*9.0 + 4.0*13.0; // = 1+10+27+52 = 90
        let result_data = result.as_slice().unwrap();
        assert!((result_data[0] - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_attention_pattern_4d_matmul() {
        // More realistic attention pattern test
        let batch_size = 1;
        let num_heads = 2; 
        let seq_len = 4;
        let d_k = 8;
        
        // Create Q and K tensors
        let q_data = (0..batch_size * num_heads * seq_len * d_k)
            .map(|i| (i as f32) * 0.1)
            .collect::<Vec<f32>>();
        let q = Tensor::from_vec(q_data, vec![batch_size, num_heads, seq_len, d_k]);
        
        let k_data = (0..batch_size * num_heads * d_k * seq_len)
            .map(|i| (i as f32) * 0.05)
            .collect::<Vec<f32>>();
        let k_t = Tensor::from_vec(k_data, vec![batch_size, num_heads, d_k, seq_len]);
        
        // Compute attention scores
        let attention_scores = q.matmul(&k_t);
        
        // Should have shape [batch_size, num_heads, seq_len, seq_len]
        assert_eq!(attention_scores.shape(), &[batch_size, num_heads, seq_len, seq_len]);
        
        // Verify some basic properties
        let scores_data = attention_scores.as_slice().unwrap();
        assert_eq!(scores_data.len(), batch_size * num_heads * seq_len * seq_len);
    }
    
    #[test]
    fn test_4d_transpose() {
        // Test 4D tensor transpose: (B, H, M, N) -> (B, H, N, M)
        let data = (0..24).map(|i| i as f32).collect::<Vec<f32>>();
        let tensor = Tensor::from_vec(data, vec![1, 2, 3, 4]); // [batch=1, heads=2, m=3, n=4]
        
        let transposed = tensor.transpose_last_two();
        assert_eq!(transposed.shape(), &[1, 2, 4, 3]); // [batch=1, heads=2, n=4, m=3]
        
        // Verify specific element: tensor[0,0,1,2] should be transposed[0,0,2,1]
        let original_val = tensor.as_slice().unwrap()[1 * 4 + 2]; // position (0,0,1,2) = 6
        let transposed_val = transposed.as_slice().unwrap()[2 * 3 + 1]; // position (0,0,2,1) = 7th element in transposed
        assert_eq!(original_val, transposed_val);
    }
    
    #[test] 
    fn test_3d_transpose() {
        // Test 3D batch matrix transpose
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 batches of 1x3 matrices
        let tensor = Tensor::from_vec(data, vec![2, 1, 3]); 
        
        let transposed = tensor.transpose_last_two();
        assert_eq!(transposed.shape(), &[2, 3, 1]);
        
        let trans_data = transposed.as_slice().unwrap();
        assert_eq!(trans_data[0], 1.0); // First batch, first element
        assert_eq!(trans_data[3], 4.0); // Second batch, first element
    }
}
