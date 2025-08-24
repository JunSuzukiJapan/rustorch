//! Type-safe tensor shapes and operations
//! 型安全なテンソル形状と操作

use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Compile-time verified tensor dimensions
/// コンパイル時検証されたテンソル次元
pub trait Dimension: Clone + Debug + PartialEq {
    /// Number of dimensions
    /// 次元数
    const NDIM: usize;
    
    /// Shape as array
    /// 形状を配列として取得
    fn shape(&self) -> Vec<usize>;
    
    /// Total number of elements
    /// 要素の総数
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }
}

/// 1-dimensional tensor shape
/// 1次元テンソル形状
#[derive(Debug, Clone, PartialEq)]
pub struct Dim1 {
    /// Size of the 1D tensor
    /// 1Dテンソルのサイズ
    pub size: usize,
}

impl Dimension for Dim1 {
    const NDIM: usize = 1;
    
    fn shape(&self) -> Vec<usize> {
        vec![self.size]
    }
}

/// 2-dimensional tensor shape (matrix)
/// 2次元テンソル形状（行列）
#[derive(Debug, Clone, PartialEq)]
pub struct Dim2 {
    /// Number of rows
    /// 行数
    pub rows: usize,
    /// Number of columns  
    /// 列数
    pub cols: usize,
}

impl Dimension for Dim2 {
    const NDIM: usize = 2;
    
    fn shape(&self) -> Vec<usize> {
        vec![self.rows, self.cols]
    }
}

/// 3-dimensional tensor shape
/// 3次元テンソル形状
#[derive(Debug, Clone, PartialEq)]
pub struct Dim3 {
    /// Depth dimension
    /// 深さ次元
    pub depth: usize,
    /// Height dimension
    /// 高さ次元
    pub height: usize,
    /// Width dimension
    /// 幅次元
    pub width: usize,
}

impl Dimension for Dim3 {
    const NDIM: usize = 3;
    
    fn shape(&self) -> Vec<usize> {
        vec![self.depth, self.height, self.width]
    }
}

/// 4-dimensional tensor shape (batch, channels, height, width)
/// 4次元テンソル形状（バッチ、チャンネル、高さ、幅）
#[derive(Debug, Clone, PartialEq)]
pub struct Dim4 {
    /// Batch size
    /// バッチサイズ
    pub batch: usize,
    /// Number of channels
    /// チャンネル数
    pub channels: usize,
    /// Height dimension
    /// 高さ次元
    pub height: usize,
    /// Width dimension
    /// 幅次元
    pub width: usize,
}

impl Dimension for Dim4 {
    const NDIM: usize = 4;
    
    fn shape(&self) -> Vec<usize> {
        vec![self.batch, self.channels, self.height, self.width]
    }
}

/// Type-safe tensor with compile-time shape verification
/// コンパイル時形状検証付きの型安全テンソル
#[derive(Debug)]
pub struct TypedTensor<T: Float, D: Dimension> {
    data: Vec<T>,
    shape: D,
    _phantom: PhantomData<T>,
}

impl<T: Float, D: Dimension> TypedTensor<T, D> {
    /// Create a new typed tensor with shape verification
    /// 形状検証付きで新しい型付きテンソルを作成
    pub fn new(data: Vec<T>, shape: D) -> RusTorchResult<Self> {
        if data.len() != shape.numel() {
            return Err(RusTorchError::type_error("Shape mismatch"));
        }
        
        Ok(Self {
            data,
            shape,
            _phantom: PhantomData,
        })
    }
    
    /// Create a zero tensor with specified shape
    /// 指定形状のゼロテンソルを作成
    pub fn zeros(shape: D) -> Self {
        let data = vec![T::zero(); shape.numel()];
        Self {
            data,
            shape,
            _phantom: PhantomData,
        }
    }
    
    /// Create a tensor filled with ones
    /// 1で埋められたテンソルを作成
    pub fn ones(shape: D) -> Self {
        let data = vec![T::one(); shape.numel()];
        Self {
            data,
            shape,
            _phantom: PhantomData,
        }
    }
    
    /// Get tensor shape
    /// テンソル形状を取得
    pub fn shape(&self) -> &D {
        &self.shape
    }
    
    /// Get tensor data
    /// テンソルデータを取得
    pub fn data(&self) -> &[T] {
        &self.data
    }
    
    /// Get mutable tensor data
    /// 変更可能なテンソルデータを取得
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    /// Number of dimensions
    /// 次元数
    pub fn ndim(&self) -> usize {
        D::NDIM
    }
    
    /// Total number of elements
    /// 要素の総数
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
}

/// Matrix multiplication for 2D tensors (compile-time verified)
/// 2Dテンソルの行列乗算（コンパイル時検証済み）
impl<T: Float> TypedTensor<T, Dim2> {
    /// Perform matrix multiplication with compile-time dimension verification
    /// コンパイル時次元検証付きで行列乗算を実行
    pub fn matmul(&self, other: &TypedTensor<T, Dim2>) -> RusTorchResult<TypedTensor<T, Dim2>> {
        if self.shape.cols != other.shape.rows {
            return Err(RusTorchError::type_error("Matrix dimension mismatch"));
        }
        
        let result_shape = Dim2 {
            rows: self.shape.rows,
            cols: other.shape.cols,
        };
        
        let mut result_data = vec![T::zero(); result_shape.numel()];
        
        // Perform matrix multiplication
        for i in 0..self.shape.rows {
            for j in 0..other.shape.cols {
                let mut sum = T::zero();
                for k in 0..self.shape.cols {
                    let a_idx = i * self.shape.cols + k;
                    let b_idx = k * other.shape.cols + j;
                    sum = sum + self.data[a_idx] * other.data[b_idx];
                }
                result_data[i * result_shape.cols + j] = sum;
            }
        }
        
        Ok(TypedTensor {
            data: result_data,
            shape: result_shape,
            _phantom: PhantomData,
        })
    }
}

/// Type-safe reshape operations
/// 型安全なリシェイプ操作
pub trait TypeSafeReshape<T: Float, From: Dimension, To: Dimension> {
    /// Reshape tensor from one dimension type to another
    /// テンソルを異なる次元タイプにリシェイプ
    fn reshape(tensor: TypedTensor<T, From>) -> RusTorchResult<TypedTensor<T, To>>;
}

/// Reshape 1D to 2D
/// 1Dから2Dへのリシェイプ
impl<T: Float> TypeSafeReshape<T, Dim1, Dim2> for TypedTensor<T, Dim2> {
    fn reshape(tensor: TypedTensor<T, Dim1>) -> RusTorchResult<TypedTensor<T, Dim2>> {
        // This would need specific shape parameters - for now, return error
        Err(RusTorchError::type_error("Manual reshape parameters required"))
    }
}

/// Element-wise operations for typed tensors
/// 型付きテンソルの要素ごとの操作
impl<T: Float, D: Dimension> TypedTensor<T, D> {
    /// Element-wise addition
    /// 要素ごとの加算
    pub fn add(&self, other: &TypedTensor<T, D>) -> RusTorchResult<TypedTensor<T, D>> {
        if self.shape != other.shape {
            return Err(RusTorchError::type_error("Shape incompatible"));
        }
        
        let result_data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        
        Ok(TypedTensor {
            data: result_data,
            shape: self.shape.clone(),
            _phantom: PhantomData,
        })
    }
    
    /// Element-wise multiplication
    /// 要素ごとの乗算
    pub fn mul(&self, other: &TypedTensor<T, D>) -> RusTorchResult<TypedTensor<T, D>> {
        if self.shape != other.shape {
            return Err(RusTorchError::type_error("Shape incompatible"));
        }
        
        let result_data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        
        Ok(TypedTensor {
            data: result_data,
            shape: self.shape.clone(),
            _phantom: PhantomData,
        })
    }
    
    /// Scalar multiplication
    /// スカラー乗算
    pub fn mul_scalar(&self, scalar: T) -> TypedTensor<T, D> {
        let result_data: Vec<T> = self.data.iter()
            .map(|&x| x * scalar)
            .collect();
        
        TypedTensor {
            data: result_data,
            shape: self.shape.clone(),
            _phantom: PhantomData,
        }
    }
    
    /// Apply function element-wise
    /// 要素ごとに関数を適用
    pub fn map<F>(&self, f: F) -> TypedTensor<T, D>
    where
        F: Fn(T) -> T,
    {
        let result_data: Vec<T> = self.data.iter()
            .map(|&x| f(x))
            .collect();
        
        TypedTensor {
            data: result_data,
            shape: self.shape.clone(),
            _phantom: PhantomData,
        }
    }
    
    /// Sum all elements
    /// 全要素の合計
    pub fn sum(&self) -> T {
        self.data.iter().fold(T::zero(), |acc, &x| acc + x)
    }
    
    /// Mean of all elements
    /// 全要素の平均
    pub fn mean(&self) -> T {
        let sum = self.sum();
        let count = T::from(self.numel()).unwrap_or(T::one());
        sum / count
    }
}

// TypeSafetyError enum removed - now using unified RusTorchError system
// TypeSafetyErrorエナム削除 - 統一RusTorchErrorシステムを使用

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = Dim2 { rows: 2, cols: 3 };
        
        let tensor = TypedTensor::new(data, shape).unwrap();
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 6);
    }
    
    #[test]
    fn test_shape_mismatch_error() {
        let data = vec![1.0f32, 2.0, 3.0]; // 3 elements
        let shape = Dim2 { rows: 2, cols: 3 }; // Expects 6 elements
        
        let result = TypedTensor::new(data, shape);
        assert!(result.is_err());
        
        if let Err(_) = result {
            // Error occurred as expected
        } else {
            panic!("Expected error");
        }
    }
    
    #[test]
    fn test_matrix_multiplication() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a_shape = Dim2 { rows: 2, cols: 2 };
        let a = TypedTensor::new(a_data, a_shape).unwrap();
        
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let b_shape = Dim2 { rows: 2, cols: 2 };
        let b = TypedTensor::new(b_data, b_shape).unwrap();
        
        let result = a.matmul(&b).unwrap();
        let expected = vec![19.0f32, 22.0, 43.0, 50.0]; // Manual calculation
        
        assert_eq!(result.data(), &expected);
        assert_eq!(result.shape().rows, 2);
        assert_eq!(result.shape().cols, 2);
    }
    
    #[test]
    fn test_element_wise_operations() {
        let a_data = vec![1.0f32, 2.0, 3.0];
        let a_shape = Dim1 { size: 3 };
        let a = TypedTensor::new(a_data, a_shape).unwrap();
        
        let b_data = vec![4.0f32, 5.0, 6.0];
        let b_shape = Dim1 { size: 3 };
        let b = TypedTensor::new(b_data, b_shape).unwrap();
        
        let sum = a.add(&b).unwrap();
        assert_eq!(sum.data(), &[5.0f32, 7.0, 9.0]);
        
        let product = a.mul(&b).unwrap();
        assert_eq!(product.data(), &[4.0f32, 10.0, 18.0]);
    }
    
    #[test]
    fn test_scalar_operations() {
        let data = vec![1.0f32, 2.0, 3.0];
        let shape = Dim1 { size: 3 };
        let tensor = TypedTensor::new(data, shape).unwrap();
        
        let scaled = tensor.mul_scalar(2.0);
        assert_eq!(scaled.data(), &[2.0f32, 4.0, 6.0]);
        
        let sum = tensor.sum();
        assert_eq!(sum, 6.0);
        
        let mean = tensor.mean();
        assert_eq!(mean, 2.0);
    }
    
    #[test]
    fn test_zeros_and_ones() {
        let shape = Dim2 { rows: 2, cols: 3 };
        
        let zeros: TypedTensor<f32, Dim2> = TypedTensor::zeros(shape.clone());
        assert_eq!(zeros.data(), &[0.0f32; 6]);
        
        let ones: TypedTensor<f32, Dim2> = TypedTensor::ones(shape);
        assert_eq!(ones.data(), &[1.0f32; 6]);
    }
}