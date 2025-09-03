//! Shared tensor operations for both regular and WASM tensors
//! 通常テンソルとWASMテンソル両方用の共通操作

use num_traits::Float;

/// Common tensor operations that both Tensor<T> and WasmTensor can implement
/// Tensor<T>とWasmTensorの両方が実装できる共通テンソル操作
pub trait CommonTensorOps<T> {
    type Error;

    /// Element-wise addition using shared implementation
    /// 共通実装を使用した要素ごと加算
    fn add_elements(&self, other: &Self) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Element-wise subtraction using shared implementation  
    /// 共通実装を使用した要素ごと減算
    fn sub_elements(&self, other: &Self) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// ReLU activation using shared implementation
    /// 共通実装を使用したReLU活性化
    fn relu_activation(&self) -> Self
    where
        Self: Sized;

    /// Sigmoid activation using shared implementation
    /// 共通実装を使用したSigmoid活性化  
    fn sigmoid_activation(&self) -> Self
    where
        Self: Sized;
}

/// Core tensor operations trait that both Tensor<T> and WasmTensor can implement
/// Tensor<T>とWasmTensorの両方が実装できるコアテンソル操作トレイト
pub trait TensorOps<T> {
    type Error;
    type Shape;

    /// Get tensor data as slice
    /// テンソルデータをスライスとして取得
    fn data(&self) -> &[T];

    /// Get tensor shape
    /// テンソル形状を取得
    fn shape(&self) -> Self::Shape;

    /// Get total number of elements
    /// 総要素数を取得
    fn size(&self) -> usize;

    /// Get number of dimensions
    /// 次元数を取得
    fn ndim(&self) -> usize;
}

/// Mathematical operations that can be shared between tensor types
/// テンソルタイプ間で共有可能な数学操作
pub mod math_ops {
    use super::*;

    /// Element-wise addition
    /// 要素ごとの加算
    pub fn element_wise_add<T: Float>(data1: &[T], data2: &[T]) -> Result<Vec<T>, &'static str> {
        if data1.len() != data2.len() {
            return Err("Data length mismatch");
        }

        Ok(data1
            .iter()
            .zip(data2.iter())
            .map(|(&a, &b)| a + b)
            .collect())
    }

    /// Element-wise subtraction
    /// 要素ごとの減算
    pub fn element_wise_sub<T: Float>(data1: &[T], data2: &[T]) -> Result<Vec<T>, &'static str> {
        if data1.len() != data2.len() {
            return Err("Data length mismatch");
        }

        Ok(data1
            .iter()
            .zip(data2.iter())
            .map(|(&a, &b)| a - b)
            .collect())
    }

    /// Element-wise multiplication
    /// 要素ごとの乗算
    pub fn element_wise_mul<T: Float>(data1: &[T], data2: &[T]) -> Result<Vec<T>, &'static str> {
        if data1.len() != data2.len() {
            return Err("Data length mismatch");
        }

        Ok(data1
            .iter()
            .zip(data2.iter())
            .map(|(&a, &b)| a * b)
            .collect())
    }

    /// Element-wise division
    /// 要素ごとの除算
    pub fn element_wise_div<T: Float>(data1: &[T], data2: &[T]) -> Result<Vec<T>, &'static str> {
        if data1.len() != data2.len() {
            return Err("Data length mismatch");
        }

        Ok(data1
            .iter()
            .zip(data2.iter())
            .map(|(&a, &b)| if b == T::zero() { T::nan() } else { a / b })
            .collect())
    }

    /// Scalar addition
    /// スカラー加算
    pub fn scalar_add<T: Float>(data: &[T], scalar: T) -> Vec<T> {
        data.iter().map(|&x| x + scalar).collect()
    }

    /// Scalar multiplication
    /// スカラー乗算
    pub fn scalar_mul<T: Float>(data: &[T], scalar: T) -> Vec<T> {
        data.iter().map(|&x| x * scalar).collect()
    }
}

/// Activation functions shared between tensor types
/// テンソルタイプ間で共有される活性化関数
pub mod activation_ops {
    use super::*;

    /// ReLU activation function
    /// ReLU活性化関数
    pub fn relu<T: Float>(data: &[T]) -> Vec<T> {
        data.iter().map(|&x| x.max(T::zero())).collect()
    }

    /// Sigmoid activation function
    /// Sigmoid活性化関数
    pub fn sigmoid<T: Float>(data: &[T]) -> Vec<T> {
        data.iter()
            .map(|&x| T::one() / (T::one() + (-x).exp()))
            .collect()
    }

    /// Tanh activation function
    /// Tanh活性化関数
    pub fn tanh<T: Float>(data: &[T]) -> Vec<T> {
        data.iter().map(|&x| x.tanh()).collect()
    }
}

/// Mathematical functions shared between tensor types
/// テンソルタイプ間で共有される数学関数
pub mod math_funcs {
    use super::*;

    /// Power function
    /// べき乗関数
    pub fn pow<T: Float>(data: &[T], exponent: T) -> Vec<T> {
        data.iter().map(|&x| x.powf(exponent)).collect()
    }

    /// Square root function
    /// 平方根関数
    pub fn sqrt<T: Float>(data: &[T]) -> Vec<T> {
        data.iter().map(|&x| x.sqrt()).collect()
    }

    /// Exponential function
    /// 指数関数
    pub fn exp<T: Float>(data: &[T]) -> Vec<T> {
        data.iter().map(|&x| x.exp()).collect()
    }

    /// Natural logarithm function
    /// 自然対数関数
    pub fn log<T: Float>(data: &[T]) -> Vec<T> {
        data.iter().map(|&x| x.ln()).collect()
    }
}

/// Statistical operations shared between tensor types
/// テンソルタイプ間で共有される統計操作
pub mod stats_ops {
    use super::*;

    /// Sum of all elements
    /// 全要素の合計
    pub fn sum<T: Float>(data: &[T]) -> T {
        data.iter().fold(T::zero(), |acc, &x| acc + x)
    }

    /// Mean of all elements
    /// 全要素の平均
    pub fn mean<T: Float>(data: &[T]) -> T {
        if data.is_empty() {
            T::zero()
        } else {
            sum(data) / T::from(data.len()).unwrap_or(T::one())
        }
    }

    /// Maximum element
    /// 最大要素
    pub fn max<T: Float>(data: &[T]) -> T {
        data.iter().fold(T::neg_infinity(), |a, &b| a.max(b))
    }

    /// Minimum element
    /// 最小要素
    pub fn min<T: Float>(data: &[T]) -> T {
        data.iter().fold(T::infinity(), |a, &b| a.min(b))
    }
}

/// Shape utility functions
/// 形状ユーティリティ関数
pub mod shape_ops {
    /// Check if two shapes are compatible for element-wise operations
    /// 要素ごと操作で2つの形状が互換性があるかチェック
    pub fn shapes_compatible(shape1: &[usize], shape2: &[usize]) -> bool {
        shape1 == shape2
    }

    /// Calculate total number of elements from shape
    /// 形状から総要素数を計算
    pub fn total_elements(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// Check if reshape is valid
    /// リシェイプが有効かチェック
    pub fn can_reshape(old_shape: &[usize], new_shape: &[usize]) -> bool {
        total_elements(old_shape) == total_elements(new_shape)
    }
}
