//! Core tensor data structure and basic operations
//! コアテンソルデータ構造と基本操作

use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::fmt;

/// A multi-dimensional array that supports automatic differentiation.
/// 自動微分をサポートする多次元配列
#[derive(Debug, Clone)]
pub struct Tensor<T: Float> {
    /// The underlying n-dimensional array data
    /// 基底のn次元配列データ
    pub data: ArrayD<T>,
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
    /// ユニーク識別用のポインターアドレスを取得
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    /// Copy data from another tensor (unsafe internal implementation)
    /// 別のテンソルからデータをコピー（unsafe内部実装）
    pub fn copy_from(&self, other: &Tensor<T>) {
        unsafe {
            let self_ptr = self.data.as_ptr() as *mut T;
            let other_ptr = other.data.as_ptr();
            let len = self.data.len().min(other.data.len());
            std::ptr::copy_nonoverlapping(other_ptr, self_ptr, len);
        }
    }
    
    /// Convert tensor to different device (mock implementation)
    /// テンソルを別のデバイスに変換（モック実装）
    pub fn to_device(&self, _device: std::sync::Arc<dyn crate::gpu::device::GpuDevice>) -> Self {
        self.clone()
    }
    
    /// Convert tensor to CPU
    /// テンソルをCPUに変換
    pub fn to_cpu(&self) -> Self {
        self.clone()
    }

    /// Creates a tensor filled with zeros.
    /// ゼロで満たされたテンソルを作成します。
    pub fn zeros(shape: &[usize]) -> Self {
        let total_size = shape.iter().product();
        let data = vec![T::zero(); total_size];
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Creates a tensor filled with ones.
    /// 1で満たされたテンソルを作成します。
    pub fn ones(shape: &[usize]) -> Self {
        let total_size = shape.iter().product();
        let data = vec![T::one(); total_size];
        Tensor::from_vec(data, shape.to_vec())
    }
    
    /// Create a scalar tensor from a single value
    /// 単一の値からスカラーテンソルを作成
    pub fn from_scalar(value: T) -> Self {
        Self::from_vec(vec![value], vec![1])
    }

    /// Creates a tensor filled with a specific value.
    /// 指定された値で満たされたテンソルを作成します。
    pub fn full(shape: &[usize], value: T) -> Self {
        let total_size = shape.iter().product();
        let data = vec![value; total_size];
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Returns the shape of the tensor.
    /// テンソルの形状を返します。
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns the number of dimensions.
    /// 次元数を返します。
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Returns the size of the tensor (total number of elements).
    /// テンソルのサイズ（要素の総数）を返します。
    pub fn size(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Returns the number of elements in the tensor.
    /// テンソル内の要素数を返します。
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the tensor is empty.
    /// テンソルが空の場合trueを返します。
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reshapes the tensor to the given shape.
    /// テンソルを指定された形状に変形します。
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, String> {
        let total_elements = self.data.len();
        let new_total_elements: usize = shape.iter().product();
        
        if total_elements != new_total_elements {
            return Err(format!(
                "Cannot reshape tensor of {} elements to shape {:?} (requires {} elements)",
                total_elements, shape, new_total_elements
            ));
        }
        
        match self.data.clone().into_shape(IxDyn(shape)) {
            Ok(reshaped) => Ok(Tensor::new(reshaped)),
            Err(e) => Err(format!("Reshape failed: {}", e)),
        }
    }

    /// Creates a view into the tensor.
    /// テンソルのビューを作成します。
    pub fn view(&self, shape: &[usize]) -> Result<Self, String> {
        self.reshape(shape)
    }

    /// Flattens the tensor to 1D.
    /// テンソルを1次元に平坦化します。
    pub fn flatten(&self) -> Self {
        let total_elements = self.data.len();
        let flattened = self.data.clone().into_shape(IxDyn(&[total_elements])).unwrap();
        Tensor::new(flattened)
    }


    /// Returns a reference to the underlying data as a slice.
    /// 基になるデータへのスライス参照を返します。
    pub fn as_slice(&self) -> Option<&[T]> {
        self.data.as_slice()
    }

    /// Returns a mutable reference to the underlying data as a slice.
    /// 基になるデータへの可変スライス参照を返します。
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.data.as_slice_mut()
    }

    /// Returns a reference to the underlying ndarray.
    /// 基になるndarrayへの参照を返します。
    pub fn as_array(&self) -> &ArrayD<T> {
        &self.data
    }

    /// Returns a mutable reference to the underlying ndarray.
    /// 基になるndarrayへの可変参照を返します。
    pub fn as_array_mut(&mut self) -> &mut ArrayD<T> {
        &mut self.data
    }

    /// Gets an element at the specified index.
    /// 指定されたインデックスの要素を取得します。
    pub fn get(&self, index: &[usize]) -> Option<T> {
        self.data.get(IxDyn(index)).copied()
    }

    /// Sets an element at the specified index.
    /// 指定されたインデックスの要素を設定します。
    pub fn set(&mut self, index: &[usize], value: T) -> Result<(), String> {
        match self.data.get_mut(IxDyn(index)) {
            Some(elem) => {
                *elem = value;
                Ok(())
            }
            None => Err(format!("Index {:?} is out of bounds for tensor with shape {:?}", index, self.shape())),
        }
    }
}

impl<T: Float + fmt::Display> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<T: Float + 'static> From<ArrayD<T>> for Tensor<T> {
    fn from(array: ArrayD<T>) -> Self {
        Tensor::new(array)
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