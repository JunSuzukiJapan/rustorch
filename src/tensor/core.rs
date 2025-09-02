//! Core tensor data structure and basic operations
//! コアテンソルデータ構造と基本操作

#[cfg(not(target_arch = "wasm32"))]
use super::memory::aligned::{SimdAllocator, SIMD_ALIGNMENT};
#[cfg(not(target_arch = "wasm32"))]
use super::memory::optimization::{MemoryOptimization, TensorMemoryInfo};
use super::operations::zero_copy::{TensorIterOps, ZeroCopyOps};
use crate::error::{RusTorchError, RusTorchResult};
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::device::GpuDevice;

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

    /// Creates a tensor from a vector and shape with error handling.
    /// エラーハンドリング付きでベクトルと形状からテンソルを作成します。
    pub fn try_from_vec(data: Vec<T>, shape: Vec<usize>) -> RusTorchResult<Self> {
        // Calculate expected size
        let expected_size = shape.iter().product::<usize>();
        let actual_size = data.len();

        if expected_size != actual_size {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![expected_size],
                actual: vec![actual_size],
            });
        }

        // Check for empty shape
        if shape.is_empty() {
            return Err(RusTorchError::TensorOp {
                message: "Shape cannot be empty".to_string(),
                source: None,
            });
        }

        // Check for zero or negative dimensions
        if shape.iter().any(|&dim| dim == 0) {
            return Err(RusTorchError::TensorOp {
                message: format!("Shape contains zero dimension: {:?}", shape),
                source: None,
            });
        }

        match ArrayD::from_shape_vec(shape, data) {
            Ok(array) => Ok(Tensor::new(array)),
            Err(e) => Err(RusTorchError::TensorOp {
                message: format!("Failed to create tensor from vector: {}", e),
                source: Some(Box::new(e)),
            }),
        }
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_device(&self, _device: std::sync::Arc<dyn crate::gpu::device::GpuDevice>) -> Self {
        self.clone()
    }

    /// Convert tensor to CPU
    /// テンソルをCPUに変換
    pub fn to_cpu(&self) -> Self {
        self.clone()
    }

    /// Automatically select the best device for this tensor operation
    /// このテンソル操作に最適なデバイスを自動選択
    #[cfg(not(target_arch = "wasm32"))]
    pub fn auto_device(&self) -> Self {
        // Simple heuristic: use GPU for large tensors, CPU for small ones
        let total_elements = self.data.len();
        const GPU_THRESHOLD: usize = 1000; // Elements

        if total_elements >= GPU_THRESHOLD {
            // Try to use GPU if available, fallback to CPU
            if let Ok(gpu_tensor) = self.try_to_gpu() {
                return gpu_tensor;
            }
        }

        // Default to CPU
        self.to_cpu()
    }

    /// Try to move tensor to GPU with error handling
    /// エラーハンドリング付きでテンソルをGPUに移動を試行
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_to_gpu(&self) -> RusTorchResult<Self> {
        // Check if GPU is available
        if !self.is_gpu_available() {
            return Err(RusTorchError::Device {
                device: "GPU".to_string(),
                message: "No GPU devices available".to_string(),
            });
        }

        // For now, just return CPU version (GPU implementation would go here)
        Ok(self.clone())
    }

    /// Check if GPU is available for this tensor
    /// このテンソルでGPUが利用可能かチェック
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_gpu_available(&self) -> bool {
        // Simple mock implementation
        // In real implementation, this would check for CUDA/Metal/OpenCL availability
        false
    }

    /// Get the current device type for this tensor
    /// このテンソルの現在のデバイスタイプを取得
    pub fn device_type(&self) -> String {
        "cpu".to_string() // Default to CPU for now
    }

    /// Check if tensor is on GPU
    /// テンソルがGPU上にあるかチェック
    pub fn is_on_gpu(&self) -> bool {
        false // Default implementation
    }

    // Memory optimization methods are now provided by the MemoryOptimization trait
    // メモリ最適化メソッドはMemoryOptimizationトレイトで提供されます

    /// Creates a tensor filled with zeros.
    /// ゼロで満たされたテンソルを作成します。
    pub fn zeros(shape: &[usize]) -> Self {
        let total_size = shape.iter().product();
        let data = vec![T::zero(); total_size];
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Creates a tensor filled with zeros with error handling.
    /// エラーハンドリング付きでゼロで満たされたテンソルを作成します。
    pub fn try_zeros(shape: &[usize]) -> RusTorchResult<Self> {
        // Check for empty shape
        if shape.is_empty() {
            return Err(RusTorchError::TensorOp {
                message: "Shape cannot be empty".to_string(),
                source: None,
            });
        }

        // Check for zero dimensions
        if shape.iter().any(|&dim| dim == 0) {
            return Err(RusTorchError::TensorOp {
                message: format!("Shape contains zero dimension: {:?}", shape),
                source: None,
            });
        }

        let total_size = shape.iter().product::<usize>();

        // Check for reasonable memory size (avoid OOM)
        const MAX_ELEMENTS: usize = 1_000_000_000; // 1 billion elements
        if total_size > MAX_ELEMENTS {
            return Err(RusTorchError::TensorOp {
                message: format!(
                    "Tensor too large: {} elements exceeds maximum of {}",
                    total_size, MAX_ELEMENTS
                ),
                source: None,
            });
        }

        let data = vec![T::zero(); total_size];
        Self::try_from_vec(data, shape.to_vec())
    }

    /// Creates a tensor filled with ones.
    /// 1で満たされたテンソルを作成します。
    pub fn ones(shape: &[usize]) -> Self {
        let total_size = shape.iter().product();
        let data = vec![T::one(); total_size];
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Creates a tensor filled with ones with error handling.
    /// エラーハンドリング付きで1で満たされたテンソルを作成します。
    pub fn try_ones(shape: &[usize]) -> RusTorchResult<Self> {
        // Check for empty shape
        if shape.is_empty() {
            return Err(RusTorchError::TensorOp {
                message: "Shape cannot be empty".to_string(),
                source: None,
            });
        }

        // Check for zero dimensions
        if shape.iter().any(|&dim| dim == 0) {
            return Err(RusTorchError::TensorOp {
                message: format!("Shape contains zero dimension: {:?}", shape),
                source: None,
            });
        }

        let total_size = shape.iter().product::<usize>();

        // Check for reasonable memory size (avoid OOM)
        const MAX_ELEMENTS: usize = 1_000_000_000; // 1 billion elements
        if total_size > MAX_ELEMENTS {
            return Err(RusTorchError::TensorOp {
                message: format!(
                    "Tensor too large: {} elements exceeds maximum of {}",
                    total_size, MAX_ELEMENTS
                ),
                source: None,
            });
        }

        let data = vec![T::one(); total_size];
        Self::try_from_vec(data, shape.to_vec())
    }

    /// Creates a tensor filled with zeros with automatic device selection
    /// 自動デバイス選択付きでゼロで満たされたテンソルを作成
    #[cfg(not(target_arch = "wasm32"))]
    pub fn zeros_auto(shape: &[usize]) -> Self {
        let tensor = Self::zeros(shape);
        tensor.auto_device()
    }

    /// Creates a tensor filled with ones with automatic device selection
    /// 自動デバイス選択付きで1で満たされたテンソルを作成
    #[cfg(not(target_arch = "wasm32"))]
    pub fn ones_auto(shape: &[usize]) -> Self {
        let tensor = Self::ones(shape);
        tensor.auto_device()
    }

    /// Creates a tensor from vector with automatic device selection
    /// 自動デバイス選択付きでベクトルからテンソルを作成
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_vec_auto(data: Vec<T>, shape: Vec<usize>) -> Self {
        let tensor = Self::from_vec(data, shape);
        tensor.auto_device()
    }

    /// Create a scalar tensor from a single value
    /// 単一の値からスカラーテンソルを作成
    pub fn from_scalar(value: T) -> Self {
        Self::from_vec(vec![value], vec![1])
    }

    /// Create tensor from ndarray
    /// ndarrayからテンソルを作成
    pub fn from_ndarray(array: ndarray::ArrayD<T>) -> Self {
        let shape = array.shape().to_vec();
        let (data, _offset) = array.into_raw_vec_and_offset();
        Self::from_vec(data, shape)
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

    /// Resolve a dimension index (supports negative indexing)
    /// 次元インデックスを解決（負のインデックスをサポート）
    pub fn resolve_dim(&self, dim: isize) -> Result<usize, String> {
        let ndim = self.shape().len() as isize;

        let resolved = if dim < 0 { ndim + dim } else { dim };

        if resolved < 0 || resolved >= ndim {
            Err(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim, ndim
            ))
        } else {
            Ok(resolved as usize)
        }
    }

    /// Reshapes the tensor to the given shape (new v2 implementation).
    /// テンソルを指定された形状に変形します。（新v2実装）
    pub fn reshape(&self, new_shape: &[usize]) -> crate::error::RusTorchResult<Self>
    where
        T: ndarray::ScalarOperand + num_traits::FromPrimitive,
    {
        let old_size: usize = self.shape().iter().product();
        let new_size: usize = new_shape.iter().product();

        if old_size != new_size {
            return Err(crate::error::RusTorchError::InvalidOperation {
                operation: "reshape".to_string(),
                message: format!(
                    "Cannot reshape tensor of size {} to size {}",
                    old_size, new_size
                ),
            });
        }

        Ok(Tensor::from_vec(
            self.data.iter().copied().collect(),
            new_shape.to_vec(),
        ))
    }

    /// Creates a view into the tensor.
    /// テンソルのビューを作成します。
    // view() method moved to ops::shape_operations as view_shape() for ownership-aware design

    /// Creates a view into the tensor with proper error handling.
    /// 適切なエラーハンドリング付きでテンソルのビューを作成します。
    pub fn try_view(&self, shape: &[usize]) -> RusTorchResult<Self> {
        let total_elements = self.data.len();
        let new_total_elements: usize = shape.iter().product();

        if total_elements != new_total_elements {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![total_elements],
                actual: vec![new_total_elements],
            });
        }

        // Check for empty shape
        if shape.is_empty() {
            return Err(RusTorchError::TensorOp {
                message: "Shape cannot be empty".to_string(),
                source: None,
            });
        }

        // Check for zero dimensions
        if shape.iter().any(|&dim| dim == 0) {
            return Err(RusTorchError::TensorOp {
                message: format!("Shape contains zero dimension: {:?}", shape),
                source: None,
            });
        }

        match self.data.clone().into_shape_with_order(IxDyn(shape)) {
            Ok(reshaped) => Ok(Tensor::new(reshaped)),
            Err(e) => Err(RusTorchError::TensorOp {
                message: format!("View failed: {}", e),
                source: Some(Box::new(e)),
            }),
        }
    }

    /// Flattens the tensor to 1D.
    /// テンソルを1次元に平坦化します。
    // flatten() method moved to ops::shape_operations for ownership-aware design

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
            None => Err(format!(
                "Index {:?} is out of bounds for tensor with shape {:?}",
                index,
                self.shape()
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

// Zero-copy operations are now provided by the ZeroCopyOps and TensorIterOps traits
// ゼロコピー操作はZeroCopyOpsとTensorIterOpsトレイトで提供されます

// Note: Standard operator implementations moved to ops/operators.rs
// to avoid conflicts and use the consolidated method implementations
// 注意: 標準演算子実装は競合を避け統合されたメソッド実装を使用するため
// ops/operators.rs に移動されました

// Note: Method aliases removed to avoid duplication with ops/ modules
// The actual implementations are in the respective ops/ modules:
// - matmul, transpose: ops/matrix.rs
// - sum, mean, sum_axis: ops/arithmetic.rs
// - maximum, minimum: ops/arithmetic.rs
// - stack, concatenate: ops/utilities.rs
// 注意: ops/モジュールとの重複を避けるためメソッドエイリアスを削除
// 実際の実装は対応するops/モジュールにあります

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Element-wise square root
    /// 要素ごとの平方根
    #[inline]
    pub fn sqrt(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.sqrt()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }
}
