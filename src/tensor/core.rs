//! Core tensor data structure and basic operations
//! コアテンソルデータ構造と基本操作

use super::device::Device;
#[cfg(not(target_arch = "wasm32"))]
use super::memory::aligned::{SimdAllocator, SIMD_ALIGNMENT};
#[cfg(not(target_arch = "wasm32"))]
use super::memory::optimization::{MemoryOptimization, TensorMemoryInfo};
use super::operations::zero_copy::{TensorIterOps, ZeroCopyOps};
use crate::error::{RusTorchError, RusTorchResult};
use crate::serialization::core::{Loadable, Saveable, SerializationResult};
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::device::GpuDevice;

/// A multi-dimensional array that supports automatic differentiation.
/// 自動微分をサポートする多次元配列
#[derive(Debug, Clone)]
pub struct Tensor<T: Float> {
    /// The underlying n-dimensional array data
    /// 基底のn次元配列データ
    pub data: ArrayD<T>,
    /// Device where tensor is stored
    /// テンソルが保存されているデバイス
    pub device: Device,
    /// Whether tensor requires gradient computation
    /// テンソルが勾配計算を必要とするか
    pub requires_grad: bool,
}

impl<T: Float + 'static> Tensor<T> {
    /// Creates a new tensor from an array.
    /// 配列から新しいテンソルを作成します。
    pub fn new(data: ArrayD<T>) -> Self {
        Tensor {
            data,
            device: Device::default(),
            requires_grad: false,
        }
    }

    /// Creates a new tensor with device and gradient settings
    /// デバイスと勾配設定付きで新しいテンソルを作成
    pub fn new_with_options(data: ArrayD<T>, device: Device, requires_grad: bool) -> Self {
        Tensor {
            data,
            device,
            requires_grad,
        }
    }

    /// Creates a tensor from a vector and shape.
    /// ベクトルと形状からテンソルを作成します。
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let array = ArrayD::from_shape_vec(shape, data).expect("Invalid shape for data");
        Tensor::new(array)
    }

    /// Creates a tensor from a vector and shape with device and gradient settings
    /// デバイスと勾配設定付きでベクトルと形状からテンソルを作成
    pub fn from_vec_with_options(
        data: Vec<T>,
        shape: Vec<usize>,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        let array = ArrayD::from_shape_vec(shape, data).expect("Invalid shape for data");
        Tensor::new_with_options(array, device, requires_grad)
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

    /// Get the device where this tensor is stored
    /// このテンソルが保存されているデバイスを取得
    pub fn device(&self) -> Device {
        self.device
    }

    /// Check if tensor requires gradient computation
    /// テンソルが勾配計算を必要とするかチェック
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set whether tensor requires gradient computation
    /// テンソルが勾配計算を必要とするかを設定
    pub fn set_requires_grad(&mut self, requires_grad: bool) -> &mut Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Move tensor to specified device type
    /// テンソルを指定デバイスタイプに移動
    pub fn with_device(&self, device: Device) -> Self {
        let mut new_tensor = self.clone();
        new_tensor.device = device;
        new_tensor
    }

    /// Get the current device type for this tensor
    /// このテンソルの現在のデバイスタイプを取得
    pub fn device_type(&self) -> String {
        self.device.to_string()
    }

    /// Check if tensor is on GPU
    /// テンソルがGPU上にあるかチェック
    pub fn is_on_gpu(&self) -> bool {
        self.device.is_cuda() || self.device.is_mps()
    }

    /// Check if tensor is on CPU
    /// テンソルがCPU上にあるかチェック
    pub fn is_cpu(&self) -> bool {
        self.device.is_cpu()
    }

    // Memory optimization methods are now provided by the MemoryOptimization trait
    // メモリ最適化メソッドはMemoryOptimizationトレイトで提供されます

    /// Creates a tensor filled with zeros.
    /// ゼロで満たされたテンソルを作成します。
    pub fn zeros(shape: &[usize]) -> Self {
        let total_size = shape.iter().product();
        let data = vec![T::zero(); total_size];
        Self {
            data: ArrayD::from_shape_vec(shape.to_vec(), data).expect("Invalid shape for data"),
            device: Device::default(),
            requires_grad: false,
        }
    }

    /// Creates a tensor filled with zeros on specified device
    /// 指定デバイス上でゼロで満たされたテンソルを作成
    pub fn zeros_on_device(shape: &[usize], device: Device) -> Self {
        let total_size = shape.iter().product();
        let data = vec![T::zero(); total_size];
        Self {
            data: ArrayD::from_shape_vec(shape.to_vec(), data).expect("Invalid shape for data"),
            device,
            requires_grad: false,
        }
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
        Self {
            data: ArrayD::from_shape_vec(shape.to_vec(), data).expect("Invalid shape for data"),
            device: Device::default(),
            requires_grad: false,
        }
    }

    /// Creates a tensor filled with ones on specified device
    /// 指定デバイス上で1で満たされたテンソルを作成
    pub fn ones_on_device(shape: &[usize], device: Device) -> Self {
        let total_size = shape.iter().product();
        let data = vec![T::one(); total_size];
        Self {
            data: ArrayD::from_shape_vec(shape.to_vec(), data).expect("Invalid shape for data"),
            device,
            requires_grad: false,
        }
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

    // Phase 8: Tensor Utilities - Conditional Operations
    /// Select elements from self or other based on condition
    /// 条件に基づいてselfまたはotherから要素を選択
    pub fn where_(&self, condition: &ArrayD<bool>, other: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        use crate::tensor::utilities::conditional::where_;
        where_(condition, self, other)
    }

    /// Select elements where mask is true
    /// マスクがtrueの位置の要素を選択
    pub fn masked_select(&self, mask: &ArrayD<bool>) -> RusTorchResult<Tensor<T>> {
        use crate::tensor::utilities::conditional::masked_select;
        masked_select(self, mask)
    }

    /// Fill tensor elements where mask is true with value
    /// マスクがtrueの位置の要素を値で埋める
    pub fn masked_fill(&self, mask: &ArrayD<bool>, value: T) -> RusTorchResult<Tensor<T>> {
        use crate::tensor::utilities::conditional::masked_fill;
        masked_fill(self, mask, value)
    }

    // Phase 8: Tensor Utilities - Index Operations
    /// Gather values along an axis
    /// 軸に沿って値を収集
    pub fn gather(&self, dim: usize, index: &ArrayD<i64>) -> RusTorchResult<Tensor<T>> {
        use crate::tensor::utilities::indexing::gather;
        gather(self, dim, index)
    }

    /// Scatter values along an axis
    /// 軸に沿って値を散布
    pub fn scatter(
        &self,
        dim: usize,
        index: &ArrayD<i64>,
        src: &Tensor<T>,
    ) -> RusTorchResult<Tensor<T>> {
        use crate::tensor::utilities::indexing::scatter;
        scatter(self, dim, index, src)
    }

    /// Select values along an axis using index
    /// インデックスを使って軸に沿って値を選択
    pub fn index_select(&self, dim: usize, index: &ArrayD<i64>) -> RusTorchResult<Tensor<T>> {
        use crate::tensor::utilities::indexing::index_select;
        index_select(self, dim, index)
    }

    // Phase 8: Tensor Utilities - Statistical Operations
    /// Get top k elements along dimension (Phase 8 utilities)
    /// 次元に沿ってトップk要素を取得（フェーズ8ユーティリティ）
    pub fn topk_util(
        &self,
        k: usize,
        dim: usize,
        largest: bool,
        sorted: bool,
    ) -> RusTorchResult<(Tensor<T>, ArrayD<i64>)> {
        use crate::tensor::utilities::statistics::topk_util;
        topk_util(self, k, dim, largest, sorted)
    }

    /// Get kth smallest/largest value
    /// k番目の最小/最大値を取得
    pub fn kthvalue(
        &self,
        k: usize,
        dim: usize,
        keepdim: bool,
    ) -> RusTorchResult<(Tensor<T>, ArrayD<i64>)> {
        use crate::tensor::utilities::statistics::kthvalue;
        kthvalue(self, k, dim, keepdim)
    }

    // Phase 8: Tensor Utilities - Advanced Operations
    /// Get unique elements
    /// 一意の要素を取得
    pub fn unique(
        &self,
        sorted: bool,
        return_inverse: bool,
        return_counts: bool,
    ) -> RusTorchResult<(Tensor<T>, Option<ArrayD<i64>>, Option<ArrayD<i64>>)> {
        use crate::tensor::utilities::advanced::unique;
        unique(self, sorted, return_inverse, return_counts, None)
    }

    /// Compute histogram
    /// ヒストグラムを計算
    pub fn histogram(
        &self,
        bins: usize,
        range: Option<(T, T)>,
    ) -> RusTorchResult<(ArrayD<i64>, Tensor<T>)> {
        use crate::tensor::utilities::advanced::histogram;
        histogram(self, bins, range, false)
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

// Serialization integration for Tensor
// Tensorのシリアライゼーション統合
impl<T: Float + 'static> Tensor<T> {
    /// Save tensor to file
    /// テンソルをファイルに保存
    pub fn save<P: AsRef<Path>>(&self, path: P) -> SerializationResult<()> {
        crate::serialization::model_io::save(self, path)
    }

    /// Load tensor from file  
    /// ファイルからテンソルを読み込み
    pub fn load<P: AsRef<Path>>(path: P) -> SerializationResult<Self> {
        crate::serialization::model_io::load(path)
    }

    /// Clone tensor with new device
    /// 新しいデバイスでテンソルをクローン
    pub fn clone_to_device(&self, device: Device) -> Self {
        Self {
            data: self.data.clone(),
            device,
            requires_grad: self.requires_grad,
        }
    }

    /// Clone tensor with new gradient requirement
    /// 新しい勾配要件でテンソルをクローン
    pub fn clone_with_grad(&self, requires_grad: bool) -> Self {
        Self {
            data: self.data.clone(),
            device: self.device,
            requires_grad,
        }
    }

    /// Get tensor metadata for serialization
    /// シリアライゼーション用テンソルメタデータを取得
    pub fn get_metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert("shape".to_string(), format!("{:?}", self.shape()));
        meta.insert("dtype".to_string(), std::any::type_name::<T>().to_string());
        meta.insert("device".to_string(), self.device.to_string());
        meta.insert("requires_grad".to_string(), self.requires_grad.to_string());
        meta.insert("numel".to_string(), self.numel().to_string());
        meta
    }
}

// Hybrid execution implementation for Tensor
// TensorのHybridExecution実装
#[cfg(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback",
    feature = "metal",
    feature = "cuda"
))]
impl<T: Float + 'static> crate::gpu::hybrid_executor::HybridExecution<T> for Tensor<T> {
    fn hybrid_operation<F, R>(
        &self,
        op_type: crate::gpu::OpType,
        operation: F,
    ) -> crate::error::RusTorchResult<R>
    where
        F: Fn(crate::gpu::DeviceType) -> crate::error::RusTorchResult<R>,
    {
        use crate::gpu::hybrid_executor::HybridExecutor;

        let tensor_info = self.tensor_info();
        let executor = HybridExecutor::new();
        executor.execute(op_type, tensor_info, operation)
    }

    fn tensor_info(&self) -> crate::gpu::hybrid_executor::TensorInfo {
        use crate::dtype::DType;
        use std::mem;

        // Determine DType based on generic type T
        let dtype = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            DType::Float32
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            DType::Float64
        } else {
            DType::Float32 // Default fallback
        };

        crate::gpu::hybrid_executor::TensorInfo {
            dtype,
            shape: self.shape().to_vec(),
            requires_custom_kernel: false, // For now, assume standard kernels
            memory_size_bytes: self.numel() * mem::size_of::<T>(),
        }
    }
}
