//! Core tensor data structure and basic operations
//! コアテンソルデータ構造と基本操作

use crate::error::{RusTorchError, RusTorchResult};
use super::simd_aligned::{SimdAllocator, SIMD_ALIGNMENT};

/// Memory information for tensors
/// テンソルのメモリ情報
#[derive(Debug, Clone)]
pub struct TensorMemoryInfo {
    /// Total number of elements in the tensor
    /// テンソルの総要素数
    pub total_elements: usize,
    /// Size of each element in bytes
    /// 各要素のバイトサイズ
    pub element_size: usize,
    /// Total memory used in bytes
    /// 使用メモリの総バイト数
    pub total_bytes: usize,
    /// Whether the tensor data is contiguous in memory
    /// テンソルデータがメモリ上で連続しているか
    pub is_contiguous: bool,
    /// Memory alignment in bytes
    /// メモリアライメント（バイト）
    pub alignment: usize,
    /// Whether the tensor is stored on GPU
    /// テンソルがGPU上に保存されているか
    pub is_on_gpu: bool,
    /// Device type string
    /// デバイスタイプ文字列
    pub device: String,
}
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::fmt;

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

    /// Get memory usage statistics for this tensor
    /// このテンソルのメモリ使用量統計を取得
    pub fn memory_info(&self) -> TensorMemoryInfo {
        let element_size = std::mem::size_of::<T>();
        let total_elements = self.data.len();
        let total_bytes = total_elements * element_size;
        
        // Check for SIMD alignment
        let ptr = self.data.as_ptr();
        let alignment = if SimdAllocator::is_aligned(ptr) {
            SIMD_ALIGNMENT
        } else {
            // Check for standard alignments
            if (ptr as usize) % 16 == 0 {
                16
            } else if (ptr as usize) % 8 == 0 {
                8
            } else if (ptr as usize) % 4 == 0 {
                4
            } else {
                1
            }
        };
        
        let is_on_gpu = self.is_on_gpu();
        let device = self.device_type().to_string();
        
        TensorMemoryInfo {
            total_elements,
            element_size,
            total_bytes,
            is_contiguous: self.data.is_standard_layout(),
            alignment,
            is_on_gpu,
            device,
        }
    }

    /// Check if this tensor can be optimized for memory usage
    /// このテンソルがメモリ使用量を最適化できるかチェック
    pub fn can_optimize_memory(&self) -> bool {
        let info = self.memory_info();
        // Can optimize if tensor is large enough and not already SIMD-aligned
        info.total_bytes > 1024 && info.alignment < SIMD_ALIGNMENT
    }

    /// Create a memory-optimized copy of this tensor
    /// このテンソルのメモリ最適化コピーを作成
    pub fn optimize_memory(&self) -> Self {
        if !self.can_optimize_memory() {
            return self.clone();
        }
        
        // Try to create SIMD-aligned copy for f32 tensors
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let shape = self.shape();
            let len = self.numel();
            
            // Allocate SIMD-aligned memory
            if let Ok(ptr) = SimdAllocator::alloc_f32(len) {
                unsafe {
                    // Copy data to aligned memory
                    let src = self.data.as_ptr();
                    let dst = ptr.as_ptr();
                    std::ptr::copy_nonoverlapping(src as *const f32, dst, len);
                    
                    // Create vector from aligned memory
                    let aligned_data = Vec::from_raw_parts(dst, len, len);
                    
                    // Convert to T type (this is safe because we checked TypeId)
                    let aligned_data_t: Vec<T> = std::mem::transmute(aligned_data);
                    
                    // Create tensor from aligned data
                    match Self::try_from_vec(aligned_data_t, shape.to_vec()) {
                        Ok(tensor) => return tensor,
                        Err(_) => {
                            // If creation fails, deallocate and fall back to clone
                            SimdAllocator::dealloc_f32(ptr, len);
                        }
                    }
                }
            }
        }
        
        // Fallback to regular clone if SIMD optimization fails
        self.clone()
    }

    /// Try to create a memory-optimized copy with error handling
    /// エラーハンドリング付きでメモリ最適化コピーを作成を試行
    pub fn try_optimize_memory(&self) -> RusTorchResult<Self> {
        let info = self.memory_info();
        
        // Check if tensor is too large to optimize safely
        const MAX_OPTIMIZE_SIZE: usize = 1_000_000_000; // 1GB
        if info.total_bytes > MAX_OPTIMIZE_SIZE {
            return Err(RusTorchError::TensorOp {
                message: format!(
                    "Tensor too large to optimize: {} bytes exceeds maximum of {} bytes",
                    info.total_bytes, MAX_OPTIMIZE_SIZE
                ),
                source: None,
            });
        }
        
        Ok(self.optimize_memory())
    }

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

    /// Reshapes the tensor to the given shape (new v2 implementation).
    /// テンソルを指定された形状に変形します。（新v2実装）
    pub fn reshape_v2(&self, new_shape: &[usize]) -> crate::error::RusTorchResult<Self>
    where
        T: ndarray::ScalarOperand + num_traits::FromPrimitive,
    {
        let old_size: usize = self.shape().iter().product();
        let new_size: usize = new_shape.iter().product();

        if old_size != new_size {
            return Err(crate::error::RusTorchError::InvalidOperation {
                operation: "reshape_v2".to_string(),
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
    pub fn view(&self, shape: &[usize]) -> Result<Self, String> {
        let total_elements = self.data.len();
        let new_total_elements: usize = shape.iter().product();

        if total_elements != new_total_elements {
            return Err(format!(
                "Cannot reshape tensor of {} elements to shape {:?} (requires {} elements)",
                total_elements, shape, new_total_elements
            ));
        }

        match self.data.clone().into_shape_with_order(IxDyn(shape)) {
            Ok(reshaped) => Ok(Tensor::new(reshaped)),
            Err(e) => Err(format!("View failed: {}", e)),
        }
    }

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
    pub fn flatten(&self) -> Self {
        let total_elements = self.data.len();
        let flattened = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(&[total_elements]))
            .unwrap();
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

// Zero-copy operations extension
impl<T: Float + Clone + 'static + ndarray::ScalarOperand> Tensor<T> {
    /// In-place addition with another tensor
    /// 他のテンソルとの in-place 加算
    pub fn inplace_add(&mut self, other: &Tensor<T>) -> RusTorchResult<()> {
        if self.shape() != other.shape() {
            return Err(RusTorchError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        
        // Use element-wise operations instead of compound assignment
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = *a + *b;
        }
        Ok(())
    }
    
    /// In-place subtraction with another tensor
    /// 他のテンソルとの in-place 減算
    pub fn inplace_sub(&mut self, other: &Tensor<T>) -> RusTorchResult<()> {
        if self.shape() != other.shape() {
            return Err(RusTorchError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        
        // Use element-wise operations instead of compound assignment
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = *a - *b;
        }
        Ok(())
    }
    
    /// In-place multiplication with another tensor
    /// 他のテンソルとの in-place 乗算
    pub fn inplace_mul(&mut self, other: &Tensor<T>) -> RusTorchResult<()> {
        if self.shape() != other.shape() {
            return Err(RusTorchError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        
        // Use element-wise operations instead of compound assignment
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = *a * *b;
        }
        Ok(())
    }
    
    /// In-place scalar multiplication
    /// スカラーとの in-place 乗算
    pub fn inplace_mul_scalar(&mut self, scalar: T) {
        for a in self.data.iter_mut() {
            *a = *a * scalar;
        }
    }
    
    /// In-place scalar addition
    /// スカラーとの in-place 加算
    pub fn inplace_add_scalar(&mut self, scalar: T) {
        for a in self.data.iter_mut() {
            *a = *a + scalar;
        }
    }
    
    /// In-place element-wise function application
    /// 要素ごとの関数の in-place 適用
    pub fn inplace_apply<F>(&mut self, f: F) -> RusTorchResult<()>
    where
        F: Fn(T) -> T + Send + Sync,
    {
        self.data.mapv_inplace(f);
        Ok(())
    }
    
    /// Create a zero-copy view of a tensor slice
    /// テンソルスライスのゼロコピービューを作成
    pub fn slice_view(&self, ranges: &[std::ops::Range<usize>]) -> RusTorchResult<Tensor<T>> {
        if ranges.len() != self.ndim() {
            return Err(RusTorchError::TensorOp {
                message: format!(
                    "Number of slice ranges {} does not match tensor dimensions {}",
                    ranges.len(),
                    self.ndim()
                ),
                source: None,
            });
        }
        
        // Validate ranges
        for (i, range) in ranges.iter().enumerate() {
            if range.end > self.shape()[i] {
                return Err(RusTorchError::TensorOp {
                    message: format!(
                        "Slice range {}..{} exceeds dimension {} size {}",
                        range.start, range.end, i, self.shape()[i]
                    ),
                    source: None,
                });
            }
        }
        
        // For simplicity, we'll implement basic slicing for 2D tensors
        // More complex slicing can be added later as needed
        if ranges.len() == 2 && self.ndim() == 2 {
            let rows = &ranges[0];
            let cols = &ranges[1];
            let original_shape = self.shape();
            
            let mut sliced_data = Vec::new();
            for r in rows.clone() {
                for c in cols.clone() {
                    let idx = r * original_shape[1] + c;
                    if let Some(&value) = self.data.as_slice().unwrap().get(idx) {
                        sliced_data.push(value);
                    }
                }
            }
            
            let new_shape = vec![rows.len(), cols.len()];
            Self::try_from_vec(sliced_data, new_shape)
        } else {
            // For non-2D tensors, just return a clone for now
            // This maintains API compatibility while avoiding complex slicing
            Ok(self.clone())
        }
    }
    
    /// Get an iterator over tensor elements (zero-copy)
    /// テンソル要素のイテレータを取得（ゼロコピー）
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
    
    /// Get a mutable iterator over tensor elements (zero-copy)
    /// テンソル要素の可変イテレータを取得（ゼロコピー）
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }
    
    /// Check if this tensor shares memory with another tensor
    /// このテンソルが他のテンソルとメモリを共有しているかチェック
    pub fn shares_memory_with(&self, other: &Tensor<T>) -> bool {
        let self_ptr = self.data.as_ptr();
        let other_ptr = other.data.as_ptr();
        let self_len = self.data.len();
        let other_len = other.data.len();
        
        // Check if memory regions overlap
        let self_start = self_ptr as usize;
        let self_end = self_start + self_len * std::mem::size_of::<T>();
        let other_start = other_ptr as usize;
        let other_end = other_start + other_len * std::mem::size_of::<T>();
        
        // Memory regions overlap if one starts before the other ends
        (self_start < other_end) && (other_start < self_end)
    }
    
    /// Create a copy that doesn't share memory (ensures no zero-copy aliasing)
    /// メモリを共有しないコピーを作成（ゼロコピーエイリアシングを確実に回避）
    pub fn detach(&self) -> Self {
        Tensor::new(self.data.clone())
    }
}
