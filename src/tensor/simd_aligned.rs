//! SIMD-aligned memory allocation and operations for maximum performance
//! 最大パフォーマンスのためのSIMDアライメントメモリ割り当てと演算

use super::Tensor;
use super::parallel_errors::{ParallelError, ParallelResult};
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::Arc;

/// SIMD alignment requirements for different architectures
/// 異なるアーキテクチャのSIMDアライメント要件
pub const SIMD_ALIGNMENT: usize = 32; // AVX2 requires 32-byte alignment

/// SIMD-aligned memory allocator
/// SIMDアライメントメモリアロケータ
pub struct SimdAllocator;

impl SimdAllocator {
    /// Allocate SIMD-aligned memory for f32 array
    /// f32配列用のSIMDアライメントメモリを割り当て
    pub fn alloc_f32(len: usize) -> Result<NonNull<f32>, std::alloc::AllocError> {
        let layout = Layout::from_size_align(len * std::mem::size_of::<f32>(), SIMD_ALIGNMENT)
            .map_err(|_| std::alloc::AllocError)?;
        
        unsafe {
            let ptr = alloc_zeroed(layout);
            if ptr.is_null() {
                Err(std::alloc::AllocError)
            } else {
                Ok(NonNull::new_unchecked(ptr as *mut f32))
            }
        }
    }

    /// Deallocate SIMD-aligned memory
    /// SIMDアライメントメモリを解放
    pub unsafe fn dealloc_f32(ptr: NonNull<f32>, len: usize) {
        let layout = Layout::from_size_align_unchecked(
            len * std::mem::size_of::<f32>(), 
            SIMD_ALIGNMENT
        );
        dealloc(ptr.as_ptr() as *mut u8, layout);
    }

    /// Check if pointer is properly aligned for SIMD operations
    /// ポインタがSIMD演算に適切にアライメントされているかチェック
    pub fn is_aligned<T>(ptr: *const T) -> bool {
        (ptr as usize) % SIMD_ALIGNMENT == 0
    }
}

/// SIMD-aligned tensor wrapper
/// SIMDアライメントテンソルラッパー
pub struct SimdTensor<T: Float> {
    data: NonNull<T>,
    shape: Vec<usize>,
    len: usize,
}

unsafe impl<T: Float + Send> Send for SimdTensor<T> {}
unsafe impl<T: Float + Sync> Sync for SimdTensor<T> {}

impl<T: Float + Clone + 'static> SimdTensor<T> {
    /// Create new SIMD-aligned tensor (f32 only for now)
    /// 新しいSIMDアライメントテンソルを作成（現在はf32のみ）
    pub fn zeros(shape: &[usize]) -> Result<Self, std::alloc::AllocError> 
    where 
        T: 'static 
    {
        let len: usize = shape.iter().product();
        
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let ptr = SimdAllocator::alloc_f32(len)?;
            Ok(SimdTensor {
                data: unsafe { std::mem::transmute(ptr) },
                shape: shape.to_vec(),
                len,
            })
        } else {
            Err(std::alloc::AllocError) // Only f32 supported for now
        }
    }

    /// Get shape of tensor
    /// テンソルの形状を取得
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get length of tensor
    /// テンソルの長さを取得
    pub fn len(&self) -> usize {
        self.len
    }

    /// Get raw pointer to data
    /// データへの生ポインタを取得
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get mutable raw pointer to data
    /// データへの可変生ポインタを取得
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_ptr()
    }

    /// Get slice view of data
    /// データのスライスビューを取得
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    /// Get mutable slice view of data
    /// データの可変スライスビューを取得
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }

    /// Convert to regular tensor
    /// 通常のテンソルに変換
    pub fn to_tensor(&self) -> Tensor<T> {
        let data = self.as_slice().to_vec();
        Tensor::from_vec(data, self.shape.clone())
    }

    /// Check if data is properly aligned for SIMD
    /// データがSIMD用に適切にアライメントされているかチェック
    pub fn is_simd_aligned(&self) -> bool {
        SimdAllocator::is_aligned(self.data.as_ptr())
    }
}

impl Drop for SimdTensor<f32> {
    fn drop(&mut self) {
        unsafe {
            SimdAllocator::dealloc_f32(
                std::mem::transmute(self.data), 
                self.len
            );
        }
    }
}

/// SIMD-optimized operations for f32 tensors
/// f32テンソル用SIMD最適化演算
impl SimdTensor<f32> {
    /// SIMD-optimized element-wise addition
    /// SIMD最適化要素ごと加算
    pub fn add_simd(&self, other: &SimdTensor<f32>) -> ParallelResult<SimdTensor<f32>> {
        if self.shape != other.shape {
            return Err(ParallelError::shape_mismatch(
                &self.shape,
                &other.shape,
                "SIMD tensor addition"
            ));
        }

        let mut result = SimdTensor::zeros(&self.shape)
            .map_err(|_| ParallelError::parallel_execution_error("SIMD allocation failed"))?;

        let self_slice = self.as_slice();
        let other_slice = other.as_slice();
        let result_slice = result.as_mut_slice();

        // Use existing SIMD operations
        crate::simd::ops::add_optimized(self_slice, other_slice, result_slice);

        Ok(result)
    }

    /// SIMD-optimized element-wise multiplication
    /// SIMD最適化要素ごと乗算
    pub fn mul_simd(&self, other: &SimdTensor<f32>) -> ParallelResult<SimdTensor<f32>> {
        if self.shape != other.shape {
            return Err(ParallelError::shape_mismatch(
                &self.shape,
                &other.shape,
                "SIMD tensor multiplication"
            ));
        }

        let mut result = SimdTensor::zeros(&self.shape)
            .map_err(|_| ParallelError::parallel_execution_error("SIMD allocation failed"))?;

        let self_slice = self.as_slice();
        let other_slice = other.as_slice();
        let result_slice = result.as_mut_slice();

        crate::simd::ops::mul_optimized(self_slice, other_slice, result_slice);

        Ok(result)
    }

    /// SIMD-optimized scalar multiplication
    /// SIMD最適化スカラー乗算
    pub fn mul_scalar_simd(&self, scalar: f32) -> SimdTensor<f32> {
        let mut result = SimdTensor::zeros(&self.shape)
            .expect("SIMD allocation should succeed");

        let self_slice = self.as_slice();
        let result_slice = result.as_mut_slice();

        crate::simd::ops::mul_scalar_optimized(self_slice, scalar, result_slice);

        result
    }

    /// SIMD-optimized matrix multiplication
    /// SIMD最適化行列乗算
    pub fn matmul_simd(&self, other: &SimdTensor<f32>) -> ParallelResult<SimdTensor<f32>> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(ParallelError::insufficient_dimensions(
                2, self.shape.len(), "SIMD matrix multiplication"
            ));
        }

        if self.shape[1] != other.shape[0] {
            return Err(ParallelError::matmul_dimension_mismatch(
                &self.shape, &other.shape
            ));
        }

        let result_shape = vec![self.shape[0], other.shape[1]];
        let mut result = SimdTensor::zeros(&result_shape)
            .map_err(|_| ParallelError::parallel_execution_error("SIMD allocation failed"))?;

        // Use SIMD matrix multiplication
        crate::simd::vectorized::matmul_f32_simd(
            self.as_slice(), self.shape[0], self.shape[1],
            other.as_slice(), other.shape[0], other.shape[1],
            result.as_mut_slice()
        );

        Ok(result)
    }

    /// In-place SIMD operations for maximum efficiency
    /// 最大効率のためのインプレースSIMD演算
    pub fn add_assign_simd(&mut self, other: &SimdTensor<f32>) -> ParallelResult<()> {
        if self.shape != other.shape {
            return Err(ParallelError::shape_mismatch(
                &self.shape,
                &other.shape,
                "SIMD in-place addition"
            ));
        }

        let self_slice = self.as_mut_slice();
        let other_slice = other.as_slice();

        // In-place SIMD addition - need to handle borrowing correctly
        let temp_result: Vec<f32> = self_slice.iter()
            .zip(other_slice.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        self_slice.copy_from_slice(&temp_result);

        Ok(())
    }

    /// Fill tensor with value using SIMD
    /// SIMDを使用してテンソルを値で埋める
    pub fn fill_simd(&mut self, value: f32) {
        let slice = self.as_mut_slice();
        
        // Use SIMD for filling large arrays
        if slice.len() >= 8 {
            // Create temporary array filled with value for SIMD operation
            let temp = vec![value; slice.len()];
            slice.copy_from_slice(&temp);
        } else {
            slice.fill(value);
        }
    }
}

/// Extensions for regular Tensor to work with SIMD-aligned tensors
/// 通常のTensorがSIMDアライメントテンソルと連携するための拡張
impl<T: Float + Clone + 'static> Tensor<T> {
    /// Convert to SIMD-aligned tensor (f32 only)
    /// SIMDアライメントテンソルに変換（f32のみ）
    pub fn to_simd_aligned(&self) -> Result<SimdTensor<T>, std::alloc::AllocError> {
        let mut simd_tensor = SimdTensor::zeros(self.data.shape())?;
        
        if let (Some(self_slice), Some(simd_slice)) = (
            self.data.as_slice(),
            Some(simd_tensor.as_mut_slice())
        ) {
            simd_slice.copy_from_slice(self_slice);
        }
        
        Ok(simd_tensor)
    }

    /// Create tensor with SIMD-aligned allocation strategy
    /// SIMDアライメント割り当て戦略でテンソルを作成
    pub fn zeros_simd_aligned(shape: &[usize]) -> Self 
    where 
        T: 'static 
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Try to create SIMD-aligned tensor and convert back
            if let Ok(simd_tensor) = SimdTensor::<T>::zeros(shape) {
                return simd_tensor.to_tensor();
            }
        }
        
        // Fallback to regular allocation
        Self::zeros(shape)
    }
}

/// Memory pool integration for SIMD-aligned tensors
/// SIMDアライメントテンソル用メモリプール統合
pub struct SimdMemoryPool {
    pools: Vec<Vec<SimdTensor<f32>>>,
    max_pool_size: usize,
}

impl SimdMemoryPool {
    /// Create new SIMD memory pool
    /// 新しいSIMDメモリプールを作成
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: Vec::new(),
            max_pool_size,
        }
    }

    /// Get pool index based on tensor size
    /// テンソルサイズに基づいてプールインデックスを取得
    fn get_pool_index(&self, total_elements: usize) -> usize {
        if total_elements <= 64 { 0 }
        else if total_elements <= 256 { 1 }
        else if total_elements <= 1024 { 2 }
        else if total_elements <= 4096 { 3 }
        else { 4 }
    }

    /// Allocate SIMD-aligned tensor from pool
    /// プールからSIMDアライメントテンソルを割り当て
    pub fn allocate(&mut self, shape: &[usize]) -> Result<SimdTensor<f32>, std::alloc::AllocError> {
        let total_elements: usize = shape.iter().product();
        let pool_index = self.get_pool_index(total_elements);
        
        // Ensure pool exists
        while self.pools.len() <= pool_index {
            self.pools.push(Vec::new());
        }
        
        // Try to reuse from pool
        if let Some(mut tensor) = self.pools[pool_index].pop() {
            if tensor.shape() == shape {
                // Zero out the tensor for reuse
                tensor.fill_simd(0.0);
                return Ok(tensor);
            } else {
                // Put back if shape doesn't match
                self.pools[pool_index].push(tensor);
            }
        }
        
        // Create new tensor if none available
        SimdTensor::zeros(shape)
    }

    /// Return tensor to pool
    /// テンソルをプールに返却
    pub fn deallocate(&mut self, tensor: SimdTensor<f32>) {
        let total_elements = tensor.len();
        let pool_index = self.get_pool_index(total_elements);
        
        // Ensure pool exists
        while self.pools.len() <= pool_index {
            self.pools.push(Vec::new());
        }
        
        if self.pools[pool_index].len() < self.max_pool_size {
            self.pools[pool_index].push(tensor);
        }
        // Otherwise, tensor is dropped automatically
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_tensor_creation() {
        let tensor = SimdTensor::<f32>::zeros(&[4, 4]);
        assert!(tensor.is_ok());
        
        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape(), &[4, 4]);
        assert_eq!(tensor.len(), 16);
        assert!(tensor.is_simd_aligned());
    }

    #[test]
    fn test_simd_operations() {
        let mut a = SimdTensor::<f32>::zeros(&[4, 4]).unwrap();
        let mut b = SimdTensor::<f32>::zeros(&[4, 4]).unwrap();
        
        a.fill_simd(2.0);
        b.fill_simd(3.0);
        
        let result = a.add_simd(&b);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.as_slice()[0], 5.0);
    }

    #[test]
    fn test_simd_matrix_multiplication() {
        let mut a = SimdTensor::<f32>::zeros(&[2, 3]).unwrap();
        let mut b = SimdTensor::<f32>::zeros(&[3, 2]).unwrap();
        
        a.fill_simd(1.0);
        b.fill_simd(2.0);
        
        let result = a.matmul_simd(&b);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice()[0], 6.0); // 1*2 + 1*2 + 1*2 = 6
    }

    #[test]
    fn test_inplace_operations() {
        let mut a = SimdTensor::<f32>::zeros(&[3, 3]).unwrap();
        let mut b = SimdTensor::<f32>::zeros(&[3, 3]).unwrap();
        
        a.fill_simd(1.0);
        b.fill_simd(2.0);
        
        let result = a.add_assign_simd(&b);
        assert!(result.is_ok());
        
        assert_eq!(a.as_slice()[0], 3.0);
    }

    #[test]
    fn test_tensor_conversion() {
        let regular_tensor = Tensor::<f32>::ones(&[2, 2]);
        let simd_tensor = regular_tensor.to_simd_aligned();
        
        assert!(simd_tensor.is_ok());
        let simd_tensor = simd_tensor.unwrap();
        assert!(simd_tensor.is_simd_aligned());
        
        let back_to_regular = simd_tensor.to_tensor();
        assert_eq!(back_to_regular.size(), vec![2, 2]);
    }

    #[test]
    fn test_simd_memory_pool() {
        let mut pool = SimdMemoryPool::new(5);
        
        let tensor1 = pool.allocate(&[4, 4]);
        assert!(tensor1.is_ok());
        
        let tensor1 = tensor1.unwrap();
        pool.deallocate(tensor1);
        
        // Should reuse from pool
        let tensor2 = pool.allocate(&[4, 4]);
        assert!(tensor2.is_ok());
    }

    #[test]
    fn test_alignment_check() {
        let tensor = SimdTensor::<f32>::zeros(&[8, 8]).unwrap();
        assert!(SimdAllocator::is_aligned(tensor.as_ptr()));
    }
}
