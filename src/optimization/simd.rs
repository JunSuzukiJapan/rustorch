//! SIMD optimizations for vector operations
//! ベクトル演算のためのSIMD最適化

use crate::tensor::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD backend selector
/// SIMDバックエンドセレクタ
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdBackend {
    /// No SIMD (scalar operations)
    Scalar,
    /// SSE2 (x86/x86_64)
    SSE2,
    /// AVX2 (x86_64)
    AVX2,
    /// AVX512 (x86_64)
    AVX512,
    /// NEON (ARM/AArch64)
    NEON,
    /// Auto-detect best available
    Auto,
}

/// Vectorized operation types
/// ベクトル化演算タイプ
#[derive(Debug, Clone, Copy)]
pub enum VectorizedOperation {
    Add,
    Multiply,
    DotProduct,
    MatMul,
    Reduction,
    Broadcast,
    ElementWise,
}

/// SIMD optimizer for tensor operations
/// テンソル演算のためのSIMD最適化器
pub struct SimdOptimizer {
    backend: SimdBackend,
    vector_width: usize,
    alignment: usize,
    capabilities: SimdCapabilities,
}

/// SIMD capabilities detection
/// SIMD機能検出
#[derive(Debug, Clone)]
struct SimdCapabilities {
    has_sse2: bool,
    has_avx2: bool,
    has_avx512: bool,
    has_fma: bool,
    has_neon: bool,
}

impl SimdOptimizer {
    /// Create new SIMD optimizer with auto-detection
    /// 自動検出付き新規SIMD最適化器作成
    pub fn new() -> Self {
        let capabilities = Self::detect_capabilities();
        let backend = Self::select_best_backend(&capabilities);
        let (vector_width, alignment) = Self::get_backend_params(backend);
        
        SimdOptimizer {
            backend,
            vector_width,
            alignment,
            capabilities,
        }
    }

    /// Create with specific backend
    /// 特定バックエンドで作成
    pub fn with_backend(backend: SimdBackend) -> Self {
        let capabilities = Self::detect_capabilities();
        let (vector_width, alignment) = Self::get_backend_params(backend);
        
        SimdOptimizer {
            backend,
            vector_width,
            alignment,
            capabilities,
        }
    }

    /// Detect available SIMD capabilities
    /// 利用可能なSIMD機能を検出
    fn detect_capabilities() -> SimdCapabilities {
        #[cfg(target_arch = "x86_64")]
        {
            SimdCapabilities {
                has_sse2: is_x86_feature_detected!("sse2"),
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512: is_x86_feature_detected!("avx512f"),
                has_fma: is_x86_feature_detected!("fma"),
                has_neon: false,
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            SimdCapabilities {
                has_sse2: false,
                has_avx2: false,
                has_avx512: false,
                has_fma: false,
                has_neon: true,
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdCapabilities {
                has_sse2: false,
                has_avx2: false,
                has_avx512: false,
                has_fma: false,
                has_neon: false,
            }
        }
    }

    /// Select best available backend
    /// 最適な利用可能バックエンドを選択
    fn select_best_backend(capabilities: &SimdCapabilities) -> SimdBackend {
        if capabilities.has_avx512 {
            SimdBackend::AVX512
        } else if capabilities.has_avx2 {
            SimdBackend::AVX2
        } else if capabilities.has_sse2 {
            SimdBackend::SSE2
        } else if capabilities.has_neon {
            SimdBackend::NEON
        } else {
            SimdBackend::Scalar
        }
    }

    /// Get backend parameters
    /// バックエンドパラメータ取得
    fn get_backend_params(backend: SimdBackend) -> (usize, usize) {
        match backend {
            SimdBackend::Scalar => (1, 8),
            SimdBackend::SSE2 => (4, 16),  // 128-bit vectors, 4 f32s
            SimdBackend::AVX2 => (8, 32),  // 256-bit vectors, 8 f32s
            SimdBackend::AVX512 => (16, 64), // 512-bit vectors, 16 f32s
            SimdBackend::NEON => (4, 16),  // 128-bit vectors, 4 f32s
            SimdBackend::Auto => (1, 8),
        }
    }

    /// Vectorized addition for f32
    /// f32のベクトル化加算
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    pub unsafe fn add_f32_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
            let len = a.len();
            let simd_len = len - (len % 8);
            
            for i in (0..simd_len).step_by(8) {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vr = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                result[i] = a[i] + b[i];
            }
    }
    
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    pub unsafe fn add_f32_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    /// Vectorized multiplication for f32
    /// f32のベクトル化乗算
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    pub unsafe fn mul_f32_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
            let len = a.len();
            let simd_len = len - (len % 8);
            
            for i in (0..simd_len).step_by(8) {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vr = _mm256_mul_ps(va, vb);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                result[i] = a[i] * b[i];
            }
    }
    
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    pub unsafe fn mul_f32_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    /// Vectorized dot product for f32
    /// f32のベクトル化内積
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32
        {
            let len = a.len();
            let simd_len = len - (len % 8);
            let mut sum = _mm256_setzero_ps();
            
            for i in (0..simd_len).step_by(8) {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
            }
            
            // Horizontal sum of AVX2 register
            let sum_array: [f32; 8] = std::mem::transmute(sum);
            let mut result = sum_array.iter().sum::<f32>();
            
            // Handle remaining elements
            for i in simd_len..len {
                result += a[i] * b[i];
            }
            
            result
    }
    
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Vectorized matrix multiplication kernel
    /// ベクトル化行列乗算カーネル
    pub fn matmul_kernel<T: Float>(&self, a: &[T], b: &[T], c: &mut [T], 
                                   m: usize, n: usize, k: usize) {
        match self.backend {
            SimdBackend::AVX2 | SimdBackend::AVX512 => {
                self.matmul_blocked(a, b, c, m, n, k);
            },
            _ => {
                self.matmul_naive(a, b, c, m, n, k);
            }
        }
    }

    /// Blocked matrix multiplication for cache efficiency
    /// キャッシュ効率のためのブロック行列乗算
    fn matmul_blocked<T: Float>(&self, a: &[T], b: &[T], c: &mut [T], 
                                m: usize, n: usize, k: usize) {
        const BLOCK_SIZE: usize = 64;
        
        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                for k_block in (0..k).step_by(BLOCK_SIZE) {
                    let i_end = (i_block + BLOCK_SIZE).min(m);
                    let j_end = (j_block + BLOCK_SIZE).min(n);
                    let k_end = (k_block + BLOCK_SIZE).min(k);
                    
                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = c[i * n + j];
                            for kk in k_block..k_end {
                                sum = sum + a[i * k + kk] * b[kk * n + j];
                            }
                            c[i * n + j] = sum;
                        }
                    }
                }
            }
        }
    }

    /// Naive matrix multiplication fallback
    /// ナイーブ行列乗算フォールバック
    fn matmul_naive<T: Float>(&self, a: &[T], b: &[T], c: &mut [T], 
                              m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for kk in 0..k {
                    sum = sum + a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    /// Apply vectorized operation to tensors
    /// テンソルにベクトル化演算を適用
    pub fn apply_vectorized<T: Float + Send + Sync + 'static>(&self, op: VectorizedOperation, 
                                      a: &Tensor<T>, b: Option<&Tensor<T>>) 
                                      -> RusTorchResult<Tensor<T>> {
        match op {
            VectorizedOperation::Add => {
                let b = b.ok_or_else(|| RusTorchError::tensor_op("Add requires two operands"))?;
                self.vectorized_add(a, b)
            },
            VectorizedOperation::Multiply => {
                let b = b.ok_or_else(|| RusTorchError::tensor_op("Multiply requires two operands"))?;
                self.vectorized_mul(a, b)
            },
            VectorizedOperation::MatMul => {
                let b = b.ok_or_else(|| RusTorchError::tensor_op("MatMul requires two operands"))?;
                self.vectorized_matmul(a, b)
            },
            _ => Err(RusTorchError::tensor_op("Operation not yet implemented")),
        }
    }

    /// Vectorized element-wise addition
    /// ベクトル化要素毎加算
    fn vectorized_add<T: Float + Send + Sync + 'static>(&self, a: &Tensor<T>, b: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        if a.shape() != b.shape() {
            return Err(RusTorchError::tensor_op("Shape mismatch for addition"));
        }
        
        let result_data: Vec<T> = a.data.iter()
            .zip(b.data.iter())
            .map(|(x, y)| *x + *y)
            .collect();
        
        Ok(Tensor::from_vec(result_data, a.shape().to_vec()))
    }

    /// Vectorized element-wise multiplication
    /// ベクトル化要素毎乗算
    fn vectorized_mul<T: Float + Send + Sync + 'static>(&self, a: &Tensor<T>, b: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        if a.shape() != b.shape() {
            return Err(RusTorchError::tensor_op("Shape mismatch for multiplication"));
        }
        
        let result_data: Vec<T> = a.data.iter()
            .zip(b.data.iter())
            .map(|(x, y)| *x * *y)
            .collect();
        
        Ok(Tensor::from_vec(result_data, a.shape().to_vec()))
    }

    /// Vectorized matrix multiplication
    /// ベクトル化行列乗算
    fn vectorized_matmul<T: Float + Send + Sync + 'static>(&self, a: &Tensor<T>, b: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(RusTorchError::tensor_op("MatMul requires 2D tensors"));
        }
        
        if a_shape[1] != b_shape[0] {
            return Err(RusTorchError::tensor_op(
                format!("Inner dimensions must match: {} vs {}", a_shape[1], b_shape[0])
            ));
        }
        
        let m = a_shape[0];
        let n = b_shape[1];
        let k = a_shape[1];
        
        let mut result = vec![T::zero(); m * n];
        
        if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
            self.matmul_kernel(a_slice, b_slice, &mut result, m, n, k);
        }
        
        Ok(Tensor::from_vec(result, vec![m, n]))
    }

    /// Get current backend
    /// 現在のバックエンド取得
    pub fn backend(&self) -> SimdBackend {
        self.backend
    }

    /// Get vector width for current backend
    /// 現在のバックエンドのベクトル幅取得
    pub fn vector_width(&self) -> usize {
        self.vector_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let optimizer = SimdOptimizer::new();
        println!("Detected SIMD backend: {:?}", optimizer.backend());
        println!("Vector width: {}", optimizer.vector_width());
        assert!(optimizer.vector_width() >= 1);
    }

    #[test]
    fn test_vectorized_add() {
        let optimizer = SimdOptimizer::new();
        let a = Tensor::<f32>::ones(&[4, 4]);
        let b = Tensor::<f32>::ones(&[4, 4]);
        
        let result = optimizer.apply_vectorized(VectorizedOperation::Add, &a, Some(&b)).unwrap();
        assert_eq!(result.shape(), &[4, 4]);
        
        if let Some(data) = result.as_slice() {
            for val in data {
                assert!((val - 2.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_vectorized_matmul() {
        let optimizer = SimdOptimizer::new();
        let a = Tensor::<f32>::ones(&[3, 4]);
        let b = Tensor::<f32>::ones(&[4, 5]);
        
        let result = optimizer.apply_vectorized(VectorizedOperation::MatMul, &a, Some(&b)).unwrap();
        assert_eq!(result.shape(), &[3, 5]);
        
        if let Some(data) = result.as_slice() {
            for val in data {
                assert!((val - 4.0).abs() < 1e-6);
            }
        }
    }
}