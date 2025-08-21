//! AVX-512 SIMD optimizations for high-performance tensor operations
//! 高性能テンソル演算のためのAVX-512 SIMD最適化

use super::Tensor;
use super::parallel_errors::{ParallelError, ParallelResult};
use num_traits::Float;
use std::arch::x86_64::*;

/// AVX-512 alignment requirements (64 bytes)
/// AVX-512アライメント要件（64バイト）
pub const AVX512_ALIGNMENT: usize = 64;

/// AVX-512 vector size for f32 (16 elements)
/// f32用のAVX-512ベクトルサイズ（16要素）
pub const AVX512_F32_LANES: usize = 16;

/// AVX-512 vector size for f64 (8 elements)
/// f64用のAVX-512ベクトルサイズ（8要素）
pub const AVX512_F64_LANES: usize = 8;

/// Check if AVX-512 is available at runtime
/// 実行時にAVX-512が利用可能かチェック
pub fn is_avx512_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX-512 optimized operations for f32 tensors
/// f32テンソル用のAVX-512最適化演算
pub struct Avx512F32Ops;

impl Avx512F32Ops {
    /// Element-wise addition using AVX-512
    /// AVX-512を使用した要素ごとの加算
    #[target_feature(enable = "avx512f")]
    pub unsafe fn add_vectorized(a: &[f32], b: &[f32], result: &mut [f32]) -> ParallelResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![a.len()],
                actual: vec![b.len(), result.len()],
            });
        }

        let len = a.len();
        let chunks = len / AVX512_F32_LANES;
        let remainder = len % AVX512_F32_LANES;

        // Process 16 elements at a time with AVX-512
        // AVX-512で一度に16要素を処理
        for i in 0..chunks {
            let offset = i * AVX512_F32_LANES;
            
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            let vresult = _mm512_add_ps(va, vb);
            
            _mm512_storeu_ps(result.as_mut_ptr().add(offset), vresult);
        }

        // Handle remaining elements
        // 残りの要素を処理
        for i in (chunks * AVX512_F32_LANES)..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Element-wise multiplication using AVX-512
    /// AVX-512を使用した要素ごとの乗算
    #[target_feature(enable = "avx512f")]
    pub unsafe fn mul_vectorized(a: &[f32], b: &[f32], result: &mut [f32]) -> ParallelResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![a.len()],
                actual: vec![b.len(), result.len()],
            });
        }

        let len = a.len();
        let chunks = len / AVX512_F32_LANES;

        for i in 0..chunks {
            let offset = i * AVX512_F32_LANES;
            
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            let vresult = _mm512_mul_ps(va, vb);
            
            _mm512_storeu_ps(result.as_mut_ptr().add(offset), vresult);
        }

        // Handle remainder
        for i in (chunks * AVX512_F32_LANES)..len {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    /// Fused multiply-add using AVX-512
    /// AVX-512を使用した融合積和演算
    #[target_feature(enable = "avx512f")]
    pub unsafe fn fma_vectorized(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) -> ParallelResult<()> {
        if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![a.len()],
                actual: vec![b.len(), c.len(), result.len()],
            });
        }

        let len = a.len();
        let chunks = len / AVX512_F32_LANES;

        for i in 0..chunks {
            let offset = i * AVX512_F32_LANES;
            
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            let vc = _mm512_loadu_ps(c.as_ptr().add(offset));
            let vresult = _mm512_fmadd_ps(va, vb, vc); // a * b + c
            
            _mm512_storeu_ps(result.as_mut_ptr().add(offset), vresult);
        }

        // Handle remainder
        for i in (chunks * AVX512_F32_LANES)..len {
            result[i] = a[i] * b[i] + c[i];
        }

        Ok(())
    }

    /// Dot product using AVX-512
    /// AVX-512を使用した内積
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_product_vectorized(a: &[f32], b: &[f32]) -> ParallelResult<f32> {
        if a.len() != b.len() {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![a.len()],
                actual: vec![b.len()],
            });
        }

        let len = a.len();
        let chunks = len / AVX512_F32_LANES;
        let mut sum_vec = _mm512_setzero_ps();

        // Vectorized accumulation
        // ベクトル化された累積
        for i in 0..chunks {
            let offset = i * AVX512_F32_LANES;
            
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            sum_vec = _mm512_fmadd_ps(va, vb, sum_vec);
        }

        // Horizontal sum of the vector
        // ベクトルの水平和
        let sum = _mm512_reduce_add_ps(sum_vec);

        // Handle remainder
        let mut remainder_sum = sum;
        for i in (chunks * AVX512_F32_LANES)..len {
            remainder_sum += a[i] * b[i];
        }

        Ok(remainder_sum)
    }

    /// Matrix multiplication using AVX-512 with blocking
    /// ブロッキングを使用したAVX-512行列乗算
    #[target_feature(enable = "avx512f")]
    pub unsafe fn matmul_blocked(
        a: &[f32], a_rows: usize, a_cols: usize,
        b: &[f32], b_rows: usize, b_cols: usize,
        c: &mut [f32]
    ) -> ParallelResult<()> {
        if a_cols != b_rows {
            return Err(ParallelError::MatrixMultiplicationError {
                a_shape: vec![a_rows, a_cols],
                b_shape: vec![b_rows, b_cols],
                message: "Inner dimensions must match".to_string(),
            });
        }

        if c.len() != a_rows * b_cols {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![a_rows * b_cols],
                actual: vec![c.len()],
            });
        }

        // Block sizes optimized for cache
        // キャッシュ最適化されたブロックサイズ
        const BLOCK_SIZE: usize = 64;

        // Initialize result matrix
        // 結果行列を初期化
        c.fill(0.0);

        // Blocked matrix multiplication
        // ブロック化行列乗算
        for i_block in (0..a_rows).step_by(BLOCK_SIZE) {
            for j_block in (0..b_cols).step_by(BLOCK_SIZE) {
                for k_block in (0..a_cols).step_by(BLOCK_SIZE) {
                    let i_end = (i_block + BLOCK_SIZE).min(a_rows);
                    let j_end = (j_block + BLOCK_SIZE).min(b_cols);
                    let k_end = (k_block + BLOCK_SIZE).min(a_cols);

                    // Process block
                    for i in i_block..i_end {
                        for k in k_block..k_end {
                            let a_val = _mm512_set1_ps(a[i * a_cols + k]);
                            
                            let j_chunks = (j_end - j_block) / AVX512_F32_LANES;
                            
                            // Vectorized inner loop
                            for j_chunk in 0..j_chunks {
                                let j = j_block + j_chunk * AVX512_F32_LANES;
                                
                                let b_vec = _mm512_loadu_ps(&b[k * b_cols + j]);
                                let c_vec = _mm512_loadu_ps(&c[i * b_cols + j]);
                                let result = _mm512_fmadd_ps(a_val, b_vec, c_vec);
                                
                                _mm512_storeu_ps(&mut c[i * b_cols + j], result);
                            }
                            
                            // Handle remaining elements in j dimension
                            for j in (j_block + j_chunks * AVX512_F32_LANES)..j_end {
                                c[i * b_cols + j] += a[i * a_cols + k] * b[k * b_cols + j];
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// AVX-512 optimized operations for f64 tensors
/// f64テンソル用のAVX-512最適化演算
pub struct Avx512F64Ops;

impl Avx512F64Ops {
    /// Element-wise addition using AVX-512 for f64
    /// f64用のAVX-512を使用した要素ごとの加算
    #[target_feature(enable = "avx512f")]
    pub unsafe fn add_vectorized(a: &[f64], b: &[f64], result: &mut [f64]) -> ParallelResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![a.len()],
                actual: vec![b.len(), result.len()],
            });
        }

        let len = a.len();
        let chunks = len / AVX512_F64_LANES;

        for i in 0..chunks {
            let offset = i * AVX512_F64_LANES;
            
            let va = _mm512_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm512_loadu_pd(b.as_ptr().add(offset));
            let vresult = _mm512_add_pd(va, vb);
            
            _mm512_storeu_pd(result.as_mut_ptr().add(offset), vresult);
        }

        // Handle remainder
        for i in (chunks * AVX512_F64_LANES)..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Dot product using AVX-512 for f64
    /// f64用のAVX-512を使用した内積
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_product_vectorized(a: &[f64], b: &[f64]) -> ParallelResult<f64> {
        if a.len() != b.len() {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![a.len()],
                actual: vec![b.len()],
            });
        }

        let len = a.len();
        let chunks = len / AVX512_F64_LANES;
        let mut sum_vec = _mm512_setzero_pd();

        for i in 0..chunks {
            let offset = i * AVX512_F64_LANES;
            
            let va = _mm512_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm512_loadu_pd(b.as_ptr().add(offset));
            sum_vec = _mm512_fmadd_pd(va, vb, sum_vec);
        }

        let sum = _mm512_reduce_add_pd(sum_vec);

        // Handle remainder
        let mut remainder_sum = sum;
        for i in (chunks * AVX512_F64_LANES)..len {
            remainder_sum += a[i] * b[i];
        }

        Ok(remainder_sum)
    }
}

/// High-level AVX-512 tensor operations
/// 高レベルAVX-512テンソル演算
pub struct Avx512TensorOps;

impl Avx512TensorOps {
    /// Perform element-wise addition with automatic fallback
    /// 自動フォールバック付き要素ごと加算
    pub fn add_tensors<T: Float + 'static>(
        a: &Tensor<T>, 
        b: &Tensor<T>
    ) -> ParallelResult<Tensor<T>> {
        if a.shape() != b.shape() {
            return Err(ParallelError::ShapeMismatch {
                expected: a.shape().to_vec(),
                actual: b.shape().to_vec(),
            });
        }

        let mut result = Tensor::zeros(a.shape());

        // Try AVX-512 for f32
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && is_avx512_available() {
            let a_data = unsafe { std::slice::from_raw_parts(a.data().as_ptr() as *const f32, a.data().len()) };
            let b_data = unsafe { std::slice::from_raw_parts(b.data().as_ptr() as *const f32, b.data().len()) };
            let result_data = unsafe { std::slice::from_raw_parts_mut(result.data_mut().as_mut_ptr() as *mut f32, result.data().len()) };

            unsafe {
                Avx512F32Ops::add_vectorized(a_data, b_data, result_data)?;
            }
        } else {
            // Fallback to scalar operations
            // スカラー演算にフォールバック
            for i in 0..a.data().len() {
                result.data_mut()[i] = a.data()[i] + b.data()[i];
            }
        }

        Ok(result)
    }

    /// Perform matrix multiplication with AVX-512 optimization
    /// AVX-512最適化付き行列乗算
    pub fn matmul_optimized(
        a: &Tensor<f32>, 
        b: &Tensor<f32>
    ) -> ParallelResult<Tensor<f32>> {
        if a.shape().len() != 2 || b.shape().len() != 2 {
            return Err(ParallelError::DimensionError {
                expected: 2,
                actual: a.shape().len().max(b.shape().len()),
            });
        }

        let (a_rows, a_cols) = (a.shape()[0], a.shape()[1]);
        let (b_rows, b_cols) = (b.shape()[0], b.shape()[1]);

        if a_cols != b_rows {
            return Err(ParallelError::MatrixMultiplicationError {
                a_shape: vec![a_rows, a_cols],
                b_shape: vec![b_rows, b_cols],
                message: "Inner dimensions must match".to_string(),
            });
        }

        let mut result = Tensor::zeros(&[a_rows, b_cols]);

        if is_avx512_available() {
            unsafe {
                Avx512F32Ops::matmul_blocked(
                    a.data(), a_rows, a_cols,
                    b.data(), b_rows, b_cols,
                    result.data_mut()
                )?;
            }
        } else {
            // Fallback implementation
            for i in 0..a_rows {
                for j in 0..b_cols {
                    let mut sum = 0.0;
                    for k in 0..a_cols {
                        sum += a.data()[i * a_cols + k] * b.data()[k * b_cols + j];
                    }
                    result.data_mut()[i * b_cols + j] = sum;
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx512_availability() {
        // This will vary by system
        // システムによって異なる
        println!("AVX-512 available: {}", is_avx512_available());
    }

    #[test]
    fn test_avx512_tensor_add() {
        let a = Tensor::ones(&[1000]);
        let b = Tensor::ones(&[1000]);
        
        let result = Avx512TensorOps::add_tensors(&a, &b).unwrap();
        
        // All elements should be 2.0
        // 全要素が2.0になるはず
        for &val in result.data() {
            assert!((val - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_avx512_matmul() {
        let a: Tensor<f32> = Tensor::ones(&[100, 50]);
        let b: Tensor<f32> = Tensor::ones(&[50, 75]);
        
        let result = Avx512TensorOps::matmul_optimized(&a, &b).unwrap();
        
        assert_eq!(result.shape(), &[100, 75]);
        
        // Each element should be 50.0 (sum of 50 ones)
        // 各要素は50.0になるはず（50個の1の和）
        for &val in result.data() {
            assert!((val - 50.0).abs() < 1e-6);
        }
    }
}
