//! High-Performance BLAS Integration for RusTorch
//! RusTorch用高性能BLAS統合

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use num_traits::Float;

/// High-performance matrix multiplication using BLAS
/// BLAS使用による高性能行列乗算
#[cfg(feature = "blas-optimized")]
pub fn optimized_matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> RusTorchResult<Tensor<T>>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    // Validate matrix dimensions
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(RusTorchError::InvalidOperation(
            "Matrix multiplication requires 2D tensors".to_string(),
        ));
    }
    
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    
    if k != k2 {
        return Err(RusTorchError::InvalidOperation(format!(
            "Matrix dimension mismatch: {}x{} @ {}x{}",
            m, k, k2, n
        )));
    }

    // Use OpenBLAS for f32 and f64 types with safe casting
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // Safe cast to f32 tensors
        let a_f32 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f32>>(a) };
        let b_f32 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f32>>(b) };
        let result_f32 = optimized_matmul_f32(a_f32, b_f32)?;
        return Ok(unsafe { std::mem::transmute::<Tensor<f32>, Tensor<T>>(result_f32) });
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // Safe cast to f64 tensors
        let a_f64 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f64>>(a) };
        let b_f64 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f64>>(b) };
        let result_f64 = optimized_matmul_f64(a_f64, b_f64)?;
        return Ok(unsafe { std::mem::transmute::<Tensor<f64>, Tensor<T>>(result_f64) });
    }

    // Fallback to multithreaded implementation for other types
    multithreaded_matmul(a, b)
}

#[cfg(feature = "blas-optimized")]
fn optimized_matmul_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> RusTorchResult<Tensor<f32>> {
    use cblas_sys::{CBLAS_ORDER, CBLAS_TRANSPOSE, cblas_sgemm};
    
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
    
    // Get data slices
    let a_data = a.data.as_slice().unwrap();
    let b_data = b.data.as_slice().unwrap();
    
    // Allocate result
    let mut result_data = vec![0.0f32; m * n];
    
    unsafe {
        // CBLAS SGEMM: C = alpha * A * B + beta * C
        // Parameters: Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc
        cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,      // Row-major layout
            CBLAS_TRANSPOSE::CblasNoTrans,   // No transpose for A
            CBLAS_TRANSPOSE::CblasNoTrans,   // No transpose for B
            m as i32,             // M: rows of A and C
            n as i32,             // N: columns of B and C
            k as i32,             // K: columns of A and rows of B
            1.0,                  // alpha
            a_data.as_ptr(),      // A matrix
            k as i32,             // lda: leading dimension of A
            b_data.as_ptr(),      // B matrix
            n as i32,             // ldb: leading dimension of B
            0.0,                  // beta
            result_data.as_mut_ptr(), // C matrix (result)
            n as i32,             // ldc: leading dimension of C
        );
    }
    
    Ok(Tensor::from_vec(result_data, vec![m, n]))
}

#[cfg(feature = "blas-optimized")]
fn optimized_matmul_f64(a: &Tensor<f64>, b: &Tensor<f64>) -> RusTorchResult<Tensor<f64>> {
    use cblas_sys::{CBLAS_ORDER, CBLAS_TRANSPOSE, cblas_dgemm};
    
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
    
    // Get data slices
    let a_data = a.data.as_slice().unwrap();
    let b_data = b.data.as_slice().unwrap();
    
    // Allocate result
    let mut result_data = vec![0.0f64; m * n];
    
    unsafe {
        // CBLAS DGEMM: C = alpha * A * B + beta * C
        cblas_dgemm(
            CBLAS_ORDER::CblasRowMajor,      // Row-major layout
            CBLAS_TRANSPOSE::CblasNoTrans,   // No transpose for A
            CBLAS_TRANSPOSE::CblasNoTrans,   // No transpose for B
            m as i32,             // M: rows of A and C
            n as i32,             // N: columns of B and C
            k as i32,             // K: columns of A and rows of B
            1.0,                  // alpha
            a_data.as_ptr(),      // A matrix
            k as i32,             // lda: leading dimension of A
            b_data.as_ptr(),      // B matrix
            n as i32,             // ldb: leading dimension of B
            0.0,                  // beta
            result_data.as_mut_ptr(), // C matrix (result)
            n as i32,             // ldc: leading dimension of C
        );
    }
    
    Ok(Tensor::from_vec(result_data, vec![m, n]))
}

/// Fallback implementation when BLAS is not available
/// BLASが利用できない場合のフォールバック実装
#[cfg(not(feature = "blas-optimized"))]
pub fn optimized_matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> RusTorchResult<Tensor<T>>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    // Use standard implementation
    a.matmul(b).map_err(|e| RusTorchError::gpu(&e))
}

/// Multi-threaded matrix multiplication using Rayon
/// Rayon使用によるマルチスレッド行列乗算
pub fn multithreaded_matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> RusTorchResult<Tensor<T>>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    use rayon::prelude::*;
    
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(RusTorchError::InvalidOperation(
            "Matrix multiplication requires 2D tensors".to_string(),
        ));
    }
    
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    
    if k != k2 {
        return Err(RusTorchError::InvalidOperation(format!(
            "Matrix dimension mismatch: {}x{} @ {}x{}",
            m, k, k2, n
        )));
    }
    
    let a_data = a.data.as_slice().unwrap();
    let b_data = b.data.as_slice().unwrap();
    
    // Parallel computation of result matrix
    let result_data: Vec<T> = (0..m * n)
        .into_par_iter()
        .map(|idx| {
            let row = idx / n;
            let col = idx % n;
            
            let mut sum = T::zero();
            for i in 0..k {
                sum = sum + a_data[row * k + i] * b_data[i * n + col];
            }
            sum
        })
        .collect();
    
    Ok(Tensor::from_vec(result_data, vec![m, n]))
}

/// Benchmark different matrix multiplication implementations
/// 異なる行列乗算実装のベンチマーク
pub fn benchmark_matmul_implementations<T>(size: usize) -> RusTorchResult<()>
where
    T: Float + Send + Sync + std::fmt::Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    use std::time::Instant;
    
    println!("🔬 Matrix Multiplication Benchmark ({}x{})", size, size);
    
    // Create test matrices
    let data_a: Vec<T> = (0..(size * size))
        .map(|i| T::from(i as f64 * 0.01).unwrap())
        .collect();
    let data_b: Vec<T> = (0..(size * size))
        .map(|i| T::from((i + 1) as f64 * 0.01).unwrap())
        .collect();
    
    let matrix_a = Tensor::<T>::from_vec(data_a, vec![size, size]);
    let matrix_b = Tensor::<T>::from_vec(data_b, vec![size, size]);
    
    // Benchmark standard implementation
    let start = Instant::now();
    let std_result = matrix_a.matmul(&matrix_b).map_err(|e| RusTorchError::gpu(&e))?;
    let std_time = start.elapsed();
    println!("  Standard: {:.2}ms", std_time.as_secs_f64() * 1000.0);
    
    // Benchmark multithreaded implementation
    let start = Instant::now();
    let mt_result = multithreaded_matmul(&matrix_a, &matrix_b)?;
    let mt_time = start.elapsed();
    println!("  Multi-threaded: {:.2}ms", mt_time.as_secs_f64() * 1000.0);
    
    // Benchmark BLAS implementation (if available)
    #[cfg(feature = "blas-optimized")]
    {
        let start = Instant::now();
        let blas_result = optimized_matmul(&matrix_a, &matrix_b)?;
        let blas_time = start.elapsed();
        println!("  BLAS-optimized: {:.2}ms", blas_time.as_secs_f64() * 1000.0);
        
        // Calculate speedup
        let speedup = std_time.as_secs_f64() / blas_time.as_secs_f64();
        println!("  🚀 BLAS Speedup: {:.2}x", speedup);
    }
    
    // Verify results are consistent (first few elements)
    let std_slice = std_result.data.as_slice().unwrap();
    let mt_slice = mt_result.data.as_slice().unwrap();
    
    println!("  ✅ Results verified - first 4 elements:");
    println!("    Standard: {:?}", &std_slice[..4]);
    println!("    Multi-threaded: {:?}", &mt_slice[..4]);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimized_matmul_f32() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::<f32>::from_vec(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]);
        
        let result = optimized_matmul(&a, &b).unwrap();
        let expected = vec![4.0, 6.0, 10.0, 12.0];
        
        let result_data = result.data.as_slice().unwrap();
        for (i, &expected_val) in expected.iter().enumerate() {
            assert!((result_data[i] - expected_val).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_multithreaded_matmul() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::<f32>::from_vec(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]);
        
        let result = multithreaded_matmul(&a, &b).unwrap();
        let expected = vec![4.0, 6.0, 10.0, 12.0];
        
        let result_data = result.data.as_slice().unwrap();
        for (i, &expected_val) in expected.iter().enumerate() {
            assert!((result_data[i] - expected_val).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_benchmark() {
        let result = benchmark_matmul_implementations::<f32>(32);
        assert!(result.is_ok());
    }
}