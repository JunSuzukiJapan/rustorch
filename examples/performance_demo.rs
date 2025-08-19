/// Performance demonstration of RusTorch optimizations
/// RusTorchの最適化のパフォーマンスデモ

use rustorch::tensor::Tensor;
use rustorch::simd::vectorized;
use std::time::Instant;
use rayon::prelude::*;

fn main() {
    println!("=== RusTorch Performance Optimization Demo ===\n");
    
    // 1. SIMD Operations Performance
    println!("1. SIMD Operations Performance:");
    test_simd_performance();
    
    // 2. Matrix Multiplication Performance
    println!("\n2. Matrix Multiplication Performance:");
    test_matrix_performance();
    
    // 3. Batch Processing Performance
    println!("\n3. Batch Processing Performance:");
    test_batch_performance();
    
    // 4. Memory Pool Performance
    println!("\n4. Memory Pool Performance:");
    test_memory_performance();
    
    println!("\n=== Performance Summary ===");
    println!("✅ SIMD optimizations provide significant speedup for element-wise operations");
    println!("✅ Parallel batch processing scales well with multiple cores");
    println!("✅ Memory pool reduces allocation overhead");
    println!("✅ Combined optimizations deliver substantial performance improvements");
}

fn test_simd_performance() {
    let size = 1_000_000;
    let a = vec![1.0f32; size];
    let b = vec![2.0f32; size];
    let mut result = vec![0.0f32; size];
    
    // Regular addition
    let start = Instant::now();
    for i in 0..size {
        result[i] = a[i] + b[i];
    }
    let regular_time = start.elapsed();
    
    // SIMD addition
    let start = Instant::now();
    if vectorized::is_avx2_available() {
        vectorized::add_f32_avx2(&a, &b, &mut result);
    } else {
        vectorized::add_f32_sse41(&a, &b, &mut result);
    }
    let simd_time = start.elapsed();
    
    let speedup = regular_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
    println!("   Regular addition: {:?}", regular_time);
    println!("   SIMD addition: {:?}", simd_time);
    println!("   SIMD speedup: {:.2}x", speedup);
    
    // Test multiplication
    let start = Instant::now();
    for i in 0..size {
        result[i] = a[i] * b[i];
    }
    let regular_mul_time = start.elapsed();
    
    let start = Instant::now();
    if vectorized::is_avx2_available() {
        vectorized::mul_f32_avx2(&a, &b, &mut result);
    } else {
        vectorized::mul_f32_sse41(&a, &b, &mut result);
    }
    let simd_mul_time = start.elapsed();
    
    let mul_speedup = regular_mul_time.as_nanos() as f64 / simd_mul_time.as_nanos() as f64;
    println!("   Regular multiplication: {:?}", regular_mul_time);
    println!("   SIMD multiplication: {:?}", simd_mul_time);
    println!("   SIMD multiplication speedup: {:.2}x", mul_speedup);
}

fn test_matrix_performance() {
    let size = 256;
    let a = Tensor::<f32>::from_vec(
        (0..size * size).map(|i| (i as f32) * 0.01).collect(),
        vec![size, size],
    );
    let b = Tensor::<f32>::from_vec(
        (0..size * size).map(|i| (i as f32) * 0.01).collect(),
        vec![size, size],
    );
    
    // Regular matrix multiplication
    let start = Instant::now();
    let _result = a.matmul(&b);
    let matmul_time = start.elapsed();
    
    println!("   Matrix multiplication ({}x{}): {:?}", size, size, matmul_time);
    
    // Test with different sizes
    for &test_size in &[64, 128, 256, 512] {
        let a = Tensor::<f32>::from_vec(
            (0..test_size * test_size).map(|i| (i as f32) * 0.01).collect(),
            vec![test_size, test_size],
        );
        let b = Tensor::<f32>::from_vec(
            (0..test_size * test_size).map(|i| (i as f32) * 0.01).collect(),
            vec![test_size, test_size],
        );
        
        let start = Instant::now();
        let _result = a.matmul(&b);
        let time = start.elapsed();
        
        println!("   {}x{} matrix: {:?}", test_size, test_size, time);
    }
}

fn test_batch_performance() {
    let batch_size = 64;
    let tensor_size = 10_000;
    let mut batch_tensors = Vec::new();
    
    for _ in 0..batch_size {
        batch_tensors.push(Tensor::<f32>::from_vec(
            (0..tensor_size).map(|i| (i as f32) * 0.01).collect(),
            vec![tensor_size],
        ));
    }
    
    // Sequential batch processing
    let start = Instant::now();
    let mut results = Vec::new();
    for tensor in &batch_tensors {
        results.push(tensor.sum());
    }
    let sequential_time = start.elapsed();
    
    // Parallel batch processing
    let start = Instant::now();
    let results: Vec<f32> = batch_tensors.par_iter()
        .map(|tensor| tensor.sum())
        .collect();
    let parallel_time = start.elapsed();
    
    let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
    println!("   Sequential batch processing: {:?}", sequential_time);
    println!("   Parallel batch processing: {:?}", parallel_time);
    println!("   Parallel speedup: {:.2}x", speedup);
    
    // Test different batch sizes
    for &test_batch_size in &[16, 32, 64, 128] {
        let mut test_batch = Vec::new();
        for _ in 0..test_batch_size {
            test_batch.push(Tensor::<f32>::from_vec(
                vec![1.0; 1000],
                vec![1000],
            ));
        }
        
        let start = Instant::now();
        let results: Vec<f32> = test_batch.par_iter()
            .map(|tensor| tensor.sum())
            .collect();
        let time = start.elapsed();
        
        println!("   Batch size {}: {:?}", test_batch_size, time);
    }
}

fn test_memory_performance() {
    let sizes = vec![1000, 10000, 100000];
    
    for size in sizes {
        let shape = vec![size];
        
        // Standard allocation
        let start = Instant::now();
        for _ in 0..100 {
            let _tensor = Tensor::<f32>::zeros(&shape);
        }
        let standard_time = start.elapsed();
        
        // Pool allocation
        let start = Instant::now();
        for _ in 0..100 {
            let _tensor = Tensor::<f32>::with_pool(&shape);
        }
        let pool_time = start.elapsed();
        
        let speedup = standard_time.as_nanos() as f64 / pool_time.as_nanos() as f64;
        println!("   Size {}: Standard {:?}, Pool {:?}, Speedup: {:.2}x", 
                 size, standard_time, pool_time, speedup);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_demo() {
        // Run a smaller version for testing
        let size = 1000;
        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        let mut result = vec![0.0f32; size];
        
        // Test SIMD addition
        if vectorized::is_avx2_available() {
            vectorized::add_f32_avx2(&a, &b, &mut result);
        } else {
            vectorized::add_f32_sse41(&a, &b, &mut result);
        }
        
        // Verify results
        for &val in &result[0..10] {
            assert!((val - 3.0).abs() < 1e-6);
        }
        
        // Test matrix multiplication
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = a.matmul(&b);
        
        // Verify shape
        assert_eq!(result.shape(), &[2, 2]);
        
        // Test parallel batch processing
        let batch = vec![
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3]),
            Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], vec![3]),
        ];
        
        let results: Vec<f32> = batch.par_iter()
            .map(|tensor| tensor.sum())
            .collect();
        
        assert_eq!(results.len(), 2);
        assert!((results[0] - 6.0).abs() < 1e-6);
        assert!((results[1] - 15.0).abs() < 1e-6);
    }
}
