/// Showcase of RusTorch performance optimizations
/// RusTorchパフォーマンス最適化のショーケース

use rustorch::tensor::Tensor;
use rustorch::simd::vectorized;
use std::time::Instant;
use rayon::prelude::*;

fn main() {
    println!("=== RusTorch Performance Optimization Showcase ===\n");
    
    // Test SIMD availability
    println!("SIMD Support:");
    println!("  AVX2 available: {}", vectorized::is_avx2_available());
    println!("  SSE4.1 available: {}", vectorized::is_sse41_available());
    println!();
    
    // 1. SIMD Element-wise Operations
    demo_simd_operations();
    
    // 2. Matrix Operations
    demo_matrix_operations();
    
    // 3. Parallel Batch Processing
    demo_parallel_batch_processing();
    
    // 4. Memory Pool Performance
    demo_memory_pool();
    
    println!("=== Optimization Summary ===");
    println!("✅ SIMD vectorization for element-wise operations");
    println!("✅ Parallel batch processing with Rayon");
    println!("✅ Memory pool for efficient allocation");
    println!("✅ GPU infrastructure foundation ready");
    println!("✅ Comprehensive test coverage");
}

fn demo_simd_operations() {
    println!("1. SIMD Element-wise Operations:");
    
    let size = 100_000;
    let a = vec![1.5f32; size];
    let b = vec![2.5f32; size];
    let mut result = vec![0.0f32; size];
    
    // Regular addition
    let start = Instant::now();
    for i in 0..size {
        result[i] = a[i] + b[i];
    }
    let regular_time = start.elapsed();
    
    // SIMD addition
    let start = Instant::now();
    unsafe {
        if vectorized::is_avx2_available() {
            vectorized::add_f32_avx2(&a, &b, &mut result);
        } else if vectorized::is_sse41_available() {
            vectorized::add_f32_sse41(&a, &b, &mut result);
        } else {
            // Fallback to regular addition
            for i in 0..size {
                result[i] = a[i] + b[i];
            }
        }
    }
    let simd_time = start.elapsed();
    
    let speedup = if simd_time.as_nanos() > 0 {
        regular_time.as_nanos() as f64 / simd_time.as_nanos() as f64
    } else {
        1.0
    };
    
    println!("   Regular addition: {:?}", regular_time);
    println!("   SIMD addition: {:?}", simd_time);
    println!("   Speedup: {:.2}x", speedup);
    
    // Verify correctness
    let expected = 4.0f32; // 1.5 + 2.5
    let actual = result[0];
    println!("   Correctness check: expected {}, got {}", expected, actual);
    assert!((actual - expected).abs() < 1e-6);
    println!();
}

fn demo_matrix_operations() {
    println!("2. Matrix Operations:");
    
    let sizes = vec![64, 128, 256];
    
    for size in sizes {
        let a = Tensor::<f32>::from_vec(
            (0..size * size).map(|i| (i as f32) * 0.01).collect(),
            vec![size, size],
        );
        let b = Tensor::<f32>::from_vec(
            (0..size * size).map(|i| (i as f32) * 0.01 + 1.0).collect(),
            vec![size, size],
        );
        
        let start = Instant::now();
        let result = a.matmul(&b);
        let time = start.elapsed();
        
        println!("   {}x{} matrix multiplication: {:?}", size, size, time);
        
        // Verify shape
        assert_eq!(result.shape(), &[size, size]);
    }
    println!();
}

fn demo_parallel_batch_processing() {
    println!("3. Parallel Batch Processing:");
    
    let batch_sizes = vec![16, 32, 64];
    let tensor_size = 10_000;
    
    for batch_size in batch_sizes {
        let mut batch_tensors = Vec::new();
        for i in 0..batch_size {
            batch_tensors.push(Tensor::<f32>::from_vec(
                vec![(i as f32 + 1.0) * 0.1; tensor_size],
                vec![tensor_size],
            ));
        }
        
        // Sequential processing
        let start = Instant::now();
        let mut sequential_results = Vec::new();
        for tensor in &batch_tensors {
            sequential_results.push(tensor.sum());
        }
        let sequential_time = start.elapsed();
        
        // Parallel processing
        let start = Instant::now();
        let parallel_results: Vec<f32> = batch_tensors.par_iter()
            .map(|tensor| tensor.sum())
            .collect();
        let parallel_time = start.elapsed();
        
        let speedup = if parallel_time.as_nanos() > 0 {
            sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64
        } else {
            1.0
        };
        
        println!("   Batch size {}: Sequential {:?}, Parallel {:?}, Speedup: {:.2}x", 
                 batch_size, sequential_time, parallel_time, speedup);
        
        // Verify correctness
        assert_eq!(sequential_results.len(), parallel_results.len());
        for (seq, par) in sequential_results.iter().zip(parallel_results.iter()) {
            assert!((seq - par).abs() < 1e-6);
        }
    }
    println!();
}

fn demo_memory_pool() {
    println!("4. Memory Pool Performance:");
    
    let sizes = vec![1000, 10000, 50000];
    let iterations = 50;
    
    for size in sizes {
        let shape = vec![size];
        
        // Standard allocation
        let start = Instant::now();
        for _ in 0..iterations {
            let _tensor = Tensor::<f32>::zeros(&shape);
        }
        let standard_time = start.elapsed();
        
        // Pool allocation
        let start = Instant::now();
        for _ in 0..iterations {
            let _tensor = Tensor::<f32>::with_pool(&shape);
        }
        let pool_time = start.elapsed();
        
        let speedup = if pool_time.as_nanos() > 0 {
            standard_time.as_nanos() as f64 / pool_time.as_nanos() as f64
        } else {
            1.0
        };
        
        println!("   Size {}: Standard {:?}, Pool {:?}, Speedup: {:.2}x", 
                 size, standard_time, pool_time, speedup);
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_showcase() {
        // Test SIMD operations
        let a = vec![1.0f32; 1000];
        let b = vec![2.0f32; 1000];
        let mut result = vec![0.0f32; 1000];
        
        unsafe {
            if vectorized::is_avx2_available() {
                vectorized::add_f32_avx2(&a, &b, &mut result);
            } else if vectorized::is_sse41_available() {
                vectorized::add_f32_sse41(&a, &b, &mut result);
            }
        }
        
        // Verify first few results
        for &val in &result[0..10] {
            assert!((val - 3.0).abs() < 1e-6);
        }
        
        // Test matrix operations
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = a.matmul(&b);
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
        
        // Test memory pool
        let shape = vec![1000];
        let _tensor1 = Tensor::<f32>::zeros(&shape);
        let _tensor2 = Tensor::<f32>::with_pool(&shape);
    }
}
