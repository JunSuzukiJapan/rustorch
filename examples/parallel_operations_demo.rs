//! Parallel Operations Demo
//! 
//! This example demonstrates the unified parallel tensor operations system
//! with automatic parallelization, SIMD acceleration, and configurable strategies.

use rustorch::prelude::*;
use rustorch::tensor::{Tensor, parallel_traits::*};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 RusTorch Parallel Operations Demo");
    println!("====================================\n");

    // Demo 1: Basic Parallel Operations
    println!("1. Basic Parallel Operations");
    println!("----------------------------");
    
    let tensor1 = Tensor::<f32>::ones(&[1000, 1000]);
    let tensor2 = Tensor::<f32>::ones(&[1000, 1000]);
    
    println!("📊 Created {}x{} tensors for parallel operations", 1000, 1000);
    
    // Automatic parallel execution for large tensors
    let start = Instant::now();
    let result = tensor1.batch_elementwise_op(&tensor2, |a, b| a + b)?;
    let parallel_time = start.elapsed();
    
    println!("✅ Parallel element-wise addition: {:?}", parallel_time);
    println!("   Result shape: {:?}", result.shape());
    
    // Parallel matrix multiplication
    let start = Instant::now();
    let matmul_result = tensor1.batch_matmul(&tensor2)?;
    let matmul_time = start.elapsed();
    
    println!("✅ Parallel matrix multiplication: {:?}", matmul_time);
    println!("   Result shape: {:?}", matmul_result.shape());

    // Demo 2: Parallel Reduction Operations
    println!("\n2. Parallel Reduction Operations");
    println!("--------------------------------");
    
    let large_tensor = Tensor::<f32>::from_vec(
        (0..1000000).map(|i| (i as f32) * 0.001).collect(),
        vec![1000, 1000]
    );
    
    println!("📊 Created tensor with 1M elements for reduction operations");
    
    let start = Instant::now();
    let sum = large_tensor.parallel_sum(0)?;
    let sum_time = start.elapsed();
    
    println!("✅ Parallel sum: {:?}", sum_time);
    println!("   Sum result shape: {:?}", sum.shape());
    
    let start = Instant::now();
    let mean = large_tensor.parallel_mean(0)?;
    let mean_time = start.elapsed();
    
    println!("✅ Parallel mean: {:?}", mean_time);
    println!("   Mean result shape: {:?}", mean.shape());
    
    let start = Instant::now();
    let variance = large_tensor.parallel_variance(0)?;
    let var_time = start.elapsed();
    
    println!("✅ Parallel variance: {:?}", var_time);
    println!("   Variance result shape: {:?}", variance.shape());

    // Demo 3: SIMD-Optimized Parallel Operations
    println!("\n3. SIMD-Optimized Parallel Operations");
    println!("-------------------------------------");
    
    let simd_tensor1 = Tensor::<f32>::ones(&[10000]);
    let simd_tensor2 = Tensor::<f32>::ones(&[10000]);
    
    println!("📊 Created f32 tensors with 10K elements for SIMD operations");
    
    // SIMD-accelerated parallel addition
    let start = Instant::now();
    let simd_add = simd_tensor1.simd_parallel_add(&simd_tensor2)?;
    let simd_add_time = start.elapsed();
    
    println!("✅ SIMD parallel addition: {:?}", simd_add_time);
    println!("   Result length: {}", simd_add.len());
    
    // SIMD-accelerated parallel matrix multiplication
    let simd_matrix1 = Tensor::<f32>::ones(&[100, 100]);
    let simd_matrix2 = Tensor::<f32>::ones(&[100, 100]);
    
    let start = Instant::now();
    let simd_matmul = simd_matrix1.simd_parallel_matmul(&simd_matrix2)?;
    let simd_matmul_time = start.elapsed();
    
    println!("✅ SIMD parallel matrix multiplication: {:?}", simd_matmul_time);
    println!("   Result shape: {:?}", simd_matmul.shape());

    // Demo 4: Configurable Parallel Execution
    println!("\n4. Configurable Parallel Execution");
    println!("----------------------------------");
    
    // Configure parallel execution strategy
    let configs = vec![
        ("Auto (default)", ParallelConfig::default()),
        ("Force Parallel", ParallelConfig {
            strategy: ParallelStrategy::ForceParallel,
            chunk_size: 2048,
            num_threads: Some(4),
        }),
        ("Sequential", ParallelConfig {
            strategy: ParallelStrategy::Sequential,
            chunk_size: 1024,
            num_threads: None,
        }),
    ];
    
    let test_tensor = Tensor::<f32>::ones(&[5000]);
    
    for (name, config) in configs {
        let context = ParallelContext::new(config);
        let should_parallelize = context.should_parallelize(test_tensor.len());
        
        println!("📋 {}: Should parallelize {} elements: {}", 
                 name, test_tensor.len(), should_parallelize);
    }

    // Demo 5: Batch Processing Operations
    println!("\n5. Batch Processing Operations");
    println!("------------------------------");
    
    let batch_size = 32;
    let matrix_size = 64;
    let batch_tensor1 = Tensor::<f32>::ones(&[batch_size, matrix_size, matrix_size]);
    let batch_tensor2 = Tensor::<f32>::ones(&[batch_size, matrix_size, matrix_size]);
    
    println!("📊 Created batch tensors: {}x{}x{}", batch_size, matrix_size, matrix_size);
    
    // Batch matrix multiplication
    let start = Instant::now();
    let batch_matmul = batch_tensor1.batch_matmul(&batch_tensor2)?;
    let batch_matmul_time = start.elapsed();
    
    println!("✅ Batch matrix multiplication: {:?}", batch_matmul_time);
    println!("   Batch result shape: {:?}", batch_matmul.shape());
    
    // Batch element-wise operations
    let start = Instant::now();
    let batch_add = batch_tensor1.batch_elementwise_op(&batch_tensor2, |a, b| a + b)?;
    let batch_add_time = start.elapsed();
    
    println!("✅ Batch element-wise addition: {:?}", batch_add_time);
    println!("   Batch result shape: {:?}", batch_add.shape());

    // Demo 6: Performance Comparison
    println!("\n6. Performance Comparison: Sequential vs Parallel");
    println!("------------------------------------------------");
    
    let test_size = [500, 500];
    let test_tensor1 = Tensor::<f32>::ones(&test_size);
    let test_tensor2 = Tensor::<f32>::ones(&test_size);
    
    // Sequential execution
    let sequential_config = ParallelConfig {
        strategy: ParallelStrategy::Sequential,
        ..Default::default()
    };
    
    let context = ParallelContext::new(sequential_config);
    let start = Instant::now();
    // Simulate sequential operation (using regular tensor ops)
    let _seq_result = &test_tensor1 + &test_tensor2;
    let sequential_time = start.elapsed();
    
    // Parallel execution
    let start = Instant::now();
    let _par_result = test_tensor1.batch_elementwise_op(&test_tensor2, |a, b| a + b)?;
    let parallel_time = start.elapsed();
    
    println!("⚡ Sequential operation: {:?}", sequential_time);
    println!("🔄 Parallel operation: {:?}", parallel_time);
    
    let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
    println!("📈 Parallel speedup: {:.2}x", speedup);

    // Demo 7: Thread Utilization
    println!("\n7. Thread Utilization Analysis");
    println!("------------------------------");
    
    let available_threads = rayon::current_num_threads();
    println!("🧵 Available threads: {}", available_threads);
    
    // Test with different tensor sizes
    let sizes = vec![
        (100, "Small (10K elements)"),
        (500, "Medium (250K elements)"),
        (1000, "Large (1M elements)"),
        (2000, "Extra Large (4M elements)"),
    ];
    
    for (size, description) in sizes {
        let tensor = Tensor::<f32>::ones(&[size, size]);
        let context = ParallelContext::new(ParallelConfig::default());
        let should_parallelize = context.should_parallelize(tensor.len());
        
        println!("📊 {}: Parallelize = {}", description, should_parallelize);
    }

    // Demo 8: Error Handling
    println!("\n8. Parallel Operations Error Handling");
    println!("-------------------------------------");
    
    // Test shape mismatch error
    let tensor_a = Tensor::<f32>::ones(&[100, 50]);
    let tensor_b = Tensor::<f32>::ones(&[100, 60]);  // Different shape
    
    match tensor_a.batch_elementwise_op(&tensor_b, |a, b| a + b) {
        Ok(_) => println!("✅ Operation succeeded"),
        Err(e) => println!("❌ Expected error caught: {}", e),
    }
    
    // Test dimension mismatch for matrix multiplication
    let matrix_a = Tensor::<f32>::ones(&[50, 100]);
    let matrix_b = Tensor::<f32>::ones(&[50, 100]);  // Wrong dimensions for matmul
    
    match matrix_a.batch_matmul(&matrix_b) {
        Ok(_) => println!("✅ Matrix multiplication succeeded"),
        Err(e) => println!("❌ Expected error caught: {}", e),
    }

    println!("\n🎉 Parallel operations demo completed successfully!");
    println!("💡 Parallel operations automatically optimize based on tensor size and available resources");
    
    Ok(())
}
