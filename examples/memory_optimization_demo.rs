//! Memory Optimization Demo
//! 
//! This example demonstrates advanced memory management strategies including
//! memory pooling, zero-copy operations, and SIMD-aligned allocation.

use rustorch::prelude::*;
use rustorch::tensor::{Tensor, memory_optimized::*, zero_copy::*};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 RusTorch Memory Optimization Demo");
    println!("====================================\n");

    // Demo 1: Memory Pool Operations
    println!("1. Memory Pool Operations");
    println!("-------------------------");
    
    let pool_config = MemoryOptimizedConfig {
        strategy: AllocationStrategy::Pool,
        enable_inplace: true,
        cache_block_size: 64,
        ..Default::default()
    };
    
    let tensor1 = Tensor::<f32>::ones(&[1000, 1000]);
    let tensor2 = Tensor::<f32>::ones(&[1000, 1000]);
    
    let start = Instant::now();
    let pool_result = tensor1.with_memory_strategy(&pool_config)
        .elementwise_with(&tensor2, |a, b| a + b)?;
    let pool_time = start.elapsed();
    
    println!("✅ Memory pool operation: {:?}", pool_time);
    println!("   Result shape: {:?}", pool_result.shape());
    
    // In-place operations
    let mut tensor = Tensor::<f32>::zeros(&[1000, 1000]);
    let start = Instant::now();
    tensor.inplace_add_with(&tensor2, &pool_config)?;
    let inplace_time = start.elapsed();
    
    println!("✅ In-place operation: {:?}", inplace_time);
    println!("   Modified tensor shape: {:?}", tensor.shape());

    // Demo 2: SIMD-Aligned Operations
    println!("\n2. SIMD-Aligned Operations");
    println!("--------------------------");
    
    let simd_config = MemoryOptimizedConfig {
        strategy: AllocationStrategy::SimdAligned,
        enable_vectorization: true,
        ..Default::default()
    };
    
    let simd_tensor1 = Tensor::<f32>::ones(&[10000]);
    let simd_tensor2 = Tensor::<f32>::ones(&[10000]);
    
    let start = Instant::now();
    let simd_result = simd_tensor1.with_memory_strategy(&simd_config)
        .vectorized_add(&simd_tensor2)?;
    let simd_time = start.elapsed();
    
    println!("✅ SIMD-aligned addition: {:?}", simd_time);
    println!("   Result length: {}", simd_result.len());
    
    // SIMD matrix multiplication
    let simd_matrix1 = Tensor::<f32>::ones(&[100, 100]);
    let simd_matrix2 = Tensor::<f32>::ones(&[100, 100]);
    
    let start = Instant::now();
    let simd_matmul = simd_matrix1.with_memory_strategy(&simd_config)
        .simd_matmul(&simd_matrix2)?;
    let simd_matmul_time = start.elapsed();
    
    println!("✅ SIMD matrix multiplication: {:?}", simd_matmul_time);
    println!("   Result shape: {:?}", simd_matmul.shape());

    // Demo 3: Zero-Copy Operations
    println!("\n3. Zero-Copy Operations");
    println!("-----------------------");
    
    let original_tensor = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
        vec![2, 4]
    );
    
    // Create zero-copy view
    let start = Instant::now();
    let view = original_tensor.zero_copy_view();
    let view_time = start.elapsed();
    
    println!("✅ Zero-copy view creation: {:?}", view_time);
    println!("   View shape: {:?}", view.shape());
    
    // Operations on views
    let view2 = original_tensor.zero_copy_view();
    let start = Instant::now();
    let view_result = view.elementwise_with(&view2, |a, b| a * b)?;
    let view_op_time = start.elapsed();
    
    println!("✅ Zero-copy view operation: {:?}", view_op_time);
    println!("   Result shape: {:?}", view_result.shape());

    // Demo 4: Shared Tensor Operations
    println!("\n4. Shared Tensor Operations");
    println!("---------------------------");
    
    let tensor_for_sharing = Tensor::<f32>::ones(&[100, 100]);
    
    let start = Instant::now();
    let shared = SharedTensor::from_tensor(tensor_for_sharing);
    let shared_time = start.elapsed();
    
    println!("✅ Shared tensor creation: {:?}", shared_time);
    
    // Multiple views
    let view1 = shared.view();
    let view2 = shared.view();
    let view3 = shared.view();
    
    println!("✅ Created 3 shared views");
    println!("   All views have same shape: {}", 
             view1.shape() == view2.shape() && view2.shape() == view3.shape());

    // Demo 5: Cache-Optimized Block Operations
    println!("\n5. Cache-Optimized Block Operations");
    println!("-----------------------------------");
    
    let block_config = MemoryOptimizedConfig {
        strategy: AllocationStrategy::Pool,
        cache_block_size: 64,
        enable_blocking: true,
        ..Default::default()
    };
    
    let large_matrix1 = Tensor::<f32>::ones(&[512, 512]);
    let large_matrix2 = Tensor::<f32>::ones(&[512, 512]);
    
    let start = Instant::now();
    let blocked_result = large_matrix1.with_memory_strategy(&block_config)
        .blocked_matmul(&large_matrix2)?;
    let blocked_time = start.elapsed();
    
    println!("✅ Cache-optimized blocked matmul: {:?}", blocked_time);
    println!("   Result shape: {:?}", blocked_result.shape());

    // Demo 6: Performance Comparison
    println!("\n6. Memory Strategy Performance Comparison");
    println!("-----------------------------------------");
    
    let test_size = [500, 500];
    let test_tensor1 = Tensor::<f32>::ones(&test_size);
    let test_tensor2 = Tensor::<f32>::ones(&test_size);
    
    // Direct allocation (baseline)
    let direct_config = MemoryOptimizedConfig {
        strategy: AllocationStrategy::Direct,
        ..Default::default()
    };
    
    let start = Instant::now();
    let _direct_result = test_tensor1.with_memory_strategy(&direct_config)
        .elementwise_with(&test_tensor2, |a, b| a + b)?;
    let direct_time = start.elapsed();
    
    // Pool allocation
    let start = Instant::now();
    let _pool_result = test_tensor1.with_memory_strategy(&pool_config)
        .elementwise_with(&test_tensor2, |a, b| a + b)?;
    let pool_time = start.elapsed();
    
    // SIMD-aligned allocation
    let start = Instant::now();
    let _simd_result = test_tensor1.with_memory_strategy(&simd_config)
        .vectorized_add(&test_tensor2)?;
    let simd_time = start.elapsed();
    
    println!("📊 Performance Results:");
    println!("   Direct allocation: {:?}", direct_time);
    println!("   Pool allocation: {:?}", pool_time);
    println!("   SIMD-aligned: {:?}", simd_time);
    
    let pool_speedup = direct_time.as_nanos() as f64 / pool_time.as_nanos() as f64;
    let simd_speedup = direct_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
    
    println!("📈 Speedup ratios:");
    println!("   Pool vs Direct: {:.2}x", pool_speedup);
    println!("   SIMD vs Direct: {:.2}x", simd_speedup);

    // Demo 7: Memory Usage Statistics
    println!("\n7. Memory Strategy Comparison Table");
    println!("-----------------------------------");
    println!("| Strategy     | Best For                    | Memory Overhead | Performance Gain |");
    println!("|--------------|-----------------------------|-----------------|--------------------|");
    println!("| Pool         | Frequent small allocations | Low             | 1.5-2x           |");
    println!("| Direct       | Large one-time allocations | Minimal         | Baseline         |");
    println!("| ZeroCopy     | View operations            | None            | Eliminates copies |");
    println!("| SimdAligned  | Vectorized operations      | 32-byte align   | 2-4x             |");

    println!("\n🎉 Memory optimization demo completed successfully!");
    println!("💡 Choose the right strategy based on your workload characteristics");
    
    Ok(())
}
