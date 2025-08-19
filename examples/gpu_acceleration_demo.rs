//! GPU Acceleration Demo
//! 
//! This example demonstrates GPU-accelerated tensor operations with automatic
//! device selection and fallback to CPU when GPU is unavailable.

use rustorch::prelude::*;
use rustorch::tensor::{Tensor, gpu_parallel::*};
use rustorch::gpu::DeviceType;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 RusTorch GPU Acceleration Demo");
    println!("==================================\n");

    // Create large tensors for demonstration
    let size = 1000;
    let tensor1 = Tensor::<f32>::ones(&[size, size]);
    let tensor2 = Tensor::<f32>::ones(&[size, size]);
    
    println!("📊 Created {}x{} tensors for GPU acceleration demo", size, size);

    // Demo 1: Basic GPU Operations
    println!("\n1. Basic GPU Operations");
    println!("-----------------------");
    
    let start = Instant::now();
    let gpu_add_result = tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b)?;
    let gpu_add_time = start.elapsed();
    
    println!("✅ GPU element-wise addition: {:?}", gpu_add_time);
    println!("   Result shape: {:?}", gpu_add_result.shape());
    
    let start = Instant::now();
    let gpu_matmul_result = tensor1.gpu_matmul(&tensor2)?;
    let gpu_matmul_time = start.elapsed();
    
    println!("✅ GPU matrix multiplication: {:?}", gpu_matmul_time);
    println!("   Result shape: {:?}", gpu_matmul_result.shape());

    // Demo 2: Device Management
    println!("\n2. Device Management");
    println!("--------------------");
    
    // Try to transfer to GPU (will fallback to CPU if no GPU available)
    match tensor1.to_device(DeviceType::Cuda(0)) {
        Ok(gpu_tensor) => {
            println!("✅ Successfully transferred tensor to CUDA device 0");
            
            // Perform operations on GPU
            let gpu_result = gpu_tensor.gpu_elementwise_op(&gpu_tensor, |a, b| a * 2.0)?;
            println!("✅ Performed GPU operations on device tensor");
            
            // Transfer back to CPU
            let cpu_result = gpu_result.to_cpu()?;
            println!("✅ Transferred result back to CPU");
            println!("   Final result shape: {:?}", cpu_result.shape());
        }
        Err(_) => {
            println!("ℹ️  No CUDA device available, operations will use CPU fallback");
        }
    }

    // Demo 3: Batch GPU Operations
    println!("\n3. Batch GPU Operations");
    println!("-----------------------");
    
    let batch_size = 32;
    let matrix_size = 128;
    let batch_tensor1 = Tensor::<f32>::ones(&[batch_size, matrix_size, matrix_size]);
    let batch_tensor2 = Tensor::<f32>::ones(&[batch_size, matrix_size, matrix_size]);
    
    println!("📊 Created batch tensors: {}x{}x{}", batch_size, matrix_size, matrix_size);
    
    let start = Instant::now();
    let batch_result = batch_tensor1.gpu_batch_matmul(&batch_tensor2)?;
    let batch_time = start.elapsed();
    
    println!("✅ GPU batch matrix multiplication: {:?}", batch_time);
    println!("   Batch result shape: {:?}", batch_result.shape());
    
    let start = Instant::now();
    let normalized = batch_tensor1.gpu_batch_normalize()?;
    let norm_time = start.elapsed();
    
    println!("✅ GPU batch normalization: {:?}", norm_time);
    println!("   Normalized shape: {:?}", normalized.shape());

    // Demo 4: GPU Execution Strategies
    println!("\n4. GPU Execution Strategies");
    println!("----------------------------");
    
    // Configure different execution strategies
    let strategies = vec![
        ("CPU Parallel", GpuExecutionStrategy::CpuParallel),
        ("GPU Preferred", GpuExecutionStrategy::GpuPreferred { fallback_threshold: 10000 }),
        ("Hybrid", GpuExecutionStrategy::Hybrid { gpu_threshold: 50000, cpu_threads: 4 }),
        ("Auto", GpuExecutionStrategy::Auto),
    ];
    
    for (name, strategy) in strategies {
        let tensor_size = 100000;
        let gpu_available = true; // Simulate GPU availability
        let should_use_gpu = strategy.should_use_gpu(tensor_size, gpu_available);
        
        println!("📋 {}: Should use GPU for {} elements: {}", 
                 name, tensor_size, should_use_gpu);
    }

    // Demo 5: Device Selection
    println!("\n5. Automatic Device Selection");
    println!("------------------------------");
    
    match select_optimal_device(&[1000, 1000]) {
        Ok(device) => {
            println!("🎯 Optimal device selected: {:?}", device);
        }
        Err(e) => {
            println!("ℹ️  Device selection fallback: {}", e);
        }
    }

    // Demo 6: Performance Comparison
    println!("\n6. Performance Comparison");
    println!("-------------------------");
    
    let test_tensor1 = Tensor::<f32>::ones(&[500, 500]);
    let test_tensor2 = Tensor::<f32>::ones(&[500, 500]);
    
    // CPU parallel operation
    let start = Instant::now();
    let _cpu_result = test_tensor1.batch_elementwise_op(&test_tensor2, |a, b| a + b)?;
    let cpu_time = start.elapsed();
    
    // GPU operation (with fallback)
    let start = Instant::now();
    let _gpu_result = test_tensor1.gpu_elementwise_op(&test_tensor2, |a, b| a + b)?;
    let gpu_time = start.elapsed();
    
    println!("⚡ CPU Parallel: {:?}", cpu_time);
    println!("🎮 GPU (with fallback): {:?}", gpu_time);
    
    let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    println!("📈 Speedup ratio: {:.2}x", speedup);

    println!("\n🎉 GPU acceleration demo completed successfully!");
    println!("💡 Note: Actual GPU performance depends on hardware availability");
    
    Ok(())
}
