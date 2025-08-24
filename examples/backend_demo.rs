//! Backend abstraction demonstration
//! バックエンド抽象化のデモンストレーション
//!
//! This example demonstrates the new unified backend architecture
//! for RusTorch tensor operations.
//! 
//! この例はRusTorchテンソル操作の新しい統一バックエンドアーキテクチャを
//! デモンストレーションします。

use rustorch::backends::{BackendFactory, ComputeBackend};
use rustorch::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RusTorch Backend Abstraction Demo");
    println!("=====================================");
    
    // Create CPU backend
    println!("\n📊 Creating CPU backend...");
    let backend = BackendFactory::create_cpu_backend()?;
    let device_info = backend.device_info();
    
    println!("✅ Backend created successfully!");
    println!("   Device: {}", device_info.name);
    println!("   Type: {}", device_info.device_type);
    println!("   Max threads: {}", device_info.max_threads);
    println!("   Supports f64: {}", device_info.supports_f64);
    println!("   Supports f16: {}", device_info.supports_f16);
    println!("   Total memory: {} MB", device_info.total_memory / 1024 / 1024);
    
    // Create sample tensors
    println!("\n🔢 Creating sample tensors...");
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    
    println!("Tensor A (2x2):");
    println!("{}", a);
    println!("Tensor B (2x2):");
    println!("{}", b);
    
    // Test basic operations using the backend
    println!("\n⚡ Testing backend operations...");
    
    // Addition
    println!("\n➕ Addition: A + B");
    let add_result = backend.add(&a, &b)?;
    println!("Result: {}", add_result);
    
    // Subtraction  
    println!("\n➖ Subtraction: A - B");
    let sub_result = backend.sub(&a, &b)?;
    println!("Result: {}", sub_result);
    
    // Multiplication
    println!("\n✖️  Element-wise multiplication: A * B");
    let mul_result = backend.mul(&a, &b)?;
    println!("Result: {}", mul_result);
    
    // Division
    println!("\n➗ Element-wise division: A / B");
    let div_result = backend.div(&a, &b)?;
    println!("Result: {}", div_result);
    
    // Matrix multiplication
    println!("\n🔢 Matrix multiplication: A @ B");
    let matmul_result = backend.matmul(&a, &b)?;
    println!("Result: {}", matmul_result);
    
    // Reduction operations
    println!("\n📈 Reduction operations on A:");
    
    let sum_result = backend.sum(&a)?;
    println!("Sum: {}", sum_result.as_slice().unwrap()[0]);
    
    let mean_result = backend.mean(&a)?;
    println!("Mean: {}", mean_result.as_slice().unwrap()[0]);
    
    let max_result = backend.max(&a)?;
    println!("Max: {}", max_result.as_slice().unwrap()[0]);
    
    let min_result = backend.min(&a)?;
    println!("Min: {}", min_result.as_slice().unwrap()[0]);
    
    // Shape operations
    println!("\n🔄 Shape operations:");
    let reshaped = backend.reshape(&a, &[4, 1])?;
    println!("Reshaped to 4x1:");
    println!("{}", reshaped);
    
    let transposed = backend.transpose(&a, &[1, 0])?;
    println!("Transposed:");
    println!("{}", transposed);
    
    // Activation functions
    println!("\n🧠 Activation functions on [-1, 0, 1]:");
    let activation_test = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0], vec![3]);
    
    let relu_result = backend.relu(&activation_test)?;
    println!("ReLU: {:?}", relu_result.as_slice().unwrap());
    
    let sigmoid_result = backend.sigmoid(&activation_test)?;
    println!("Sigmoid: {:?}", sigmoid_result.as_slice().unwrap());
    
    let tanh_result = backend.tanh(&activation_test)?;
    println!("Tanh: {:?}", tanh_result.as_slice().unwrap());
    
    // Memory operations
    println!("\n💾 Memory operations:");
    let buffer = backend.allocate_memory(1024)?;
    println!("Allocated buffer: {} bytes", buffer.size());
    
    // Synchronization
    backend.synchronize()?;
    println!("✅ Backend synchronized");
    
    println!("\n🎉 Backend abstraction demo completed successfully!");
    println!("    This demonstrates the unified compute backend architecture");
    println!("    that will support CPU, CUDA, Metal, and OpenCL backends.");
    
    Ok(())
}