//! Performance profiler demonstration
//! パフォーマンスプロファイラーのデモンストレーション

use rustorch::profiler::{
    enable_profiler, disable_profiler, print_profiler_report, 
    clear_profiler, export_chrome_trace, ProfileContext
};
use rustorch::tensor::Tensor;
use rustorch::nn::{Module, Linear};
use rustorch::autograd::Variable;
use std::thread;
use std::time::Duration;

fn main() {
    println!("🔍 RusTorch Profiler Demo");
    println!("=========================\n");

    // Clear any previous profiling data
    clear_profiler();
    
    // Enable profiler
    enable_profiler();
    
    // Run different profiling scenarios
    profile_basic_operations();
    profile_neural_network();
    profile_nested_operations();
    profile_parallel_operations();
    
    // Disable profiler
    disable_profiler();
    
    // Print profiling report
    println!("\n📊 Profiling Report:");
    println!("====================");
    print_profiler_report();
    
    // Export Chrome trace
    if let Some(trace) = export_chrome_trace() {
        // Save to file for Chrome tracing
        std::fs::write("profile_trace.json", trace)
            .expect("Failed to write trace file");
        println!("\n✅ Chrome trace saved to profile_trace.json");
        println!("   Open chrome://tracing and load the file to visualize");
    }
}

/// Profile basic tensor operations
fn profile_basic_operations() {
    println!("1️⃣ Profiling Basic Tensor Operations");
    println!("-------------------------------------");
    
    {
        let _ctx = ProfileContext::new("tensor_creation");
        let tensor1 = Tensor::<f32>::randn(&[1000, 1000]);
        let tensor2 = Tensor::<f32>::randn(&[1000, 1000]);
        
        {
            let _ctx = ProfileContext::new("tensor_addition");
            let _result = &tensor1 + &tensor2;
        }
        
        {
            let _ctx = ProfileContext::new("tensor_multiplication");
            let _result = &tensor1 * &tensor2;
        }
        
        {
            let _ctx = ProfileContext::new("matrix_multiplication");
            let _result = tensor1.matmul(&tensor2);
        }
    }
    
    println!("✓ Basic operations profiled\n");
}

/// Profile neural network operations
fn profile_neural_network() {
    println!("2️⃣ Profiling Neural Network Operations");
    println!("---------------------------------------");
    
    let _ctx = ProfileContext::new("neural_network");
    
    // Create simple layers
    let layer1: Linear<f32> = Linear::new(784, 256);
    let layer2: Linear<f32> = Linear::new(256, 128);
    let layer3: Linear<f32> = Linear::new(128, 10);
    
    // Forward pass
    {
        let _ctx = ProfileContext::new("forward_pass");
        let input = Variable::new(Tensor::<f32>::randn(&[32, 784]), false);
        
        {
            let _ctx = ProfileContext::new("layer1_forward");
            let hidden1 = layer1.forward(&input);
            // Simulate additional processing
            thread::sleep(Duration::from_millis(5));
            
            {
                let _ctx = ProfileContext::new("layer2_forward");
                let hidden2 = layer2.forward(&hidden1);
                thread::sleep(Duration::from_millis(3));
                
                {
                    let _ctx = ProfileContext::new("layer3_forward");
                    let _output = layer3.forward(&hidden2);
                    thread::sleep(Duration::from_millis(2));
                }
            }
        }
    }
    
    println!("✓ Neural network operations profiled\n");
}

/// Profile nested operations
fn profile_nested_operations() {
    println!("3️⃣ Profiling Nested Operations");
    println!("-------------------------------");
    
    let _ctx = ProfileContext::new("nested_operations");
    
    {
        let _ctx = ProfileContext::new("outer_operation");
        
        for i in 0..3 {
            let _ctx = ProfileContext::new(&format!("iteration_{}", i));
            
            {
                let _ctx = ProfileContext::new("inner_computation");
                let tensor = Tensor::<f32>::randn(&[100, 100]);
                let _result = tensor.sum();
            }
            
            thread::sleep(Duration::from_millis(2));
        }
    }
    
    println!("✓ Nested operations profiled\n");
}

/// Profile parallel operations
fn profile_parallel_operations() {
    println!("4️⃣ Profiling Parallel Operations");
    println!("---------------------------------");
    
    let _ctx = ProfileContext::new("parallel_operations");
    
    // Use the profile! macro
    rustorch::profile!("macro_profiled_operation", {
        let tensors: Vec<_> = (0..4).map(|i| {
            let _ctx = ProfileContext::new(&format!("parallel_tensor_{}", i));
            Tensor::<f32>::randn(&[500, 500])
        }).collect();
        
        // Parallel reduction
        let _ctx = ProfileContext::new("parallel_reduction");
        let _sum: f32 = tensors.iter()
            .map(|t| t.sum())
            .sum();
    });
    
    println!("✓ Parallel operations profiled\n");
}

/// Demonstrate memory profiling
fn profile_memory_operations() {
    println!("5️⃣ Profiling Memory Operations");
    println!("-------------------------------");
    
    let _ctx = ProfileContext::new("memory_operations");
    
    // Large allocation
    {
        let _ctx = ProfileContext::new("large_allocation");
        let _large_tensor = Tensor::<f32>::zeros(&[10000, 10000]);
    }
    
    // Many small allocations
    {
        let _ctx = ProfileContext::new("small_allocations");
        let _small_tensors: Vec<_> = (0..1000)
            .map(|_| Tensor::<f32>::zeros(&[10, 10]))
            .collect();
    }
    
    println!("✓ Memory operations profiled\n");
}