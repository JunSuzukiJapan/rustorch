//! Performance profiler demonstration
//! パフォーマンスプロファイラーのデモンストレーション

use rustorch::autograd::Variable;
use rustorch::nn::Linear;
use rustorch::profiler::{
    clear_profiler, disable_profiler, enable_profiler, export_chrome_trace, print_profiler_report,
    ProfileContext,
};
use rustorch::tensor::Tensor;
use std::thread;
use std::time::Duration;

fn main() {
    println!("🔍 RusTorch Profiler Demo");
    println!("=========================\n");

    // Clear any previous profiling data
    clear_profiler();

    // Enable profiler
    enable_profiler();

    // Run different profiling scenarios (simplified)
    profile_basic_operations();
    profile_neural_network();
    // Skip time-intensive demos
    // profile_nested_operations();
    // profile_parallel_operations();

    // Disable profiler
    disable_profiler();

    // Print profiling report
    println!("\n📊 Profiling Report:");
    println!("====================");
    print_profiler_report();

    // Export Chrome trace
    if let Some(trace) = export_chrome_trace() {
        // Save to file for Chrome tracing
        std::fs::write("profile_trace.json", trace).expect("Failed to write trace file");
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
        let tensor1 = Tensor::<f32>::randn(&[100, 100]);
        let tensor2 = Tensor::<f32>::randn(&[100, 100]);

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
            thread::sleep(Duration::from_millis(1));

            {
                let _ctx = ProfileContext::new("layer2_forward");
                let hidden2 = layer2.forward(&hidden1);
                // Removed sleep for faster execution

                {
                    let _ctx = ProfileContext::new("layer3_forward");
                    layer3.forward(&hidden2);
                    // Removed sleep for faster execution
                }
            }
        }
    }

    println!("✓ Neural network operations profiled\n");
}
