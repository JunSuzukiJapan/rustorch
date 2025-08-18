//! Debug gradient computation to identify the issue

use rustorch::prelude::*;
use rustorch::nn::Linear;
use rustorch::optim::{SGD, Optimizer};
use rustorch::nn::loss::mse_loss;

fn main() {
    println!("=== Gradient Debug Test ===");
    
    // Create simple data
    let x_tensor = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
    let y_tensor = Tensor::from_vec(vec![3.0], vec![1, 1]);
    
    let x_var = Variable::new(x_tensor, false);
    let y_var = Variable::new(y_tensor, false);
    
    // Create simple linear model
    let model = Linear::new(2, 1);
    let params = model.parameters();
    
    println!("Initial parameters:");
    for (i, param) in params.iter().enumerate() {
        let data = param.data();
        let param_data = data.read().unwrap();
        println!("  Param {}: {:?}", i, param_data.as_array());
        println!("  Requires grad: {}", param.requires_grad());
    }
    
    // Forward pass
    println!("\n=== Forward Pass ===");
    let output = model.forward(&x_var);
    let output_data = output.data();
    let output_val = output_data.read().unwrap();
    println!("Output: {:?}", output_val.as_array());
    println!("Output requires grad: {}", output.requires_grad());
    
    // Compute loss
    println!("\n=== Loss Computation ===");
    let loss_var = mse_loss(&output, &y_var);
    let loss_data = loss_var.data();
    let loss_val = loss_data.read().unwrap();
    println!("Loss: {:?}", loss_val.as_array());
    println!("Loss requires grad: {}", loss_var.requires_grad());
    
    // Check gradients before backward
    println!("\n=== Before Backward ===");
    for (i, param) in params.iter().enumerate() {
        let grad = param.grad();
        let grad_data = grad.read().unwrap();
        println!("  Param {} grad: {:?}", i, grad_data.as_ref().map(|g| g.as_array()));
    }
    
    // Backward pass
    println!("\n=== Backward Pass ===");
    loss_var.backward();
    
    // Check gradients after backward
    println!("\n=== After Backward ===");
    for (i, param) in params.iter().enumerate() {
        let grad = param.grad();
        let grad_data = grad.read().unwrap();
        println!("  Param {} grad: {:?}", i, grad_data.as_ref().map(|g| g.as_array()));
    }
    
    // Test optimizer step
    println!("\n=== Optimizer Test ===");
    let mut optimizer = SGD::new(params.clone(), 0.1, None, None, None, None);
    
    println!("Before optimizer step:");
    for (i, param) in params.iter().enumerate() {
        let data = param.data();
        let param_data = data.read().unwrap();
        println!("  Param {}: {:?}", i, param_data.as_array());
    }
    
    optimizer.step();
    
    println!("After optimizer step:");
    for (i, param) in params.iter().enumerate() {
        let data = param.data();
        let param_data = data.read().unwrap();
        println!("  Param {}: {:?}", i, param_data.as_array());
    }
}
