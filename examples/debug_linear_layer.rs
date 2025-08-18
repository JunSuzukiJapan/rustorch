//! Debug Linear layer gradient propagation

use rustorch::prelude::*;
use rustorch::nn::Linear;
use rustorch::nn::loss::mse_loss;

fn main() {
    println!("=== Debug Linear Layer ===");
    
    // Simple data: 2 features -> 1 output
    let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]), false);
    let y_true = Variable::new(Tensor::from_vec(vec![5.0], vec![1, 1]), false);
    
    // Create linear layer
    let model = Linear::new(2, 1);
    let params = model.parameters();
    
    println!("Initial parameters:");
    for (i, param) in params.iter().enumerate() {
        let data_binding = param.data();
        let data = data_binding.read().unwrap();
        println!("  param[{}]: {:?}, requires_grad: {}, is_leaf: {}", 
            i, data.as_array(), param.requires_grad(), param.grad_fn().is_none());
    }
    
    // Forward pass
    let output = model.forward(&x);
    println!("\nForward pass:");
    let output_binding = output.data();
    let output_data = output_binding.read().unwrap();
    println!("  output: {:?}, requires_grad: {}, is_leaf: {}", 
        output_data.as_array(), 
        output.requires_grad(), 
        output.grad_fn().is_none());
    drop(output_data);
    drop(output_binding);
    
    // Compute loss
    let loss = mse_loss(&output, &y_true);
    let loss_binding = loss.data();
    let loss_data = loss_binding.read().unwrap();
    println!("  loss: {:?}, requires_grad: {}, is_leaf: {}", 
        loss_data.as_array(), 
        loss.requires_grad(), 
        loss.grad_fn().is_none());
    drop(loss_data);
    drop(loss_binding);
    
    // Clear gradients
    for param in &params {
        param.zero_grad();
    }
    
    println!("\nBefore backward:");
    for (i, param) in params.iter().enumerate() {
        let grad_data = param.grad();
        let grad = grad_data.read().unwrap();
        println!("  param[{}] grad: {:?}", i, grad.as_ref().map(|g| g.as_array()));
    }
    
    // Backward pass
    loss.backward();
    
    println!("\nAfter backward:");
    for (i, param) in params.iter().enumerate() {
        let grad_data = param.grad();
        let grad = grad_data.read().unwrap();
        println!("  param[{}] grad: {:?}", i, grad.as_ref().map(|g| g.as_array()));
    }
}
