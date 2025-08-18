//! Manual linear regression implementation to test basic gradient flow

use rustorch::prelude::*;
use rustorch::optim::{SGD, Optimizer};

fn main() {
    println!("=== Manual Linear Regression ===");
    
    // Create simple data: y = 2*x + 1
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let y_data = vec![3.0, 5.0, 7.0, 9.0]; // y = 2*x + 1
    
    let x = Variable::new(Tensor::from_vec(x_data, vec![4, 1]), false);
    let y = Variable::new(Tensor::from_vec(y_data, vec![4, 1]), false);
    
    // Initialize parameters manually
    let w = Variable::new(Tensor::from_vec(vec![0.5], vec![1, 1]), true);
    let b = Variable::new(Tensor::from_vec(vec![0.0], vec![1]), true);
    
    println!("Initial w: {:?}", w.data().read().unwrap().as_array());
    println!("Initial b: {:?}", b.data().read().unwrap().as_array());
    
    // Create optimizer
    let mut optimizer = SGD::new(vec![w.clone(), b.clone()], 0.01, None, None, None, None);
    
    // Training loop
    for epoch in 0..50 {
        // Zero gradients
        optimizer.zero_grad();
        
        // Manual forward pass: y_pred = x * w + b
        let x_w = x.matmul(&w);
        let y_pred = &x_w + &Variable::new(
            Tensor::ones(&[4, 1]), false
        ).matmul(&Variable::new(
            Tensor::from_vec(vec![*b.data().read().unwrap().as_array().iter().next().unwrap()], vec![1, 1]), 
            false
        ));
        
        // Compute loss: MSE = mean((y_pred - y)^2)
        let diff = &y_pred - &y;
        let squared_diff = &diff * &diff;
        let mean_factor = Variable::new(Tensor::from_vec(vec![0.25], vec![]), false);
        let loss = &squared_diff.sum() * &mean_factor;
        
        // Backward pass
        loss.backward();
        
        // Update parameters
        optimizer.step();
        
        if epoch % 10 == 0 {
            let loss_data = loss.data();
            let loss_val = loss_data.read().unwrap();
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val.as_array().iter().next().unwrap_or(&0.0));
            println!("  w: {:?}", w.data().read().unwrap().as_array());
            println!("  b: {:?}", b.data().read().unwrap().as_array());
            println!("  w_grad: {:?}", w.grad().read().unwrap().as_ref().map(|g| g.as_array()));
            println!("  b_grad: {:?}", b.grad().read().unwrap().as_ref().map(|g| g.as_array()));
        }
    }
    
    println!("\nFinal parameters:");
    println!("w: {:?} (expected: ~2.0)", w.data().read().unwrap().as_array());
    println!("b: {:?} (expected: ~1.0)", b.data().read().unwrap().as_array());
}
