//! Simple linear regression that should work with current autograd

use rustorch::prelude::*;
use rustorch::optim::{SGD, Optimizer};

fn main() {
    println!("=== Simple Linear Regression ===");
    
    // Simple data: y = 2*x + 1
    let x = Variable::new(Tensor::from_vec(vec![1.0], vec![1, 1]), false);
    let y_true = Variable::new(Tensor::from_vec(vec![3.0], vec![1, 1]), false);
    
    // Parameters
    let w = Variable::new(Tensor::from_vec(vec![0.5], vec![1, 1]), true);
    let b = Variable::new(Tensor::from_vec(vec![0.0], vec![1, 1]), true);
    
    println!("Initial w: {:?}", w.data().read().unwrap().as_array());
    println!("Initial b: {:?}", b.data().read().unwrap().as_array());
    
    // Create optimizer
    let mut optimizer = SGD::new(vec![w.clone(), b.clone()], 0.1, None, None, None, None);
    
    // Training loop
    for epoch in 0..20 {
        // Zero gradients
        optimizer.zero_grad();
        
        // Forward pass: y_pred = x * w + b
        let y_pred = &x.matmul(&w) + &b;
        
        // Loss: (y_pred - y_true)^2
        let diff = &y_pred - &y_true;
        let loss = &diff * &diff;
        
        // Backward pass
        loss.backward();
        
        // Update parameters
        optimizer.step();
        
        if epoch % 5 == 0 {
            let loss_data = loss.data();
            let loss_guard = loss_data.read().unwrap();
            let loss_val = *loss_guard.as_array().iter().next().unwrap_or(&0.0);
            drop(loss_guard);
            
            let w_data = w.data();
            let w_guard = w_data.read().unwrap();
            let w_val = *w_guard.as_array().iter().next().unwrap_or(&0.0);
            drop(w_guard);
            
            let b_data = b.data();
            let b_guard = b_data.read().unwrap();
            let b_val = *b_guard.as_array().iter().next().unwrap_or(&0.0);
            drop(b_guard);
            
            println!("Epoch {}: Loss = {:.6}, w = {:.3}, b = {:.3}", 
                epoch, loss_val, w_val, b_val);
            
            // Check gradients
            let w_grad_data = w.grad();
            let w_grad_guard = w_grad_data.read().unwrap();
            let w_grad_val = w_grad_guard.as_ref().map(|g| *g.as_array().iter().next().unwrap_or(&0.0));
            drop(w_grad_guard);
            
            let b_grad_data = b.grad();
            let b_grad_guard = b_grad_data.read().unwrap();
            let b_grad_val = b_grad_guard.as_ref().map(|g| *g.as_array().iter().next().unwrap_or(&0.0));
            drop(b_grad_guard);
            
            println!("  w_grad: {:?}", w_grad_val);
            println!("  b_grad: {:?}", b_grad_val);
        }
    }
    
    println!("\nFinal parameters:");
    println!("w: {:?} (expected: ~2.0)", w.data().read().unwrap().as_array());
    println!("b: {:?} (expected: ~1.0)", b.data().read().unwrap().as_array());
}
