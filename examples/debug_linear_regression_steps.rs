//! Debug each step of linear regression computation

use rustorch::prelude::*;

fn main() {
    println!("=== Debug Linear Regression Steps ===");
    
    // Data: y = 2*x + 1, so x=1 -> y=3
    let x = Variable::new(Tensor::from_vec(vec![1.0], vec![1, 1]), false);
    let y_true = Variable::new(Tensor::from_vec(vec![3.0], vec![1, 1]), false);
    
    // Parameters
    let w = Variable::new(Tensor::from_vec(vec![0.5], vec![1, 1]), true);
    let b = Variable::new(Tensor::from_vec(vec![0.0], vec![1, 1]), true);
    
    println!("x: {:?}, is_leaf: {}", x.data().read().unwrap().as_array(), x.grad_fn().is_none());
    println!("w: {:?}, is_leaf: {}", w.data().read().unwrap().as_array(), w.grad_fn().is_none());
    println!("b: {:?}, is_leaf: {}", b.data().read().unwrap().as_array(), b.grad_fn().is_none());
    
    // Step 1: matmul
    println!("\n=== Step 1: y_pred = x.matmul(&w) ===");
    let matmul_result = x.matmul(&w);
    println!("matmul_result: {:?}, is_leaf: {}", 
        matmul_result.data().read().unwrap().as_array(), 
        matmul_result.grad_fn().is_none());
    
    // Step 2: add bias
    println!("\n=== Step 2: y_pred = matmul_result + b ===");
    let y_pred = &matmul_result + &b;
    println!("y_pred: {:?}, is_leaf: {}", 
        y_pred.data().read().unwrap().as_array(), 
        y_pred.grad_fn().is_none());
    
    // Step 3: subtract target
    println!("\n=== Step 3: diff = y_pred - y_true ===");
    let diff = &y_pred - &y_true;
    println!("diff: {:?}, is_leaf: {}", 
        diff.data().read().unwrap().as_array(), 
        diff.grad_fn().is_none());
    
    // Step 4: square
    println!("\n=== Step 4: loss = diff * diff ===");
    let loss = &diff * &diff;
    println!("loss: {:?}, is_leaf: {}", 
        loss.data().read().unwrap().as_array(), 
        loss.grad_fn().is_none());
    
    // Test backward propagation
    println!("\n=== Testing Backward Propagation ===");
    w.zero_grad();
    b.zero_grad();
    
    println!("Before backward:");
    {
        let w_grad_data = w.grad();
        let w_grad = w_grad_data.read().unwrap();
        let b_grad_data = b.grad();
        let b_grad = b_grad_data.read().unwrap();
        println!("  w_grad: {:?}", w_grad.as_ref().map(|g| g.as_array()));
        println!("  b_grad: {:?}", b_grad.as_ref().map(|g| g.as_array()));
    }
    
    loss.backward();
    
    println!("After backward:");
    {
        let w_grad_data = w.grad();
        let w_grad = w_grad_data.read().unwrap();
        let b_grad_data = b.grad();
        let b_grad = b_grad_data.read().unwrap();
        println!("  w_grad: {:?}", w_grad.as_ref().map(|g| g.as_array()));
        println!("  b_grad: {:?}", b_grad.as_ref().map(|g| g.as_array()));
    }
    
    // Expected gradients:
    // loss = (y_pred - y_true)^2 = (x*w + b - y_true)^2
    // d_loss/d_w = 2 * (x*w + b - y_true) * x = 2 * (0.5 + 0 - 3) * 1 = 2 * (-2.5) = -5
    // d_loss/d_b = 2 * (x*w + b - y_true) * 1 = 2 * (-2.5) = -5
    println!("Expected w_grad: -5.0");
    println!("Expected b_grad: -5.0");
}
