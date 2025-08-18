//! Debug leaf nodes and gradient propagation

use rustorch::prelude::*;

fn main() {
    println!("=== Debug Leaf Nodes ===");
    
    // Create parameters (should be leaf nodes)
    let w = Variable::new(Tensor::from_vec(vec![2.0], vec![1, 1]), true);
    let b = Variable::new(Tensor::from_vec(vec![1.0], vec![1, 1]), true);
    
    println!("w is_leaf: {}", w.grad_fn().is_none());
    println!("b is_leaf: {}", b.grad_fn().is_none());
    
    // Test direct gradient assignment
    println!("\n=== Testing Direct Gradient Assignment ===");
    w.backward_with_grad(Some(Tensor::from_vec(vec![5.0], vec![1, 1])));
    
    {
        let w_grad_data = w.grad();
        let w_grad = w_grad_data.read().unwrap();
        println!("w grad after direct assignment: {:?}", w_grad.as_ref().map(|g| g.as_array()));
    }
    
    // Clear gradient
    w.zero_grad();
    
    // Test simple computation
    println!("\n=== Testing Simple Computation ===");
    let x = Variable::new(Tensor::from_vec(vec![3.0], vec![1, 1]), false);
    let y = &x * &w;  // y = 3 * 2 = 6
    
    println!("y value: {:?}", y.data().read().unwrap().as_array());
    println!("y is_leaf: {}", y.grad_fn().is_none());
    
    y.backward();
    
    {
        let w_grad_data = w.grad();
        let w_grad_after = w_grad_data.read().unwrap();
        println!("w grad after y.backward(): {:?}", w_grad_after.as_ref().map(|g| g.as_array()));
    }
    println!("Expected w grad: x = 3.0");
}
