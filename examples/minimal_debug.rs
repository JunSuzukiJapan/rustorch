//! Minimal debug to trace gradient flow

use rustorch::prelude::*;

fn main() {
    println!("=== Minimal Debug ===");

    // Create a simple variable
    let x = Variable::new(Tensor::from_vec(vec![2.0], vec![1]), true);
    println!(
        "x: {:?}, requires_grad: {}",
        x.data().read().unwrap().as_array(),
        x.requires_grad()
    );

    // Test sum operation
    let sum_result = x.sum();
    println!(
        "sum_result: {:?}, requires_grad: {}",
        sum_result.data().read().unwrap().as_array(),
        sum_result.requires_grad()
    );

    // Check if gradient function exists (can't access private field directly)

    // Test backward
    println!("\n=== Testing Backward ===");
    println!(
        "Before backward - x grad: {:?}",
        x.grad().read().unwrap().as_ref().map(|g| g.as_array())
    );

    sum_result.backward();

    println!(
        "After backward - x grad: {:?}",
        x.grad().read().unwrap().as_ref().map(|g| g.as_array())
    );

    // Check sum_result gradient
    println!(
        "sum_result grad: {:?}",
        sum_result
            .grad()
            .read()
            .unwrap()
            .as_ref()
            .map(|g| g.as_array())
    );
}
