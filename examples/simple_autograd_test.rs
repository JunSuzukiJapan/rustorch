//! Simple test to verify basic autograd operations

use rustorch::prelude::*;

fn main() {
    println!("=== Simple Autograd Test ===");

    // Create simple variables
    let x = Variable::new(Tensor::from_vec(vec![2.0], vec![1]), true);
    let y = Variable::new(Tensor::from_vec(vec![3.0], vec![1]), true);

    println!(
        "x: {:?}, requires_grad: {}",
        x.data().read().unwrap().as_array(),
        x.requires_grad()
    );
    println!(
        "y: {:?}, requires_grad: {}",
        y.data().read().unwrap().as_array(),
        y.requires_grad()
    );

    // Test subtraction
    println!("\n=== Testing Subtraction ===");
    let diff = &x - &y;
    println!(
        "diff = x - y: {:?}, requires_grad: {}",
        diff.data().read().unwrap().as_array(),
        diff.requires_grad()
    );

    // Test multiplication
    println!("\n=== Testing Multiplication ===");
    let squared = &diff * &diff;
    println!(
        "squared = diff * diff: {:?}, requires_grad: {}",
        squared.data().read().unwrap().as_array(),
        squared.requires_grad()
    );

    // Test sum
    println!("\n=== Testing Sum ===");
    let sum_result = squared.sum();
    println!(
        "sum: {:?}, requires_grad: {}",
        sum_result.data().read().unwrap().as_array(),
        sum_result.requires_grad()
    );

    // Test backward
    println!("\n=== Testing Backward ===");
    println!(
        "Before backward - x grad: {:?}",
        x.grad().read().unwrap().as_ref().map(|g| g.as_array())
    );
    println!(
        "Before backward - y grad: {:?}",
        y.grad().read().unwrap().as_ref().map(|g| g.as_array())
    );

    sum_result.backward();

    println!(
        "After backward - x grad: {:?}",
        x.grad().read().unwrap().as_ref().map(|g| g.as_array())
    );
    println!(
        "After backward - y grad: {:?}",
        y.grad().read().unwrap().as_ref().map(|g| g.as_array())
    );
}
