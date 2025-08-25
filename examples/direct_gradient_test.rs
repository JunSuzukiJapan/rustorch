//! Direct gradient test with simple operations

use rustorch::prelude::*;

fn main() {
    println!("=== Direct Gradient Test ===");

    // Test matmul gradient
    let w = Variable::new(Tensor::from_vec(vec![2.0], vec![1, 1]), true);
    let x = Variable::new(Tensor::from_vec(vec![3.0], vec![1, 1]), false);

    println!(
        "w: {:?}, requires_grad: {}",
        w.data().read().unwrap().as_array(),
        w.requires_grad()
    );
    println!(
        "x: {:?}, requires_grad: {}",
        x.data().read().unwrap().as_array(),
        x.requires_grad()
    );

    // Test matmul: y = x @ w
    let y = x.matmul(&w);
    println!(
        "y = x @ w: {:?}, requires_grad: {}",
        y.data().read().unwrap().as_array(),
        y.requires_grad()
    );

    // Backward
    println!("\n=== Testing Matmul Backward ===");
    println!(
        "Before backward - w grad: {:?}",
        w.grad().read().unwrap().as_ref().map(|g| g.as_array())
    );

    y.backward();

    println!(
        "After backward - w grad: {:?}",
        w.grad().read().unwrap().as_ref().map(|g| g.as_array())
    );
    println!("Expected w grad: x = 3.0");

    // Test with loss
    println!("\n=== Testing with Loss ===");
    let w2 = Variable::new(Tensor::from_vec(vec![1.0], vec![1, 1]), true);
    let x2 = Variable::new(Tensor::from_vec(vec![2.0], vec![1, 1]), false);
    let target = Variable::new(Tensor::from_vec(vec![5.0], vec![1, 1]), false);

    w2.zero_grad();

    let pred = x2.matmul(&w2); // pred = 2 * 1 = 2
    let diff = &pred - &target; // diff = 2 - 5 = -3
    let loss = &diff * &diff; // loss = (-3)^2 = 9

    println!("pred: {:?}", pred.data().read().unwrap().as_array());
    println!("diff: {:?}", diff.data().read().unwrap().as_array());
    println!("loss: {:?}", loss.data().read().unwrap().as_array());

    println!(
        "Before backward - w2 grad: {:?}",
        w2.grad().read().unwrap().as_ref().map(|g| g.as_array())
    );

    loss.backward();

    println!(
        "After backward - w2 grad: {:?}",
        w2.grad().read().unwrap().as_ref().map(|g| g.as_array())
    );
    println!("Expected w2 grad: 2 * (pred - target) * x2 = 2 * (-3) * 2 = -12");
}
