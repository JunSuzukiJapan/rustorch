//! Demonstration of automatic differentiation capabilities
//! 自動微分機能のデモンストレーション

use rustorch::prelude::*;

fn main() {
    println!("🚀 RusTorch Automatic Differentiation Demo");
    println!("==========================================");

    // Example 1: Simple scalar computation
    println!("\n📊 Example 1: Scalar computation with gradients");
    let x = Variable::new(Tensor::from_vec(vec![2.0], vec![]), true);
    let y = Variable::new(Tensor::from_vec(vec![3.0], vec![]), true);

    // Compute z = x * y + x
    let xy = &x * &y; // x * y = 6.0
    let z = &xy + &x; // z = 6.0 + 2.0 = 8.0

    println!("x = 2.0, y = 3.0");
    println!(
        "z = x * y + x = {:.1}",
        z.data().read().unwrap().as_array().iter().next().unwrap()
    );

    // Backward pass
    z.backward();

    // Check gradients
    let x_grad_binding = x.grad();
    let x_grad = x_grad_binding.read().unwrap();
    let y_grad_binding = y.grad();
    let y_grad = y_grad_binding.read().unwrap();

    if let Some(ref grad) = *x_grad {
        println!("dz/dx = {:.1}", grad.as_array().iter().next().unwrap());
    }
    if let Some(ref grad) = *y_grad {
        println!("dz/dy = {:.1}", grad.as_array().iter().next().unwrap());
    }

    // Example 2: Vector operations
    println!("\n📊 Example 2: Vector operations");
    let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]), true);
    let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]), true);

    // Element-wise multiplication and sum
    let c = &a * &b; // [4.0, 10.0, 18.0]
    let sum = c.sum(); // 32.0

    println!("a = [1.0, 2.0, 3.0]");
    println!("b = [4.0, 5.0, 6.0]");
    println!(
        "sum(a * b) = {:.1}",
        sum.data().read().unwrap().as_array().iter().next().unwrap()
    );

    // Backward pass
    sum.backward();

    // Check gradients
    let a_grad_binding = a.grad();
    let a_grad = a_grad_binding.read().unwrap();
    let b_grad_binding = b.grad();
    let b_grad = b_grad_binding.read().unwrap();

    if let Some(ref grad) = *a_grad {
        print!("da/dsum = [");
        for (i, val) in grad.as_array().iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.1}", val);
        }
        println!("]");
    }

    if let Some(ref grad) = *b_grad {
        print!("db/dsum = [");
        for (i, val) in grad.as_array().iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.1}", val);
        }
        println!("]");
    }

    // Example 3: Matrix multiplication
    println!("\n📊 Example 3: Matrix multiplication");
    let m1 = Variable::new(Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]), true);
    let m2 = Variable::new(Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]), true);

    let result = m1.matmul(&m2);

    println!("m1 = [1.0, 2.0] (1x2)");
    println!("m2 = [3.0; 4.0] (2x1)");
    println!(
        "m1 @ m2 = {:.1}",
        result
            .data()
            .read()
            .unwrap()
            .as_array()
            .iter()
            .next()
            .unwrap()
    );

    // Backward pass
    result.backward();

    println!("\n✅ Automatic differentiation demo completed!");
    println!("All gradients computed successfully using backward propagation.");
}
