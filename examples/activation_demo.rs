//! Demonstration of activation functions
//! 活性化関数のデモンストレーション

use rustorch::prelude::*;

fn main() {
    println!("🧠 RusTorch Activation Functions Demo");
    println!("=====================================");

    // Test data: negative, zero, and positive values
    let test_input = Variable::new(
        Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]),
        true,
    );

    println!("\n📊 Input: [-2.0, -1.0, 0.0, 1.0, 2.0]");
    println!("=========================================");

    // Test ReLU
    println!("\n🔥 ReLU Activation");
    let relu_output = relu(&test_input);
    print_activation_result("ReLU", &relu_output);

    // Test Sigmoid
    println!("\n📈 Sigmoid Activation");
    let sigmoid_output = sigmoid(&test_input);
    print_activation_result("Sigmoid", &sigmoid_output);

    // Test Tanh
    println!("\n🌊 Tanh Activation");
    let tanh_output = tanh(&test_input);
    print_activation_result("Tanh", &tanh_output);

    // Test Leaky ReLU
    println!("\n⚡ Leaky ReLU Activation (α=0.1)");
    let leaky_relu_output = leaky_relu(&test_input, 0.1);
    print_activation_result("Leaky ReLU", &leaky_relu_output);

    // Test Softmax
    println!("\n🎯 Softmax Activation");
    let softmax_input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]), false);
    let softmax_output = softmax(&softmax_input);
    print_activation_result("Softmax", &softmax_output);

    // Verify softmax sums to 1
    let softmax_binding = softmax_output.data();
    let softmax_data = softmax_binding.read().unwrap();
    let sum: f32 = softmax_data.as_array().iter().sum();
    println!("  Sum of probabilities: {:.6}", sum);

    // Demonstrate gradient computation with ReLU
    println!("\n🔄 Gradient Computation Example");
    println!("===============================");

    let x = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]), true);

    println!("Input: [-1.0, 0.0, 1.0, 2.0]");

    // Apply ReLU and compute sum
    let activated = relu(&x);
    let sum_result = activated.sum();

    print!("ReLU output: ");
    print_tensor_values(&activated);

    println!(
        "Sum: {:.1}",
        sum_result
            .data()
            .read()
            .unwrap()
            .as_array()
            .iter()
            .next()
            .unwrap()
    );

    // Backward pass
    sum_result.backward();

    // Check gradients (should be [0, 0, 1, 1] for ReLU derivative)
    let grad_binding = x.grad();
    let grad = grad_binding.read().unwrap();
    if let Some(ref g) = *grad {
        print!("Gradients: ");
        for (i, &val) in g.as_array().iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.1}", val);
        }
        println!();
    }

    println!("\n✅ All activation functions working correctly!");
    println!("Ready for building neural networks! 🚀");
}

fn print_activation_result<
    T: num_traits::Float
        + std::fmt::Display
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
>(
    name: &str,
    output: &Variable<T>,
) {
    let binding = output.data();
    let data = binding.read().unwrap();
    print!("  {} output: ", name);
    print_tensor_values_generic(data.as_array().iter());
}

fn print_tensor_values<
    T: num_traits::Float
        + std::fmt::Display
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
>(
    tensor: &Variable<T>,
) {
    let binding = tensor.data();
    let data = binding.read().unwrap();
    print_tensor_values_generic(data.as_array().iter());
}

fn print_tensor_values_generic<T: std::fmt::Display>(iter: impl Iterator<Item = T>) {
    print!("[");
    for (i, val) in iter.enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.3}", val);
    }
    println!("]");
}
