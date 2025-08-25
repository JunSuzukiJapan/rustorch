//! Demonstration of a simple neural network using activation functions
//! 活性化関数を使用したシンプルなニューラルネットワークのデモ

use rustorch::nn::Linear;
use rustorch::prelude::*;

fn main() {
    println!("🧠 Neural Network with Activation Functions Demo");
    println!("===============================================");

    // Create a simple 2-layer neural network
    // Input: 3 features -> Hidden: 4 neurons -> Output: 2 classes
    let linear1 = Linear::new(3, 4); // First layer: 3 -> 4
    let linear2 = Linear::new(4, 2); // Second layer: 4 -> 2

    // Sample input data (batch size = 2, features = 3)
    let input_data = Variable::new(
        Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, // Sample 1
                0.5, -1.0, 2.5, // Sample 2
            ],
            vec![2, 3],
        ),
        true,
    );

    println!("📊 Input Data (2 samples, 3 features):");
    print_variable_data(&input_data);

    // Forward pass through the network
    println!("\n🔄 Forward Pass:");
    println!("================");

    // Layer 1: Linear + ReLU
    println!("\n1️⃣ First Layer (Linear 3->4)");
    let hidden1 = linear1.forward(&input_data);
    println!("   Linear output:");
    print_variable_data(&hidden1);

    println!("\n   After ReLU activation:");
    let activated1 = relu(&hidden1);
    print_variable_data(&activated1);

    // Layer 2: Linear + Sigmoid (for binary classification)
    println!("\n2️⃣ Second Layer (Linear 4->2)");
    let hidden2 = linear2.forward(&activated1);
    println!("   Linear output:");
    print_variable_data(&hidden2);

    println!("\n   After Sigmoid activation:");
    let output = sigmoid(&hidden2);
    print_variable_data(&output);

    // Apply softmax for probability distribution
    println!("\n🎯 Final Softmax Probabilities:");
    let probabilities = softmax(&output);
    print_variable_data(&probabilities);

    // Compute a simple loss (sum of outputs for demonstration)
    println!("\n📈 Loss Computation:");
    let loss = probabilities.sum();
    println!(
        "   Total sum (loss): {:.6}",
        loss.data()
            .read()
            .unwrap()
            .as_array()
            .iter()
            .next()
            .unwrap()
    );

    // Backward pass
    println!("\n🔄 Backward Pass (Gradient Computation):");
    println!("========================================");

    loss.backward();

    // Check if gradients were computed for input
    let input_grad_binding = input_data.grad();
    let input_grad = input_grad_binding.read().unwrap();

    if let Some(ref grad) = *input_grad {
        println!("✅ Input gradients computed:");
        print_tensor_data(grad);
    } else {
        println!("❌ No gradients computed for input");
    }

    // Demonstrate different activation functions on the same data
    println!("\n🔬 Activation Function Comparison:");
    println!("==================================");

    let test_data = Variable::new(
        Tensor::from_vec(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5]),
        false,
    );

    println!("Input: [-2.0, -0.5, 0.0, 0.5, 2.0]");

    let relu_result = relu(&test_data);
    let sigmoid_result = sigmoid(&test_data);
    let tanh_result = tanh(&test_data);

    println!("ReLU:    {:?}", format_tensor_values(&relu_result));
    println!("Sigmoid: {:?}", format_tensor_values(&sigmoid_result));
    println!("Tanh:    {:?}", format_tensor_values(&tanh_result));

    println!("\n✅ Neural Network Demo Complete!");
    println!("🚀 Ready to build more complex architectures!");
}

fn print_variable_data<
    T: num_traits::Float
        + std::fmt::Display
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
>(
    var: &Variable<T>,
) {
    let binding = var.data();
    let data = binding.read().unwrap();
    print_tensor_data(&*data);
}

fn print_tensor_data<T: num_traits::Float + std::fmt::Display + 'static>(tensor: &Tensor<T>) {
    let shape = tensor.shape();
    println!("   Shape: {:?}", shape);

    if shape.len() == 1 {
        // 1D tensor
        print!("   Values: [");
        for (i, val) in tensor.as_array().iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.3}", val);
        }
        println!("]");
    } else if shape.len() == 2 {
        // 2D tensor
        println!("   Values:");
        for row in 0..shape[0] {
            print!("     [");
            for col in 0..shape[1] {
                if col > 0 {
                    print!(", ");
                }
                let idx = row * shape[1] + col;
                if let Some(val) = tensor.as_array().iter().nth(idx) {
                    print!("{:.3}", val);
                }
            }
            println!("]");
        }
    }
}

fn format_tensor_values<
    T: num_traits::Float
        + std::fmt::Display
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
>(
    var: &Variable<T>,
) -> String {
    let binding = var.data();
    let data = binding.read().unwrap();
    let values: Vec<String> = data
        .as_array()
        .iter()
        .map(|x| format!("{:.3}", x))
        .collect();
    format!("[{}]", values.join(", "))
}
