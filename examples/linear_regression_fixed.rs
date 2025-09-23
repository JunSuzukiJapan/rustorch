//! A proper linear regression example with actual training using rustorch

use rand::Rng;
use rustorch::nn::loss::mse_loss;
use rustorch::nn::Linear;
use rustorch::optim::{Optimizer, SGD};
use rustorch::prelude::*;

fn generate_data(n_samples: usize, n_features: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();

    // Generate random input features
    let x_data: Vec<f32> = (0..n_samples * n_features)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // Generate random weights for the true model
    let true_weights: Vec<f32> = (0..n_features).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Generate target values with some noise
    let mut y_data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut y = 0.0;
        for j in 0..n_features {
            y += x_data[i * n_features + j] * true_weights[j];
        }
        // Add some noise
        y += rng.gen_range(-0.1..0.1);
        y_data.push(y);
    }

    (x_data, y_data)
}

fn main() {
    // Generate some random data
    let n_samples = 100;
    let n_features = 3;
    let (x_data, y_data) = generate_data(n_samples, n_features);

    // Create tensors
    let x_tensor = Tensor::from_vec(x_data, vec![n_samples, n_features]);
    let y_tensor = Tensor::from_vec(y_data, vec![n_samples, 1]);

    // Create variables
    let x_var = Variable::new(x_tensor, false);
    let y_var = Variable::new(y_tensor, false);

    // Create a simple linear model
    let model = Linear::new(n_features, 1);

    // Get model parameters
    let params = model.parameters();

    // Create optimizer
    let mut optimizer = SGD::new(0.01);

    // Training parameters
    let n_epochs = 100;

    // Training loop
    println!("Starting training...");
    for epoch in 0..n_epochs {
        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        let output = model.forward(&x_var);

        // Compute loss using autograd
        let loss_var = mse_loss(&output, &y_var);

        // Backward pass
        loss_var.backward();

        // Update parameters
        for param in &params {
            let param_data = param.data();
            let param_tensor = param_data.read().unwrap();
            let grad_data = param.grad();
            let grad_guard = grad_data.read().unwrap();
            if let Some(ref grad_tensor) = *grad_guard {
                optimizer.step(&param_tensor, grad_tensor);
            }
        }

        // Print loss every 10 epochs
        if epoch % 10 == 0 {
            let loss_data = loss_var.data();
            let loss_value = loss_data.read().unwrap();
            println!(
                "Epoch {}: Loss = {:.6}",
                epoch,
                loss_value.as_array().iter().next().unwrap_or(&0.0)
            );
        }
    }

    println!("Training completed!");
}
