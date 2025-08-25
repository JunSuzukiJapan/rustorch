//! A simple linear regression example using rustorch

use rand::Rng;
use rustorch::nn::Linear;
use rustorch::prelude::*;
use rustorch::utils::mse_loss;

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

    // Training parameters
    let n_epochs = 10;

    // Training loop (simplified)
    println!("Starting training...");
    for epoch in 0..n_epochs {
        // Forward pass
        let output = model.forward(&x_var);

        // Compute loss - simplified version
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let y_binding = y_var.data();
        let y_data = y_binding.read().unwrap();
        let loss = mse_loss(output_data.as_array(), y_data.as_array());

        // Print loss every epoch
        println!("Epoch {}: Loss = {:.6}", epoch, loss);
    }

    println!("Training completed!");
}
