//! A simple linear regression example using rustorch

use rustorch::prelude::*;
use rustorch::nn::{Module, Linear};
use rustorch::utils::mse_loss;
use ndarray::ArrayD;
use rand::Rng;

fn generate_data(n_samples: usize, n_features: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut rng = rand::thread_rng();
    
    // Generate random input features
    let x_data: Vec<f32> = (0..n_samples * n_features)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    // Generate random weights for the true model
    let true_weights: Vec<f32> = (0..n_features)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
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
    
    // Reshape the data
    let x = ArrayD::from_shape_vec(vec![n_samples, n_features], x_data).unwrap();
    let y = ArrayD::from_shape_vec(vec![n_samples, 1], y_data).unwrap();
    
    (x, y)
}

fn main() {
    // Generate some random data
    let n_samples = 1000;
    let n_features = 5;
    let (x_train, y_train) = generate_data(n_samples, n_features);
    
    // Create a simple linear model (mutable for parameter updates)
    let mut model = Linear::new(n_features, 1);
    
    // Training parameters
    let learning_rate = 0.01;
    let n_epochs = 100;
    
    // Training loop
    println!("Starting training...");
    for epoch in 0..n_epochs {
        let x_var = Variable::new(Tensor::from_ndarray(x_train.clone()), true);
        let y_var = Variable::new(Tensor::from_ndarray(y_train.clone()), false);
        
        // Forward pass
        let output = model.forward(&x_var);
        
        // Compute loss - get the underlying ArrayD from the Tensors
        let loss = mse_loss(output.data().data(), y_var.data().data());
        
        // Backward pass - compute gradients
        let mut output_var = output;
        output_var.backward(None, true);  // retain_graph=true to keep computation graph
        
        // Track the maximum gradient magnitude for debugging
        let mut max_grad_mag: f32 = 0.0;
        
        // Update parameters using gradient descent
        for param in model.parameters_mut() {
            // Get the gradient first and clone it to release the borrow
            let grad_opt = param.grad().cloned();
            
            if let Some(grad) = grad_opt {
                // Get the gradient data
                let grad_data = grad.data();
                
                // Calculate gradient magnitude for debugging
                let grad_mag = grad_data.iter().map(|&x| x * x).sum::<f32>().sqrt();
                max_grad_mag = max_grad_mag.max(grad_mag);
                
                if epoch % 10 == 0 {
                    println!("  Param shape: {:?}, Grad mag: {:.6}", grad_data.shape(), grad_mag);
                }
                
                // Get mutable access to parameter data
                let param_data = param.data_mut();
                let param_array = param_data.data_mut();
                
                // Print parameter statistics before update
                if epoch % 10 == 0 {
                    let param_min = param_array.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let param_max = param_array.iter().fold(-f32::INFINITY, |a, &b| a.max(b));
                    let param_mean = param_array.mean().unwrap_or(0.0);
                    println!("  Before update - Min: {:.6}, Max: {:.6}, Mean: {:.6}", 
                             param_min, param_max, param_mean);
                }
                
                // Update parameters: param = param - learning_rate * grad
                *param_array = &*param_array - &(grad_data * learning_rate);
            }
        }
        
        // 10エポックごとに損失を表示
        if epoch % 10 == 0 {
            println!("エポック {}: 損失 = {:.6}", epoch, loss);
        }
    }
    
    println!("トレーニングが完了しました！");
    
    // TODO: Add evaluation on test data
}
