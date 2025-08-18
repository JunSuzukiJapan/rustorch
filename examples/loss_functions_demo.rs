//! Demonstration of loss functions
//! 損失関数のデモンストレーション

use rustorch::prelude::*;

fn main() {
    println!("📊 RusTorch Loss Functions Demo");
    println!("===============================");

    // Example 1: Mean Squared Error (MSE) Loss
    println!("\n🎯 Example 1: Mean Squared Error (MSE) Loss");
    println!("===========================================");
    
    let predictions = Variable::new(
        Tensor::from_vec(vec![2.5, 0.0, 2.1, 1.8], vec![4]),
        true
    );
    let targets = Variable::new(
        Tensor::from_vec(vec![3.0, -0.5, 2.0, 1.2], vec![4]),
        false
    );
    
    println!("Predictions: [2.5, 0.0, 2.1, 1.8]");
    println!("Targets:     [3.0, -0.5, 2.0, 1.2]");
    
    let mse = mse_loss(&predictions, &targets);
    println!("MSE Loss: {:.6}", 
             mse.data().read().unwrap().as_array().iter().next().unwrap());

    // Example 2: Binary Cross Entropy Loss
    println!("\n🎯 Example 2: Binary Cross Entropy Loss");
    println!("=======================================");
    
    let binary_predictions = Variable::new(
        Tensor::from_vec(vec![0.9, 0.1, 0.8, 0.3], vec![4]),
        true
    );
    let binary_targets = Variable::new(
        Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4]),
        false
    );
    
    println!("Predictions: [0.9, 0.1, 0.8, 0.3] (probabilities)");
    println!("Targets:     [1.0, 0.0, 1.0, 0.0] (binary labels)");
    
    let bce = binary_cross_entropy(&binary_predictions, &binary_targets);
    println!("BCE Loss: {:.6}", 
             bce.data().read().unwrap().as_array().iter().next().unwrap());

    // Example 3: Huber Loss (robust to outliers)
    println!("\n🎯 Example 3: Huber Loss");
    println!("========================");
    
    let robust_predictions = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 10.0, 4.0], vec![4]), // Note: 10.0 is an outlier
        true
    );
    let robust_targets = Variable::new(
        Tensor::from_vec(vec![1.1, 2.1, 3.0, 4.1], vec![4]),
        false
    );
    
    println!("Predictions: [1.0, 2.0, 10.0, 4.0] (with outlier)");
    println!("Targets:     [1.1, 2.1, 3.0, 4.1]");
    
    let huber = huber_loss(&robust_predictions, &robust_targets, 1.0);
    let mse_with_outlier = mse_loss(&robust_predictions, &robust_targets);
    
    println!("Huber Loss (δ=1.0): {:.6}", 
             huber.data().read().unwrap().as_array().iter().next().unwrap());
    println!("MSE Loss:            {:.6}", 
             mse_with_outlier.data().read().unwrap().as_array().iter().next().unwrap());
    println!("→ Huber loss is more robust to the outlier (10.0)");

    // Example 4: Training simulation with gradient computation
    println!("\n🎯 Example 4: Training Simulation");
    println!("=================================");
    
    // Simple regression problem
    let x = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]),
        true
    );
    let y_true = Variable::new(
        Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![4]), // y = 2x
        false
    );
    
    // Simulate a simple linear model: y = w * x + b
    let w = Variable::new(Tensor::from_vec(vec![1.5], vec![1]), true); // Initial weight
    let b = Variable::new(Tensor::from_vec(vec![0.5], vec![1]), true); // Initial bias
    
    println!("True relationship: y = 2x");
    println!("Initial model: y = 1.5x + 0.5");
    
    // Forward pass: y_pred = w * x + b
    let w_expanded = Variable::new(
        Tensor::from_vec(vec![1.5, 1.5, 1.5, 1.5], vec![4]),
        true
    );
    let b_expanded = Variable::new(
        Tensor::from_vec(vec![0.5, 0.5, 0.5, 0.5], vec![4]),
        true
    );
    
    let y_pred = &(&x * &w_expanded) + &b_expanded;
    
    // Compute loss
    let loss = mse_loss(&y_pred, &y_true);
    
    println!("Predictions: {:?}", format_tensor(&y_pred));
    println!("Targets:     [2.0, 4.0, 6.0, 8.0]");
    println!("Loss: {:.6}", 
             loss.data().read().unwrap().as_array().iter().next().unwrap());
    
    // Backward pass
    loss.backward();
    
    // Check gradients (in a real training loop, we'd use these to update parameters)
    println!("✅ Gradients computed successfully!");
    println!("   (In real training, these would be used to update w and b)");

    // Example 5: Comparison of different loss functions
    println!("\n🎯 Example 5: Loss Function Comparison");
    println!("=====================================");
    
    let pred = Variable::new(
        Tensor::from_vec(vec![0.1, 0.4, 0.8, 0.9], vec![4]),
        false
    );
    let target = Variable::new(
        Tensor::from_vec(vec![0.0, 0.5, 1.0, 1.0], vec![4]),
        false
    );
    
    println!("Predictions: [0.1, 0.4, 0.8, 0.9]");
    println!("Targets:     [0.0, 0.5, 1.0, 1.0]");
    
    let mse_result = mse_loss(&pred, &target);
    let bce_result = binary_cross_entropy(&pred, &target);
    let huber_result = huber_loss(&pred, &target, 0.5);
    
    println!("MSE Loss:   {:.6}", 
             mse_result.data().read().unwrap().as_array().iter().next().unwrap());
    println!("BCE Loss:   {:.6}", 
             bce_result.data().read().unwrap().as_array().iter().next().unwrap());
    println!("Huber Loss: {:.6}", 
             huber_result.data().read().unwrap().as_array().iter().next().unwrap());

    println!("\n✅ Loss Functions Demo Complete!");
    println!("🚀 Ready for training neural networks!");
}

fn format_tensor<T: num_traits::Float + std::fmt::Display + Send + Sync + 'static>(
    var: &Variable<T>
) -> String {
    let binding = var.data();
    let data = binding.read().unwrap();
    let values: Vec<String> = data.as_array().iter()
        .map(|x| format!("{:.1}", x))
        .collect();
    format!("[{}]", values.join(", "))
}