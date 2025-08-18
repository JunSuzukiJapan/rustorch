//! Complete neural network training demonstration
//! 完全なニューラルネットワーク学習のデモンストレーション

use rustorch::prelude::*;
use rustorch::nn::Linear;

fn main() {
    println!("🧠 Complete Neural Network Training Demo");
    println!("========================================");

    // Create a simple classification dataset
    // 2D input -> 2 classes (binary classification)
    println!("\n📊 Dataset: 2D Binary Classification");
    println!("====================================");
    
    // Training data: [x1, x2] -> class (0 or 1)
    // Class 0: points near (0, 0)
    // Class 1: points near (1, 1)
    let train_x = Variable::new(
        Tensor::from_vec(
            vec![
                0.1, 0.2,   // Class 0
                0.2, 0.1,   // Class 0
                0.9, 0.8,   // Class 1
                0.8, 0.9,   // Class 1
                0.0, 0.3,   // Class 0
                1.0, 0.7,   // Class 1
            ],
            vec![6, 2]
        ),
        true
    );
    
    let train_y = Variable::new(
        Tensor::from_vec(
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0], // Binary labels
            vec![6]
        ),
        false
    );
    
    println!("Training samples: 6");
    println!("Input features: 2 (x1, x2)");
    println!("Classes: 2 (binary classification)");

    // Create a simple neural network: 2 -> 4 -> 1
    println!("\n🏗️ Neural Network Architecture");
    println!("==============================");
    
    let linear1 = Linear::new(2, 4);  // Input layer: 2 -> 4
    let linear2 = Linear::new(4, 1);  // Output layer: 4 -> 1
    
    println!("Layer 1: Linear(2 -> 4) + ReLU");
    println!("Layer 2: Linear(4 -> 1) + Sigmoid");
    println!("Loss: Binary Cross Entropy");

    // Training parameters
    let learning_rate = 0.1;
    let epochs = 10;
    
    println!("\n🎯 Training Configuration");
    println!("========================");
    println!("Learning rate: {}", learning_rate);
    println!("Epochs: {}", epochs);

    // Training loop
    println!("\n🔄 Training Progress");
    println!("===================");
    
    for epoch in 0..epochs {
        // Forward pass
        let h1 = linear1.forward(&train_x);
        let a1 = relu(&h1);
        let h2 = linear2.forward(&a1);
        let predictions = sigmoid(&h2);
        
        // Reshape predictions to match targets
        let pred_reshaped = Variable::new(
            {
                let pred_binding = predictions.data();
                let pred_data = pred_binding.read().unwrap();
                let values: Vec<f32> = pred_data.as_array().iter().cloned().collect();
                Tensor::from_vec(values, vec![6])
            },
            predictions.requires_grad()
        );
        
        // Compute loss
        let loss = binary_cross_entropy(&pred_reshaped, &train_y);
        
        // Print progress
        let loss_binding = loss.data();
        let loss_data = loss_binding.read().unwrap();
        let loss_val = *loss_data.as_array().iter().next().unwrap();
        println!("Epoch {}: Loss = {:.6}", epoch + 1, loss_val);
        
        // Backward pass (compute gradients)
        loss.backward();
        
        // In a real implementation, we would update parameters here using an optimizer
        // For now, we just demonstrate that gradients are computed
        
        // Clear gradients for next iteration (in real training)
        // optimizer.zero_grad(); // This would be done by an optimizer
    }

    // Test the trained network
    println!("\n🧪 Testing Trained Network");
    println!("==========================");
    
    let test_x = Variable::new(
        Tensor::from_vec(
            vec![
                0.05, 0.05,  // Should predict class 0
                0.95, 0.95,  // Should predict class 1
            ],
            vec![2, 2]
        ),
        false
    );
    
    // Forward pass on test data
    let test_h1 = linear1.forward(&test_x);
    let test_a1 = relu(&test_h1);
    let test_h2 = linear2.forward(&test_a1);
    let test_predictions = sigmoid(&test_h2);
    
    println!("Test input 1: [0.05, 0.05] (expected: class 0)");
    println!("Test input 2: [0.95, 0.95] (expected: class 1)");
    
    let test_binding = test_predictions.data();
    let test_data = test_binding.read().unwrap();
    println!("Predictions: {:?}", 
             test_data.as_array().iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>());

    // Demonstrate different loss functions on the same problem
    println!("\n📊 Loss Function Comparison");
    println!("===========================");
    
    let sample_pred = Variable::new(
        Tensor::from_vec(vec![0.3, 0.7, 0.9, 0.1], vec![4]),
        false
    );
    let sample_target = Variable::new(
        Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![4]),
        false
    );
    
    let mse = mse_loss(&sample_pred, &sample_target);
    let bce = binary_cross_entropy(&sample_pred, &sample_target);
    let huber = huber_loss(&sample_pred, &sample_target, 0.5);
    
    println!("Sample predictions: [0.3, 0.7, 0.9, 0.1]");
    println!("Sample targets:     [0.0, 1.0, 1.0, 0.0]");
    println!("MSE Loss:   {:.6}", mse.data().read().unwrap().as_array().iter().next().unwrap());
    println!("BCE Loss:   {:.6}", bce.data().read().unwrap().as_array().iter().next().unwrap());
    println!("Huber Loss: {:.6}", huber.data().read().unwrap().as_array().iter().next().unwrap());

    // Summary
    println!("\n✅ Training Demo Complete!");
    println!("==========================");
    println!("🎯 Demonstrated features:");
    println!("   ✓ Neural network forward pass");
    println!("   ✓ Activation functions (ReLU, Sigmoid)");
    println!("   ✓ Loss functions (MSE, BCE, Huber)");
    println!("   ✓ Automatic differentiation (backward pass)");
    println!("   ✓ Multi-layer architecture");
    println!("   ✓ Binary classification setup");
    println!("\n🚀 Next steps: Implement optimizers for parameter updates!");
}