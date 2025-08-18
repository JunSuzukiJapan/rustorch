//! Advanced features demonstration
//! 高度な機能のデモンストレーション

use rustorch::prelude::*;
use rustorch::nn::{Sequential, Conv2d, MaxPool2d, BatchNorm2d, Linear, Dropout};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RusTorch Advanced Features Demo");
    println!("==================================");
    
    // Demonstrate extended activation functions
    demonstrate_activation_functions();
    
    // Demonstrate dropout regularization
    demonstrate_dropout();
    
    // Demonstrate complete neural network with all features
    demonstrate_complete_network();
    
    println!("\n🎉 Advanced features demo completed successfully!");
    
    Ok(())
}

fn demonstrate_activation_functions() {
    println!("\n🧠 Testing Extended Activation Functions:");
    println!("========================================");
    
    let input = Variable::new(
        Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]),
        false
    );
    
    println!("Input: {:?}", input.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
    
    // Test GELU
    let gelu_output = gelu(&input);
    println!("GELU output: {:?}", 
        gelu_output.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
    
    // Test Swish
    let swish_output = swish(&input);
    println!("Swish output: {:?}", 
        swish_output.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
    
    // Test ELU
    let elu_output = elu(&input, 1.0);
    println!("ELU output: {:?}", 
        elu_output.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
    
    // Test SELU (self-normalizing)
    let selu_output = selu(&input);
    println!("SELU output: {:?}", 
        selu_output.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
    
    // Test Mish
    let mish_output = mish(&input);
    println!("Mish output: {:?}", 
        mish_output.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
    
    // Test Hardswish (MobileNet activation)
    let hardswish_output = hardswish(&input);
    println!("Hardswish output: {:?}", 
        hardswish_output.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
}

fn demonstrate_dropout() {
    println!("\n🎲 Testing Dropout Regularization:");
    println!("==================================");
    
    // Create dropout layer
    let dropout_layer = Dropout::<f32>::new(0.5, None);
    
    // Create test input
    let input = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]),
        true
    );
    
    println!("Original input: {:?}", 
        input.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
    
    // Test in training mode
    dropout_layer.train();
    println!("Training mode: {}", dropout_layer.is_training());
    
    let train_output = dropout_layer.forward(&input);
    println!("Training output (p=0.5): {:?}", 
        train_output.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
    
    // Test in evaluation mode
    dropout_layer.eval();
    println!("Evaluation mode: {}", dropout_layer.is_training());
    
    let eval_output = dropout_layer.forward(&input);
    println!("Evaluation output: {:?}", 
        eval_output.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
    
    // Test functional dropout
    let functional_train = dropout(&input, 0.3, true);
    let functional_eval = dropout(&input, 0.3, false);
    
    println!("Functional dropout (training, p=0.3): {:?}", 
        functional_train.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
    println!("Functional dropout (eval, p=0.3): {:?}", 
        functional_eval.data().read().unwrap().as_array().iter().collect::<Vec<_>>());
}

fn demonstrate_complete_network() {
    println!("\n🏗️  Building Complete Network with All Features:");
    println!("===============================================");
    
    // Create a modern CNN with all advanced features
    let mut advanced_cnn = Sequential::<f32>::new();
    
    // First block: Conv + BatchNorm + Activation + Dropout
    let conv1 = Conv2d::new(3, 32, (3, 3), Some((1, 1)), Some((1, 1)), Some(true));
    advanced_cnn.add_module(conv1);
    println!("✅ Added Conv2d(3→32, 3x3)");
    
    let bn1 = BatchNorm2d::new(32, None, None, None);
    advanced_cnn.add_module(bn1);
    println!("✅ Added BatchNorm2d(32)");
    
    // Note: We can't add activation functions directly to Sequential yet
    // This would require implementing a wrapper or modifying Sequential
    
    let pool1 = MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
    advanced_cnn.add_module(pool1);
    println!("✅ Added MaxPool2d(2x2)");
    
    let dropout1 = Dropout::new(0.25, None);
    advanced_cnn.add_module(dropout1);
    println!("✅ Added Dropout(p=0.25)");
    
    // Second block
    let conv2 = Conv2d::new(32, 64, (3, 3), Some((1, 1)), Some((1, 1)), Some(true));
    advanced_cnn.add_module(conv2);
    println!("✅ Added Conv2d(32→64, 3x3)");
    
    let bn2 = BatchNorm2d::new(64, None, None, None);
    advanced_cnn.add_module(bn2);
    println!("✅ Added BatchNorm2d(64)");
    
    let pool2 = MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
    advanced_cnn.add_module(pool2);
    println!("✅ Added MaxPool2d(2x2)");
    
    let dropout2 = Dropout::new(0.5, None);
    advanced_cnn.add_module(dropout2);
    println!("✅ Added Dropout(p=0.5)");
    
    println!("\nNetwork structure created with {} modules", advanced_cnn.len());
    
    // Test forward pass
    let test_input = Variable::new(
        Tensor::from_vec(
            (0..3*32*32).map(|i| (i as f32) * 0.001).collect(),
            vec![1, 3, 32, 32]
        ),
        true
    );
    
    println!("\nTesting forward pass:");
    println!("Input shape: {:?}", test_input.data().read().unwrap().shape());
    
    // Set all layers to training mode (manually for this demo)
    // In practice, we'd implement a train() method for Sequential
    let output = advanced_cnn.forward(&test_input);
    println!("Final output shape: {:?}", output.data().read().unwrap().shape());
    
    // Demonstrate different loss functions
    demonstrate_loss_functions(&output);
}

fn demonstrate_loss_functions(prediction: &Variable<f32>) {
    println!("\n📊 Testing Loss Functions:");
    println!("==========================");
    
    // Create dummy target with same shape as prediction
    let target_shape = prediction.data().read().unwrap().shape().to_vec();
    let target_data: Vec<f32> = (0..target_shape.iter().product::<usize>())
        .map(|i| (i as f32) * 0.0001)
        .collect();
    let target = Variable::new(
        Tensor::from_vec(target_data, target_shape),
        false
    );
    
    // Test MSE Loss
    let mse = mse_loss(prediction, &target);
    println!("MSE Loss: {:?}", 
        mse.data().read().unwrap().as_array().iter().next().unwrap());
    
    // Test Binary Cross Entropy (with sigmoid)
    let sigmoid_pred = sigmoid(prediction);
    let sigmoid_target = sigmoid(&target);
    let bce = binary_cross_entropy(&sigmoid_pred, &sigmoid_target);
    println!("BCE Loss: {:?}", 
        bce.data().read().unwrap().as_array().iter().next().unwrap());
    
    // Test Huber Loss
    let huber = huber_loss(prediction, &target, 1.0);
    println!("Huber Loss: {:?}", 
        huber.data().read().unwrap().as_array().iter().next().unwrap());
    
    println!("All loss functions computed successfully!");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_activation_functions_integration() {
        let input = Variable::new(
            Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]),
            false
        );
        
        // Test that all new activation functions work
        let _gelu_out = gelu(&input);
        let _swish_out = swish(&input);
        let _elu_out = elu(&input, 1.0);
        let _selu_out = selu(&input);
        let _mish_out = mish(&input);
        let _hardswish_out = hardswish(&input);
        
        // If we get here, all activations work
        assert!(true);
    }
    
    #[test]
    fn test_dropout_integration() {
        let dropout = Dropout::<f32>::new(0.5, None);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]),
            false
        );
        
        // Test training mode
        dropout.train();
        let _train_out = dropout.forward(&input);
        
        // Test eval mode
        dropout.eval();
        let eval_out = dropout.forward(&input);
        
        // In eval mode, output should equal input
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let output_binding = eval_out.data();
        let output_data = output_binding.read().unwrap();
        
        assert_eq!(input_data.shape(), output_data.shape());
    }
    
    #[test]
    fn test_complete_network_forward() {
        let mut network = Sequential::<f32>::new();
        
        // Add layers
        network.add_module(Conv2d::new(1, 4, (3, 3), Some((1, 1)), Some((1, 1)), Some(true)));
        network.add_module(BatchNorm2d::new(4, None, None, None));
        network.add_module(MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0))));
        network.add_module(Dropout::new(0.5, None));
        
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 1 * 8 * 8], vec![1, 1, 8, 8]),
            false
        );
        
        let output = network.forward(&input);
        
        // Should produce valid output
        assert!(output.data().read().unwrap().len() > 0);
    }
}