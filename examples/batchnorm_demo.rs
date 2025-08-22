//! Batch Normalization Demo with CNN
//! CNNでのバッチ正規化デモ

use rustorch::prelude::*;
use rustorch::nn::{Sequential, Conv2d, MaxPool2d, BatchNorm2d, BatchNorm1d, Linear};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 RusTorch BatchNorm Demo");
    println!("========================");
    
    // Create a modern CNN with BatchNorm for CIFAR-10 like data (3x32x32)
    // CIFAR-10風データ用のBatchNorm付きモダンCNNを作成 (3x32x32)
    let mut cnn_with_bn = Sequential::<f32>::new();
    
    println!("🏗️  Building CNN with BatchNorm layers:");
    
    // First conv block: 3 -> 32 channels
    // 最初の畳み込みブロック: 3 -> 32チャンネル
    let conv1 = Conv2d::new(3, 32, (3, 3), Some((1, 1)), Some((1, 1)), Some(true));
    cnn_with_bn.add_module(conv1);
    println!("   ✅ Conv2d(3→32, 3x3, padding=1)");
    
    let bn1 = BatchNorm2d::new(32, None, None, None);
    cnn_with_bn.add_module(bn1);
    println!("   ✅ BatchNorm2d(32)");
    
    let pool1 = MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
    cnn_with_bn.add_module(pool1);
    println!("   ✅ MaxPool2d(2x2, stride=2)");
    
    // Second conv block: 32 -> 64 channels
    // 2番目の畳み込みブロック: 32 -> 64チャンネル
    let conv2 = Conv2d::new(32, 64, (3, 3), Some((1, 1)), Some((1, 1)), Some(true));
    cnn_with_bn.add_module(conv2);
    println!("   ✅ Conv2d(32→64, 3x3, padding=1)");
    
    let bn2 = BatchNorm2d::new(64, None, None, None);
    cnn_with_bn.add_module(bn2);
    println!("   ✅ BatchNorm2d(64)");
    
    let pool2 = MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
    cnn_with_bn.add_module(pool2);
    println!("   ✅ MaxPool2d(2x2, stride=2)");
    
    // Third conv block: 64 -> 128 channels
    // 3番目の畳み込みブロック: 64 -> 128チャンネル
    let conv3 = Conv2d::new(64, 128, (3, 3), Some((1, 1)), Some((1, 1)), Some(true));
    cnn_with_bn.add_module(conv3);
    println!("   ✅ Conv2d(64→128, 3x3, padding=1)");
    
    let bn3 = BatchNorm2d::new(128, None, None, None);
    cnn_with_bn.add_module(bn3);
    println!("   ✅ BatchNorm2d(128)");
    
    println!("\n📊 Testing BatchNorm components individually:");
    
    // Test BatchNorm2d independently
    // BatchNorm2dを独立してテスト
    let test_bn2d = BatchNorm2d::<f32>::new(16, None, None, None);
    
    println!("🧪 BatchNorm2d test:");
    println!("   - Created with 16 channels");
    println!("   - Training mode: {}", test_bn2d.is_training());
    println!("   - Epsilon: {:.2e}", test_bn2d.eps());
    println!("   - Momentum: {:.1}", test_bn2d.momentum());
    
    // Create test input for BatchNorm2d: batch=4, channels=16, height=8, width=8
    let test_input_2d = Variable::new(
        Tensor::from_vec(
            (0..4*16*8*8).map(|i| (i as f32) * 0.01).collect(),
            vec![4, 16, 8, 8]
        ),
        true
    );
    
    println!("   - Input shape: {:?}", test_input_2d.data().read().unwrap().shape());
    
    // Forward pass in training mode
    let output_train = test_bn2d.forward(&test_input_2d);
    println!("   - Training output shape: {:?}", output_train.data().read().unwrap().shape());
    
    // Switch to eval mode and test
    test_bn2d.eval();
    let output_eval = test_bn2d.forward(&test_input_2d);
    println!("   - Evaluation output shape: {:?}", output_eval.data().read().unwrap().shape());
    println!("   - Evaluation mode: {}", !test_bn2d.is_training());
    
    // Test BatchNorm1d for fully connected layers
    // 全結合層用のBatchNorm1dをテスト
    println!("\n🧪 BatchNorm1d test:");
    let test_bn1d = BatchNorm1d::<f32>::new(128, None, None, None);
    
    println!("   - Created with 128 features");
    println!("   - Training mode: {}", test_bn1d.is_training());
    
    // Create test input for BatchNorm1d: batch=32, features=128
    let test_input_1d = Variable::new(
        Tensor::from_vec(
            (0..32*128).map(|i| (i as f32) * 0.001).collect(),
            vec![32, 128]
        ),
        true
    );
    
    println!("   - Input shape: {:?}", test_input_1d.data().read().unwrap().shape());
    
    let output_1d = test_bn1d.forward(&test_input_1d);
    println!("   - Output shape: {:?}", output_1d.data().read().unwrap().shape());
    
    // Test parameter management
    // パラメータ管理のテスト
    println!("\n📈 Parameter Analysis:");
    
    let bn2d_params = test_bn2d.parameters();
    println!("   BatchNorm2d parameters: {}", bn2d_params.len());
    for (i, param) in bn2d_params.iter().enumerate() {
        let param_binding = param.data();
        let param_data = param_binding.read().unwrap();
        let param_count: usize = param_data.shape().iter().product();
        println!("     Parameter {}: shape {:?}, count: {}", i, param_data.shape(), param_count);
    }
    
    let bn1d_params = test_bn1d.parameters();
    println!("   BatchNorm1d parameters: {}", bn1d_params.len());
    for (i, param) in bn1d_params.iter().enumerate() {
        let param_binding = param.data();
        let param_data = param_binding.read().unwrap();
        let param_count: usize = param_data.shape().iter().product();
        println!("     Parameter {}: shape {:?}, count: {}", i, param_data.shape(), param_count);
    }
    
    // Demonstrate training with BatchNorm
    // BatchNormを使った学習のデモ
    println!("\n🎯 Training Simulation:");
    
    // Create a small CNN with BatchNorm
    let conv_layer = Conv2d::<f32>::new(3, 16, (3, 3), Some((1, 1)), Some((1, 1)), Some(true));
    let bn_layer = BatchNorm2d::<f32>::new(16, None, None, None);
    
    // Create training data batch
    let train_input = Variable::new(
        Tensor::from_vec(
            (0..8*3*16*16).map(|i| ((i as f32) / 1000.0).sin()).collect(),
            vec![8, 3, 16, 16]
        ),
        true
    );
    
    let target = Variable::new(
        Tensor::ones(&[8, 16, 16, 16]),
        false
    );
    
    println!("   Training batch input shape: {:?}", train_input.data().read().unwrap().shape());
    
    // Collect all parameters
    let mut all_params = conv_layer.parameters();
    all_params.extend(bn_layer.parameters());
    
    let mut optimizer = SGD::new(0.01, 0.9);
    
    // Training loop simulation
    for epoch in 0..3 {
        // Set training mode
        bn_layer.train();
        
        // Forward pass
        let conv_out = conv_layer.forward(&train_input);
        let bn_out = bn_layer.forward(&conv_out);
        
        // Simple loss computation
        let diff = &bn_out - &target;
        let loss = (&diff * &diff).sum().mean_autograd();
        
        // Backward pass
        loss.backward();
        
        // Update parameters
        for param in &all_params {
            let param_data = param.data();
            let param_tensor = param_data.read().unwrap();
            let grad_data = param.grad();
            let grad_guard = grad_data.read().unwrap();
            if let Some(ref grad_tensor) = *grad_guard {
                optimizer.step(&param_tensor, &grad_tensor);
            }
        }
        
        println!("   Epoch {}: Loss shape: {:?}", 
                epoch + 1, loss.data().read().unwrap().shape());
    }
    
    // Switch to evaluation mode
    println!("\n🔍 Evaluation Mode Test:");
    bn_layer.eval();
    
    let eval_out = conv_layer.forward(&train_input);
    let eval_bn_out = bn_layer.forward(&eval_out);
    
    println!("   Evaluation output shape: {:?}", eval_bn_out.data().read().unwrap().shape());
    println!("   BatchNorm in eval mode: {}", !bn_layer.is_training());
    
    // Show running statistics
    println!("\n📊 Running Statistics:");
    let running_mean = bn_layer.running_mean();
    let running_var = bn_layer.running_var();
    
    println!("   Running mean shape: {:?}", running_mean.shape());
    println!("   Running var shape: {:?}", running_var.shape());
    
    println!("\n🎉 BatchNorm Demo completed successfully!");
    println!("   - BatchNorm1d for fully connected layers ✅");
    println!("   - BatchNorm2d for convolutional layers ✅");
    println!("   - Training/Evaluation mode switching ✅");
    println!("   - Running statistics tracking ✅");
    println!("   - Parameter management ✅");
    println!("   - Integration with optimizers ✅");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batchnorm_integration() {
        // Test BatchNorm2d with Conv2d
        let conv = Conv2d::<f32>::new(1, 4, (3, 3), Some((1, 1)), Some((1, 1)), Some(true));
        let bn = BatchNorm2d::<f32>::new(4, None, None, None);
        
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 1 * 8 * 8], vec![1, 1, 8, 8]),
            false
        );
        
        let conv_out = conv.forward(&input);
        let bn_out = bn.forward(&conv_out);
        
        assert_eq!(bn_out.data().read().unwrap().shape(), &[1, 4, 8, 8]);
        
        // Test BatchNorm1d with Linear (simulated)
        let bn1d = BatchNorm1d::<f32>::new(10, None, None, None);
        let input_1d = Variable::new(
            Tensor::from_vec(vec![0.1; 5 * 10], vec![5, 10]),
            false
        );
        
        let output_1d = bn1d.forward(&input_1d);
        assert_eq!(output_1d.data().read().unwrap().shape(), &[5, 10]);
    }
}