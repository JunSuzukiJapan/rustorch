//! Convolutional Neural Network Demo
//! 畳み込みニューラルネットワークのデモ

use rustorch::nn::{Conv2d, MaxPool2d, Sequential};
use rustorch::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 RusTorch CNN Demo");
    println!("==================");

    // Create a simple CNN for MNIST-like data (1x28x28)
    // MNIST風データ用のシンプルなCNNを作成 (1x28x28)
    let mut cnn = Sequential::<f32>::new();

    // First conv block: 1 -> 32 channels
    // 最初の畳み込みブロック: 1 -> 32チャンネル
    let conv1 = Conv2d::new(
        1,            // in_channels
        32,           // out_channels
        (3, 3),       // kernel_size
        Some((1, 1)), // stride
        Some((1, 1)), // padding
        Some(true),   // bias
    );
    cnn.add_module(conv1);

    // First pooling: 28x28 -> 14x14
    // 最初のプーリング: 28x28 -> 14x14
    let pool1 = MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
    cnn.add_module(pool1);

    // Second conv block: 32 -> 64 channels
    // 2番目の畳み込みブロック: 32 -> 64チャンネル
    let conv2 = Conv2d::new(
        32,           // in_channels
        64,           // out_channels
        (3, 3),       // kernel_size
        Some((1, 1)), // stride
        Some((1, 1)), // padding
        Some(true),   // bias
    );
    cnn.add_module(conv2);

    // Second pooling: 14x14 -> 7x7
    // 2番目のプーリング: 14x14 -> 7x7
    let pool2 = MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
    cnn.add_module(pool2);

    // Note: In a real implementation, we would need a Flatten layer here
    // 注意: 実際の実装では、ここでFlattenレイヤーが必要です

    println!("✅ CNN Architecture created");
    println!("   - Conv2d(1→32, 3x3, padding=1)");
    println!("   - MaxPool2d(2x2, stride=2)");
    println!("   - Conv2d(32→64, 3x3, padding=1)");
    println!("   - MaxPool2d(2x2, stride=2)");

    // Create sample input: batch_size=2, channels=1, height=28, width=28
    // サンプル入力を作成: batch_size=2, channels=1, height=28, width=28
    let batch_size = 2;
    let input_data: Vec<f32> = (0..batch_size * 1 * 28 * 28)
        .map(|i| (i as f32) / 1000.0) // Normalize to small values
        .collect();

    let input = Variable::new(
        Tensor::from_vec(input_data, vec![batch_size, 1, 28, 28]),
        true, // requires_grad for training
    );

    println!(
        "\n📊 Input tensor shape: {:?}",
        input.data().read().unwrap().shape()
    );

    // Test individual layers
    // 個別レイヤーをテスト

    // Test Conv2d
    let conv_test = Conv2d::<f32>::new(1, 16, (3, 3), Some((1, 1)), Some((1, 1)), Some(true));
    let conv_output = conv_test.forward(&input);
    println!(
        "🔍 Conv2d output shape: {:?}",
        conv_output.data().read().unwrap().shape()
    );

    // Test MaxPool2d
    let pool_test = MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
    let pool_output = pool_test.forward(&conv_output);
    println!(
        "🏊 MaxPool2d output shape: {:?}",
        pool_output.data().read().unwrap().shape()
    );

    // Demonstrate parameter counting
    // パラメータ数のデモ
    let conv_params = conv_test.parameters();
    println!("\n📈 Conv2d parameters:");
    for (i, param) in conv_params.iter().enumerate() {
        let param_binding = param.data();
        let param_data = param_binding.read().unwrap();
        let param_count: usize = param_data.shape().iter().product();
        println!(
            "   Parameter {}: shape {:?}, count: {}",
            i,
            param_data.shape(),
            param_count
        );
    }

    // Test with different input sizes
    // 異なる入力サイズでのテスト
    println!("\n🧪 Testing different input sizes:");

    let test_sizes = vec![(32, 32), (64, 64), (128, 128)];

    for (h, w) in test_sizes {
        let test_input = Variable::new(
            Tensor::from_vec(vec![0.0; 1 * 1 * h * w], vec![1, 1, h, w]),
            false,
        );

        let conv_out = conv_test.forward(&test_input);
        let pool_out = pool_test.forward(&conv_out);

        println!(
            "   Input {}x{} → Conv: {:?} → Pool: {:?}",
            h,
            w,
            conv_out.data().read().unwrap().shape(),
            pool_out.data().read().unwrap().shape()
        );
    }

    // Demonstrate training setup
    // 学習設定のデモ
    println!("\n🎯 Training Setup Demo:");

    // Create optimizer
    let all_params = conv_test.parameters();
    let mut optimizer = SGD::new(0.01);

    // Simulate training step
    let target = Variable::new(
        Tensor::ones(&[batch_size, 16, 28, 28]), // Dummy target
        false,
    );

    // Forward pass
    let prediction = conv_test.forward(&input);

    // Compute loss (simplified)
    let diff = &prediction - &target;
    let loss = (&diff * &diff).sum().mean_autograd();

    println!("   Forward pass completed");
    println!("   Loss shape: {:?}", loss.data().read().unwrap().shape());

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

    println!("   Backward pass and optimization step completed");

    println!("\n🎉 CNN Demo completed successfully!");
    println!("   - Conv2d layers working ✅");
    println!("   - MaxPool2d layers working ✅");
    println!("   - Parameter management working ✅");
    println!("   - Basic training loop working ✅");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cnn_components() {
        // Test Conv2d
        let conv = Conv2d::<f32>::new(3, 16, (3, 3), Some((1, 1)), Some((1, 1)), Some(true));
        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 1 * 3 * 32 * 32], vec![1, 3, 32, 32]),
            false,
        );
        let output = conv.forward(&input);
        assert_eq!(output.data().read().unwrap().shape(), &[1, 16, 32, 32]);

        // Test MaxPool2d
        let pool = MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
        let pooled = pool.forward(&output);
        assert_eq!(pooled.data().read().unwrap().shape(), &[1, 16, 16, 16]);
    }
}
