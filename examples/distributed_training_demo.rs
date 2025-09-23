//! Distributed training demonstration
//! 分散学習デモ
//!
//! This example demonstrates how to use RusTorch's distributed training capabilities
//! with PyTorch-compatible APIs.
//!
//! この例では、PyTorch互換APIを使用したRusTorchの分散学習機能の
//! 使用方法を実演します。

use rustorch::{
    autograd::Variable,
    distributed::{self, DistributedBackend, DistributedDataParallel},
    error::RusTorchResult,
    nn::{Linear, Sequential},
    tensor::Tensor,
};
use std::time::Duration;

fn main() -> RusTorchResult<()> {
    println!("🚀 RusTorch Distributed Training Demo");
    println!("=====================================");

    // Initialize distributed training
    setup_distributed_training()?;

    // Create and train model
    let model = create_model()?;
    let ddp_model = setup_ddp_model(model)?;

    // Run training simulation
    run_training_simulation(&ddp_model)?;

    // Cleanup
    distributed::destroy_process_group()?;

    println!("✅ Distributed training demo completed successfully!");
    Ok(())
}

/// Setup distributed training environment
/// 分散学習環境をセットアップ
fn setup_distributed_training() -> RusTorchResult<()> {
    println!("📡 Setting up distributed training...");

    // Set default environment variables if not set
    if std::env::var("RANK").is_err() {
        std::env::set_var("RANK", "0");
    }
    if std::env::var("WORLD_SIZE").is_err() {
        std::env::set_var("WORLD_SIZE", "1");
    }
    if std::env::var("MASTER_ADDR").is_err() {
        std::env::set_var("MASTER_ADDR", "localhost");
    }
    if std::env::var("MASTER_PORT").is_err() {
        std::env::set_var("MASTER_PORT", "29500");
    }

    // Initialize process group
    distributed::init_process_group(
        DistributedBackend::TCP, // Use TCP for simplicity in demo
        None,                    // Use environment variables
        None,                    // Use environment variables
        None,                    // Use environment variables
        Some(Duration::from_secs(30)),
    )?;

    println!("  ✓ Process group initialized");
    println!("  ✓ Rank: {:?}", distributed::get_rank());
    println!("  ✓ World size: {:?}", distributed::get_world_size());

    Ok(())
}

/// Create a simple neural network model
/// シンプルなニューラルネットワークモデルを作成
fn create_model() -> RusTorchResult<Sequential<f32>> {
    println!("🧠 Creating neural network model...");

    let mut model = Sequential::<f32>::new();
    model.add_module(Linear::<f32>::new(784, 256));
    model.add_module(Linear::<f32>::new(256, 128));
    model.add_module(Linear::<f32>::new(128, 10));

    println!("  ✓ Model created with 3 linear layers");
    println!("  ✓ Architecture: 784 → 256 → 128 → 10");

    Ok(model)
}

/// Setup DistributedDataParallel wrapper
/// DistributedDataParallelラッパーをセットアップ
fn setup_ddp_model(
    model: Sequential<f32>,
) -> RusTorchResult<DistributedDataParallel<f32, Sequential<f32>>> {
    println!("⚙️  Setting up DistributedDataParallel...");

    let device_ids = vec![0]; // Use device 0 for demo
    let ddp_model = distributed::wrap_module(model, Some(device_ids))?;

    println!("  ✓ DDP wrapper created");
    println!("  ✓ Device IDs: {:?}", ddp_model.device_ids());

    Ok(ddp_model)
}

/// Run training simulation
/// 学習シミュレーションを実行
fn run_training_simulation(
    ddp_model: &DistributedDataParallel<f32, Sequential<f32>>,
) -> RusTorchResult<()> {
    println!("🏃 Running training simulation...");

    let batch_size = 32;
    let num_batches = 5;

    for epoch in 1..=3 {
        println!("  📊 Epoch {}/3", epoch);

        for batch in 1..=num_batches {
            // Create synthetic training data
            let input: Tensor<f32> = Tensor::randn(&[batch_size, 784]);
            let _target: Tensor<f32> = Tensor::randn(&[batch_size, 10]);

            // Forward pass
            let input_var = Variable::new(input, false);
            let output = ddp_model.forward(&input_var)?;
            // Simple shape verification without complex borrowing
            let expected_shape = [batch_size, 10];
            assert!(output.data().read().unwrap().shape() == &expected_shape[..]);

            // Simulate backward pass and gradient synchronization
            ddp_model.sync_gradients()?;

            if batch % 2 == 0 {
                println!("    ✓ Batch {}/{} completed", batch, num_batches);
            }
        }
    }

    Ok(())
}

/// Demonstrate async gradient synchronization
/// 非同期勾配同期のデモ
#[cfg(feature = "nccl")]
fn demo_async_gradient_sync() -> RusTorchResult<()> {
    use rustorch::distributed::async_gradient::{AsyncConfig, AsyncGradientSynchronizer, Priority};

    println!("⚡ Demonstrating async gradient synchronization...");

    let config = AsyncConfig {
        max_concurrent_ops: 8,
        sync_timeout: Duration::from_secs(10),
        enable_compression: false,
        compression_threshold: 1024 * 1024,
        enable_bucketing: true,
        bucket_size_mb: 25,
    };

    let synchronizer = AsyncGradientSynchronizer::new(config)?;

    // Submit several gradients for async sync
    let mut request_ids = Vec::new();
    for i in 0..5 {
        let gradient: Tensor<f32> = Tensor::randn(&[100, 100]);
        let param_name = format!("layer_{}", i);

        let request_id = synchronizer.submit_gradient(param_name, gradient, Priority::Normal)?;
        request_ids.push(request_id);
    }

    // Wait for completion
    for request_id in request_ids {
        synchronizer.wait_for_completion(request_id, Duration::from_secs(5))?;
    }

    println!("  ✓ Async gradient sync completed");
    Ok(())
}
