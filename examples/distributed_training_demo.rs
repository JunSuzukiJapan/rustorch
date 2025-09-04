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
            let expected_shape = vec![batch_size, 10];
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

/// Benchmark different communication patterns
/// 異なる通信パターンのベンチマーク
fn benchmark_communication() -> RusTorchResult<()> {
    println!("⏱️  Benchmarking communication patterns...");

    let tensor_sizes = vec![
        (vec![100], "small"),
        (vec![1000], "medium"),
        (vec![10000], "large"),
    ];

    for (shape, label) in tensor_sizes {
        let tensor: Tensor<f32> = Tensor::randn(&shape);
        let start = std::time::Instant::now();

        // Test all-reduce
        let mut tensor_copy = tensor.clone();
        distributed::all_reduce(&mut tensor_copy, distributed::ReduceOp::Sum, None, false)?;
        let all_reduce_time = start.elapsed();

        // Test broadcast
        let start = std::time::Instant::now();
        let mut tensor_copy = tensor.clone();
        distributed::broadcast(&mut tensor_copy, 0, None, false)?;
        let broadcast_time = start.elapsed();

        // Test all-gather
        let start = std::time::Instant::now();
        let mut tensor_list = Vec::new();
        distributed::all_gather(&mut tensor_list, &tensor, None, false)?;
        let all_gather_time = start.elapsed();

        println!(
            "  {} tensor ({} elements):",
            label,
            shape.iter().product::<usize>()
        );
        println!("    All-reduce: {:?}", all_reduce_time);
        println!("    Broadcast:  {:?}", broadcast_time);
        println!("    All-gather: {:?}", all_gather_time);
    }

    Ok(())
}

/// Demonstrate different backends
/// 異なるバックエンドのデモ
fn demo_backends() -> RusTorchResult<()> {
    println!("🔧 Testing different communication backends...");

    let backends = vec![
        (DistributedBackend::TCP, "TCP"),
        (DistributedBackend::Gloo, "Gloo"),
    ];

    for (backend, name) in backends {
        println!("  Testing {} backend:", name);

        // Clean up any previous state
        let _ = distributed::destroy_process_group();

        // Try to initialize with this backend
        let result = distributed::init_process_group(
            backend,
            Some("tcp://localhost:29500"),
            Some(1),
            Some(0),
            Some(Duration::from_secs(5)),
        );

        match result {
            Ok(()) => {
                println!("    ✓ {} backend initialized successfully", name);

                // Test basic operation
                let mut tensor: Tensor<f32> = Tensor::ones(&[10]);
                let op_result =
                    distributed::all_reduce(&mut tensor, distributed::ReduceOp::Sum, None, false);

                if op_result.is_ok() {
                    println!("    ✓ All-reduce operation successful");
                } else {
                    println!("    ⚠️  All-reduce operation failed: {:?}", op_result);
                }
            }
            Err(e) => {
                println!("    ❌ {} backend failed to initialize: {}", name, e);
            }
        }
    }

    Ok(())
}

/// Advanced features demonstration
/// 高度機能のデモ
fn demo_advanced_features() -> RusTorchResult<()> {
    println!("🚀 Demonstrating advanced distributed features...");

    // Test custom process groups
    let ranks = vec![0];
    let custom_group = distributed::new_group(ranks, Some(Duration::from_secs(10)), None)?;
    println!(
        "  ✓ Custom process group created (size: {})",
        custom_group.size()
    );

    // Test monitoring
    let stats = distributed::monitoring::get_communication_stats()?;
    println!("  ✓ Communication stats retrieved");
    println!("    Operations: {}", stats.total_operations);
    println!("    Avg latency: {:.2}ms", stats.average_latency_ms);

    #[cfg(feature = "nccl")]
    {
        // Test NCCL-specific features
        use rustorch::distributed::nccl_integration::NCCLOps;

        let nccl_config = NCCLOps::get_optimal_config(4, 16.0);
        println!("  ✓ NCCL optimal config generated");
        println!("    Bucket size: {}MB", nccl_config.bucket_size_mb);
        println!("    Streams: {}", nccl_config.num_streams);
        println!("    Compression: {}", nccl_config.compression_enabled);
    }

    #[cfg(feature = "nccl")]
    {
        // Test async features
        demo_async_gradient_sync()?;
    }

    Ok(())
}

/// Print system information
/// システム情報を印刷
fn print_system_info() {
    println!("💻 System Information:");
    println!("  OS: {}", std::env::consts::OS);
    println!("  Architecture: {}", std::env::consts::ARCH);

    if let Ok(hostname) = hostname::get() {
        println!("  Hostname: {:?}", hostname);
    }

    #[cfg(feature = "cuda")]
    {
        use rustorch::gpu::DeviceType;
        let cuda_available = DeviceType::Cuda(0).is_available();
        println!("  CUDA available: {}", cuda_available);
    }

    #[cfg(feature = "nccl")]
    {
        println!("  NCCL support: enabled");
    }
    #[cfg(not(feature = "nccl"))]
    {
        println!("  NCCL support: disabled");
    }

    println!();
}
