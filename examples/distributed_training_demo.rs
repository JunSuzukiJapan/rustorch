//! Distributed training demonstration
//! 分散学習デモンストレーション
//! 
//! This example demonstrates how to use RusTorch's distributed training capabilities
//! including data parallel training, model parallel training, and cluster management.

use rustorch::tensor::Tensor;
use rustorch::autograd::Variable;
use rustorch::nn::{Linear, Module};
use rustorch::distributed::{
    init_process_group, get_world_size, get_rank, DistributedBackend, ProcessGroup,
    data_parallel::DataParallel,
    model_parallel::ModelParallel,
    optimizer::{DistributedOptimizer, GradientSyncStrategy, DistributedOptimizerBuilder},
    backends::{BackendFactory, GlooBackend},
    cluster::{ClusterManager, ClusterConfig, NodeInfo, NodeCapabilities, NodeStatus, ClusterTopology, FaultToleranceConfig},
};
use std::sync::Arc;
use std::any::Any;

/// Simple neural network for demonstration
/// デモンストレーション用シンプルニューラルネットワーク
#[derive(Debug)]
struct SimpleNet {
    fc1: Linear<f32>,
    fc2: Linear<f32>,
    fc3: Linear<f32>,
}

impl SimpleNet {
    fn new() -> Self {
        Self {
            fc1: Linear::new(784, 256),
            fc2: Linear::new(256, 128),
            fc3: Linear::new(128, 10),
        }
    }
}

impl Module<f32> for SimpleNet {
    fn forward(&self, input: &Variable<f32>) -> Variable<f32> {
        let x = self.fc1.forward(input);
        // Note: ReLU activation would be applied here in a real implementation
        let x = self.fc2.forward(&x);
        // Note: ReLU activation would be applied here in a real implementation
        self.fc3.forward(&x)
    }
    
    fn parameters(&self) -> Vec<Variable<f32>> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RusTorch Distributed Training Demo");
    println!("=====================================");

    // Demo 1: Basic Process Group Setup
    // デモ1: 基本プロセスグループセットアップ
    demo_process_group_setup()?;

    // Demo 2: Data Parallel Training
    // デモ2: データ並列学習
    demo_data_parallel_training()?;

    // Demo 3: Model Parallel Training
    // デモ3: モデル並列学習
    demo_model_parallel_training()?;

    // Demo 4: Distributed Optimizer
    // デモ4: 分散オプティマイザー
    demo_distributed_optimizer()?;

    // Demo 5: Cluster Management
    // デモ5: クラスター管理
    demo_cluster_management()?;

    // Demo 6: Fault Tolerance
    // デモ6: 障害耐性
    demo_fault_tolerance()?;

    println!("\n✅ All distributed training demos completed successfully!");
    Ok(())
}

/// Demonstrate process group setup
/// プロセスグループセットアップのデモンストレーション
fn demo_process_group_setup() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📡 Demo 1: Process Group Setup");
    println!("------------------------------");

    // Initialize process group with TCP backend
    // TCPバックエンドでプロセスグループを初期化
    let result = init_process_group(
        DistributedBackend::TCP,
        0,    // rank
        4,    // world_size
        "localhost".to_string(),
        12345,
    );

    match result {
        Ok(_) => {
            println!("✅ Process group initialized successfully");
            println!("   Backend: TCP");
            println!("   Rank: {}", get_rank().unwrap_or(0));
            println!("   World Size: {}", get_world_size().unwrap_or(1));
        },
        Err(e) => {
            println!("⚠️  Process group initialization failed: {:?}", e);
            println!("   This is expected in single-process demo mode");
        }
    }

    // Demonstrate different backends
    // 異なるバックエンドのデモンストレーション
    let backends = [
        DistributedBackend::TCP,
        DistributedBackend::Gloo,
        #[cfg(feature = "nccl")]
        DistributedBackend::NCCL,
    ];

    for backend in &backends {
        println!("   Supported backend: {:?}", backend);
    }

    Ok(())
}

/// Demonstrate data parallel training
/// データ並列学習のデモンストレーション
fn demo_data_parallel_training() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demo 2: Data Parallel Training");
    println!("----------------------------------");

    // Create model
    // モデルを作成
    let model = SimpleNet::new();
    println!("✅ Created simple neural network model");

    // Create process group for data parallel
    // データ並列用プロセスグループを作成
    let process_group = ProcessGroup::new(
        0, 4, DistributedBackend::Gloo,
        "localhost".to_string(), 12345
    );

    // Create data parallel wrapper
    // データ並列ラッパーを作成
    let mut data_parallel = DataParallel::new(
        Box::new(model),
        vec![0, 1, 2, 3], // device_ids
        process_group,
    )?;

    println!("✅ Created data parallel wrapper");
    println!("   Devices: [0, 1, 2, 3]");
    println!("   Sync strategy: {:?}", data_parallel.get_sync_strategy());

    // Create sample input batch
    // サンプル入力バッチを作成
    let batch_size = 32;
    let input = Variable::new(Tensor::zeros(&[batch_size, 784]));
    let target = Variable::new(Tensor::zeros(&[batch_size, 10]));

    println!("✅ Created sample batch");
    println!("   Input shape: {:?}", input.data().shape());
    println!("   Target shape: {:?}", target.data().shape());

    // Forward pass
    // フォワードパス
    let output = data_parallel.forward(&input);
    println!("✅ Forward pass completed");
    println!("   Output shape: {:?}", output.data().shape());

    // Simulate gradient synchronization
    // 勾配同期をシミュレート
    match data_parallel.sync_gradients() {
        Ok(_) => println!("✅ Gradient synchronization completed"),
        Err(e) => println!("⚠️  Gradient sync failed (expected in demo): {:?}", e),
    }

    Ok(())
}

/// Demonstrate model parallel training
/// モデル並列学習のデモンストレーション
fn demo_model_parallel_training() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧩 Demo 3: Model Parallel Training");
    println!("-----------------------------------");

    // Create process group for model parallel
    // モデル並列用プロセスグループを作成
    let process_group = ProcessGroup::new(
        0, 4, DistributedBackend::Gloo,
        "localhost".to_string(), 12346
    );

    // Create model parallel manager
    // モデル並列マネージャーを作成
    let mut model_parallel = ModelParallel::new(process_group)?;

    println!("✅ Created model parallel manager");

    // Add model partitions
    // モデルパーティションを追加
    let fc1 = Linear::new(784, 256);
    let fc2 = Linear::new(256, 128);
    let fc3 = Linear::new(128, 10);

    model_parallel.add_partition(0, Box::new(fc1), vec![1])?;
    model_parallel.add_partition(1, Box::new(fc2), vec![2])?;
    model_parallel.add_partition(2, Box::new(fc3), vec![])?;

    println!("✅ Added model partitions");
    println!("   Partition 0: Linear(784 -> 256) -> Device 1");
    println!("   Partition 1: Linear(256 -> 128) -> Device 2");
    println!("   Partition 2: Linear(128 -> 10) -> Output");

    // Create sample input
    // サンプル入力を作成
    let input = Variable::new(Tensor::zeros(&[32, 784]));
    println!("✅ Created sample input: {:?}", input.data().shape());

    // Forward pass through partitioned model
    // パーティション化モデルでフォワードパス
    match model_parallel.forward(&input) {
        Ok(output) => {
            println!("✅ Model parallel forward pass completed");
            println!("   Output shape: {:?}", output.data().shape());
        },
        Err(e) => {
            println!("⚠️  Model parallel forward failed (expected in demo): {:?}", e);
        }
    }

    // Demonstrate pipeline parallelism
    // パイプライン並列性のデモンストレーション
    let pipeline_config = model_parallel.create_pipeline_config(4, 2)?;
    println!("✅ Created pipeline configuration");
    println!("   Micro-batches: {}", pipeline_config.num_microbatches);
    println!("   Pipeline stages: {}", pipeline_config.num_stages);

    Ok(())
}

/// Demonstrate distributed optimizer
/// 分散オプティマイザーのデモンストレーション
fn demo_distributed_optimizer() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Demo 4: Distributed Optimizer");
    println!("--------------------------------");

    // Create process group
    // プロセスグループを作成
    let process_group = ProcessGroup::new(
        0, 4, DistributedBackend::Gloo,
        "localhost".to_string(), 12347
    );

    // Create backend
    // バックエンドを作成
    let backend = Arc::new(GlooBackend::new(process_group)?);
    println!("✅ Created Gloo communication backend");

    // Create distributed SGD optimizer
    // 分散SGDオプティマイザーを作成
    let mut sgd_optimizer = DistributedOptimizerBuilder::sgd(0.01, 0.9, 0.0001)
        .backend(backend.clone())
        .sync_strategy(GradientSyncStrategy::Synchronous)
        .bucket_size(25 * 1024 * 1024) // 25MB buckets
        .build()?;

    println!("✅ Created distributed SGD optimizer");
    println!("   Learning rate: 0.01");
    println!("   Momentum: 0.9");
    println!("   Sync strategy: Synchronous");
    println!("   Bucket size: 25MB");

    // Create distributed Adam optimizer
    // 分散Adamオプティマイザーを作成
    let mut adam_optimizer = DistributedOptimizerBuilder::adam(0.001, 0.9, 0.999, 1e-8, 0.0001)
        .backend(backend.clone())
        .sync_strategy(GradientSyncStrategy::LocalSGD { sync_frequency: 10 })
        .build()?;

    println!("✅ Created distributed Adam optimizer");
    println!("   Learning rate: 0.001");
    println!("   Sync strategy: Local SGD (freq=10)");

    // Demonstrate different sync strategies
    // 異なる同期戦略のデモンストレーション
    let strategies = [
        GradientSyncStrategy::Synchronous,
        GradientSyncStrategy::Asynchronous,
        GradientSyncStrategy::LocalSGD { sync_frequency: 5 },
        GradientSyncStrategy::Compressed { compression_ratio: 0.1 },
        GradientSyncStrategy::Hierarchical,
    ];

    for strategy in &strategies {
        println!("   Supported strategy: {:?}", strategy);
    }

    // Simulate training step
    // 学習ステップをシミュレート
    let mut parameters = vec![
        Tensor::zeros(&[256, 784]),
        Tensor::zeros(&[256]),
        Tensor::zeros(&[128, 256]),
        Tensor::zeros(&[128]),
    ];

    match sgd_optimizer.step(&mut parameters) {
        Ok(_) => println!("✅ Distributed optimizer step completed"),
        Err(e) => println!("⚠️  Optimizer step failed (expected in demo): {:?}", e),
    }

    println!("   Step count: {}", sgd_optimizer.step_count());

    Ok(())
}

/// Demonstrate cluster management
/// クラスター管理のデモンストレーション
fn demo_cluster_management() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🖥️  Demo 5: Cluster Management");
    println!("------------------------------");

    // Create cluster configuration
    // クラスター設定を作成
    let node1 = NodeInfo {
        node_id: 0,
        address: "192.168.1.100".to_string(),
        port: 12345,
        gpu_count: 4,
        capabilities: NodeCapabilities {
            memory_gb: 64.0,
            cpu_cores: 16,
            gpu_memory_gb: 32.0,
            network_bandwidth_gbps: 10.0,
        },
        status: NodeStatus::Available,
    };

    let node2 = NodeInfo {
        node_id: 1,
        address: "192.168.1.101".to_string(),
        port: 12345,
        gpu_count: 8,
        capabilities: NodeCapabilities {
            memory_gb: 128.0,
            cpu_cores: 32,
            gpu_memory_gb: 80.0,
            network_bandwidth_gbps: 25.0,
        },
        status: NodeStatus::Available,
    };

    let cluster_config = ClusterConfig {
        master_addr: "192.168.1.1".to_string(),
        master_port: 12345,
        worker_nodes: vec![node1, node2],
        topology: ClusterTopology::Flat,
        fault_tolerance: FaultToleranceConfig {
            enable_failover: true,
            heartbeat_interval: 30,
            node_timeout: 120,
            max_retries: 3,
            checkpoint_frequency: 100,
        },
    };

    println!("✅ Created cluster configuration");
    println!("   Master: {}:{}", cluster_config.master_addr, cluster_config.master_port);
    println!("   Worker nodes: {}", cluster_config.worker_nodes.len());
    println!("   Topology: {:?}", cluster_config.topology);
    println!("   Fault tolerance: enabled");

    // Create cluster manager
    // クラスターマネージャーを作成
    let mut cluster_manager = ClusterManager::new(cluster_config)?;
    println!("✅ Created cluster manager");

    // Get cluster status
    // クラスターステータスを取得
    let status = cluster_manager.get_cluster_status();
    println!("✅ Cluster status:");
    println!("   Total nodes: {}", status.total_nodes);
    println!("   Available nodes: {}", status.available_nodes);
    println!("   Total GPUs: {}", status.total_gpus);
    println!("   Active jobs: {}", status.active_jobs);

    // Create process group for distributed job
    // 分散ジョブ用プロセスグループを作成
    match cluster_manager.create_process_group(
        "training_job_1".to_string(),
        4,
        DistributedBackend::Gloo,
    ) {
        Ok(pg) => {
            println!("✅ Created process group for distributed job");
            println!("   Job ID: training_job_1");
            println!("   World size: {}", pg.world_size);
            println!("   Backend: {:?}", pg.backend);
        },
        Err(e) => {
            println!("⚠️  Process group creation failed (expected in demo): {:?}", e);
        }
    }

    Ok(())
}

/// Demonstrate fault tolerance
/// 障害耐性のデモンストレーション
fn demo_fault_tolerance() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🛡️  Demo 6: Fault Tolerance");
    println!("---------------------------");

    // Create cluster with fault tolerance enabled
    // 障害耐性を有効にしたクラスターを作成
    let fault_config = FaultToleranceConfig {
        enable_failover: true,
        heartbeat_interval: 10, // 10 seconds
        node_timeout: 30,       // 30 seconds
        max_retries: 5,
        checkpoint_frequency: 50, // Every 50 steps
    };

    println!("✅ Fault tolerance configuration:");
    println!("   Failover: enabled");
    println!("   Heartbeat interval: {} seconds", fault_config.heartbeat_interval);
    println!("   Node timeout: {} seconds", fault_config.node_timeout);
    println!("   Max retries: {}", fault_config.max_retries);
    println!("   Checkpoint frequency: {} steps", fault_config.checkpoint_frequency);

    // Simulate node failure and recovery
    // ノード障害と復旧をシミュレート
    let node = NodeInfo {
        node_id: 0,
        address: "192.168.1.100".to_string(),
        port: 12345,
        gpu_count: 4,
        capabilities: NodeCapabilities {
            memory_gb: 64.0,
            cpu_cores: 16,
            gpu_memory_gb: 32.0,
            network_bandwidth_gbps: 10.0,
        },
        status: NodeStatus::Available,
    };

    let cluster_config = ClusterConfig {
        master_addr: "192.168.1.1".to_string(),
        master_port: 12345,
        worker_nodes: vec![node],
        topology: ClusterTopology::Flat,
        fault_tolerance: fault_config,
    };

    let mut cluster_manager = ClusterManager::new(cluster_config)?;

    // Simulate node failure
    // ノード障害をシミュレート
    println!("🔥 Simulating node failure...");
    match cluster_manager.handle_node_failure(0) {
        Ok(_) => println!("✅ Node failure handled successfully"),
        Err(e) => println!("⚠️  Node failure handling failed (expected in demo): {:?}", e),
    }

    // Show recovery mechanisms
    // 復旧メカニズムを表示
    println!("🔄 Recovery mechanisms:");
    println!("   • Automatic failover to backup nodes");
    println!("   • Job migration and state restoration");
    println!("   • Gradient synchronization recovery");
    println!("   • Checkpoint-based model recovery");

    Ok(())
}

/// Helper function to print section separator
/// セクション区切りを印刷するヘルパー関数
#[allow(dead_code)]
fn print_separator(title: &str) {
    println!("\n{}", "=".repeat(50));
    println!("{}", title);
    println!("{}", "=".repeat(50));
}
