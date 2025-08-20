# RusTorch GPU Acceleration Guide
## GPU加速利用ガイド

**RusTorch GPU カーネル実装完了版 - 本番環境対応**

## 📋 概要

RusTorch は CUDA、Metal、OpenCL の統一 GPU カーネルインターフェースを提供し、自動デバイス選択による透過的な GPU 加速を実現します。

### サポート GPU バックエンド
- **CUDA**: NVIDIA GPU 向け cuBLAS 統合高性能カーネル
- **Metal**: Apple Silicon 向け Metal Performance Shaders 最適化
- **OpenCL**: クロスプラットフォーム GPU 対応

## 🚀 クイックスタート

### 1. 依存関係の設定

```toml
[dependencies]
rustorch = "0.1.8"

[features]
# 単一 GPU バックエンド
cuda = ["rustorch/cuda"]      # NVIDIA CUDA
metal = ["rustorch/metal"]    # Apple Metal
opencl = ["rustorch/opencl"]  # OpenCL

# 全 GPU バックエンド有効化
all-gpu = ["rustorch/all-gpu"]
```

### 2. 基本的な GPU カーネル使用例

```rust
use rustorch::gpu::{
    DeviceType, 
    kernels::{KernelExecutor, AddKernel, MatMulKernel}
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 利用可能デバイスの自動検出
    let available_devices = DeviceType::available_devices();
    println!("利用可能デバイス: {:?}", available_devices);
    
    // 最適デバイスの自動選択
    let device = DeviceType::best_available();
    let executor = KernelExecutor::new(device);
    
    // GPU 上での要素ごと加算
    let a = vec![1.0f32; 1024];
    let b = vec![2.0f32; 1024];
    let mut c = vec![0.0f32; 1024];
    
    let kernel = AddKernel;
    let inputs = [a.as_slice(), b.as_slice()];
    let mut outputs = [c.as_mut_slice()];
    
    executor.execute_kernel(&kernel, &inputs, &mut outputs)?;
    
    println!("GPU 計算完了: 結果の最初の5要素 {:?}", &c[..5]);
    
    Ok(())
}
```

## 🎯 高度な使用例

### GPU 行列乗算

```rust
use rustorch::gpu::{DeviceType, kernels::{KernelExecutor, MatMulKernel}};

fn gpu_matrix_multiplication() -> Result<(), Box<dyn std::error::Error>> {
    let device = DeviceType::best_available();
    let executor = KernelExecutor::new(device);
    
    // 行列サイズ設定
    let m = 512;
    let n = 512; 
    let k = 512;
    
    // 行列データ準備
    let a = vec![1.0f32; m * k];
    let b = vec![2.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    
    let kernel = MatMulKernel;
    let inputs = [a.as_slice(), b.as_slice()];
    let mut outputs = [c.as_mut_slice()];
    
    // GPU での高性能行列乗算実行
    executor.execute_kernel(&kernel, &inputs, &mut outputs)?;
    
    println!("{}x{} 行列乗算完了", m, n);
    
    Ok(())
}
```

### デバイス固有の実行

```rust
use rustorch::gpu::{DeviceType, kernels::KernelExecutor};

fn device_specific_execution() -> Result<(), Box<dyn std::error::Error>> {
    // CUDA デバイス指定
    if let Ok(executor) = KernelExecutor::new(DeviceType::Cuda(0)) {
        println!("CUDA GPU で実行中...");
        // CUDA 固有の処理
    }
    
    // Metal デバイス指定
    if let Ok(executor) = KernelExecutor::new(DeviceType::Metal(0)) {
        println!("Metal GPU で実行中...");
        // Metal 固有の処理
    }
    
    // OpenCL デバイス指定
    if let Ok(executor) = KernelExecutor::new(DeviceType::OpenCl(0)) {
        println!("OpenCL GPU で実行中...");
        // OpenCL 固有の処理
    }
    
    // CPU フォールバック
    let executor = KernelExecutor::new(DeviceType::Cpu);
    println!("CPU で実行中...");
    
    Ok(())
}
```

## 🔧 GPU カーネル検証

### 検証フレームワークの使用

```rust
use rustorch::gpu::validation::{GpuValidator, print_gpu_validation_report};

fn validate_gpu_kernels() -> Result<(), Box<dyn std::error::Error>> {
    // GPU カーネル検証の実行
    let validator = GpuValidator::new();
    let report = validator.run_validation()?;
    
    // 検証結果の表示
    print_gpu_validation_report(&report);
    
    // 個別結果の確認
    for result in &report.results {
        if !result.passed {
            println!("検証失敗: {} on {:?}", result.operation, result.device);
            if let Some(error) = &result.error_message {
                println!("エラー: {}", error);
            }
        }
    }
    
    Ok(())
}
```

## 📊 パフォーマンス測定

### GPU vs CPU 性能比較

```rust
use rustorch::gpu::{DeviceType, kernels::{KernelExecutor, AddKernel}};
use std::time::Instant;

fn performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let size = 1_000_000;
    let a = vec![1.0f32; size];
    let b = vec![2.0f32; size];
    let mut c_gpu = vec![0.0f32; size];
    let mut c_cpu = vec![0.0f32; size];
    
    // GPU 実行時間測定
    let gpu_device = DeviceType::best_available();
    let gpu_executor = KernelExecutor::new(gpu_device);
    let kernel = AddKernel;
    
    let start = Instant::now();
    let inputs = [a.as_slice(), b.as_slice()];
    let mut outputs = [c_gpu.as_mut_slice()];
    gpu_executor.execute_kernel(&kernel, &inputs, &mut outputs)?;
    let gpu_time = start.elapsed();
    
    // CPU 実行時間測定
    let cpu_executor = KernelExecutor::new(DeviceType::Cpu);
    let start = Instant::now();
    let inputs = [a.as_slice(), b.as_slice()];
    let mut outputs = [c_cpu.as_mut_slice()];
    cpu_executor.execute_kernel(&kernel, &inputs, &mut outputs)?;
    let cpu_time = start.elapsed();
    
    println!("GPU 実行時間: {:?}", gpu_time);
    println!("CPU 実行時間: {:?}", cpu_time);
    println!("GPU 高速化率: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    
    Ok(())
}
```

## 🛠️ トラブルシューティング

### よくある問題と解決方法

#### 1. GPU デバイスが検出されない

```rust
use rustorch::gpu::DeviceType;

fn debug_device_detection() {
    let devices = DeviceType::available_devices();
    
    if devices.is_empty() {
        println!("GPU デバイスが検出されませんでした");
        println!("- GPU ドライバーがインストールされているか確認");
        println!("- 適切な機能フラグが有効化されているか確認");
        println!("- CPU フォールバックを使用します");
    } else {
        println!("検出されたデバイス: {:?}", devices);
    }
}
```

#### 2. CUDA エラーの処理

```rust
use rustorch::gpu::{DeviceType, kernels::KernelExecutor, GpuError};

fn handle_cuda_errors() {
    match KernelExecutor::new(DeviceType::Cuda(0)) {
        Ok(executor) => {
            println!("CUDA 初期化成功");
        }
        Err(GpuError::InitializationError(msg)) => {
            println!("CUDA 初期化失敗: {}", msg);
            println!("CPU フォールバックに切り替え");
        }
        Err(e) => {
            println!("予期しないエラー: {:?}", e);
        }
    }
}
```

#### 3. メモリ不足エラーの対処

```rust
use rustorch::gpu::{GpuError, kernels::KernelExecutor};

fn handle_memory_errors(executor: &KernelExecutor) -> Result<(), GpuError> {
    // 大きなデータの処理時
    let large_data = vec![1.0f32; 100_000_000]; // 100M 要素
    
    match process_large_data(executor, &large_data) {
        Err(GpuError::MemoryAllocationError(_)) => {
            println!("GPU メモリ不足、バッチサイズを削減");
            // バッチ処理に分割
            process_in_batches(executor, &large_data)?;
        }
        result => result?,
    }
    
    Ok(())
}

fn process_large_data(executor: &KernelExecutor, data: &[f32]) -> Result<(), GpuError> {
    // 大きなデータの処理実装
    Ok(())
}

fn process_in_batches(executor: &KernelExecutor, data: &[f32]) -> Result<(), GpuError> {
    let batch_size = 1_000_000;
    for chunk in data.chunks(batch_size) {
        // バッチごとの処理
    }
    Ok(())
}
```

## 📈 最適化のベストプラクティス

### 1. 適切なバッチサイズの選択

```rust
fn optimize_batch_size() {
    // 小さすぎるバッチ: GPU の並列性を活用できない
    let small_batch = 32;
    
    // 大きすぎるバッチ: メモリ不足の可能性
    let large_batch = 100_000_000;
    
    // 推奨バッチサイズ: GPU メモリとワークロードに応じて調整
    let optimal_batch = 10_000;
}
```

### 2. メモリ転送の最小化

```rust
fn minimize_memory_transfers() {
    // 悪い例: 頻繁な CPU-GPU 間転送
    // for i in 0..1000 {
    //     gpu_operation(small_data[i]);
    // }
    
    // 良い例: バッチ処理でメモリ転送を最小化
    // gpu_batch_operation(large_batch_data);
}
```

### 3. 非同期実行の活用

```rust
use rustorch::gpu::kernels::KernelExecutor;

fn async_gpu_execution() {
    // 複数の独立した GPU 操作を並列実行
    // (将来の実装で非同期サポート予定)
}
```

## 🧪 テストとベンチマーク

### GPU カーネルテストの実行

```bash
# 全テスト実行 (GPU 検証含む)
cargo test --release

# GPU 固有テスト
cargo test gpu --release

# GPU カーネルベンチマーク
cargo bench --bench gpu_kernel_performance

# GPU vs CPU 比較ベンチマーク
cargo bench --bench gpu_cpu_performance
```

### カスタムベンチマークの作成

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rustorch::gpu::{DeviceType, kernels::{KernelExecutor, AddKernel}};

fn gpu_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU Operations");
    
    for size in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("GPU Add", size),
            size,
            |b, &size| {
                let a = vec![1.0f32; size];
                let b_vec = vec![2.0f32; size];
                let mut c = vec![0.0f32; size];
                
                let executor = KernelExecutor::new(DeviceType::best_available());
                let kernel = AddKernel;
                
                b.iter(|| {
                    let inputs = [a.as_slice(), b_vec.as_slice()];
                    let mut outputs = [c.as_mut_slice()];
                    executor.execute_kernel(&kernel, &inputs, &mut outputs)
                        .expect("GPU kernel execution failed");
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, gpu_benchmark);
criterion_main!(benches);
```

## 📚 API リファレンス

### 主要な型とトレイト

- `DeviceType`: GPU デバイスタイプの列挙型
- `KernelExecutor`: GPU カーネル実行器
- `GpuKernel`: GPU カーネルトレイト
- `GpuError`: GPU エラー型
- `GpuValidator`: GPU カーネル検証器

### 利用可能なカーネル

- `AddKernel`: 要素ごと加算
- `MatMulKernel`: 行列乗算
- `ReduceKernel`: リダクション操作

## 🔗 関連リソース

- [RusTorch メインドキュメント](../README.md)
- [パフォーマンス分析レポート](../PERFORMANCE_ANALYSIS.md)
- [GPU カーネルデモ](../examples/gpu_kernel_demo.rs)
- [GPU vs CPU ベンチマーク](../benches/gpu_cpu_performance.rs)

---

**RusTorch GPU 加速は本番環境対応の完全実装です。CUDA、Metal、OpenCL の統一インターフェースにより、最適な GPU 性能を簡単に活用できます。**
