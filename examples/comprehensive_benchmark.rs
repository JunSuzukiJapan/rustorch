//! Comprehensive benchmark for RusTorch performance
//! RusTorchの包括的性能ベンチマーク

use rustorch::autograd::Variable;
use rustorch::nn::{Linear, Module};
use rustorch::tensor::Tensor;
use std::time::Instant;

fn main() {
    println!("🚀 RusTorch Comprehensive Benchmark 🚀");
    println!("=====================================\n");

    // 1. Tensor Operations Benchmark
    println!("📊 1. Tensor Operations Performance");
    benchmark_tensor_operations();

    // 2. Matrix Operations Benchmark
    println!("\n📊 2. Matrix Operations Performance");
    benchmark_matrix_operations();

    // 3. Neural Network Benchmark
    println!("\n📊 3. Neural Network Performance");
    benchmark_neural_networks();

    // 4. Memory Operations Benchmark
    println!("\n📊 4. Memory Operations Performance");
    benchmark_memory_operations();

    // 5. Advanced Operations Benchmark
    println!("\n📊 5. Advanced Operations Performance");
    benchmark_advanced_operations();

    println!("\n✅ Comprehensive Benchmark Complete!");
    println!("🎉 RusTorch is ready for production use!");
}

fn benchmark_tensor_operations() {
    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let a = Tensor::<f32>::from_vec((0..size).map(|i| i as f32).collect(), vec![size]);
        let b = Tensor::<f32>::from_vec((0..size).map(|i| (i + 1) as f32).collect(), vec![size]);

        // Addition benchmark
        let start = Instant::now();
        for _ in 0..1000 {
            let _result = &a + &b;
        }
        let add_time = start.elapsed();

        // Sum benchmark
        let start = Instant::now();
        for _ in 0..1000 {
            let _sum = a.sum();
        }
        let sum_time = start.elapsed();

        // Mean benchmark
        let start = Instant::now();
        for _ in 0..1000 {
            let _mean = a.mean();
        }
        let mean_time = start.elapsed();

        println!("  Size: {} elements", size);
        println!(
            "    Addition (1000x): {:?} ({:.2} Mops/sec)",
            add_time,
            1000.0 / add_time.as_secs_f64() / 1_000_000.0
        );
        println!(
            "    Sum (1000x): {:?} ({:.2} Mops/sec)",
            sum_time,
            1000.0 / sum_time.as_secs_f64() / 1_000_000.0
        );
        println!(
            "    Mean (1000x): {:?} ({:.2} Mops/sec)",
            mean_time,
            1000.0 / mean_time.as_secs_f64() / 1_000_000.0
        );
    }
}

fn benchmark_matrix_operations() {
    let sizes = vec![32, 64, 128, 256];

    for size in sizes {
        let a = Tensor::<f32>::from_vec(
            (0..size * size).map(|i| i as f32 * 0.01).collect(),
            vec![size, size],
        );
        let b = Tensor::<f32>::from_vec(
            (0..size * size).map(|i| (i + 1) as f32 * 0.01).collect(),
            vec![size, size],
        );

        // Matrix multiplication benchmark
        let start = Instant::now();
        let iterations = if size <= 128 { 100 } else { 10 };
        for _ in 0..iterations {
            let _result = a.matmul(&b).unwrap();
        }
        let matmul_time = start.elapsed();

        // Calculate GFLOPS
        let ops_per_matmul = 2.0 * (size as f64).powi(3);
        let total_ops = ops_per_matmul * iterations as f64;
        let gflops = total_ops / matmul_time.as_secs_f64() / 1e9;

        // Transpose benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _result = a.transpose().unwrap();
        }
        let transpose_time = start.elapsed();

        println!("  Matrix size: {}×{}", size, size);
        println!(
            "    MatMul ({}x): {:?} ({:.2} GFLOPS)",
            iterations, matmul_time, gflops
        );
        println!(
            "    Transpose ({}x): {:?} ({:.2} ops/sec)",
            iterations,
            transpose_time,
            iterations as f64 / transpose_time.as_secs_f64()
        );
    }
}

fn benchmark_neural_networks() {
    let batch_sizes = vec![16, 32, 64, 128];
    let input_dim = 128;
    let hidden_dim = 256;
    let output_dim = 64;

    for batch_size in batch_sizes {
        let linear1 = Linear::<f32>::new(input_dim, hidden_dim);
        let linear2 = Linear::<f32>::new(hidden_dim, output_dim);

        let input_data = (0..batch_size * input_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![batch_size, input_dim]),
            false,
        );

        // Forward pass benchmark
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let hidden = linear1.forward(&input);
            let _output = linear2.forward(&hidden);
        }
        let forward_time = start.elapsed();

        // Parameter count
        let params = linear1.parameters().len() + linear2.parameters().len();

        println!("  Batch size: {}", batch_size);
        println!(
            "    Forward pass ({}x): {:?} ({:.2} inferences/sec)",
            iterations,
            forward_time,
            iterations as f64 / forward_time.as_secs_f64()
        );
        println!("    Parameters: {}", params);
        println!(
            "    Throughput: {:.2} samples/sec",
            (iterations * batch_size) as f64 / forward_time.as_secs_f64()
        );
    }
}

fn benchmark_memory_operations() {
    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        // Tensor creation benchmark
        let start = Instant::now();
        for _ in 0..1000 {
            let _tensor = Tensor::<f32>::zeros(&[size]);
        }
        let creation_time = start.elapsed();

        // Tensor cloning benchmark
        let tensor = Tensor::<f32>::from_vec((0..size).map(|i| i as f32).collect(), vec![size]);

        let start = Instant::now();
        for _ in 0..1000 {
            let _clone = tensor.clone();
        }
        let clone_time = start.elapsed();

        // Memory access benchmark
        let start = Instant::now();
        for _ in 0..1000 {
            let _shape = tensor.shape();
            let _numel = tensor.numel();
        }
        let access_time = start.elapsed();

        println!("  Size: {} elements", size);
        println!(
            "    Creation (1000x): {:?} ({:.2} MB/sec)",
            creation_time,
            (size * 4 * 1000) as f64 / creation_time.as_secs_f64() / 1_000_000.0
        );
        println!(
            "    Clone (1000x): {:?} ({:.2} MB/sec)",
            clone_time,
            (size * 4 * 1000) as f64 / clone_time.as_secs_f64() / 1_000_000.0
        );
        println!(
            "    Access (1000x): {:?} ({:.2} Mops/sec)",
            access_time,
            1000.0 / access_time.as_secs_f64() / 1_000_000.0
        );
    }
}

fn benchmark_advanced_operations() {
    let size = 10000;
    let tensor = Tensor::<f32>::from_vec((0..size).map(|i| (i as f32).sin()).collect(), vec![size]);

    // Broadcasting benchmark
    let scalar_tensor = Tensor::<f32>::from_vec(vec![2.0], vec![1]);
    let start = Instant::now();
    for _ in 0..1000 {
        let _result = tensor.add_v2(&scalar_tensor).unwrap();
    }
    let broadcast_time = start.elapsed();

    // Reshape benchmark
    let matrix_tensor =
        Tensor::<f32>::from_vec((0..10000).map(|i| i as f32).collect(), vec![100, 100]);

    let start = Instant::now();
    for _ in 0..1000 {
        let _reshaped = matrix_tensor.reshape(&[200, 50]).unwrap();
    }
    let reshape_time = start.elapsed();

    // Statistical operations benchmark
    let start = Instant::now();
    for _ in 0..1000 {
        let _sum = tensor.sum();
        let _mean = tensor.mean();
    }
    let stats_time = start.elapsed();

    println!("  Advanced Operations:");
    println!(
        "    Broadcasting (1000x): {:?} ({:.2} Mops/sec)",
        broadcast_time,
        1000.0 / broadcast_time.as_secs_f64() / 1_000_000.0
    );
    println!(
        "    Reshape (1000x): {:?} ({:.2} Mops/sec)",
        reshape_time,
        1000.0 / reshape_time.as_secs_f64() / 1_000_000.0
    );
    println!(
        "    Statistics (1000x): {:?} ({:.2} Mops/sec)",
        stats_time,
        1000.0 / stats_time.as_secs_f64() / 1_000_000.0
    );
}
