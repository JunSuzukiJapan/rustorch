use rustorch::autograd::Variable;
use rustorch::nn::Linear;
use rustorch::tensor::Tensor;
use std::time::Instant;

fn main() {
    println!("\n🚀 RusTorch Performance Benchmark 🚀\n");

    // Test 1: Basic Tensor Operations
    println!("📊 1. Basic Tensor Operations");
    benchmark_basic_ops();

    // Test 2: Matrix Multiplication
    println!("\n📊 2. Matrix Multiplication");
    benchmark_matmul();

    // Test 3: Neural Network Operations
    println!("\n📊 3. Neural Network Operations");
    benchmark_nn_ops();

    println!("\n✅ Benchmark Complete!");
}

fn benchmark_basic_ops() {
    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        println!("  Size: {} elements", size);

        let a = Tensor::<f32>::from_vec((0..size).map(|i| i as f32 * 0.01).collect(), vec![size]);
        let b = Tensor::<f32>::from_vec(
            (0..size).map(|i| (i + 1) as f32 * 0.01).collect(),
            vec![size],
        );

        // Addition benchmark
        let start = Instant::now();
        for _ in 0..100 {
            let _result = a.add_v2(&b).unwrap();
        }
        let add_time = start.elapsed();

        // Sum benchmark
        let start = Instant::now();
        for _ in 0..100 {
            let _sum = a.sum_v2();
        }
        let sum_time = start.elapsed();

        println!(
            "    Addition (100x): {:?} ({:.2} ops/sec)",
            add_time,
            100.0 / add_time.as_secs_f64()
        );
        println!(
            "    Sum (100x): {:?} ({:.2} ops/sec)",
            sum_time,
            100.0 / sum_time.as_secs_f64()
        );
    }
}

fn benchmark_matmul() {
    let sizes = vec![64, 128, 256, 512];

    for size in sizes {
        println!("  Matrix size: {}x{}", size, size);

        let a = Tensor::<f32>::from_vec(
            (0..size * size).map(|i| i as f32 * 0.01).collect(),
            vec![size, size],
        );
        let b = Tensor::<f32>::from_vec(
            (0..size * size).map(|i| (i + 1) as f32 * 0.01).collect(),
            vec![size, size],
        );

        let start = Instant::now();
        for _ in 0..10 {
            let _result = a.matmul_v2(&b).unwrap();
        }
        let matmul_time = start.elapsed();

        let ops_per_matmul = 2.0 * (size as f64).powi(3);
        let total_ops = ops_per_matmul * 10.0;
        let gflops = total_ops / matmul_time.as_secs_f64() / 1e9;

        println!("    MatMul (10x): {:?} ({:.2} GFLOPS)", matmul_time, gflops);
    }
}

fn benchmark_nn_ops() {
    let batch_sizes = vec![32, 64, 128];
    let input_dim = 256;
    let hidden_dim = 512;
    let output_dim = 128;

    for batch_size in batch_sizes {
        println!("  Batch size: {}", batch_size);

        let linear1 = Linear::<f32>::new(input_dim, hidden_dim);
        let linear2 = Linear::<f32>::new(hidden_dim, output_dim);

        let input_data = (0..batch_size * input_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![batch_size, input_dim]),
            false,
        );

        let start = Instant::now();
        for _ in 0..50 {
            let hidden = linear1.forward(&input);
            let _output = linear2.forward(&hidden);
        }
        let forward_time = start.elapsed();

        println!(
            "    Forward pass (50x): {:?} ({:.2} inferences/sec)",
            forward_time,
            50.0 / forward_time.as_secs_f64()
        );
    }
}
