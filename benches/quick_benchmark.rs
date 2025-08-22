//! Quick performance benchmarks
//! 簡単なパフォーマンステスト

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustorch::tensor::Tensor;
use rustorch::autograd::Variable;
use rustorch::nn::Linear;

fn bench_basic_tensor_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_tensor_ops");
    
    let sizes = vec![100, 500, 1000];
    
    for size in sizes {
        // Tensor creation
        group.bench_with_input(
            BenchmarkId::new("tensor_creation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data: Vec<f32> = (0..size*size).map(|i| i as f32).collect();
                    black_box(Tensor::from_vec(data, vec![size, size]))
                })
            }
        );
        
        // Basic arithmetic
        let a = Tensor::from_vec((0..size*size).map(|i| i as f32).collect(), vec![size, size]);
        let b = Tensor::from_vec((0..size*size).map(|i| (i*2) as f32).collect(), vec![size, size]);
        
        group.bench_with_input(
            BenchmarkId::new("tensor_add", size),
            &size,
            |bencher, _| {
                bencher.iter(|| black_box(&a + &b))
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("tensor_mul", size),
            &size,
            |bencher, _| {
                bencher.iter(|| black_box(&a * &b))
            }
        );
        
        // Matrix multiplication (smaller sizes for performance)
        if size <= 500 {
            group.bench_with_input(
                BenchmarkId::new("matmul", size),
                &size,
                |bencher, _| {
                    bencher.iter(|| black_box(a.matmul(&b)))
                }
            );
        }
    }
    
    group.finish();
}

fn bench_4d_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("4d_tensor_ops");
    
    // Test new 4D matmul functionality
    let batch_size = 2;
    let num_heads = 4;
    let seq_len = 32;
    let d_k = 16;
    
    let q_data = (0..batch_size * num_heads * seq_len * d_k)
        .map(|i| i as f32 * 0.1)
        .collect::<Vec<f32>>();
    let q = Tensor::from_vec(q_data, vec![batch_size, num_heads, seq_len, d_k]);
    
    let k_data = (0..batch_size * num_heads * d_k * seq_len)  
        .map(|i| i as f32 * 0.05)
        .collect::<Vec<f32>>();
    let k_t = Tensor::from_vec(k_data, vec![batch_size, num_heads, d_k, seq_len]);
    
    group.bench_function("4d_matmul_attention", |bencher| {
        bencher.iter(|| black_box(q.matmul(&k_t)))
    });
    
    // Test 4D transpose
    let tensor_4d = Tensor::from_vec(
        (0..batch_size * num_heads * seq_len * d_k).map(|i| i as f32).collect(),
        vec![batch_size, num_heads, seq_len, d_k]
    );
    
    group.bench_function("4d_transpose", |bencher| {
        bencher.iter(|| black_box(tensor_4d.transpose_last_two()))
    });
    
    group.finish();
}

fn bench_neural_networks(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_networks");
    
    let input_size = 784;
    let hidden_size = 256;
    let batch_size = 32;
    
    let layer = Linear::<f32>::new(input_size, hidden_size);
    let input_data: Vec<f32> = (0..batch_size * input_size).map(|i| i as f32 * 0.01).collect();
    let input_tensor = Tensor::from_vec(input_data, vec![batch_size, input_size]);
    let input_var = Variable::new(input_tensor, false);
    
    group.bench_function("linear_layer_forward", |bencher| {
        bencher.iter(|| black_box(layer.forward(&input_var)))
    });
    
    group.finish();
}

fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_ops");
    
    let size = 1000;
    let tensor = Tensor::from_vec((0..size*size).map(|i| i as f32).collect(), vec![size, size]);
    
    group.bench_function("tensor_clone", |bencher| {
        bencher.iter(|| black_box(tensor.clone()))
    });
    
    group.bench_function("tensor_sum", |bencher| {
        bencher.iter(|| black_box(tensor.sum()))
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_basic_tensor_ops,
    bench_4d_tensor_operations,
    bench_neural_networks,
    bench_memory_operations
);
criterion_main!(benches);