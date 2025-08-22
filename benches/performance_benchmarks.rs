use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rustorch::tensor::Tensor;
use rustorch::nn::{Linear, Conv2d, BatchNorm2d, MultiHeadAttention};
use rustorch::optim::{SGD, Adam};
use rustorch::parallel::ParallelTensor;
use rustorch::simd::ops;
use std::time::Duration;

fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");
    
    for size in &[10, 100, 1000, 10000] {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("zeros", size), size, |b, &size| {
            b.iter(|| {
                let tensor = Tensor::zeros(&[size]);
                black_box(tensor);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("ones", size), size, |b, &size| {
            b.iter(|| {
                let tensor = Tensor::ones(&[size]);
                black_box(tensor);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("randn", size), size, |b, &size| {
            b.iter(|| {
                let tensor = Tensor::randn(&[size]);
                black_box(tensor);
            });
        });
    }
    group.finish();
}

fn bench_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");
    group.measurement_time(Duration::from_secs(10));
    
    for size in &[32, 64, 128, 256, 512] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));
        
        let a = Tensor::randn(&[*size, *size]);
        let b = Tensor::randn(&[*size, *size]);
        
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bench, _| {
            bench.iter(|| {
                let c = a.matmul(&b);
                black_box(c);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("element_add", size), size, |bench, _| {
            bench.iter(|| {
                let c = &a + &b;
                black_box(c);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("element_mul", size), size, |bench, _| {
            bench.iter(|| {
                let c = &a * &b;
                black_box(c);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("transpose", size), size, |bench, _| {
            bench.iter(|| {
                let c = a.transpose();
                black_box(c);
            });
        });
    }
    group.finish();
}

fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    for size in &[128, 512, 1024, 4096, 16384] {
        group.throughput(Throughput::Elements(*size as u64));
        
        let a = vec![1.0f32; *size];
        let b = vec![2.0f32; *size];
        
        group.bench_with_input(BenchmarkId::new("simd_add", size), size, |bench, _| {
            bench.iter(|| {
                let mut result = vec![0.0; a.len()];
                ops::add_optimized(&a, &b, &mut result);
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("simd_mul", size), size, |bench, _| {
            bench.iter(|| {
                let mut result = vec![0.0; a.len()];
                ops::mul_optimized(&a, &b, &mut result);
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("simd_dot", size), size, |bench, _| {
            bench.iter(|| {
                let result = ops::dot_product_optimized(&a, &b);
                black_box(result);
            });
        });
        
        // Compare with scalar operations
        group.bench_with_input(BenchmarkId::new("scalar_add", size), size, |bench, _| {
            bench.iter(|| {
                let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                black_box(c);
            });
        });
    }
    group.finish();
}

fn bench_neural_network_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("nn_layers");
    
    // Linear layer benchmarks
    for (in_features, out_features, batch_size) in &[
        (784, 128, 32),
        (128, 64, 64),
        (1024, 512, 16),
        (512, 256, 128),
    ] {
        let input = Tensor::randn(&[*batch_size, *in_features]);
        let layer = Linear::new(*in_features, *out_features);
        
        let id = format!("linear_{}x{}x{}", batch_size, in_features, out_features);
        group.bench_with_input(BenchmarkId::new("forward", &id), &input, |b, input| {
            b.iter(|| {
                let output = layer.forward(input);
                black_box(output);
            });
        });
    }
    
    // Conv2d layer benchmarks
    for (channels, kernel_size, image_size, batch_size) in &[
        (3, 3, 32, 8),
        (64, 3, 28, 16),
        (128, 5, 14, 32),
    ] {
        let input = Tensor::randn(&[*batch_size, *channels, *image_size, *image_size]);
        let layer = Conv2d::new(*channels, channels * 2, *kernel_size, 1, kernel_size / 2);
        
        let id = format!("conv2d_{}x{}x{}x{}", batch_size, channels, image_size, kernel_size);
        group.bench_with_input(BenchmarkId::new("forward", &id), &input, |b, input| {
            b.iter(|| {
                let output = layer.forward(input);
                black_box(output);
            });
        });
    }
    
    // BatchNorm benchmarks
    for (channels, size, batch_size) in &[
        (64, 28, 16),
        (128, 14, 32),
        (256, 7, 64),
    ] {
        let input = Tensor::randn(&[*batch_size, *channels, *size, *size]);
        let layer = BatchNorm2d::new(*channels);
        
        let id = format!("batchnorm_{}x{}x{}", batch_size, channels, size);
        group.bench_with_input(BenchmarkId::new("forward", &id), &input, |b, input| {
            b.iter(|| {
                let output = layer.forward(input);
                black_box(output);
            });
        });
    }
    
    group.finish();
}

fn bench_attention_mechanisms(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");
    
    for (seq_len, d_model, num_heads, batch_size) in &[
        (128, 256, 8, 4),
        (256, 512, 8, 2),
        (512, 768, 12, 1),
    ] {
        let attention = MultiHeadAttention::new(*d_model, *num_heads);
        let query = Tensor::randn(&[*batch_size, *seq_len, *d_model]);
        let key = Tensor::randn(&[*batch_size, *seq_len, *d_model]);
        let value = Tensor::randn(&[*batch_size, *seq_len, *d_model]);
        
        let id = format!("mha_{}x{}x{}x{}", batch_size, seq_len, d_model, num_heads);
        group.bench_function(&id, |b| {
            b.iter(|| {
                let output = attention.forward(&query, &key, &value, None);
                black_box(output);
            });
        });
    }
    
    group.finish();
}

fn bench_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_ops");
    
    for size in &[1000, 10000, 100000, 1000000] {
        group.throughput(Throughput::Elements(*size as u64));
        
        let tensor = Tensor::randn(&[*size]);
        let parallel_tensor = ParallelTensor::from_tensor(tensor.clone());
        
        group.bench_with_input(BenchmarkId::new("parallel_sum", size), size, |b, _| {
            b.iter(|| {
                let sum = parallel_tensor.sum();
                black_box(sum);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("sequential_sum", size), size, |b, _| {
            b.iter(|| {
                let sum = tensor.sum();
                black_box(sum);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("parallel_map", size), size, |b, _| {
            b.iter(|| {
                let result = parallel_tensor.map(|x| x * 2.0 + 1.0);
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("sequential_map", size), size, |b, _| {
            b.iter(|| {
                let result = tensor.map(|x| x * 2.0 + 1.0);
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn bench_optimizer_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizers");
    
    for param_count in &[100, 1000, 10000, 100000] {
        let params = vec![Tensor::randn(&[*param_count]); 10];
        let grads = vec![Tensor::randn(&[*param_count]); 10];
        
        // SGD optimizer
        let mut sgd = SGD::new(0.01, 0.9);
        group.bench_with_input(BenchmarkId::new("sgd_step", param_count), param_count, |b, _| {
            b.iter(|| {
                for (param, grad) in params.iter().zip(grads.iter()) {
                    sgd.step(param, grad);
                }
            });
        });
        
        // Adam optimizer
        let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
        group.bench_with_input(BenchmarkId::new("adam_step", param_count), param_count, |b, _| {
            b.iter(|| {
                for (param, grad) in params.iter().zip(grads.iter()) {
                    adam.step(param, grad);
                }
            });
        });
    }
    
    group.finish();
}

fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_ops");
    
    for size in &[1024, 10240, 102400, 1024000] {
        group.throughput(Throughput::Bytes(*size as u64 * 4)); // f32 = 4 bytes
        
        let src = Tensor::randn(&[*size]);
        let mut dst = Tensor::zeros(&[*size]);
        
        group.bench_with_input(BenchmarkId::new("copy", size), size, |b, _| {
            b.iter(|| {
                dst.copy_from(&src);
                black_box(&dst);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("clone", size), size, |b, _| {
            b.iter(|| {
                let cloned = src.clone();
                black_box(cloned);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("view", size), size, |b, _| {
            b.iter(|| {
                let view = src.view(&[*size / 2, 2]);
                black_box(view);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("reshape", size), size, |b, _| {
            b.iter(|| {
                let reshaped = src.reshape(&[*size / 2, 2]);
                black_box(reshaped);
            });
        });
    }
    
    group.finish();
}

fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");
    
    for size in &[1000, 10000, 100000] {
        group.throughput(Throughput::Elements(*size as u64));
        
        let input = Tensor::randn(&[*size]);
        
        group.bench_with_input(BenchmarkId::new("relu", size), size, |b, _| {
            b.iter(|| {
                let output = input.relu();
                black_box(output);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("sigmoid", size), size, |b, _| {
            b.iter(|| {
                let output = input.sigmoid();
                black_box(output);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("tanh", size), size, |b, _| {
            b.iter(|| {
                let output = input.tanh();
                black_box(output);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("softmax", size), size, |b, _| {
            b.iter(|| {
                let output = input.softmax(-1);
                black_box(output);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("gelu", size), size, |b, _| {
            b.iter(|| {
                let output = input.gelu();
                black_box(output);
            });
        });
    }
    
    group.finish();
}

fn bench_broadcasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting");
    
    let cases = vec![
        (vec![1, 100], vec![100, 100], "1x100_to_100x100"),
        (vec![100, 1], vec![100, 100], "100x1_to_100x100"),
        (vec![1], vec![1000000], "1_to_1000000"),
        (vec![100, 1, 100], vec![100, 100, 100], "100x1x100_to_100x100x100"),
    ];
    
    for (from_shape, to_shape, name) in cases {
        let small = Tensor::randn(&from_shape);
        let large = Tensor::randn(&to_shape);
        
        group.bench_function(format!("broadcast_{}", name), |b| {
            b.iter(|| {
                let result = &small + &large;
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn bench_reduction_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");
    
    for size in &[100, 1000, 10000, 100000] {
        let tensor_1d = Tensor::randn(&[*size]);
        let size_sqrt = ((*size as f64).sqrt() as usize).max(1);
        let tensor_2d = Tensor::randn(&[size_sqrt, size_sqrt]);
        
        group.bench_with_input(BenchmarkId::new("sum_1d", size), size, |b, _| {
            b.iter(|| {
                let sum = tensor_1d.sum();
                black_box(sum);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("mean_1d", size), size, |b, _| {
            b.iter(|| {
                let mean = tensor_1d.mean();
                black_box(mean);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("max_1d", size), size, |b, _| {
            b.iter(|| {
                let max = tensor_1d.max();
                black_box(max);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("sum_2d_dim0", size), size, |b, _| {
            b.iter(|| {
                let sum = tensor_2d.sum_dim(&[0], false);
                black_box(sum);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("mean_2d_all", size), size, |b, _| {
            b.iter(|| {
                let mean = tensor_2d.mean();
                black_box(mean);
            });
        });
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_gpu_operations(c: &mut Criterion) {
    use rustorch::gpu::cuda::CudaDevice;
    
    let mut group = c.benchmark_group("gpu_ops");
    
    if let Ok(device) = CudaDevice::new(0) {
        for size in &[100, 1000, 10000] {
            let cpu_tensor = Tensor::randn(&[*size, *size]);
            let gpu_tensor = cpu_tensor.to_device(device.clone());
            
            group.bench_with_input(BenchmarkId::new("gpu_matmul", size), size, |b, _| {
                b.iter(|| {
                    let result = gpu_tensor.matmul(&gpu_tensor);
                    device.synchronize();
                    black_box(result);
                });
            });
            
            group.bench_with_input(BenchmarkId::new("cpu_matmul", size), size, |b, _| {
                b.iter(|| {
                    let result = cpu_tensor.matmul(&cpu_tensor);
                    black_box(result);
                });
            });
            
            group.bench_with_input(BenchmarkId::new("gpu_transfer_to", size), size, |b, _| {
                b.iter(|| {
                    let gpu = cpu_tensor.to_device(device.clone());
                    black_box(gpu);
                });
            });
            
            group.bench_with_input(BenchmarkId::new("gpu_transfer_from", size), size, |b, _| {
                b.iter(|| {
                    let cpu = gpu_tensor.to_cpu();
                    black_box(cpu);
                });
            });
        }
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_creation,
    bench_matrix_operations,
    bench_simd_operations,
    bench_neural_network_layers,
    bench_attention_mechanisms,
    bench_parallel_operations,
    bench_optimizer_step,
    bench_memory_operations,
    bench_activation_functions,
    bench_broadcasting,
    bench_reduction_operations,
);

#[cfg(feature = "cuda")]
criterion_group!(gpu_benches, bench_gpu_operations);

#[cfg(not(feature = "cuda"))]
criterion_main!(benches);

#[cfg(feature = "cuda")]
criterion_main!(benches, gpu_benches);