//! Mixed Precision Training benchmarks
//! 混合精度学習のベンチマーク

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch::amp::{
    autocast, cast_to_fp16, cast_to_fp32, maybe_autocast_f32, AMPOptimizer, GradScaler, ParamGroup,
};
use rustorch::dtype::DType;
use rustorch::optim::{sgd::SGD, Optimizer};
use rustorch::tensor::Tensor;

fn bench_dtype_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_conversions");

    // Test different tensor sizes
    let sizes = vec![
        (1000, "1K"),
        (10000, "10K"),
        (100000, "100K"),
        (1000000, "1M"),
    ];

    for (size, name) in sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
        let tensor = Tensor::from_vec(data, vec![size]);

        group.bench_with_input(
            BenchmarkId::new("fp32_to_fp16", name),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let _fp16 = cast_to_fp16(tensor);
                });
            },
        );

        let fp16_tensor = cast_to_fp16(&tensor);
        group.bench_with_input(
            BenchmarkId::new("fp16_to_fp32", name),
            &fp16_tensor,
            |b, fp16_tensor| {
                b.iter(|| {
                    let _fp32 = cast_to_fp32(fp16_tensor);
                });
            },
        );
    }

    group.finish();
}

fn bench_autocast_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("autocast_operations");

    let tensor = Tensor::from_vec(
        (0..10000).map(|i| i as f32 * 0.001).collect(),
        vec![100, 100],
    );

    // Benchmark without autocast
    group.bench_function("without_autocast", |b| {
        b.iter(|| {
            let _result = tensor.clone();
        });
    });

    // Benchmark with FP16 autocast
    group.bench_function("with_fp16_autocast", |b| {
        b.iter(|| {
            let _ctx = autocast("cuda", true, Some(DType::Float16));
            let _result = maybe_autocast_f32(&tensor);
        });
    });

    // Benchmark with BF16 autocast
    group.bench_function("with_bf16_autocast", |b| {
        b.iter(|| {
            let _ctx = autocast("cuda", true, Some(DType::BFloat16));
            let _result = maybe_autocast_f32(&tensor);
        });
    });

    group.finish();
}

fn bench_gradient_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_scaling");

    let mut scaler = GradScaler::default();

    // Test different numbers of gradients
    let grad_counts = vec![1, 10, 100, 1000];

    for count in grad_counts {
        let gradients: Vec<Tensor<f32>> = (0..count)
            .map(|i| Tensor::from_vec(vec![0.1 + i as f32 * 0.01; 100], vec![10, 10]))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("check_overflow", count),
            &gradients,
            |b, grads| {
                b.iter(|| {
                    scaler.check_overflow(grads);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scale_tensors", count),
            &gradients,
            |b, grads| {
                b.iter(|| {
                    for grad in grads {
                        let _scaled = scaler.scale_tensor(grad);
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_amp_optimizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("amp_optimizer");

    // Create test data
    let params = vec![
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
    ];

    let grads = vec![
        Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        Tensor::from_vec(vec![0.5, 0.6, 0.7, 0.8], vec![2, 2]),
    ];

    // Benchmark regular SGD
    group.bench_function("regular_sgd", |b| {
        let mut sgd = SGD::new(0.01);
        b.iter(|| {
            for (param, grad) in params.iter().zip(grads.iter()) {
                sgd.step(param, grad);
            }
        });
    });

    // Benchmark AMP optimizer without parameter groups
    group.bench_function("amp_without_groups", |b| {
        let sgd = SGD::new(0.01);
        let mut amp_optimizer = AMPOptimizer::new(sgd, None);
        b.iter(|| {
            let mut grads_copy = grads.clone();
            amp_optimizer.step(&params, &mut grads_copy);
        });
    });

    // Benchmark AMP optimizer with parameter groups
    group.bench_function("amp_with_groups", |b| {
        let sgd = SGD::new(0.01);
        let mut amp_optimizer = AMPOptimizer::new(sgd, None);

        let param_group = ParamGroup {
            param_ids: vec![0, 1],
            clip_gradients: true,
            max_grad_norm: Some(1.0),
            use_amp: true,
        };
        amp_optimizer.add_param_group(param_group);

        b.iter(|| {
            let mut grads_copy = grads.clone();
            amp_optimizer.step(&params, &mut grads_copy);
        });
    });

    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        group.bench_with_input(BenchmarkId::new("fp32_creation", size), &data, |b, data| {
            b.iter(|| {
                let _tensor = Tensor::from_vec(data.clone(), vec![size]);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("fp16_roundtrip", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let fp32_tensor = Tensor::from_vec(data.clone(), vec![size]);
                    let fp16_tensor = cast_to_fp16(&fp32_tensor);
                    let _restored = cast_to_fp32(&fp16_tensor);
                });
            },
        );
    }

    group.finish();
}

fn bench_overflow_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("overflow_detection");

    let mut scaler = GradScaler::default();

    // Normal gradients
    let normal_grads = vec![
        Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        Tensor::from_vec(vec![0.5, 0.6, 0.7, 0.8], vec![2, 2]),
    ];

    // Overflow gradients
    let overflow_grads = vec![
        Tensor::from_vec(vec![1e10, 2e10, 3e10, 4e10], vec![2, 2]),
        Tensor::from_vec(vec![5e10, 6e10, 7e10, 8e10], vec![2, 2]),
    ];

    // NaN gradients
    let nan_grads = vec![
        Tensor::from_vec(vec![f32::NAN, 0.2, 0.3, 0.4], vec![2, 2]),
        Tensor::from_vec(vec![0.5, f32::INFINITY, 0.7, 0.8], vec![2, 2]),
    ];

    group.bench_function("normal_gradients", |b| {
        b.iter(|| {
            scaler.check_overflow(&normal_grads);
        });
    });

    group.bench_function("overflow_gradients", |b| {
        b.iter(|| {
            scaler.check_overflow(&overflow_grads);
        });
    });

    group.bench_function("nan_gradients", |b| {
        b.iter(|| {
            scaler.check_overflow(&nan_grads);
        });
    });

    group.finish();
}

criterion_group!(
    mixed_precision_benches,
    bench_dtype_conversions,
    bench_autocast_operations,
    bench_gradient_scaling,
    bench_amp_optimizer,
    bench_memory_usage,
    bench_overflow_detection
);

criterion_main!(mixed_precision_benches);
