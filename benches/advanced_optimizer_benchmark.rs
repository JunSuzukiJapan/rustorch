//! Advanced Optimizer Benchmark
//! 高度なオプティマイザーベンチマーク
//!
//! Performance benchmarks for LAMB, AdaBound, and L-BFGS optimizers

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch::optim::{AdaBound, Adam, LineSearchMethod, Optimizer, LAMB, LBFGS, SGD};
use rustorch::tensor::Tensor;
use std::time::Duration;

fn create_test_problem(size: usize) -> (Tensor<f32>, Tensor<f32>) {
    let param = Tensor::<f32>::ones(&[size, size]) * 2.0;
    let grad = Tensor::<f32>::ones(&[size, size]) * 0.1;
    (param, grad)
}

fn benchmark_optimizer_step(c: &mut Criterion) {
    let sizes = vec![10, 50, 100, 200];

    let mut group = c.benchmark_group("optimizer_single_step");
    group.measurement_time(Duration::from_secs(10));

    for size in sizes {
        let (param, grad) = create_test_problem(size);

        // Benchmark SGD (baseline)
        group.bench_with_input(BenchmarkId::new("SGD", size), &size, |b, _| {
            let mut optimizer = SGD::new(0.01);
            let test_param = param.clone();
            let test_grad = grad.clone();
            b.iter(|| {
                optimizer.step(black_box(&test_param), black_box(&test_grad));
            });
        });

        // Benchmark Adam
        group.bench_with_input(BenchmarkId::new("Adam", size), &size, |b, _| {
            let mut optimizer = Adam::default_params(0.01);
            let test_param = param.clone();
            let test_grad = grad.clone();
            b.iter(|| {
                optimizer.step(black_box(&test_param), black_box(&test_grad));
            });
        });

        // Benchmark LAMB
        group.bench_with_input(BenchmarkId::new("LAMB", size), &size, |b, _| {
            let mut optimizer = LAMB::new(0.01);
            let test_param = param.clone();
            let test_grad = grad.clone();
            b.iter(|| {
                optimizer.step(black_box(&test_param), black_box(&test_grad));
            });
        });

        // Benchmark AdaBound
        group.bench_with_input(BenchmarkId::new("AdaBound", size), &size, |b, _| {
            let mut optimizer = AdaBound::new(0.01);
            let test_param = param.clone();
            let test_grad = grad.clone();
            b.iter(|| {
                optimizer.step(black_box(&test_param), black_box(&test_grad));
            });
        });

        // Benchmark L-BFGS
        group.bench_with_input(BenchmarkId::new("L-BFGS", size), &size, |b, _| {
            let mut optimizer = LBFGS::new(0.1);
            let test_param = param.clone();
            let test_grad = grad.clone();
            b.iter(|| {
                optimizer.step(black_box(&test_param), black_box(&test_grad));
            });
        });
    }

    group.finish();
}

fn benchmark_optimizer_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_convergence");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    let problem_size = 50;
    let max_iterations = 100;
    let (target_param, _) = create_test_problem(problem_size);

    // Benchmark convergence speed for quadratic problem
    // Minimize ||param - target||^2

    group.bench_function("SGD_convergence", |b| {
        b.iter(|| {
            let mut optimizer = SGD::new(0.01);
            let param = Tensor::<f32>::ones(&[problem_size, problem_size]) * 5.0;

            for _ in 0..max_iterations {
                let grad = (&param - &target_param) * 2.0;
                optimizer.step(black_box(&param), black_box(&grad));

                // Early stopping for convergence
                let diff = &param - &target_param;
                let loss = diff.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if loss < 1e-3 {
                    break;
                }
            }

            black_box(param)
        });
    });

    group.bench_function("Adam_convergence", |b| {
        b.iter(|| {
            let mut optimizer = Adam::default_params(0.01);
            let param = Tensor::<f32>::ones(&[problem_size, problem_size]) * 5.0;

            for _ in 0..max_iterations {
                let grad = (&param - &target_param) * 2.0;
                optimizer.step(black_box(&param), black_box(&grad));

                let diff = &param - &target_param;
                let loss = diff.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if loss < 1e-3 {
                    break;
                }
            }

            black_box(param)
        });
    });

    group.bench_function("LAMB_convergence", |b| {
        b.iter(|| {
            let mut optimizer = LAMB::new(0.01);
            let param = Tensor::<f32>::ones(&[problem_size, problem_size]) * 5.0;

            for _ in 0..max_iterations {
                let grad = (&param - &target_param) * 2.0;
                optimizer.step(black_box(&param), black_box(&grad));

                let diff = &param - &target_param;
                let loss = diff.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if loss < 1e-3 {
                    break;
                }
            }

            black_box(param)
        });
    });

    group.bench_function("AdaBound_convergence", |b| {
        b.iter(|| {
            let mut optimizer = AdaBound::new(0.01);
            let param = Tensor::<f32>::ones(&[problem_size, problem_size]) * 5.0;

            for _ in 0..max_iterations {
                let grad = (&param - &target_param) * 2.0;
                optimizer.step(black_box(&param), black_box(&grad));

                let diff = &param - &target_param;
                let loss = diff.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if loss < 1e-3 {
                    break;
                }
            }

            black_box(param)
        });
    });

    group.bench_function("L-BFGS_convergence", |b| {
        b.iter(|| {
            let mut optimizer = LBFGS::new(0.1);
            let param = Tensor::<f32>::ones(&[problem_size, problem_size]) * 5.0;

            for _ in 0..max_iterations {
                let grad = (&param - &target_param) * 2.0;
                optimizer.step(black_box(&param), black_box(&grad));

                let diff = &param - &target_param;
                let loss = diff.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if loss < 1e-3 {
                    break;
                }
            }

            black_box(param)
        });
    });

    group.finish();
}

fn benchmark_lamb_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("LAMB_variants");
    group.measurement_time(Duration::from_secs(8));

    let (param, grad) = create_test_problem(100);

    group.bench_function("LAMB_default", |b| {
        let mut optimizer = LAMB::new(0.01);
        let test_param = param.clone();
        let test_grad = grad.clone();
        b.iter(|| {
            optimizer.step(black_box(&test_param), black_box(&test_grad));
        });
    });

    group.bench_function("LAMB_no_bias_correction", |b| {
        let mut optimizer = LAMB::without_bias_correction(0.01, 0.9, 0.999, 1e-6, 0.01);
        let test_param = param.clone();
        let test_grad = grad.clone();
        b.iter(|| {
            optimizer.step(black_box(&test_param), black_box(&test_grad));
        });
    });

    group.bench_function("LAMB_high_weight_decay", |b| {
        let mut optimizer = LAMB::with_params(0.01, 0.9, 0.999, 1e-6, 0.1);
        let test_param = param.clone();
        let test_grad = grad.clone();
        b.iter(|| {
            optimizer.step(black_box(&test_param), black_box(&test_grad));
        });
    });

    group.finish();
}

fn benchmark_lbfgs_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("L-BFGS_variants");
    group.measurement_time(Duration::from_secs(8));

    let (param, grad) = create_test_problem(50);

    group.bench_function("L-BFGS_no_line_search", |b| {
        let mut optimizer = LBFGS::with_params(0.1, 20, 20, 1e-5, 1e-9, 10, LineSearchMethod::None);
        let test_param = param.clone();
        let test_grad = grad.clone();
        b.iter(|| {
            optimizer.step(black_box(&test_param), black_box(&test_grad));
        });
    });

    group.bench_function("L-BFGS_backtracking", |b| {
        let mut optimizer =
            LBFGS::with_params(0.1, 20, 20, 1e-5, 1e-9, 10, LineSearchMethod::Backtracking);
        let test_param = param.clone();
        let test_grad = grad.clone();
        b.iter(|| {
            optimizer.step(black_box(&test_param), black_box(&test_grad));
        });
    });

    group.bench_function("L-BFGS_strong_wolfe", |b| {
        let mut optimizer =
            LBFGS::with_params(0.1, 20, 20, 1e-5, 1e-9, 10, LineSearchMethod::StrongWolfe);
        let test_param = param.clone();
        let test_grad = grad.clone();
        b.iter(|| {
            optimizer.step(black_box(&test_param), black_box(&test_grad));
        });
    });

    group.bench_function("L-BFGS_small_memory", |b| {
        let mut optimizer = LBFGS::with_params(0.1, 20, 20, 1e-5, 1e-9, 3, LineSearchMethod::None);
        let test_param = param.clone();
        let test_grad = grad.clone();
        b.iter(|| {
            optimizer.step(black_box(&test_param), black_box(&test_grad));
        });
    });

    group.bench_function("L-BFGS_large_memory", |b| {
        let mut optimizer = LBFGS::with_params(0.1, 20, 20, 1e-5, 1e-9, 20, LineSearchMethod::None);
        let test_param = param.clone();
        let test_grad = grad.clone();
        b.iter(|| {
            optimizer.step(black_box(&test_param), black_box(&test_grad));
        });
    });

    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_memory");
    group.measurement_time(Duration::from_secs(8));

    let (param, grad) = create_test_problem(100);
    let iterations = 50;

    // Benchmark memory-efficient optimizers
    group.bench_function("SGD_memory", |b| {
        b.iter(|| {
            let mut optimizer = SGD::new(0.01);
            let test_param = param.clone();
            let test_grad = grad.clone();

            for _ in 0..iterations {
                optimizer.step(black_box(&test_param), black_box(&test_grad));
            }

            black_box(optimizer)
        });
    });

    group.bench_function("LAMB_memory", |b| {
        b.iter(|| {
            let mut optimizer = LAMB::new(0.01);
            let test_param = param.clone();
            let test_grad = grad.clone();

            for _ in 0..iterations {
                optimizer.step(black_box(&test_param), black_box(&test_grad));
            }

            black_box(optimizer)
        });
    });

    group.bench_function("L-BFGS_memory", |b| {
        b.iter(|| {
            let mut optimizer = LBFGS::new(0.1);
            let test_param = param.clone();
            let test_grad = grad.clone();

            for _ in 0..iterations {
                optimizer.step(black_box(&test_param), black_box(&test_grad));
            }

            black_box(optimizer)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_optimizer_step,
    benchmark_optimizer_convergence,
    benchmark_lamb_variants,
    benchmark_lbfgs_variants,
    benchmark_memory_usage
);

criterion_main!(benches);
