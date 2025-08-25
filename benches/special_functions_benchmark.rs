use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch::special::*;
use rustorch::tensor::Tensor;

fn benchmark_gamma_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("gamma_functions");

    // Scalar benchmarks
    let test_values = vec![0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0];

    for value in test_values {
        group.bench_with_input(BenchmarkId::new("gamma_scalar", value), &value, |b, &x| {
            b.iter(|| gamma::gamma_scalar::<f64>(black_box(x)))
        });

        group.bench_with_input(BenchmarkId::new("lgamma_scalar", value), &value, |b, &x| {
            b.iter(|| gamma::lgamma_scalar::<f64>(black_box(x)))
        });

        group.bench_with_input(
            BenchmarkId::new("digamma_scalar", value),
            &value,
            |b, &x| b.iter(|| gamma::digamma_scalar::<f64>(black_box(x))),
        );
    }

    // Beta function benchmarks
    group.bench_function("beta_2_3", |b| {
        b.iter(|| gamma::beta(black_box(2.0_f64), black_box(3.0_f64)))
    });

    group.bench_function("lbeta_5_7", |b| {
        b.iter(|| gamma::lbeta(black_box(5.0_f64), black_box(7.0_f64)))
    });

    // Tensor benchmarks
    let tensor_sizes = vec![10, 100, 1000];

    for size in tensor_sizes {
        let data: Vec<f64> = (1..=size).map(|i| i as f64 * 0.5).collect();
        let tensor = Tensor::from_vec(data, vec![size]);

        group.bench_with_input(BenchmarkId::new("gamma_tensor", size), &tensor, |b, t| {
            b.iter(|| gamma::gamma(black_box(t)))
        });

        group.bench_with_input(BenchmarkId::new("lgamma_tensor", size), &tensor, |b, t| {
            b.iter(|| gamma::lgamma(black_box(t)))
        });
    }

    group.finish();
}

fn benchmark_bessel_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("bessel_functions");

    let test_values = vec![0.5, 1.0, 2.0, 5.0, 10.0];
    let orders = vec![0.0, 1.0, 2.0, 3.5];

    for order in orders {
        for value in &test_values {
            let id = format!("J_{}_{}", order, value);
            group.bench_with_input(
                BenchmarkId::new("bessel_j", &id),
                &(*value, order),
                |b, &(x, n)| b.iter(|| bessel::bessel_j_scalar(black_box(n), black_box(x))),
            );

            if *value > 0.0 {
                let id = format!("Y_{}_{}", order, value);
                group.bench_with_input(
                    BenchmarkId::new("bessel_y", &id),
                    &(*value, order),
                    |b, &(x, n)| b.iter(|| bessel::bessel_y_scalar(black_box(n), black_box(x))),
                );
            }

            let id = format!("I_{}_{}", order, value);
            group.bench_with_input(
                BenchmarkId::new("bessel_i", &id),
                &(*value, order),
                |b, &(x, n)| b.iter(|| bessel::bessel_i_scalar(black_box(n), black_box(x))),
            );

            if *value > 0.0 {
                let id = format!("K_{}_{}", order, value);
                group.bench_with_input(
                    BenchmarkId::new("bessel_k", &id),
                    &(*value, order),
                    |b, &(x, n)| b.iter(|| bessel::bessel_k_scalar(black_box(n), black_box(x))),
                );
            }
        }
    }

    // Tensor benchmarks
    let tensor_sizes = vec![10, 100, 1000];

    for size in tensor_sizes {
        let data: Vec<f64> = (1..=size).map(|i| i as f64 * 0.1).collect();
        let tensor = Tensor::from_vec(data, vec![size]);

        group.bench_with_input(
            BenchmarkId::new("bessel_j_tensor", size),
            &tensor,
            |b, t| b.iter(|| bessel::bessel_j(black_box(0.0), black_box(t))),
        );

        group.bench_with_input(
            BenchmarkId::new("bessel_i_tensor", size),
            &tensor,
            |b, t| b.iter(|| bessel::bessel_i(black_box(1.0), black_box(t))),
        );
    }

    group.finish();
}

fn benchmark_error_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_functions");

    let test_values = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];

    for value in test_values {
        group.bench_with_input(BenchmarkId::new("erf", value), &value, |b, &x| {
            b.iter(|| error::erf_scalar::<f64>(black_box(x)))
        });

        group.bench_with_input(BenchmarkId::new("erfc", value), &value, |b, &x| {
            b.iter(|| error::erfc_scalar::<f64>(black_box(x)))
        });

        if value.abs() < 1.0 {
            group.bench_with_input(BenchmarkId::new("erfinv", value), &value, |b, &x| {
                b.iter(|| error::erfinv_scalar::<f64>(black_box(x)))
            });
        }

        if value > 0.0 && value < 2.0 {
            group.bench_with_input(BenchmarkId::new("erfcinv", value), &value, |b, &x| {
                b.iter(|| error::erfcinv_scalar::<f64>(black_box(x)))
            });
        }

        group.bench_with_input(BenchmarkId::new("erfcx", value), &value, |b, &x| {
            b.iter(|| error::erfcx_scalar::<f64>(black_box(x)))
        });
    }

    // Compare series vs approximation
    group.bench_function("erf_series_0.5", |b| {
        b.iter(|| error::erf_series(black_box(0.5_f64)))
    });

    group.bench_function("erf_approx_0.5", |b| {
        b.iter(|| error::erf_scalar(black_box(0.5_f64)))
    });

    // Tensor benchmarks
    let tensor_sizes = vec![10, 100, 1000];

    for size in tensor_sizes {
        let data: Vec<f64> = (0..size)
            .map(|i| (i as f64 - size as f64 / 2.0) * 0.1)
            .collect();
        let tensor = Tensor::from_vec(data, vec![size]);

        group.bench_with_input(BenchmarkId::new("erf_tensor", size), &tensor, |b, t| {
            b.iter(|| error::erf(black_box(t)))
        });

        group.bench_with_input(BenchmarkId::new("erfc_tensor", size), &tensor, |b, t| {
            b.iter(|| error::erfc(black_box(t)))
        });
    }

    group.finish();
}

fn benchmark_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("utils");

    // Factorial benchmarks
    let factorial_inputs = vec![5, 10, 15, 20];
    for n in factorial_inputs {
        group.bench_with_input(BenchmarkId::new("factorial", n), &n, |b, &n| {
            b.iter(|| utils::factorial(black_box(n)))
        });

        group.bench_with_input(BenchmarkId::new("double_factorial", n), &n, |b, &n| {
            b.iter(|| utils::double_factorial(black_box(n)))
        });
    }

    // Binomial coefficient benchmarks
    group.bench_function("binomial_10_5", |b| {
        b.iter(|| utils::binomial_coefficient(black_box(10), black_box(5)))
    });

    group.bench_function("binomial_20_10", |b| {
        b.iter(|| utils::binomial_coefficient(black_box(20), black_box(10)))
    });

    // Pochhammer symbol benchmarks
    group.bench_function("pochhammer_2.5_5", |b| {
        b.iter(|| utils::pochhammer(black_box(2.5_f64), black_box(5)))
    });

    // Harmonic number benchmarks
    let harmonic_inputs = vec![10, 50, 100, 500];
    for n in harmonic_inputs {
        group.bench_with_input(BenchmarkId::new("harmonic_number", n), &n, |b, &n| {
            b.iter(|| utils::harmonic_number(black_box(n)))
        });
    }

    group.finish();
}

fn benchmark_combined_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_operations");

    // Benchmark typical use cases
    group.bench_function("normal_cdf_computation", |b| {
        b.iter(|| {
            let x = black_box(1.5_f64);
            let sqrt_2 = 2.0_f64.sqrt();
            (1.0 + error::erf_scalar(x / sqrt_2)) / 2.0
        })
    });

    group.bench_function("gamma_ratio", |b| {
        b.iter(|| {
            let a = black_box(5.0_f64);
            let b = black_box(3.0_f64);
            let gamma_a = gamma::gamma_scalar(a);
            let gamma_b = gamma::gamma_scalar(b);
            let gamma_ab = gamma::gamma_scalar(a + b);
            gamma_a.and_then(|ga| gamma_b.and_then(|gb| gamma_ab.map(|gab| ga * gb / gab)))
        })
    });

    group.bench_function("bessel_ratio_j0_j1", |b| {
        b.iter(|| {
            let x = black_box(2.5_f64);
            let j0 = bessel::bessel_j_scalar(0.0, x);
            let j1 = bessel::bessel_j_scalar(1.0, x);
            j0.and_then(|v0| j1.map(|v1| v0 / v1))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_gamma_functions,
    benchmark_bessel_functions,
    benchmark_error_functions,
    benchmark_utils,
    benchmark_combined_operations
);
criterion_main!(benches);
