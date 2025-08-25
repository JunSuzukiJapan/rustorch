use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch::distributions::*;
use rustorch::tensor::Tensor;

fn bench_normal_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_sampling");

    // Different sample sizes
    let sizes = [100, 1000, 10000];

    for size in sizes.iter() {
        let normal = Normal::<f32>::standard(false).unwrap();

        group.bench_with_input(BenchmarkId::new("standard", size), size, |b, &size| {
            b.iter(|| {
                let samples = normal.sample(Some(&[size])).unwrap();
                black_box(samples)
            })
        });

        // Custom parameters
        let custom_normal = Normal::<f32>::from_scalars(2.0, 1.5, false).unwrap();
        group.bench_with_input(BenchmarkId::new("custom_params", size), size, |b, &size| {
            b.iter(|| {
                let samples = custom_normal.sample(Some(&[size])).unwrap();
                black_box(samples)
            })
        });
    }

    group.finish();
}

fn bench_normal_log_prob(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_log_prob");

    let sizes = [100, 1000, 10000];
    let normal = Normal::<f32>::standard(false).unwrap();

    for size in sizes.iter() {
        let values = Tensor::from_vec(
            (0..*size).map(|i| (i as f32) * 0.01 - 5.0).collect(),
            vec![*size],
        );

        group.bench_with_input(BenchmarkId::new("log_prob", size), size, |b, _| {
            b.iter(|| {
                let log_probs = normal.log_prob(&values).unwrap();
                black_box(log_probs)
            })
        });
    }

    group.finish();
}

fn bench_bernoulli_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("bernoulli_sampling");

    let sizes = [100, 1000, 10000];
    let bernoulli = Bernoulli::<f32>::from_scalar_prob(0.3, false).unwrap();

    for size in sizes.iter() {
        group.bench_with_input(BenchmarkId::new("sampling", size), size, |b, &size| {
            b.iter(|| {
                let samples = bernoulli.sample(Some(&[size])).unwrap();
                black_box(samples)
            })
        });
    }

    group.finish();
}

fn bench_categorical_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("categorical_sampling");

    let sizes = [100, 1000, 5000]; // Smaller sizes due to computational complexity
    let num_categories = 10;

    // Create uniform categorical distribution
    let probs = Tensor::from_vec(vec![0.1f32; num_categories], vec![num_categories]);
    let categorical = Categorical::from_probs(probs, false).unwrap();

    for size in sizes.iter() {
        group.bench_with_input(BenchmarkId::new("uniform", size), size, |b, &size| {
            b.iter(|| {
                let samples = categorical.sample(Some(&[size])).unwrap();
                black_box(samples)
            })
        });
    }

    // Create skewed categorical distribution
    let skewed_probs = Tensor::from_vec(
        vec![0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01],
        vec![num_categories],
    );
    let skewed_categorical = Categorical::from_probs(skewed_probs, false).unwrap();

    for size in sizes.iter() {
        group.bench_with_input(BenchmarkId::new("skewed", size), size, |b, &size| {
            b.iter(|| {
                let samples = skewed_categorical.sample(Some(&[size])).unwrap();
                black_box(samples)
            })
        });
    }

    group.finish();
}

fn bench_gamma_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gamma_sampling");

    let sizes = [100, 1000, 5000];

    // Exponential (shape=1)
    let exponential = Gamma::<f32>::exponential(1.0, false).unwrap();

    for size in sizes.iter() {
        group.bench_with_input(BenchmarkId::new("exponential", size), size, |b, &size| {
            b.iter(|| {
                let samples = exponential.sample(Some(&[size])).unwrap();
                black_box(samples)
            })
        });
    }

    // General Gamma (shape > 1)
    let concentration = Tensor::from_vec(vec![2.0f32], vec![]);
    let rate = Tensor::from_vec(vec![1.0f32], vec![]);
    let gamma = Gamma::from_concentration_rate(concentration, rate, false).unwrap();

    for size in sizes.iter() {
        group.bench_with_input(BenchmarkId::new("shape_2", size), size, |b, &size| {
            b.iter(|| {
                let samples = gamma.sample(Some(&[size])).unwrap();
                black_box(samples)
            })
        });
    }

    group.finish();
}

fn bench_uniform_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("uniform_sampling");

    let sizes = [100, 1000, 10000];
    let uniform = Uniform::<f32>::from_scalars(-2.0, 5.0, false).unwrap();

    for size in sizes.iter() {
        group.bench_with_input(BenchmarkId::new("sampling", size), size, |b, &size| {
            b.iter(|| {
                let samples = uniform.sample(Some(&[size])).unwrap();
                black_box(samples)
            })
        });
    }

    group.finish();
}

fn bench_beta_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("beta_sampling");

    let sizes = [100, 1000, 2500]; // Smaller sizes due to complexity
    let beta = Beta::<f32>::from_scalars(2.0, 5.0, false).unwrap();

    for size in sizes.iter() {
        group.bench_with_input(BenchmarkId::new("sampling", size), size, |b, &size| {
            b.iter(|| {
                let samples = beta.sample(Some(&[size])).unwrap();
                black_box(samples)
            })
        });
    }

    group.finish();
}

fn bench_exponential_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("exponential_sampling");

    let sizes = [100, 1000, 10000];
    let exponential = Exponential::<f32>::from_scalar(2.0, false).unwrap();

    for size in sizes.iter() {
        group.bench_with_input(BenchmarkId::new("sampling", size), size, |b, &size| {
            b.iter(|| {
                let samples = exponential.sample(Some(&[size])).unwrap();
                black_box(samples)
            })
        });
    }

    group.finish();
}

fn bench_distribution_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution_statistics");

    // Test statistical computations for different distributions
    let normal = Normal::<f32>::standard(false).unwrap();
    let bernoulli = Bernoulli::<f32>::from_scalar_prob(0.7, false).unwrap();
    let uniform = Uniform::<f32>::standard(false).unwrap();

    group.bench_function("normal_mean_variance", |b| {
        b.iter(|| {
            let mean = normal.mean().unwrap();
            let var = normal.variance().unwrap();
            let entropy = normal.entropy().unwrap();
            black_box((mean, var, entropy))
        })
    });

    group.bench_function("bernoulli_mean_variance", |b| {
        b.iter(|| {
            let mean = bernoulli.mean().unwrap();
            let var = bernoulli.variance().unwrap();
            let entropy = bernoulli.entropy().unwrap();
            black_box((mean, var, entropy))
        })
    });

    group.bench_function("uniform_mean_variance", |b| {
        b.iter(|| {
            let mean = uniform.mean().unwrap();
            let var = uniform.variance().unwrap();
            let entropy = uniform.entropy().unwrap();
            black_box((mean, var, entropy))
        })
    });

    group.finish();
}

fn bench_batch_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distributions");

    let batch_sizes = [10, 100, 500];

    for batch_size in batch_sizes.iter() {
        // Batch Normal distribution
        let locs = Tensor::from_vec(
            (0..*batch_size).map(|i| i as f32 * 0.1).collect(),
            vec![*batch_size],
        );
        let scales = Tensor::from_vec(vec![1.0f32; *batch_size], vec![*batch_size]);
        let batch_normal = Normal::new(locs, scales, false).unwrap();

        group.bench_with_input(
            BenchmarkId::new("batch_normal_sampling", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let samples = batch_normal.sample(Some(&[100])).unwrap();
                    black_box(samples)
                })
            },
        );

        // Batch Bernoulli distribution
        let probs = Tensor::from_vec(
            (0..*batch_size).map(|i| 0.3 + (i as f32) * 0.004).collect(),
            vec![*batch_size],
        );
        let batch_bernoulli = Bernoulli::from_probs(probs, false).unwrap();

        group.bench_with_input(
            BenchmarkId::new("batch_bernoulli_sampling", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let samples = batch_bernoulli.sample(Some(&[100])).unwrap();
                    black_box(samples)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_normal_sampling,
    bench_normal_log_prob,
    bench_bernoulli_sampling,
    bench_categorical_sampling,
    bench_gamma_sampling,
    bench_uniform_sampling,
    bench_beta_sampling,
    bench_exponential_sampling,
    bench_distribution_statistics,
    bench_batch_distributions
);

criterion_main!(benches);
