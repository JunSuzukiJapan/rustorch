use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::distributions::*;
use rustorch::tensor::Tensor;

fn simple_sampling_bench(c: &mut Criterion) {
    // Very minimal benchmark to get basic timings
    c.bench_function("normal_100", |b| {
        let normal = Normal::<f32>::standard(false).unwrap();
        b.iter(|| black_box(normal.sample(Some(&[100])).unwrap()))
    });
    
    c.bench_function("bernoulli_100", |b| {
        let bernoulli = Bernoulli::<f32>::from_scalar_prob(0.5, false).unwrap();
        b.iter(|| black_box(bernoulli.sample(Some(&[100])).unwrap()))
    });
    
    c.bench_function("uniform_100", |b| {
        let uniform = Uniform::<f32>::standard(false).unwrap();
        b.iter(|| black_box(uniform.sample(Some(&[100])).unwrap()))
    });
    
    c.bench_function("exponential_100", |b| {
        let exponential = Exponential::<f32>::standard(false).unwrap();
        b.iter(|| black_box(exponential.sample(Some(&[100])).unwrap()))
    });
    
    // Test Gamma distribution (the challenging one)
    c.bench_function("gamma_exponential_100", |b| {
        let gamma = Gamma::<f32>::exponential(1.0, false).unwrap();
        b.iter(|| black_box(gamma.sample(Some(&[100])).unwrap()))
    });
    
    c.bench_function("gamma_shape2_100", |b| {
        let concentration = Tensor::from_vec(vec![2.0f32], vec![]);
        let rate = Tensor::from_vec(vec![1.0f32], vec![]);
        let gamma = Gamma::from_concentration_rate(concentration, rate, false).unwrap();
        b.iter(|| black_box(gamma.sample(Some(&[100])).unwrap()))
    });
    
    // Test Beta distribution
    c.bench_function("beta_uniform_100", |b| {
        let beta = Beta::<f32>::uniform(false).unwrap();
        b.iter(|| black_box(beta.sample(Some(&[100])).unwrap()))
    });
    
    c.bench_function("beta_25_100", |b| {
        let beta = Beta::<f32>::from_scalars(2.0, 5.0, false).unwrap();
        b.iter(|| black_box(beta.sample(Some(&[100])).unwrap()))
    });
}

fn simple_log_prob_bench(c: &mut Criterion) {
    let values = Tensor::from_vec(vec![0.5f32; 100], vec![100]);
    let positive_values = Tensor::from_vec(
        (0..100).map(|i| 0.1 + (i as f32) * 0.01).collect(),
        vec![100]
    );
    let unit_values = Tensor::from_vec(
        (0..100).map(|i| 0.01 + (i as f32) * 0.0098).collect(), // Values in (0,1)
        vec![100]
    );
    
    c.bench_function("normal_log_prob_100", |b| {
        let normal = Normal::<f32>::standard(false).unwrap();
        b.iter(|| black_box(normal.log_prob(&values).unwrap()))
    });
    
    c.bench_function("exponential_log_prob_100", |b| {
        let exponential = Exponential::<f32>::standard(false).unwrap();
        b.iter(|| black_box(exponential.log_prob(&positive_values).unwrap()))
    });
    
    c.bench_function("gamma_log_prob_100", |b| {
        let gamma = Gamma::<f32>::exponential(1.0, false).unwrap();
        b.iter(|| black_box(gamma.log_prob(&positive_values).unwrap()))
    });
    
    c.bench_function("beta_log_prob_100", |b| {
        let beta = Beta::<f32>::uniform(false).unwrap();
        b.iter(|| black_box(beta.log_prob(&unit_values).unwrap()))
    });
}

fn simple_stats_bench(c: &mut Criterion) {
    c.bench_function("normal_stats", |b| {
        let normal = Normal::<f32>::standard(false).unwrap();
        b.iter(|| {
            black_box((
                normal.mean().unwrap(),
                normal.variance().unwrap(),
            ))
        })
    });
    
    c.bench_function("gamma_stats", |b| {
        let gamma = Gamma::<f32>::exponential(1.0, false).unwrap();
        b.iter(|| {
            black_box((
                gamma.mean().unwrap(),
                gamma.variance().unwrap(),
            ))
        })
    });
    
    c.bench_function("beta_stats", |b| {
        let beta = Beta::<f32>::from_scalars(2.0, 3.0, false).unwrap();
        b.iter(|| {
            black_box((
                beta.mean().unwrap(),
                beta.variance().unwrap(),
            ))
        })
    });
}

// Configure for faster execution
criterion_group!{
    name = benches;
    config = Criterion::default()
        .sample_size(10)           // Very small sample size
        .measurement_time(std::time::Duration::from_secs(5))  // Short measurement time
        .warm_up_time(std::time::Duration::from_secs(1));     // Short warm up
    targets = simple_sampling_bench, simple_log_prob_bench, simple_stats_bench
}

criterion_main!(benches);