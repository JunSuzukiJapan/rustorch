use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rustorch::distributions::*;
use rustorch::tensor::Tensor;

fn quick_distribution_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_sampling");
    
    // Test different distributions with fixed sample size for speed
    let sample_size = 1000;
    group.throughput(Throughput::Elements(sample_size as u64));
    
    // Normal distribution
    let normal = Normal::<f32>::standard(false).unwrap();
    group.bench_function("normal", |b| {
        b.iter(|| {
            let samples = normal.sample(Some(&[sample_size])).unwrap();
            black_box(samples)
        })
    });
    
    // Bernoulli distribution
    let bernoulli = Bernoulli::<f32>::from_scalar_prob(0.5, false).unwrap();
    group.bench_function("bernoulli", |b| {
        b.iter(|| {
            let samples = bernoulli.sample(Some(&[sample_size])).unwrap();
            black_box(samples)
        })
    });
    
    // Uniform distribution
    let uniform = Uniform::<f32>::standard(false).unwrap();
    group.bench_function("uniform", |b| {
        b.iter(|| {
            let samples = uniform.sample(Some(&[sample_size])).unwrap();
            black_box(samples)
        })
    });
    
    // Exponential distribution
    let exponential = Exponential::<f32>::standard(false).unwrap();
    group.bench_function("exponential", |b| {
        b.iter(|| {
            let samples = exponential.sample(Some(&[sample_size])).unwrap();
            black_box(samples)
        })
    });
    
    group.finish();
}

fn quick_log_prob_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_log_prob");
    
    let sample_size = 1000;
    let values = Tensor::from_vec(
        (0..sample_size).map(|i| (i as f32) * 0.01 - 5.0).collect(),
        vec![sample_size]
    );
    group.throughput(Throughput::Elements(sample_size as u64));
    
    // Normal log_prob
    let normal = Normal::<f32>::standard(false).unwrap();
    group.bench_function("normal_log_prob", |b| {
        b.iter(|| {
            let log_probs = normal.log_prob(&values).unwrap();
            black_box(log_probs)
        })
    });
    
    // Uniform log_prob
    let uniform = Uniform::<f32>::from_scalars(-10.0, 10.0, false).unwrap();
    group.bench_function("uniform_log_prob", |b| {
        b.iter(|| {
            let log_probs = uniform.log_prob(&values).unwrap();
            black_box(log_probs)
        })
    });
    
    // Exponential log_prob (need positive values)
    let positive_values = Tensor::from_vec(
        (0..sample_size).map(|i| (i as f32) * 0.01 + 0.1).collect(),
        vec![sample_size]
    );
    let exponential = Exponential::<f32>::standard(false).unwrap();
    group.bench_function("exponential_log_prob", |b| {
        b.iter(|| {
            let log_probs = exponential.log_prob(&positive_values).unwrap();
            black_box(log_probs)
        })
    });
    
    group.finish();
}

fn statistical_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics");
    
    let normal = Normal::<f32>::standard(false).unwrap();
    let uniform = Uniform::<f32>::standard(false).unwrap();
    let exponential = Exponential::<f32>::standard(false).unwrap();
    
    group.bench_function("normal_statistics", |b| {
        b.iter(|| {
            let mean = normal.mean().unwrap();
            let var = normal.variance().unwrap();
            let entropy = normal.entropy().unwrap();
            black_box((mean, var, entropy))
        })
    });
    
    group.bench_function("uniform_statistics", |b| {
        b.iter(|| {
            let mean = uniform.mean().unwrap();
            let var = uniform.variance().unwrap();
            let entropy = uniform.entropy().unwrap();
            black_box((mean, var, entropy))
        })
    });
    
    group.bench_function("exponential_statistics", |b| {
        b.iter(|| {
            let mean = exponential.mean().unwrap();
            let var = exponential.variance().unwrap();
            let entropy = exponential.entropy().unwrap();
            black_box((mean, var, entropy))
        })
    });
    
    group.finish();
}

fn batch_sampling_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_comparison");
    
    // Compare scalar vs batch distributions
    let sample_size = 500;
    let batch_size = 10;
    group.throughput(Throughput::Elements((sample_size * batch_size) as u64));
    
    // Scalar distribution (sampled multiple times)
    let scalar_normal = Normal::<f32>::standard(false).unwrap();
    group.bench_function("scalar_multiple", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for _ in 0..batch_size {
                let samples = scalar_normal.sample(Some(&[sample_size])).unwrap();
                results.push(samples);
            }
            black_box(results)
        })
    });
    
    // Batch distribution (single sample call)
    let means = Tensor::from_vec(vec![0.0f32; batch_size], vec![batch_size]);
    let scales = Tensor::from_vec(vec![1.0f32; batch_size], vec![batch_size]);
    let batch_normal = Normal::new(means, scales, false).unwrap();
    group.bench_function("batch_single", |b| {
        b.iter(|| {
            let samples = batch_normal.sample(Some(&[sample_size])).unwrap();
            black_box(samples)
        })
    });
    
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(30); // Smaller sample size for speed
    targets = 
        quick_distribution_sampling,
        quick_log_prob_benchmark,
        statistical_operations_benchmark,
        batch_sampling_comparison
);

criterion_main!(benches);