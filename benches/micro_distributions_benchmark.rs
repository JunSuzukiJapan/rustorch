use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rustorch::distributions::*;
use rustorch::tensor::Tensor;

fn micro_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("micro_sampling");
    group.sample_size(20); // Very small sample size for speed
    group.measurement_time(std::time::Duration::from_secs(10)); // Shorter measurement time
    
    let sample_size = 1000;
    group.throughput(Throughput::Elements(sample_size as u64));
    
    // Normal distribution - Box-Muller method
    let normal = Normal::<f32>::standard(false).unwrap();
    group.bench_function("normal_1000", |b| {
        b.iter(|| black_box(normal.sample(Some(&[sample_size])).unwrap()))
    });
    
    // Bernoulli - simple inverse transform
    let bernoulli = Bernoulli::<f32>::from_scalar_prob(0.5, false).unwrap();
    group.bench_function("bernoulli_1000", |b| {
        b.iter(|| black_box(bernoulli.sample(Some(&[sample_size])).unwrap()))
    });
    
    // Uniform - direct transformation
    let uniform = Uniform::<f32>::standard(false).unwrap();
    group.bench_function("uniform_1000", |b| {
        b.iter(|| black_box(uniform.sample(Some(&[sample_size])).unwrap()))
    });
    
    group.finish();
}

fn micro_log_prob(c: &mut Criterion) {
    let mut group = c.benchmark_group("micro_log_prob");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(10));
    
    let sample_size = 1000;
    let values = Tensor::from_vec(vec![0.5f32; sample_size], vec![sample_size]);
    group.throughput(Throughput::Elements(sample_size as u64));
    
    let normal = Normal::<f32>::standard(false).unwrap();
    group.bench_function("normal_log_prob", |b| {
        b.iter(|| black_box(normal.log_prob(&values).unwrap()))
    });
    
    let uniform = Uniform::<f32>::standard(false).unwrap();
    group.bench_function("uniform_log_prob", |b| {
        b.iter(|| black_box(uniform.log_prob(&values).unwrap()))
    });
    
    group.finish();
}

fn micro_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("micro_statistics");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(5));
    
    let normal = Normal::<f32>::standard(false).unwrap();
    group.bench_function("normal_stats", |b| {
        b.iter(|| {
            let mean = normal.mean().unwrap();
            let var = normal.variance().unwrap();
            black_box((mean, var))
        })
    });
    
    group.finish();
}

criterion_group!(benches, micro_sampling, micro_log_prob, micro_statistics);
criterion_main!(benches);