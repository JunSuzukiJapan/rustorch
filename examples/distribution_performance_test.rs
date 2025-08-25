use rustorch::distributions::*;
use rustorch::tensor::Tensor;
use std::time::Instant;

fn main() {
    println!("🚀 RusTorch統計分布パフォーマンステスト\n");

    let sample_size = 1000;
    let iterations = 1000;

    println!(
        "サンプルサイズ: {}, 繰り返し: {}\n",
        sample_size, iterations
    );

    // Normal distribution
    test_distribution_sampling(
        "Normal (Box-Muller)",
        || {
            let normal = Normal::<f32>::standard(false).unwrap();
            normal.sample(Some(&[sample_size])).unwrap()
        },
        iterations,
    );

    // Bernoulli distribution
    test_distribution_sampling(
        "Bernoulli",
        || {
            let bernoulli = Bernoulli::<f32>::from_scalar_prob(0.5, false).unwrap();
            bernoulli.sample(Some(&[sample_size])).unwrap()
        },
        iterations,
    );

    // Uniform distribution
    test_distribution_sampling(
        "Uniform",
        || {
            let uniform = Uniform::<f32>::standard(false).unwrap();
            uniform.sample(Some(&[sample_size])).unwrap()
        },
        iterations,
    );

    // Exponential distribution
    test_distribution_sampling(
        "Exponential",
        || {
            let exponential = Exponential::<f32>::standard(false).unwrap();
            exponential.sample(Some(&[sample_size])).unwrap()
        },
        iterations,
    );

    // Gamma distribution (exponential case, shape=1)
    test_distribution_sampling(
        "Gamma (Exponential, shape=1)",
        || {
            let gamma = Gamma::<f32>::exponential(1.0, false).unwrap();
            gamma.sample(Some(&[sample_size])).unwrap()
        },
        iterations / 2,
    ); // Half iterations due to complexity

    // Gamma distribution (general case, shape=2)
    test_distribution_sampling(
        "Gamma (shape=2)",
        || {
            let concentration = Tensor::from_vec(vec![2.0f32], vec![]);
            let rate = Tensor::from_vec(vec![1.0f32], vec![]);
            let gamma = Gamma::from_concentration_rate(concentration, rate, false).unwrap();
            gamma.sample(Some(&[sample_size])).unwrap()
        },
        iterations / 4,
    ); // Quarter iterations due to complexity

    // Beta distribution (uniform case)
    test_distribution_sampling(
        "Beta (Uniform, α=1, β=1)",
        || {
            let beta = Beta::<f32>::uniform(false).unwrap();
            beta.sample(Some(&[sample_size])).unwrap()
        },
        iterations / 4,
    ); // Quarter iterations due to complexity

    // Beta distribution (general case)
    test_distribution_sampling(
        "Beta (α=2, β=5)",
        || {
            let beta = Beta::<f32>::from_scalars(2.0, 5.0, false).unwrap();
            beta.sample(Some(&[sample_size])).unwrap()
        },
        iterations / 8,
    ); // Eighth iterations due to complexity

    println!("\n📊 Log確率密度関数のテスト\n");

    // Test log_prob performance
    let values = Tensor::from_vec(vec![0.5f32; sample_size], vec![sample_size]);
    let positive_values = Tensor::from_vec(
        (0..sample_size).map(|i| 0.1 + (i as f32) * 0.001).collect(),
        vec![sample_size],
    );
    let unit_values = Tensor::from_vec(
        (0..sample_size)
            .map(|i| 0.01 + (i as f32) * 0.0098)
            .collect(),
        vec![sample_size],
    );

    test_log_prob(
        "Normal log_prob",
        || {
            let normal = Normal::<f32>::standard(false).unwrap();
            normal.log_prob(&values).unwrap()
        },
        iterations,
    );

    test_log_prob(
        "Exponential log_prob",
        || {
            let exponential = Exponential::<f32>::standard(false).unwrap();
            exponential.log_prob(&positive_values).unwrap()
        },
        iterations,
    );

    test_log_prob(
        "Gamma log_prob",
        || {
            let gamma = Gamma::<f32>::exponential(1.0, false).unwrap();
            gamma.log_prob(&positive_values).unwrap()
        },
        iterations / 2,
    );

    test_log_prob(
        "Beta log_prob",
        || {
            let beta = Beta::<f32>::uniform(false).unwrap();
            beta.log_prob(&unit_values).unwrap()
        },
        iterations / 2,
    );

    println!("\n⚡ 統計計算のテスト\n");

    test_stats(
        "Normal統計計算",
        || {
            let normal = Normal::<f32>::standard(false).unwrap();
            (normal.mean().unwrap(), normal.variance().unwrap())
        },
        iterations * 10,
    );

    test_stats(
        "Gamma統計計算",
        || {
            let gamma = Gamma::<f32>::exponential(1.0, false).unwrap();
            (gamma.mean().unwrap(), gamma.variance().unwrap())
        },
        iterations,
    );

    test_stats(
        "Beta統計計算",
        || {
            let beta = Beta::<f32>::from_scalars(2.0, 3.0, false).unwrap();
            (beta.mean().unwrap(), beta.variance().unwrap())
        },
        iterations,
    );
}

fn test_distribution_sampling<F>(name: &str, mut f: F, iterations: usize)
where
    F: FnMut() -> Tensor<f32>,
{
    let start = Instant::now();

    for _ in 0..iterations {
        let _result = f();
    }

    let duration = start.elapsed();
    let avg_time = duration.as_nanos() as f64 / iterations as f64;
    let throughput = (1000.0 * 1_000_000.0) / avg_time; // samples per second

    println!(
        "{:<30} | {:>8.1} ns/op | {:>8.0} samples/s",
        name, avg_time, throughput
    );
}

fn test_log_prob<F>(name: &str, mut f: F, iterations: usize)
where
    F: FnMut() -> Tensor<f32>,
{
    let start = Instant::now();

    for _ in 0..iterations {
        let _result = f();
    }

    let duration = start.elapsed();
    let avg_time = duration.as_nanos() as f64 / iterations as f64;

    println!("{:<30} | {:>8.1} ns/op", name, avg_time);
}

fn test_stats<F>(name: &str, mut f: F, iterations: usize)
where
    F: FnMut() -> (Tensor<f32>, Tensor<f32>),
{
    let start = Instant::now();

    for _ in 0..iterations {
        let _result = f();
    }

    let duration = start.elapsed();
    let avg_time = duration.as_nanos() as f64 / iterations as f64;

    println!("{:<30} | {:>8.1} ns/op", name, avg_time);
}
