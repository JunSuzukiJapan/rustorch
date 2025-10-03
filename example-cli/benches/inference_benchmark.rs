// Inference performance benchmark for RusTorch CLI

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch_cli::metrics::{timing::InferenceTimer, MetricsCollector};

fn benchmark_metrics_collection(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_collection");

    group.bench_function("create_metrics", |b| {
        b.iter(|| {
            let metrics = MetricsCollector::new();
            black_box(metrics);
        });
    });

    group.bench_function("set_all_metrics", |b| {
        b.iter(|| {
            let mut metrics = MetricsCollector::new();
            metrics.set_ttft(150.0);
            metrics.set_tokens_per_sec(25.0);
            metrics.set_total_time(5000.0);
            metrics.set_memory_usage(1_200_000_000);
            metrics.set_model_size(1_000_000_000);
            metrics.set_backend("test".to_string());
            black_box(metrics);
        });
    });

    group.bench_function("check_targets", |b| {
        let mut metrics = MetricsCollector::new();
        metrics.set_ttft(150.0);
        metrics.set_tokens_per_sec(25.0);
        metrics.set_model_size(1_000_000_000);
        metrics.set_memory_usage(1_200_000_000);

        b.iter(|| {
            let result = metrics.meets_all_targets();
            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_inference_timer(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_timer");

    group.bench_function("create_timer", |b| {
        b.iter(|| {
            let timer = InferenceTimer::new();
            black_box(timer);
        });
    });

    group.bench_function("full_timing_cycle", |b| {
        b.iter(|| {
            let mut timer = InferenceTimer::new();
            timer.start();
            timer.mark_first_token();
            for _ in 0..10 {
                timer.mark_token();
            }
            timer.end();
            black_box(timer);
        });
    });

    group.bench_function("calculate_metrics", |b| {
        let mut timer = InferenceTimer::new();
        timer.start();
        timer.mark_first_token();
        for _ in 0..100 {
            timer.mark_token();
        }
        timer.end();

        b.iter(|| {
            let ttft = timer.ttft_ms();
            let tps = timer.tokens_per_sec();
            let total = timer.total_ms();
            black_box((ttft, tps, total));
        });
    });

    group.finish();
}

fn benchmark_token_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_generation_simulation");

    for token_count in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(token_count),
            token_count,
            |b, &count| {
                b.iter(|| {
                    let mut timer = InferenceTimer::new();
                    timer.start();
                    timer.mark_first_token();
                    for _ in 0..count {
                        timer.mark_token();
                    }
                    timer.end();
                    black_box(timer);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_metrics_collection,
    benchmark_inference_timer,
    benchmark_token_generation
);
criterion_main!(benches);
