use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::autograd::Variable;
use rustorch::nn::{AdaptiveMaxPool2d, Conv1d, Conv2d, Conv3d, ConvTranspose2d};
use rustorch::tensor::Tensor;

fn benchmark_conv1d_creation(c: &mut Criterion) {
    c.bench_function("conv1d_creation", |b| {
        b.iter(|| {
            black_box(Conv1d::<f32>::create(
                64, 128, 3, None, None, None, None, None,
            ))
        })
    });
}

fn benchmark_conv2d_creation(c: &mut Criterion) {
    c.bench_function("conv2d_creation", |b| {
        b.iter(|| {
            black_box(Conv2d::<f32>::new(
                64,
                128,
                (3, 3),
                Some((1, 1)),
                Some((1, 1)),
                None,
            ))
        })
    });
}

fn benchmark_conv3d_creation(c: &mut Criterion) {
    c.bench_function("conv3d_creation", |b| {
        b.iter(|| {
            black_box(Conv3d::<f32>::new(
                64,
                128,
                (3, 3, 3),
                None,
                None,
                None,
                None,
                None,
            ))
        })
    });
}

fn benchmark_conv_transpose_creation(c: &mut Criterion) {
    c.bench_function("conv_transpose_creation", |b| {
        b.iter(|| {
            black_box(ConvTranspose2d::<f32>::new(
                64,
                128,
                (3, 3),
                None,
                None,
                None,
                None,
                None,
                None,
            ))
        })
    });
}

fn benchmark_adaptive_pool_creation(c: &mut Criterion) {
    c.bench_function("adaptive_pool_creation", |b| {
        b.iter(|| black_box(AdaptiveMaxPool2d::<f32>::new((7, 7), None)))
    });
}

fn benchmark_parameter_count(c: &mut Criterion) {
    let conv = Conv1d::<f32>::create(64, 128, 3, None, None, None, None, None);

    c.bench_function("parameter_count", |b| {
        b.iter(|| black_box(conv.num_parameters()))
    });
}

fn benchmark_forward_pass_overhead(c: &mut Criterion) {
    let conv = Conv1d::<f32>::create(64, 128, 3, None, None, None, None, None);
    let input = Variable::new(
        Tensor::from_vec(vec![1.0f32; 64 * 100], vec![1, 64, 100]),
        false,
    );

    c.bench_function("forward_pass_placeholder", |b| {
        b.iter(|| black_box(conv.forward(&input)))
    });
}

criterion_group!(
    nn_benches,
    benchmark_conv1d_creation,
    benchmark_conv2d_creation,
    benchmark_conv3d_creation,
    benchmark_conv_transpose_creation,
    benchmark_adaptive_pool_creation,
    benchmark_parameter_count,
    benchmark_forward_pass_overhead
);

criterion_main!(nn_benches);
