use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::autograd::Variable;
use rustorch::nn::{AdaptiveMaxPool2d, Conv1d, Conv2d, Conv3d, ConvTranspose2d};
use rustorch::tensor::Tensor;

#[cfg(not(target_arch = "wasm32"))]
use rustorch::gpu::DeviceManager;

/// Check if GPU is available for GPU-specific benchmarks
#[cfg(not(target_arch = "wasm32"))]
fn is_gpu_available() -> bool {
    DeviceManager::is_cuda_available() || DeviceManager::is_metal_available()
}

#[cfg(target_arch = "wasm32")]
fn is_gpu_available() -> bool {
    false
}

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

// GPU-specific benchmarks that only run when GPU is available
fn benchmark_gpu_operations(c: &mut Criterion) {
    if !is_gpu_available() {
        println!("Skipping GPU benchmarks - no GPU available");
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        c.bench_function("gpu_conv2d_creation", |b| {
            b.iter(|| {
                black_box(Conv2d::<f32>::new(
                    32,
                    64,
                    (3, 3),
                    Some((1, 1)),
                    Some((1, 1)),
                    None,
                ))
            })
        });

        // Add more GPU-specific benchmarks if the layers support them
        let conv = Conv2d::<f32>::new(32, 64, (3, 3), Some((1, 1)), Some((1, 1)), None);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0f32; 32 * 224 * 224], vec![1, 32, 224, 224]),
            false,
        );

        c.bench_function("gpu_conv2d_forward", |b| {
            b.iter(|| black_box(conv.forward(&input)))
        });
    }
}

criterion_group!(
    nn_benches,
    benchmark_conv1d_creation,
    benchmark_conv2d_creation,
    benchmark_conv3d_creation,
    benchmark_conv_transpose_creation,
    benchmark_adaptive_pool_creation,
    benchmark_parameter_count,
    benchmark_forward_pass_overhead,
    benchmark_gpu_operations
);

criterion_main!(nn_benches);