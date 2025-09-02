//! Phase 3 Advanced Layers Benchmarks
//! フェーズ3高度レイヤーのベンチマーク

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::autograd::Variable;
use rustorch::nn::{
    ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
    InstanceNorm1d, InstanceNorm2d, InstanceNorm3d,
    glu, swiglu, geglu, reglu,
};
use rustorch::tensor::Tensor;

fn bench_conv_transpose_1d(c: &mut Criterion) {
    let layer = ConvTranspose1d::<f32>::new(
        64,        // in_channels
        32,        // out_channels
        3,         // kernel_size
        Some(2),   // stride
        Some(1),   // padding
        None,      // output_padding
        None,      // dilation
        None,      // groups
        Some(true), // bias
    );
    
    let input = Variable::new(
        Tensor::from_vec(vec![1.0f32; 64 * 100], vec![1, 64, 100]),
        false,
    );
    
    c.bench_function("conv_transpose_1d_forward", |b| {
        b.iter(|| {
            black_box(layer.forward(black_box(&input)));
        })
    });
}

fn bench_conv_transpose_2d(c: &mut Criterion) {
    let layer = ConvTranspose2d::<f32>::new(
        32,              // in_channels
        16,              // out_channels
        (3, 3),          // kernel_size
        Some((2, 2)),    // stride
        Some((1, 1)),    // padding
        None,            // output_padding
        None,            // dilation
        None,            // groups
        Some(true),      // bias
    );
    
    let input = Variable::new(
        Tensor::from_vec(vec![1.0f32; 32 * 32 * 32], vec![1, 32, 32, 32]),
        false,
    );
    
    c.bench_function("conv_transpose_2d_forward", |b| {
        b.iter(|| {
            black_box(layer.forward(black_box(&input)));
        })
    });
}

fn bench_conv_transpose_3d(c: &mut Criterion) {
    let layer = ConvTranspose3d::<f32>::new(
        16,                 // in_channels
        8,                  // out_channels
        (3, 3, 3),          // kernel_size
        Some((2, 2, 2)),    // stride
        Some((1, 1, 1)),    // padding
        None,               // output_padding
        None,               // dilation
        None,               // groups
        Some(true),         // bias
    );
    
    let input = Variable::new(
        Tensor::from_vec(vec![1.0f32; 16 * 16 * 16 * 16], vec![1, 16, 16, 16, 16]),
        false,
    );
    
    c.bench_function("conv_transpose_3d_forward", |b| {
        b.iter(|| {
            black_box(layer.forward(black_box(&input)));
        })
    });
}

fn bench_instance_norm_1d(c: &mut Criterion) {
    let layer = InstanceNorm1d::<f32>::new(
        64,         // num_features
        None,       // eps
        None,       // momentum
        Some(true), // affine
        None,       // track_running_stats
    );
    
    let input = Variable::new(
        Tensor::from_vec(vec![1.0f32; 64 * 1000], vec![1, 64, 1000]),
        false,
    );
    
    c.bench_function("instance_norm_1d_forward", |b| {
        b.iter(|| {
            black_box(layer.forward(black_box(&input)));
        })
    });
}

fn bench_instance_norm_2d(c: &mut Criterion) {
    let layer = InstanceNorm2d::<f32>::new(
        32,         // num_features
        None,       // eps
        None,       // momentum
        Some(true), // affine
        None,       // track_running_stats
    );
    
    let input = Variable::new(
        Tensor::from_vec(vec![1.0f32; 32 * 64 * 64], vec![1, 32, 64, 64]),
        false,
    );
    
    c.bench_function("instance_norm_2d_forward", |b| {
        b.iter(|| {
            black_box(layer.forward(black_box(&input)));
        })
    });
}

fn bench_instance_norm_3d(c: &mut Criterion) {
    let layer = InstanceNorm3d::<f32>::new(
        16,         // num_features
        None,       // eps
        None,       // momentum
        Some(true), // affine
        None,       // track_running_stats
    );
    
    let input = Variable::new(
        Tensor::from_vec(vec![1.0f32; 16 * 32 * 32 * 32], vec![1, 16, 32, 32, 32]),
        false,
    );
    
    c.bench_function("instance_norm_3d_forward", |b| {
        b.iter(|| {
            black_box(layer.forward(black_box(&input)));
        })
    });
}

fn bench_glu_variants(c: &mut Criterion) {
    let input_data = vec![1.0f32; 1024];
    let input = Variable::new(
        Tensor::from_vec(input_data, vec![1, 1024]),
        false,
    );
    
    c.bench_function("glu_forward", |b| {
        b.iter(|| {
            black_box(glu(black_box(&input)));
        })
    });
    
    c.bench_function("swiglu_forward", |b| {
        b.iter(|| {
            black_box(swiglu(black_box(&input)));
        })
    });
    
    c.bench_function("geglu_forward", |b| {
        b.iter(|| {
            black_box(geglu(black_box(&input)));
        })
    });
    
    c.bench_function("reglu_forward", |b| {
        b.iter(|| {
            black_box(reglu(black_box(&input)));
        })
    });
}

criterion_group!(
    phase3_benches,
    bench_conv_transpose_1d,
    bench_conv_transpose_2d,
    bench_conv_transpose_3d,
    bench_instance_norm_1d,
    bench_instance_norm_2d,
    bench_instance_norm_3d,
    bench_glu_variants
);
criterion_main!(phase3_benches);