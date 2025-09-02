//! Phase 4 Gradient Utilities Benchmarks
//! フェーズ4勾配ユーティリティベンチマーク

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::autograd::{Variable, grad, jacobian, gradcheck_simple, no_grad};
use rustorch::tensor::Tensor;

fn benchmark_grad_computation(c: &mut Criterion) {
    c.bench_function("grad_computation", |b| {
        let x = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true);
        let y = Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true);
        
        b.iter(|| {
            let z1 = &x * &x;
            let z2 = &y * &y;
            let output = &z1 + &z2;
            
            let _gradients = grad(
                &[black_box(output)], 
                &[x.clone(), y.clone()], 
                None, 
                false, 
                false
            ).unwrap();
        });
    });
}

fn benchmark_jacobian_computation(c: &mut Criterion) {
    c.bench_function("jacobian_computation", |b| {
        let input = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true);
        
        b.iter(|| {
            let _jacobian_result = jacobian(
                |x| x * x,
                black_box(&input),
                false,
            ).unwrap();
        });
    });
}

fn benchmark_gradcheck(c: &mut Criterion) {
    c.bench_function("gradcheck", |b| {
        let input = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true);
        
        b.iter(|| {
            let _passed = gradcheck_simple(
                |inputs| {
                    let x = &inputs[0];
                    x * x
                },
                &[black_box(input.clone())],
            );
        });
    });
}

fn benchmark_no_grad_context(c: &mut Criterion) {
    c.bench_function("no_grad_context", |b| {
        let x = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true);
        
        b.iter(|| {
            no_grad(|| {
                let y = &x * &x;
                y.backward();
                black_box(y)
            })
        });
    });
}

criterion_group!(
    phase4_benchmarks,
    benchmark_grad_computation,
    benchmark_jacobian_computation,
    benchmark_gradcheck,
    benchmark_no_grad_context
);
criterion_main!(phase4_benchmarks);