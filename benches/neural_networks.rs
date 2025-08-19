//! Neural network benchmarks
//! ニューラルネットワークのベンチマーク

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::prelude::*;
use rustorch::nn::{Linear, loss::mse_loss};
use rustorch::optim::{SGD, Optimizer};

fn bench_linear_layer(c: &mut Criterion) {
    let model = Linear::new(784, 128);
    let input = Variable::new(
        Tensor::from_vec((0..784).map(|i| i as f32 * 0.01).collect(), vec![1, 784]),
        false
    );
    
    c.bench_function("linear_forward_784_to_128", |b| {
        b.iter(|| black_box(model.forward(&input)))
    });
    
    // Batch processing
    let batch_input = Variable::new(
        Tensor::from_vec((0..78400).map(|i| i as f32 * 0.01).collect(), vec![100, 784]),
        false
    );
    
    c.bench_function("linear_forward_batch100_784_to_128", |b| {
        b.iter(|| black_box(model.forward(&batch_input)))
    });
}

fn bench_neural_network_training(c: &mut Criterion) {
    c.bench_function("simple_training_step", |b| {
        b.iter(|| {
            let model = Linear::new(10, 1);
            let params = model.parameters();
            let mut optimizer = SGD::new(params, 0.01, None, None, None, None);
            
            let input = Variable::new(
                Tensor::from_vec((0..10).map(|i| i as f32 * 0.1).collect(), vec![1, 10]),
                false
            );
            let target = Variable::new(
                Tensor::from_vec(vec![1.0], vec![1, 1]),
                false
            );
            
            // Forward pass
            let output = model.forward(&input);
            let loss = mse_loss(&output, &target);
            
            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            black_box(loss)
        })
    });
}

fn bench_multi_layer_network(c: &mut Criterion) {
    c.bench_function("three_layer_forward", |b| {
        b.iter(|| {
            let layer1 = Linear::new(784, 256);
            let layer2 = Linear::new(256, 128);
            let layer3 = Linear::new(128, 10);
            
            let input = Variable::new(
                Tensor::from_vec((0..784).map(|i| i as f32 * 0.01).collect(), vec![1, 784]),
                false
            );
            
            let h1 = layer1.forward(&input);
            let h2 = layer2.forward(&h1);
            let output = layer3.forward(&h2);
            
            black_box(output)
        })
    });
}

fn bench_batch_training(c: &mut Criterion) {
    c.bench_function("batch_training_mnist_like", |b| {
        b.iter(|| {
            let model = Linear::new(784, 10);
            let params = model.parameters();
            let mut optimizer = SGD::new(params, 0.01, None, None, None, None);
            
            // Simulate MNIST-like batch
            let batch_size = 32;
            let input = Variable::new(
                Tensor::from_vec(
                    (0..784 * batch_size).map(|i| (i as f32 * 0.001) % 1.0).collect(),
                    vec![batch_size, 784]
                ),
                false
            );
            let target = Variable::new(
                Tensor::from_vec(
                    (0..10 * batch_size).map(|i| if i % 11 == 0 { 1.0 } else { 0.0 }).collect(),
                    vec![batch_size, 10]
                ),
                false
            );
            
            // Training step
            optimizer.zero_grad();
            let output = model.forward(&input);
            let loss = mse_loss(&output, &target);
            loss.backward();
            optimizer.step();
            
            black_box(loss)
        })
    });
}

fn bench_rnn_operations(c: &mut Criterion) {
    use rustorch::nn::GRU;
    
    c.bench_function("gru_forward_seq10_batch8", |b| {
        b.iter(|| {
            let gru = GRU::<f32>::new(
                50,    // input_size
                100,   // hidden_size
                Some(1), // num_layers
                Some(true), // bias
                Some(true), // batch_first
                None,   // dropout
                Some(false), // bidirectional
            );
            
            let input = Variable::new(
                Tensor::from_vec(
                    (0..4000).map(|i| i as f32 * 0.001).collect(),
                    vec![10, 8, 50] // [seq_len, batch, input_size]
                ),
                false
            );
            
            let (output, _hidden) = gru.forward_with_hidden(&input, None);
            black_box(output)
        })
    });
}

criterion_group!(
    neural_network_benches,
    bench_linear_layer,
    bench_neural_network_training,
    bench_multi_layer_network,
    bench_batch_training,
    bench_rnn_operations
);
criterion_main!(neural_network_benches);
