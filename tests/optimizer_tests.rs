/// Tests for optimizer implementations
/// オプティマイザー実装のテスト

use rustorch::tensor::Tensor;
use rustorch::optim::{Optimizer, SGD, Adam, RMSprop, AdaGrad};

#[test]
fn test_sgd_optimizer() {
    let mut sgd = SGD::new(0.01);
    
    // Test basic functionality
    assert_eq!(sgd.learning_rate(), 0.01);
    
    // Test parameter update - use scalar tensor
    let param = Tensor::ones(&[1]);
    let grad = Tensor::ones(&[1]);
    
    sgd.step(&param, &grad);
    
    // Parameters should be updated
    let expected = 1.0 - 0.01; // 1.0 - lr * grad
    assert!((param.item() - expected).abs() < 1e-6);
}

#[test]
fn test_sgd_with_weight_decay() {
    let mut sgd = SGD::with_weight_decay(0.01, 0.9, 0.001);
    
    let param = Tensor::ones(&[1]);
    let grad = Tensor::ones(&[1]);
    
    sgd.step(&param, &grad);
    
    // Check that weight decay was applied
    // grad_total = grad + weight_decay * param = 1.0 + 0.001 * 1.0 = 1.001
    let expected = 1.0 - 0.01 * 1.001; // param - lr * grad_total
    assert!((param.item() - expected).abs() < 1e-6);
}

#[test]
fn test_adam_optimizer() {
    let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
    
    assert_eq!(adam.learning_rate(), 0.001);
    
    let param = Tensor::ones(&[1]);
    let grad = Tensor::ones(&[1]);
    
    let original_value = param.item();
    adam.step(&param, &grad);
    
    // Parameters should be updated (exact value depends on Adam formula)
    assert_ne!(param.item(), original_value);
}

#[test]
fn test_adam_with_weight_decay() {
    let mut adam = Adam::with_weight_decay(0.001, 0.9, 0.999, 1e-8, 0.01);
    
    let param = Tensor::ones(&[1]);
    let grad = Tensor::ones(&[1]);
    
    let original_value = param.item();
    adam.step(&param, &grad);
    
    // Parameters should be updated with weight decay
    assert_ne!(param.item(), original_value);
}

#[test]
fn test_rmsprop_optimizer() {
    let mut rmsprop = RMSprop::new(0.01, 0.99, 1e-8);
    
    assert_eq!(rmsprop.learning_rate(), 0.01);
    
    let param = Tensor::ones(&[1]);
    let grad = Tensor::ones(&[1]);
    
    let original_value = param.item();
    rmsprop.step(&param, &grad);
    
    // Parameters should be updated
    assert_ne!(param.item(), original_value);
}

#[test]
fn test_rmsprop_with_momentum() {
    let mut rmsprop = RMSprop::with_momentum(0.01, 0.99, 1e-8, 0.9);
    
    let param = Tensor::ones(&[1]);
    let grad = Tensor::ones(&[1]);
    
    let original_value = param.item();
    rmsprop.step(&param, &grad);
    
    // Parameters should be updated with momentum
    assert_ne!(param.item(), original_value);
}

#[test]
fn test_rmsprop_centered() {
    let mut rmsprop = RMSprop::centered(0.01, 0.99, 1e-8, true);
    
    let param = Tensor::ones(&[1]);
    let grad = Tensor::ones(&[1]);
    
    let original_value = param.item();
    rmsprop.step(&param, &grad);
    
    // Parameters should be updated with centered variant
    assert_ne!(param.item(), original_value);
}

#[test]
fn test_adagrad_optimizer() {
    let mut adagrad = AdaGrad::new(0.01, 1e-10);
    
    assert_eq!(adagrad.learning_rate(), 0.01);
    
    let param = Tensor::ones(&[1]);
    let grad = Tensor::ones(&[1]);
    
    let original_value = param.item();
    adagrad.step(&param, &grad);
    
    // Parameters should be updated
    assert_ne!(param.item(), original_value);
}

#[test]
fn test_adagrad_with_weight_decay() {
    let mut adagrad = AdaGrad::with_weight_decay(0.01, 1e-10, 0.001);
    
    let param = Tensor::ones(&[1]);
    let grad = Tensor::ones(&[1]);
    
    let original_value = param.item();
    adagrad.step(&param, &grad);
    
    // Parameters should be updated with weight decay
    assert_ne!(param.item(), original_value);
}

#[test]
fn test_adagrad_with_initial_accumulator() {
    let mut adagrad = AdaGrad::with_initial_accumulator(0.01, 1e-10, 0.1);
    
    let param = Tensor::ones(&[1]);
    let grad = Tensor::ones(&[1]);
    
    let original_value = param.item();
    adagrad.step(&param, &grad);
    
    // Parameters should be updated with initial accumulator
    assert_ne!(param.item(), original_value);
}

#[test]
fn test_optimizer_state_dict() {
    let mut sgd = SGD::new(0.01);
    
    // Get initial state
    let state = sgd.state_dict();
    assert_eq!(state.get("learning_rate"), Some(&0.01));
    assert_eq!(state.get("momentum"), Some(&0.9));
    
    // Modify optimizer
    sgd.set_learning_rate(0.001);
    
    // Load state back
    sgd.load_state_dict(state);
    assert_eq!(sgd.learning_rate(), 0.01);
}

#[test]
fn test_multiple_step_convergence() {
    let mut sgd = SGD::new(0.1); // No momentum for simple test
    
    // Simple quadratic function: f(x) = (x - 2)^2
    // Gradient: f'(x) = 2(x - 2)
    // Minimum at x = 2
    
    let param = Tensor::zeros(&[1]); // Start at x = 0
    
    for _ in 0..50 {
        // Compute gradient: 2 * (param - 2)
        let grad_value = 2.0 * (param.item() - 2.0);
        let grad = Tensor::from_vec(vec![grad_value], vec![1]);
        
        sgd.step(&param, &grad);
    }
    
    // Should converge close to 2
    assert!((param.item() - 2.0).abs() < 0.1);
}

#[test]
fn test_learning_rate_modification() {
    let mut optimizers: Vec<Box<dyn Optimizer>> = vec![
        Box::new(SGD::new(0.01)),
        Box::new(Adam::new(0.001, 0.9, 0.999, 1e-8)),
        Box::new(RMSprop::new(0.01, 0.99, 1e-8)),
        Box::new(AdaGrad::new(0.01, 1e-10)),
    ];
    
    for optimizer in &mut optimizers {
        // Test learning rate modification
        optimizer.set_learning_rate(0.005);
        assert_eq!(optimizer.learning_rate(), 0.005);
    }
}