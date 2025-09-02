//! Advanced Optimizer Tests
//! 高度なオプティマイザーテスト
//!
//! Comprehensive tests for LAMB, AdaBound, and L-BFGS optimizers

use rustorch::optim::{AdaBound, LineSearchMethod, Optimizer, LAMB, LBFGS};
use rustorch::tensor::Tensor;
use std::collections::HashMap;

#[test]
fn test_lamb_basic_functionality() {
    let mut optimizer = LAMB::new(0.01);
    let param = Tensor::<f32>::ones(&[4, 4]);
    let grad = Tensor::<f32>::ones(&[4, 4]) * 0.1;

    let initial_data = param.data.as_slice().unwrap().to_vec();

    // Perform optimization step
    optimizer.step(&param, &grad);

    let updated_data = param.data.as_slice().unwrap();

    // Verify parameters were updated
    assert_ne!(initial_data[0], updated_data[0]);
    assert_eq!(optimizer.step_count(), 1);
}

#[test]
fn test_lamb_large_batch_adaptation() {
    let mut optimizer = LAMB::with_params(0.01, 0.9, 0.999, 1e-6, 0.01);
    let param = Tensor::<f32>::ones(&[8, 8]);
    let grad = Tensor::<f32>::ones(&[8, 8]) * 0.2;

    // Simulate multiple steps for large batch training
    for i in 0..10 {
        let step_grad = &grad * (1.0 + i as f32 * 0.1);
        optimizer.step(&param, &step_grad);
    }

    assert_eq!(optimizer.step_count(), 10);

    // Test state persistence
    let state = optimizer.state_dict();
    assert_eq!(state["step_count"], 10.0);
    assert_eq!(state["learning_rate"], 0.01);
}

#[test]
fn test_lamb_without_bias_correction() {
    let mut optimizer = LAMB::without_bias_correction(0.05, 0.8, 0.95, 1e-5, 0.02);
    let param = Tensor::<f32>::zeros(&[3, 3]);
    let grad = Tensor::<f32>::ones(&[3, 3]) * 0.05;

    optimizer.step(&param, &grad);

    // Verify it works without bias correction
    assert_eq!(optimizer.step_count(), 1);

    // Test setting bias correction
    optimizer.set_bias_correction(true);
    optimizer.step(&param, &grad);

    assert_eq!(optimizer.step_count(), 2);
}

#[test]
fn test_adabound_basic_functionality() {
    let mut optimizer = AdaBound::new(0.01);
    let param = Tensor::<f32>::ones(&[5, 5]);
    let grad = Tensor::<f32>::ones(&[5, 5]) * 0.1;

    let initial_data = param.data.as_slice().unwrap().to_vec();

    // Perform optimization step
    optimizer.step(&param, &grad);

    let updated_data = param.data.as_slice().unwrap();

    // Verify parameters were updated
    assert_ne!(initial_data[0], updated_data[0]);
    assert_eq!(optimizer.step_count(), 1);
}

#[test]
fn test_adabound_convergence_behavior() {
    let mut optimizer = AdaBound::with_params(0.1, 0.05, 0.9, 0.999, 1e-8, 0.01, 1e-3);
    let param = Tensor::<f32>::ones(&[2, 2]) * 2.0;
    let grad = Tensor::<f32>::ones(&[2, 2]) * 0.2;

    let initial_data = param.data.as_slice().unwrap().to_vec();

    // Perform multiple steps to test convergence behavior
    for _ in 0..20 {
        optimizer.step(&param, &grad);
    }

    // Verify optimization occurred
    let final_data = param.data.as_slice().unwrap();
    assert_ne!(initial_data[0], final_data[0]);
    assert_eq!(optimizer.step_count(), 20);

    // Test that the optimizer is still functional after many steps
    let pre_step_data = param.data.as_slice().unwrap().to_vec();
    optimizer.step(&param, &grad);
    let post_step_data = param.data.as_slice().unwrap();
    assert_ne!(pre_step_data[0], post_step_data[0]);
}

#[test]
fn test_adabound_parameter_customization() {
    let mut optimizer = AdaBound::with_params(0.02, 0.08, 0.85, 0.95, 1e-6, 0.05, 2e-3);

    // Test parameter setting
    optimizer.set_final_lr(0.04);
    optimizer.set_gamma(1e-3);

    let param = Tensor::<f32>::ones(&[3, 3]);
    let grad = Tensor::<f32>::ones(&[3, 3]) * 0.1;

    optimizer.step(&param, &grad);

    // Verify state dictionary
    let state = optimizer.state_dict();
    assert_eq!(state["learning_rate"], 0.02);
    assert_eq!(state["final_lr"], 0.04);
    assert_eq!(state["gamma"], 1e-3);
}

#[test]
fn test_lbfgs_basic_functionality() {
    let mut optimizer = LBFGS::new(0.1).unwrap();
    let param = Tensor::<f32>::ones(&[3, 3]);
    let grad = Tensor::<f32>::ones(&[3, 3]) * 0.1;

    let initial_data = param.data.as_slice().unwrap().to_vec();

    // First step (steepest descent)
    optimizer.step(&param, &grad);

    let updated_data = param.data.as_slice().unwrap();

    // Verify parameters were updated
    assert_ne!(initial_data[0], updated_data[0]);
    // Note: LBFGS doesn't expose step_count() method, so we verify functionality differently
    let state = optimizer.state_dict();
    assert_eq!(state["step_count"], 1.0);
}

#[test]
fn test_lbfgs_memory_building() {
    let mut optimizer = LBFGS::with_params(0.1, 20, 20, 1e-5, 1e-9, 5, LineSearchMethod::None).unwrap();
    let param = Tensor::<f32>::ones(&[4, 4]) * 3.0;

    // Perform multiple steps to build L-BFGS memory
    for i in 0..8 {
        let grad = Tensor::<f32>::ones(&[4, 4]) * (0.1 * (i + 1) as f32);
        optimizer.step(&param, &grad);
    }

    // Verify steps through state dictionary
    let state = optimizer.state_dict();
    assert_eq!(state["step_count"], 8.0);

    // Memory should be limited to history_size (5)
    // This is tested internally in the implementation

    // Verify state dictionary
    assert_eq!(state["history_size"], 5.0);
}

#[test]
fn test_lbfgs_convergence_detection() {
    let mut optimizer = LBFGS::new(0.1).unwrap();
    optimizer.set_tolerance_grad(1e-2).unwrap();

    let param = Tensor::<f32>::ones(&[2, 2]);
    let small_grad = Tensor::<f32>::ones(&[2, 2]) * 1e-4; // Below tolerance

    let initial_param = param.clone();

    // Should detect convergence and not update
    optimizer.step(&param, &small_grad);

    let final_data = param.data.as_slice().unwrap();
    let initial_data = initial_param.data.as_slice().unwrap();

    // Parameters should remain unchanged due to convergence
    for (final_val, initial_val) in final_data.iter().zip(initial_data.iter()) {
        assert!((final_val - initial_val).abs() < 1e-6);
    }
}

#[test]
fn test_lbfgs_line_search_methods() {
    let optimizers = vec![
        LBFGS::with_params(0.1, 20, 20, 1e-5, 1e-9, 5, LineSearchMethod::None).unwrap(),
        LBFGS::with_params(0.1, 20, 20, 1e-5, 1e-9, 5, 
            LineSearchMethod::Backtracking { c1: 1e-4, rho: 0.5 }).unwrap(),
        LBFGS::with_params(0.1, 20, 20, 1e-5, 1e-9, 5,
            LineSearchMethod::StrongWolfe { c1: 1e-4, c2: 0.9 }).unwrap(),
    ];

    for mut optimizer in optimizers {
        let param = Tensor::<f32>::ones(&[3, 3]);
        let grad = Tensor::<f32>::ones(&[3, 3]) * 0.05;

        let initial_data = param.data.as_slice().unwrap().to_vec();

        optimizer.step(&param, &grad);

        let updated_data = param.data.as_slice().unwrap();

        // All line search methods should work
        assert_ne!(initial_data[0], updated_data[0]);
        let state = optimizer.state_dict();
        assert_eq!(state["step_count"], 1.0);
    }
}

#[test]
fn test_all_optimizers_state_management() {
    let mut lamb = LAMB::new(0.01);
    let mut adabound = AdaBound::new(0.01);
    let mut lbfgs = LBFGS::new(0.1).unwrap();

    let param = Tensor::<f32>::ones(&[2, 2]);
    let grad = Tensor::<f32>::ones(&[2, 2]) * 0.1;

    // Test all optimizers can handle state management
    let optimizers: Vec<&mut dyn Optimizer> = vec![&mut lamb, &mut adabound, &mut lbfgs];

    for optimizer in optimizers {
        // Test state saving
        let _initial_state = optimizer.state_dict();

        // Perform optimization
        optimizer.step(&param, &grad);

        // Test state loading
        let mut new_state = HashMap::new();
        new_state.insert("learning_rate".to_string(), 0.05);
        optimizer.load_state_dict(new_state);

        assert_eq!(optimizer.learning_rate(), 0.05);
    }
}

#[test]
fn test_optimizer_performance_comparison() {
    let param = Tensor::<f32>::ones(&[10, 10]) * 5.0;
    let target = Tensor::<f32>::zeros(&[10, 10]);

    let mut lamb = LAMB::new(0.01);
    let mut adabound = AdaBound::new(0.01);
    let mut lbfgs = LBFGS::new(0.1).unwrap();

    // Simple quadratic optimization problem
    // Minimize ||param - target||^2

    let initial_param = param.clone();
    let param_lamb = initial_param.clone();
    let param_adabound = initial_param.clone();
    let param_lbfgs = initial_param.clone();

    // Run each optimizer for multiple steps
    for _ in 0..10 {
        // Gradient = 2 * (param - target)
        let grad_lamb = (&param_lamb - &target) * 2.0;
        let grad_adabound = (&param_adabound - &target) * 2.0;
        let grad_lbfgs = (&param_lbfgs - &target) * 2.0;

        lamb.step(&param_lamb, &grad_lamb);
        adabound.step(&param_adabound, &grad_adabound);
        lbfgs.step(&param_lbfgs, &grad_lbfgs);
    }

    // All optimizers should have made progress toward the target
    // Calculate distances using sum of squares (manual norm calculation)
    let initial_sum_squares = initial_param
        .data
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt();
    let lamb_sum_squares = param_lamb.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let adabound_sum_squares = param_adabound
        .data
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt();
    let lbfgs_sum_squares = param_lbfgs.data.iter().map(|&x| x * x).sum::<f32>().sqrt();

    // All should reduce the distance to target (though rates may vary)
    assert!(lamb_sum_squares < initial_sum_squares);
    assert!(adabound_sum_squares < initial_sum_squares);
    assert!(lbfgs_sum_squares < initial_sum_squares);
}

#[test]
fn test_optimizers_with_different_tensor_shapes() {
    let shapes = vec![
        vec![1],          // 1D
        vec![5, 5],       // 2D
        vec![2, 3, 4],    // 3D
        vec![2, 2, 2, 2], // 4D
    ];

    for shape in shapes {
        let param = Tensor::<f32>::ones(&shape);
        let grad = Tensor::<f32>::ones(&shape) * 0.1;

        let mut lamb = LAMB::new(0.01);
        let mut adabound = AdaBound::new(0.01);
        let mut lbfgs = LBFGS::new(0.05).unwrap();

        let initial_data = param.data.as_slice().unwrap().to_vec();

        // Test all optimizers with this shape
        lamb.step(&param, &grad);
        let lamb_data = param.data.as_slice().unwrap();
        assert_ne!(initial_data[0], lamb_data[0]);

        // Reset parameter
        param.copy_from(&Tensor::<f32>::ones(&shape));

        adabound.step(&param, &grad);
        let adabound_data = param.data.as_slice().unwrap();
        assert_ne!(initial_data[0], adabound_data[0]);

        // Reset parameter
        param.copy_from(&Tensor::<f32>::ones(&shape));

        lbfgs.step(&param, &grad);
        let lbfgs_data = param.data.as_slice().unwrap();
        assert_ne!(initial_data[0], lbfgs_data[0]);
    }
}
