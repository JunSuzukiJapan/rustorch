//! Phase 4 Integration Tests: Gradient Utilities
//! フェーズ4統合テスト：勾配ユーティリティ

use rustorch::autograd::{
    Variable, grad, jacobian, hessian, hvp, gradcheck, gradcheck_simple,
    GradCheckConfig, no_grad, detect_anomaly
};
use rustorch::error::RusTorchError;
use rustorch::tensor::Tensor;

#[test]
fn test_grad_integration() {
    // Test the enhanced grad() function with multiple outputs
    let x = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true);
    let y = Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true);
    
    // f(x, y) = x^2 + y^2
    let z1 = &x * &x;
    let z2 = &y * &y;
    let output = &z1 + &z2;
    
    let gradients = grad(&[output], &[x.clone(), y.clone()], None, false, false).unwrap();
    
    assert!(gradients[0].is_some());
    assert!(gradients[1].is_some());
    
    let grad_x = gradients[0].as_ref().unwrap().as_array()[0];
    let grad_y = gradients[1].as_ref().unwrap().as_array()[0];
    
    assert!((grad_x - 4.0).abs() < 1e-6); // df/dx = 2x = 4
    assert!((grad_y - 6.0).abs() < 1e-6); // df/dy = 2y = 6
}

#[test]
fn test_jacobian_integration() {
    // Test Jacobian computation for a simple function
    let input = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true);
    
    // f(x) = x^2, df/dx = 2x = 4
    let jacobian_result = jacobian(
        |x| x * x,
        &input,
        false,
    ).unwrap();
    
    let jac_data = jacobian_result.as_array().as_slice().unwrap();
    assert!((jac_data[0] - 4.0).abs() < 1e-6);
}

#[test]
fn test_hessian_integration() {
    // Test Hessian computation for a quadratic function
    let input = Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true);
    
    // f(x) = x^2 (Hessian should be 2)
    let hessian_result = hessian(|x| x * x, &input).unwrap();
    
    let hessian_val = hessian_result.as_array().as_slice().unwrap()[0];
    assert!((hessian_val - 2.0).abs() < 1e-1); // Relaxed tolerance for finite differences
}

#[test]
fn test_hvp_integration() {
    // Test Hessian-Vector Product for efficiency
    let input = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true);
    let v = Variable::new(Tensor::from_vec(vec![1.0f32], vec![1]), false);
    
    // f(x) = x^2, Hessian is 2, HVP with v=[1] should be 2
    let hvp_result = hvp(|x| x * x, &input, &v).unwrap();
    
    let hvp_data_guard = hvp_result.data();
    let hvp_data = hvp_data_guard.read().unwrap();
    let hvp_val = hvp_data.as_array().as_slice().unwrap()[0];
    assert!((hvp_val - 2.0).abs() < 1e-1); // Relaxed tolerance for finite differences
}

#[test]
fn test_gradcheck_integration() {
    // Test numerical gradient validation
    let input = Variable::new(Tensor::from_vec(vec![1.5f32], vec![1]), true);
    
    // f(x) = x^2 (simpler function for better numerical stability)
    let config = GradCheckConfig {
        eps: 1e-4f32,
        atol: 1e-2f32,
        rtol: 1e-1f32,
        nondet_tol: 0.0f32,
        check_sparse_nnz: true,
    };
    
    let result = gradcheck(
        |inputs| {
            let x = &inputs[0];
            x * x
        },
        &[input],
        Some(config),
    ).unwrap();
    
    assert!(result.passed, "Gradient check failed: {:?}", result.error_details);
    assert!(result.max_error < 0.1); // Relaxed for finite precision
}

#[test]
fn test_gradcheck_simple_integration() {
    // Test simplified gradient checking
    let input = Variable::new(Tensor::from_vec(vec![2.5f32], vec![1]), true);
    
    // f(x) = x^2
    let passed = gradcheck_simple(
        |inputs| {
            let x = &inputs[0];
            x * x
        },
        &[input],
    );
    
    assert!(passed);
}

#[test]
fn test_context_managers_integration() {
    // Test gradient context management
    assert!(rustorch::autograd::is_grad_enabled()); // Default state
    
    // Test no_grad context
    let result = no_grad(|| {
        assert!(!rustorch::autograd::is_grad_enabled());
        let x = Variable::new(Tensor::from_vec(vec![1.0f32], vec![1]), true);
        let y = &x * &x;
        y.backward();
        
        // Gradient should not be computed in no_grad context
        let grad_guard = x.grad();
        let grad = grad_guard.read().unwrap();
        grad.is_none()
    });
    
    assert!(result);
    assert!(rustorch::autograd::is_grad_enabled()); // Restored
    
    // Test anomaly detection
    let anomaly_result = detect_anomaly(|| {
        assert!(rustorch::autograd::is_anomaly_detection_enabled());
        "anomaly detection active"
    });
    
    assert_eq!(anomaly_result, "anomaly detection active");
    assert!(!rustorch::autograd::is_anomaly_detection_enabled()); // Restored
}

#[test]
fn test_performance_benchmark() {
    // Basic performance benchmark for gradient computation
    use std::time::Instant;
    
    let input = Variable::new(Tensor::from_vec(vec![1.0f32; 100], vec![100]), true);
    
    let start = Instant::now();
    for _ in 0..10 {
        let sum_var = input.sum();
        sum_var.backward();
        input.zero_grad();
    }
    let duration = start.elapsed();
    
    // Should complete reasonably quickly (under 1 second for this simple case)
    assert!(duration.as_millis() < 1000, "Performance too slow: {:?}", duration);
}

#[test]
fn test_error_handling_integration() {
    // Test error cases are properly handled
    let x = Variable::new(Tensor::from_vec(vec![1.0f32, 2.0f32], vec![2]), true);
    
    // Test with non-scalar output (should fail)
    let result = gradcheck(
        |inputs| {
            let input = &inputs[0];
            input.clone() // Non-scalar output
        },
        &[x],
        None,
    );
    
    assert!(result.is_err());
    match result.unwrap_err() {
        RusTorchError::InvalidParameters { operation, message } => {
            assert_eq!(operation, "gradcheck");
            assert!(message.contains("scalar"));
        }
        _ => panic!("Expected InvalidParameters error"),
    }
}

#[test]
fn test_complex_computation_graph() {
    // Test complex computation graph with multiple operations
    let x = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true);
    let y = Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true);
    
    // Function: f(x, y) = x^2 + y^2
    let x_squared = &x * &x;
    let y_squared = &y * &y;
    let result = &x_squared + &y_squared;
    
    let gradients = grad(&[result], &[x.clone(), y.clone()], None, false, false).unwrap();
    
    // Verify gradients are computed
    assert!(gradients[0].is_some());
    assert!(gradients[1].is_some());
    
    let grad_x = gradients[0].as_ref().unwrap().as_array()[0];
    let grad_y = gradients[1].as_ref().unwrap().as_array()[0];
    
    assert!((grad_x - 4.0).abs() < 1e-6); // df/dx = 2x = 4
    assert!((grad_y - 6.0).abs() < 1e-6); // df/dy = 2y = 6
}