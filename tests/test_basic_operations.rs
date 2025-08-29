//! Tests for basic tensor operations
//! 基本的なテンソル演算のテスト

use rustorch::tensor::Tensor;

#[test]
fn test_arithmetic_operators() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

    // Test addition operator
    let c = &a + &b;
    assert_eq!(c.as_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);

    // Test subtraction operator
    let d = &a - &b;
    assert_eq!(d.as_slice().unwrap(), &[-4.0, -4.0, -4.0, -4.0]);

    // Test multiplication operator
    let e = &a * &b;
    assert_eq!(e.as_slice().unwrap(), &[5.0, 12.0, 21.0, 32.0]);

    // Test division operator
    let f = &b / &a;
    assert_eq!(f.as_slice().unwrap(), &[5.0, 3.0, 7.0 / 3.0, 2.0]);
}

#[test]
fn test_scalar_operators() {
    let a = Tensor::from_vec(vec![2.0f32, 4.0, 6.0], vec![3]);

    // Test scalar multiplication
    let b = &a * 2.0;
    assert_eq!(b.as_slice().unwrap(), &[4.0, 8.0, 12.0]);

    // Test scalar division
    let c = &a / 2.0;
    assert_eq!(c.as_slice().unwrap(), &[1.0, 2.0, 3.0]);

    // Test scalar addition
    let d = &a + 10.0;
    assert_eq!(d.as_slice().unwrap(), &[12.0, 14.0, 16.0]);

    // Test scalar subtraction
    let e = &a - 1.0;
    assert_eq!(e.as_slice().unwrap(), &[1.0, 3.0, 5.0]);
}

#[test]
fn test_matrix_multiplication() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

    // Test matmul
    let c = a.matmul(&b).unwrap();
    // [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3*5+4*7, 3*6+4*8] = [43, 50]
    assert_eq!(c.as_slice().unwrap(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_transpose() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = a.transpose().unwrap();

    assert_eq!(b.shape(), &[3, 2]);
    assert_eq!(b.as_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_sum_operations() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    // Test total sum
    let total = a.sum();
    assert_eq!(total, 21.0);

    // Test sum along axis 0 (sum rows)
    let sum_axis0 = a.sum_axis(0).unwrap();
    assert_eq!(sum_axis0.shape(), &[3]);
    assert_eq!(sum_axis0.as_slice().unwrap(), &[5.0, 7.0, 9.0]);

    // Test sum along axis 1 (sum columns)
    let sum_axis1 = a.sum_axis(1).unwrap();
    assert_eq!(sum_axis1.shape(), &[2]);
    assert_eq!(sum_axis1.as_slice().unwrap(), &[6.0, 15.0]);
}

#[test]
fn test_mathematical_functions() {
    use std::f32::consts::E;

    let a = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], vec![3]);

    // Test exp
    let exp_result = a.exp();
    let expected_exp = [1.0, E, E * E];
    for (actual, expected) in exp_result
        .as_slice()
        .unwrap()
        .iter()
        .zip(expected_exp.iter())
    {
        assert!((actual - expected).abs() < 1e-5);
    }

    // Test ln
    let b = Tensor::from_vec(vec![1.0f32, E, E * E], vec![3]);
    let ln_result = b.ln();
    let expected_ln = [0.0, 1.0, 2.0];
    for (actual, expected) in ln_result.as_slice().unwrap().iter().zip(expected_ln.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }

    // Test sqrt
    let c = Tensor::from_vec(vec![4.0f32, 9.0, 16.0], vec![3]);
    let sqrt_result = c.sqrt();
    assert_eq!(sqrt_result.as_slice().unwrap(), &[2.0, 3.0, 4.0]);

    // Test abs
    let d = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0], vec![3]);
    let abs_result = d.abs();
    assert_eq!(abs_result.as_slice().unwrap(), &[1.0, 0.0, 1.0]);
}
