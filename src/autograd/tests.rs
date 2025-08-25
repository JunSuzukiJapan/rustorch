//! Tests for automatic differentiation
//! 自動微分のテスト

#[cfg(test)]
mod tests {
    use crate::autograd::Variable;
    use crate::tensor::Tensor;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_variable_creation() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let var = Variable::new(data, true);

        assert!(var.requires_grad());
        assert_eq!(var.data().read().unwrap().shape(), &[3]);
    }

    #[test]
    fn test_zero_grad() {
        let data = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let var = Variable::new(data, true);

        // Manually set some gradient
        {
            let grad_binding = var.grad();
            let mut grad = grad_binding.write().unwrap();
            *grad = Some(Tensor::from_vec(vec![0.5, 1.0], vec![2]));
        }

        // Zero the gradient
        var.zero_grad();

        // Check that gradient is None
        let grad_binding = var.grad();
        let grad = grad_binding.read().unwrap();
        assert!(grad.is_none());
    }

    #[test]
    fn test_backward_scalar() {
        let data = Tensor::from_vec(vec![2.0], vec![]);
        let var = Variable::new(data, true);

        // Backward pass
        var.backward();

        // Check gradient
        let grad_binding = var.grad();
        let grad = grad_binding.read().unwrap();
        assert!(grad.is_some());
        if let Some(ref g) = *grad {
            if let Some(first_val) = g.as_array().iter().next() {
                assert_abs_diff_eq!(*first_val, 1.0f32, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_addition() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], vec![2]), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0, 4.0], vec![2]), true);

        let c = &a + &b;

        // Check result
        let data_binding = c.data();
        let result_data = data_binding.read().unwrap();
        let expected = vec![4.0, 6.0];
        for (actual, expected) in result_data.as_array().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }

        assert!(c.requires_grad());
    }

    #[test]
    fn test_multiplication() {
        let a = Variable::new(Tensor::from_vec(vec![2.0, 3.0], vec![2]), true);
        let b = Variable::new(Tensor::from_vec(vec![4.0, 5.0], vec![2]), true);

        let c = &a * &b;

        // Check result
        let data_binding = c.data();
        let result_data = data_binding.read().unwrap();
        let expected = vec![8.0, 15.0];
        for (actual, expected) in result_data.as_array().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }

        assert!(c.requires_grad());
    }

    #[test]
    fn test_sum() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]), true);

        let sum = a.sum();

        // Check result
        let data_binding = sum.data();
        let result_data = data_binding.read().unwrap();
        assert_abs_diff_eq!(
            result_data.as_array().iter().next().unwrap(),
            &6.0,
            epsilon = 1e-6
        );

        assert!(sum.requires_grad());
    }

    #[test]
    fn test_matmul() {
        let a = Variable::new(Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]), true);

        let c = a.matmul(&b);

        // Check result: [1, 2] @ [3; 4] = [11]
        let data_binding = c.data();
        let result_data = data_binding.read().unwrap();
        assert_abs_diff_eq!(
            result_data.as_array().iter().next().unwrap(),
            &11.0,
            epsilon = 1e-6
        );

        assert!(c.requires_grad());
    }
}
