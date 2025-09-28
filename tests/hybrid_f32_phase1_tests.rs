//! hybrid_f32 フェーズ1基本演算テスト
//! hybrid_f32 Phase 1 Basic Operations Tests

#[cfg(feature = "hybrid-f32")]
mod tests {
    use rustorch::hybrid_f32::tensor::F32Tensor;

    #[test]
    fn test_tensor_creation_methods() {
        rustorch::hybrid_f32_experimental!();

        // zeros
        let zeros = F32Tensor::zeros(&[2, 3]);
        assert_eq!(zeros.shape(), &[2, 3]);
        assert_eq!(zeros.len(), 6);
        assert!(zeros.as_slice().iter().all(|&x| x == 0.0));

        // ones
        let ones = F32Tensor::ones(&[2, 2]);
        assert_eq!(ones.shape(), &[2, 2]);
        assert!(ones.as_slice().iter().all(|&x| x == 1.0));

        // from_vec
        let from_vec = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(from_vec.shape(), &[2, 2]);
        assert_eq!(from_vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        // arange
        let arange = F32Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(arange.shape(), &[5]);
        assert_eq!(arange.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0]);

        // linspace
        let linspace = F32Tensor::linspace(0.0, 10.0, 5);
        assert_eq!(linspace.shape(), &[5]);
        assert_eq!(linspace.as_slice(), &[0.0, 2.5, 5.0, 7.5, 10.0]);

        // eye
        let eye = F32Tensor::eye(3);
        assert_eq!(eye.shape(), &[3, 3]);
        let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(eye.as_slice(), &expected);
    }

    #[test]
    fn test_random_tensor_creation() {
        rustorch::hybrid_f32_experimental!();

        // rand
        let rand_tensor = F32Tensor::rand(&[10, 10]);
        assert_eq!(rand_tensor.shape(), &[10, 10]);
        assert!(rand_tensor.as_slice().iter().all(|&x| x >= 0.0 && x < 1.0));

        // randn
        let randn_tensor = F32Tensor::randn(&[5, 5]);
        assert_eq!(randn_tensor.shape(), &[5, 5]);
        // randn は -1.0 から 1.0 の範囲
        assert!(randn_tensor.as_slice().iter().all(|&x| x >= -1.0 && x <= 1.0));

        // uniform
        let uniform_tensor = F32Tensor::uniform(&[3, 3], 5.0, 15.0);
        assert_eq!(uniform_tensor.shape(), &[3, 3]);
        assert!(uniform_tensor.as_slice().iter().all(|&x| x >= 5.0 && x < 15.0));
    }

    #[test]
    fn test_basic_arithmetic() {
        rustorch::hybrid_f32_experimental!();

        let a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = F32Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();

        // add
        let add_result = a.add(&b).unwrap();
        assert_eq!(add_result.as_slice(), &[3.0, 5.0, 7.0, 9.0]);

        // sub
        let sub_result = a.sub(&b).unwrap();
        assert_eq!(sub_result.as_slice(), &[-1.0, -1.0, -1.0, -1.0]);

        // mul
        let mul_result = a.mul(&b).unwrap();
        assert_eq!(mul_result.as_slice(), &[2.0, 6.0, 12.0, 20.0]);

        // div
        let div_result = b.div(&a).unwrap();
        assert_eq!(div_result.as_slice(), &[2.0, 1.5, 4.0/3.0, 1.25]);
    }

    #[test]
    fn test_scalar_operations() {
        rustorch::hybrid_f32_experimental!();

        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // add_scalar
        let add_result = tensor.add_scalar(10.0).unwrap();
        assert_eq!(add_result.as_slice(), &[11.0, 12.0, 13.0, 14.0]);

        // sub_scalar
        let sub_result = tensor.sub_scalar(0.5).unwrap();
        assert_eq!(sub_result.as_slice(), &[0.5, 1.5, 2.5, 3.5]);

        // mul_scalar
        let mul_result = tensor.mul_scalar(2.0).unwrap();
        assert_eq!(mul_result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);

        // div_scalar
        let div_result = tensor.div_scalar(2.0).unwrap();
        assert_eq!(div_result.as_slice(), &[0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_unary_operations() {
        rustorch::hybrid_f32_experimental!();

        let tensor = F32Tensor::from_vec(vec![-2.0, -1.0, 4.0, 9.0], vec![2, 2]).unwrap();

        // neg
        let neg_result = tensor.neg().unwrap();
        assert_eq!(neg_result.as_slice(), &[2.0, 1.0, -4.0, -9.0]);

        // abs
        let abs_result = tensor.abs().unwrap();
        assert_eq!(abs_result.as_slice(), &[2.0, 1.0, 4.0, 9.0]);

        // sqrt (正の値のみでテスト)
        let positive_tensor = F32Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2]).unwrap();
        let sqrt_result = positive_tensor.sqrt().unwrap();
        assert_eq!(sqrt_result.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        // pow
        let pow_result = positive_tensor.pow(2.0).unwrap();
        assert_eq!(pow_result.as_slice(), &[1.0, 16.0, 81.0, 256.0]);
    }

    #[test]
    fn test_statistics() {
        rustorch::hybrid_f32_experimental!();

        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // sum
        let sum = tensor.sum().unwrap();
        assert_eq!(sum, 21.0);

        // mean
        let mean = tensor.mean().unwrap();
        assert_eq!(mean, 3.5);

        // max
        let max = tensor.max().unwrap();
        assert_eq!(max, 6.0);

        // min
        let min = tensor.min().unwrap();
        assert_eq!(min, 1.0);

        // std and var
        let std = tensor.std().unwrap();
        let var = tensor.var().unwrap();
        assert!((std - (var.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn test_axis_operations() {
        rustorch::hybrid_f32_experimental!();

        let tensor = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // sum_axis(0) - 行方向に合計
        let sum_axis0 = tensor.sum_axis(0).unwrap();
        assert_eq!(sum_axis0.shape(), &[3]);
        assert_eq!(sum_axis0.as_slice(), &[5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]

        // sum_axis(1) - 列方向に合計
        let sum_axis1 = tensor.sum_axis(1).unwrap();
        assert_eq!(sum_axis1.shape(), &[2]);
        assert_eq!(sum_axis1.as_slice(), &[6.0, 15.0]); // [1+2+3, 4+5+6]

        // mean_axis
        let mean_axis0 = tensor.mean_axis(0).unwrap();
        assert_eq!(mean_axis0.as_slice(), &[2.5, 3.5, 4.5]); // sum_axis0 / 2

        let mean_axis1 = tensor.mean_axis(1).unwrap();
        assert_eq!(mean_axis1.as_slice(), &[2.0, 5.0]); // sum_axis1 / 3
    }

    #[test]
    fn test_matrix_operations() {
        rustorch::hybrid_f32_experimental!();

        let a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = F32Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        // matmul
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(result.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_error_handling() {
        rustorch::hybrid_f32_experimental!();

        // Shape mismatch for element-wise operations
        let a = F32Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = F32Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        assert!(a.add(&b).is_err());
        assert!(a.sub(&b).is_err());
        assert!(a.mul(&b).is_err());
        assert!(a.div(&b).is_err());

        // Empty tensor statistics
        let empty = F32Tensor::from_vec(vec![], vec![0]).unwrap();
        assert!(empty.mean().is_err());
        assert!(empty.max().is_err());
        assert!(empty.min().is_err());
        assert!(empty.std().is_err());
        assert!(empty.var().is_err());

        // Out of bounds axis
        let tensor = F32Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(tensor.sum_axis(1).is_err());
        assert!(tensor.mean_axis(1).is_err());
    }
}