//! Complex number support for tensors - Legacy compatibility layer
//! テンソルの複素数サポート - レガシー互換レイヤー
//!
//! This file provides backward compatibility by re-exporting everything
//! from the new modular complex number implementation.

// Import the new modular implementation
pub use super::complex_impl::*;

// Re-export tests module for backward compatibility
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use approx::assert_relative_eq;

    #[test]
    fn test_complex_creation() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.real(), 3.0);
        assert_eq!(z.imag(), 4.0);

        let real = Complex::from_real(5.0);
        assert_eq!(real, Complex::new(5.0, 0.0));

        let imag = Complex::from_imag(2.0);
        assert_eq!(imag, Complex::new(0.0, 2.0));
    }

    #[test]
    fn test_complex_arithmetic() {
        let z1 = Complex::new(3.0, 4.0);
        let z2 = Complex::new(1.0, 2.0);

        // Addition
        assert_eq!(z1 + z2, Complex::new(4.0, 6.0));

        // Subtraction
        assert_eq!(z1 - z2, Complex::new(2.0, 2.0));

        // Multiplication
        assert_eq!(z1 * z2, Complex::new(-5.0, 10.0));

        // Division
        let div = z1 / z2;
        assert_relative_eq!(div.real(), 2.2, epsilon = 1e-10);
        assert_relative_eq!(div.imag(), -0.4, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_properties() {
        let z = Complex::new(3.0, 4.0);

        // Magnitude
        assert_relative_eq!(Complex::abs(&z), 5.0, epsilon = 1e-10);

        // Magnitude squared
        assert_relative_eq!(z.abs_sq(), 25.0, epsilon = 1e-10);

        // Conjugate
        assert_eq!(z.conj(), Complex::new(3.0, -4.0));

        // Phase
        let expected_phase = 4.0_f64.atan2(3.0);
        assert_relative_eq!(z.arg(), expected_phase, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_functions() {
        let z = Complex::new(1.0, 1.0);

        // Exponential
        let exp_z = z.exp();
        let expected_real = 1.0_f64.exp() * 1.0_f64.cos();
        let expected_imag = 1.0_f64.exp() * 1.0_f64.sin();
        assert_relative_eq!(exp_z.real(), expected_real, epsilon = 1e-10);
        assert_relative_eq!(exp_z.imag(), expected_imag, epsilon = 1e-10);

        // Square root
        let sqrt_z = z.sqrt();
        assert_relative_eq!((sqrt_z * sqrt_z).real(), z.real(), epsilon = 1e-10);
        assert_relative_eq!((sqrt_z * sqrt_z).imag(), z.imag(), epsilon = 1e-10);
    }

    #[test]
    fn test_polar_conversion() {
        let z = Complex::new(3.0, 4.0);
        let (r, theta) = z.to_polar();
        let z_converted = Complex::from_polar(r, theta);

        assert_relative_eq!(z_converted.real(), z.real(), epsilon = 1e-10);
        assert_relative_eq!(z_converted.imag(), z.imag(), epsilon = 1e-10);
    }

    #[test]
    fn test_trigonometric_functions() {
        let z = Complex::new(0.5, 0.3);

        // Test sin^2 + cos^2 = 1
        let sin_z = z.sin();
        let cos_z = z.cos();
        let identity = sin_z * sin_z + cos_z * cos_z;

        assert_relative_eq!(identity.real(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity.imag(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_constants() {
        let zero = Complex::<f64>::zero_const();
        assert_eq!(zero.real(), 0.0);
        assert_eq!(zero.imag(), 0.0);

        let one = Complex::<f64>::one_const();
        assert_eq!(one.real(), 1.0);
        assert_eq!(one.imag(), 0.0);

        let i = Complex::<f64>::i();
        assert_eq!(i.real(), 0.0);
        assert_eq!(i.imag(), 1.0);
    }

    #[test]
    fn test_complex_tensor_creation() {
        let real = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let imag = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);

        let complex_tensor = Complex::from_tensors(&real, &imag).unwrap();
        assert_eq!(complex_tensor.shape(), &[3]);
        assert_eq!(complex_tensor.data[0].real(), 1.0);
        assert_eq!(complex_tensor.data[0].imag(), 4.0);
        assert_eq!(complex_tensor.data[1].real(), 2.0);
        assert_eq!(complex_tensor.data[1].imag(), 5.0);
        assert_eq!(complex_tensor.data[2].real(), 3.0);
        assert_eq!(complex_tensor.data[2].imag(), 6.0);
    }

    #[test]
    fn test_complex_tensor_extraction() {
        let complex_data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let complex_tensor = Tensor::from_vec(complex_data, vec![3]);

        let real_part = Complex::tensor_real_part(&complex_tensor);
        assert_eq!(real_part.data.as_slice().unwrap(), &[1.0, 3.0, 5.0]);

        let imag_part = Complex::tensor_imag_part(&complex_tensor);
        assert_eq!(imag_part.data.as_slice().unwrap(), &[2.0, 4.0, 6.0]);

        let abs_part = Complex::tensor_abs(&complex_tensor);
        assert_relative_eq!(abs_part.data[0], 5.0_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(abs_part.data[1], 25.0_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(abs_part.data[2], 61.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_complex_tensor_factory_functions() {
        let zeros = Tensor::<Complex<f64>>::complex_zeros(&[2, 3]);
        assert_eq!(zeros.shape(), &[2, 3]);
        for z in zeros.data.iter() {
            assert_eq!(z.real(), 0.0);
            assert_eq!(z.imag(), 0.0);
        }

        let ones = Tensor::<Complex<f64>>::complex_ones(&[2, 2]);
        assert_eq!(ones.shape(), &[2, 2]);
        for z in ones.data.iter() {
            assert_eq!(z.real(), 1.0);
            assert_eq!(z.imag(), 0.0);
        }

        let i_tensor = Tensor::<Complex<f64>>::complex_i(&[1, 4]);
        assert_eq!(i_tensor.shape(), &[1, 4]);
        for z in i_tensor.data.iter() {
            assert_eq!(z.real(), 0.0);
            assert_eq!(z.imag(), 1.0);
        }
    }

    #[test]
    fn test_complex_tensor_polar_conversion() {
        let magnitude = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let phase = Tensor::from_vec(vec![0.0, std::f64::consts::PI / 2.0], vec![2]);

        let complex_tensor = Tensor::from_polar(&magnitude, &phase).unwrap();
        assert_eq!(complex_tensor.shape(), &[2]);

        assert_relative_eq!(complex_tensor.data[0].real(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(complex_tensor.data[0].imag(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(complex_tensor.data[1].real(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(complex_tensor.data[1].imag(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_tensor_conjugate() {
        let complex_data = vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)];
        let complex_tensor = Tensor::from_vec(complex_data, vec![2]);

        let conj_tensor = Complex::tensor_conj(&complex_tensor);
        assert_eq!(conj_tensor.data[0].real(), 1.0);
        assert_eq!(conj_tensor.data[0].imag(), -2.0);
        assert_eq!(conj_tensor.data[1].real(), -3.0);
        assert_eq!(conj_tensor.data[1].imag(), -4.0);
    }

    #[test]
    fn test_complex_mathematical_functions() {
        let complex_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 1.0),
        ];
        let complex_tensor = Tensor::from_vec(complex_data, vec![3]);

        // Test exponential
        let exp_result = complex_tensor.exp();
        assert_relative_eq!(
            exp_result.data[0].real(),
            std::f64::consts::E,
            epsilon = 1e-10
        );
        assert_relative_eq!(exp_result.data[0].imag(), 0.0, epsilon = 1e-10);

        // Test natural logarithm
        let ln_result = complex_tensor.ln();
        assert_relative_eq!(ln_result.data[0].real(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ln_result.data[0].imag(), 0.0, epsilon = 1e-10);

        // Test square root
        let sqrt_result = complex_tensor.sqrt();
        let sqrt_1_1 = sqrt_result.data[2];
        assert_relative_eq!((sqrt_1_1 * sqrt_1_1).real(), 1.0, epsilon = 1e-10);
        assert_relative_eq!((sqrt_1_1 * sqrt_1_1).imag(), 1.0, epsilon = 1e-10);

        // Test trigonometric functions
        let sin_result = complex_tensor.sin();
        let cos_result = complex_tensor.cos();

        // Test sin^2 + cos^2 = 1 for complex numbers
        for i in 0..3 {
            let sin_val = sin_result.data[i];
            let cos_val = cos_result.data[i];
            let identity = sin_val * sin_val + cos_val * cos_val;
            assert_relative_eq!(identity.real(), 1.0, epsilon = 1e-10);
            assert_relative_eq!(identity.imag(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_complex_matrix_multiplication() {
        // Create 2x2 complex matrices
        let a_data = vec![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0), // First row
            Complex::new(0.0, 1.0),
            Complex::new(1.0, -1.0), // Second row
        ];
        let a = Tensor::from_vec(a_data, vec![2, 2]);

        let b_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0), // First row
            Complex::new(1.0, 1.0),
            Complex::new(1.0, 0.0), // Second row
        ];
        let b = Tensor::from_vec(b_data, vec![2, 2]);

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // Verify matrix multiplication result using proper ndarray indexing
        // [1+i, 2] * [1, i] = [1+i+2+2i, i-1+2] = [3+3i, 1+i]
        // [i, 1-i]   [1+i, 1]   [i+1-i-i^2, -1+i] = [1+i, -1+i]
        assert_relative_eq!(result.data[[0, 0]].real(), 3.0, epsilon = 1e-10); // [0,0]
        assert_relative_eq!(result.data[[0, 0]].imag(), 3.0, epsilon = 1e-10); // [0,0]
        assert_relative_eq!(result.data[[0, 1]].real(), 1.0, epsilon = 1e-10); // [0,1]
        assert_relative_eq!(result.data[[0, 1]].imag(), 1.0, epsilon = 1e-10); // [0,1]
    }

    #[test]
    fn test_complex_matrix_transpose() {
        let data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        let matrix = Tensor::from_vec(data, vec![2, 2]);

        let transposed = matrix.transpose().unwrap();
        assert_eq!(transposed.shape(), &[2, 2]);

        // Check transposition using proper ndarray indexing
        assert_eq!(transposed.data[[0, 0]], Complex::new(1.0, 2.0)); // [0,0] -> [0,0]
        assert_eq!(transposed.data[[0, 1]], Complex::new(5.0, 6.0)); // [1,0] -> [0,1]
        assert_eq!(transposed.data[[1, 0]], Complex::new(3.0, 4.0)); // [0,1] -> [1,0]
        assert_eq!(transposed.data[[1, 1]], Complex::new(7.0, 8.0)); // [1,1] -> [1,1]
    }

    #[test]
    fn test_complex_matrix_conjugate_transpose() {
        let data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        let matrix = Tensor::from_vec(data, vec![2, 2]);

        let conj_transposed = matrix.conj_transpose().unwrap();
        assert_eq!(conj_transposed.shape(), &[2, 2]);

        // Check conjugate transposition using proper ndarray indexing
        assert_eq!(conj_transposed.data[[0, 0]], Complex::new(1.0, -2.0)); // [0,0] -> conj([0,0])
        assert_eq!(conj_transposed.data[[0, 1]], Complex::new(5.0, -6.0)); // [1,0] -> conj([0,1])
        assert_eq!(conj_transposed.data[[1, 0]], Complex::new(3.0, -4.0)); // [0,1] -> conj([1,0])
        assert_eq!(conj_transposed.data[[1, 1]], Complex::new(7.0, -8.0)); // [1,1] -> conj([1,1])
    }

    #[test]
    fn test_complex_matrix_trace() {
        let data = vec![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 2.0),
        ];
        let matrix = Tensor::from_vec(data, vec![2, 2]);

        let trace = matrix.trace().unwrap();
        // Trace = (1+i) + (4+2i) = 5+3i
        assert_eq!(trace.real(), 5.0);
        assert_eq!(trace.imag(), 3.0);
    }

    #[test]
    fn test_complex_matrix_determinant() {
        let data = vec![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(1.0, -1.0),
        ];
        let matrix = Tensor::from_vec(data, vec![2, 2]);

        let det = matrix.determinant().unwrap();
        // det = (1+i)(1-i) - (2)(i) = (1-i^2) - 2i = 2 - 2i
        assert_eq!(det.real(), 2.0);
        assert_eq!(det.imag(), -2.0);
    }

    #[test]
    fn test_complex_fft_basic() {
        // Create a simple complex signal
        let signal_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let signal = Tensor::from_vec(signal_data, vec![4]);

        let fft_result = signal.fft(None, None, None);
        assert!(
            fft_result.is_ok(),
            "Complex FFT should work on basic signal"
        );

        let fft_tensor = fft_result.unwrap();
        assert_eq!(fft_tensor.shape(), &[4]);

        // Test FFT-IFFT round trip
        let ifft_result = fft_tensor.ifft(None, None, None).unwrap();

        for i in 0..4 {
            assert_relative_eq!(
                ifft_result.data[i].real(),
                signal.data[i].real(),
                epsilon = 1e-6
            );
            assert_relative_eq!(
                ifft_result.data[i].imag(),
                signal.data[i].imag(),
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn test_complex_fft_shift() {
        let data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let tensor = Tensor::from_vec(data, vec![4]);

        let shifted = tensor.fftshift(None).unwrap();

        // For [1, 2, 3, 4], fftshift should give [3, 4, 1, 2]
        assert_eq!(shifted.data[0].real(), 3.0);
        assert_eq!(shifted.data[1].real(), 4.0);
        assert_eq!(shifted.data[2].real(), 1.0);
        assert_eq!(shifted.data[3].real(), 2.0);

        // Test ifftshift
        let unshifted = shifted.ifftshift(None).unwrap();
        for i in 0..4 {
            assert_relative_eq!(
                unshifted.data[i].real(),
                tensor.data[i].real(),
                epsilon = 1e-10
            );
            assert_relative_eq!(
                unshifted.data[i].imag(),
                tensor.data[i].imag(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_complex_power_operations() {
        let base_data = vec![
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 1.0),
        ];
        let base = Tensor::from_vec(base_data, vec![3]);

        // Test scalar power
        let squared = base.pow_scalar(Complex::new(2.0, 0.0));
        assert_relative_eq!(squared.data[0].real(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(squared.data[0].imag(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(squared.data[1].real(), -1.0, epsilon = 1e-10); // i^2 = -1
        assert_relative_eq!(squared.data[1].imag(), 0.0, epsilon = 1e-10);

        // Test tensor power
        let exp_data = vec![
            Complex::new(0.5, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        let exponent = Tensor::from_vec(exp_data, vec![3]);

        let powered = base.pow(&exponent).unwrap();
        assert_relative_eq!(powered.data[0].real(), 2.0_f64.sqrt(), epsilon = 1e-10); // 2^0.5
        assert_relative_eq!(powered.data[0].imag(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(powered.data[1].real(), -1.0, epsilon = 1e-10); // i^2 = -1
        assert_relative_eq!(powered.data[1].imag(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(powered.data[2].real(), 1.0, epsilon = 1e-10); // (1+i)^1 = 1+i, real part = 1
        assert_relative_eq!(powered.data[2].imag(), 1.0, epsilon = 1e-10); // (1+i)^1 = 1+i, imag part = 1
    }
}
