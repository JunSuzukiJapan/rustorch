//! Advanced mathematical functions for tensors
//! テンソル用高度数学関数

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;
use std::f64::consts::PI;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    // Hyperbolic functions
    // 双曲線関数

    /// Apply hyperbolic sine element-wise
    /// 要素ごとに双曲線サインを適用
    pub fn sinh(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.sinh()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply hyperbolic cosine element-wise
    /// 要素ごとに双曲線コサインを適用
    pub fn cosh(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.cosh()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply hyperbolic tangent element-wise
    /// 要素ごとに双曲線タンジェントを適用
    pub fn tanh(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.tanh()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply inverse hyperbolic sine element-wise
    /// 要素ごとに逆双曲線サインを適用
    pub fn asinh(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.asinh()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply inverse hyperbolic cosine element-wise
    /// 要素ごとに逆双曲線コサインを適用
    pub fn acosh(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.acosh()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply inverse hyperbolic tangent element-wise
    /// 要素ごとに逆双曲線タンジェントを適用
    pub fn atanh(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.atanh()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    // Inverse trigonometric functions
    // 逆三角関数

    /// Apply arcsine element-wise
    /// 要素ごとにアークサインを適用
    pub fn asin(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.asin()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply arccosine element-wise
    /// 要素ごとにアークコサインを適用
    pub fn acos(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.acos()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply arctangent element-wise
    /// 要素ごとにアークタンジェントを適用
    pub fn atan(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.atan()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply two-argument arctangent element-wise
    /// 要素ごとに二引数アークタンジェントを適用
    pub fn atan2(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
        if self.shape() != other.shape() && !self.can_broadcast_with(other) {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&y, &x)| y.atan2(x))
                .collect();
            Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
        } else {
            let (broadcasted_self, broadcasted_other) = self.broadcast_with(other)?;
            let result_data: Vec<T> = broadcasted_self
                .data
                .iter()
                .zip(broadcasted_other.data.iter())
                .map(|(&y, &x)| y.atan2(x))
                .collect();
            Ok(Tensor::from_vec(
                result_data,
                broadcasted_self.shape().to_vec(),
            ))
        }
    }

    // Rounding functions
    // 丸め関数

    /// Apply floor function element-wise
    /// 要素ごとにフロア関数を適用
    pub fn floor(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.floor()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply ceiling function element-wise
    /// 要素ごとにシーリング関数を適用
    pub fn ceil(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.ceil()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply round function element-wise
    /// 要素ごとにラウンド関数を適用
    pub fn round(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.round()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply truncation function element-wise
    /// 要素ごとに切り捨て関数を適用
    pub fn trunc(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.trunc()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    // Power and logarithmic functions
    // べき乗と対数関数

    // Note: pow method is defined in mathematical.rs to avoid duplication
    // 注意: powメソッドは重複を避けるためmathematical.rsで定義されています

    /// Raise elements to powers from another tensor
    /// 他のテンソルの要素をべき指数として使用
    pub fn pow_tensor(&self, exponent: &Tensor<T>) -> RusTorchResult<Self> {
        if self.shape() != exponent.shape() && !self.can_broadcast_with(exponent) {
            return Err(RusTorchError::shape_mismatch(
                self.shape(),
                exponent.shape(),
            ));
        }

        if self.shape() == exponent.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(exponent.data.iter())
                .map(|(&base, &exp)| base.powf(exp))
                .collect();
            Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
        } else {
            let (broadcasted_self, broadcasted_exp) = self.broadcast_with(exponent)?;
            let result_data: Vec<T> = broadcasted_self
                .data
                .iter()
                .zip(broadcasted_exp.data.iter())
                .map(|(&base, &exp)| base.powf(exp))
                .collect();
            Ok(Tensor::from_vec(
                result_data,
                broadcasted_self.shape().to_vec(),
            ))
        }
    }

    // Note: sqrt method is defined in core.rs to avoid duplication
    // 注意: sqrtメソッドは重複を避けるためcore.rsで定義されています

    /// Apply square element-wise
    /// 要素ごとに平方を適用
    pub fn square(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x * x).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply logarithm base 10 element-wise
    /// 要素ごとに常用対数を適用
    pub fn log10(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.log10()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply logarithm base 2 element-wise
    /// 要素ごとに二進対数を適用
    pub fn log2(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.log2()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Apply logarithm with custom base element-wise
    /// 要素ごとにカスタム底の対数を適用
    pub fn log(&self, base: T) -> Self {
        let log_base = base.ln();
        let result_data: Vec<T> = self.data.iter().map(|&x| x.ln() / log_base).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    // Comparison and logical functions
    // 比較と論理関数

    // Note: maximum method is defined in arithmetic.rs to avoid duplication
    // 注意: maximumメソッドは重複を避けるためarithmetic.rsで定義されています

    // Note: minimum method is defined in arithmetic.rs to avoid duplication
    // 注意: minimumメソッドは重複を避けるためarithmetic.rsで定義されています

    /// Clamp values between min and max
    /// 値をminとmaxの間にクランプ
    pub fn clamp(&self, min_val: T, max_val: T) -> Self {
        let result_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| {
                if x < min_val {
                    min_val
                } else if x > max_val {
                    max_val
                } else {
                    x
                }
            })
            .collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Sign function: returns -1, 0, or 1
    /// 符号関数: -1, 0, 1を返す
    pub fn sign(&self) -> Self {
        let result_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| {
                if x > T::zero() {
                    T::one()
                } else if x < T::zero() {
                    -T::one()
                } else {
                    T::zero()
                }
            })
            .collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Linear interpolation between two tensors
    /// 2つのテンソル間の線形補間
    pub fn lerp(&self, other: &Tensor<T>, weight: T) -> RusTorchResult<Self> {
        if self.shape() != other.shape() && !self.can_broadcast_with(other) {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + weight * (b - a))
                .collect();
            Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
        } else {
            let (broadcasted_self, broadcasted_other) = self.broadcast_with(other)?;
            let result_data: Vec<T> = broadcasted_self
                .data
                .iter()
                .zip(broadcasted_other.data.iter())
                .map(|(&a, &b)| a + weight * (b - a))
                .collect();
            Ok(Tensor::from_vec(
                result_data,
                broadcasted_self.shape().to_vec(),
            ))
        }
    }
}

// Special functions implementation for f64/f32 specific functionality
// Note: Special functions (gamma, erf) are disabled for now due to libm dependency
// f64/f32固有機能の特殊関数実装
// 注意: 特殊関数（gamma, erf）はlibm依存関係のため現在無効化されています

/*
Special functions will be enabled when libm dependency is added:

impl Tensor<f64> {
    pub fn gamma(&self) -> Self {
        // Implementation with libm::tgamma
    }
    
    pub fn lgamma(&self) -> Self {
        // Implementation with libm::lgamma  
    }
    
    pub fn erf(&self) -> Self {
        // Implementation with libm::erf
    }
    
    pub fn erfc(&self) -> Self {
        // Implementation with libm::erfc
    }
}

impl Tensor<f32> {
    pub fn gamma(&self) -> Self {
        // Implementation with libm::tgammaf
    }
    
    pub fn lgamma(&self) -> Self {
        // Implementation with libm::lgammaf
    }
    
    pub fn erf(&self) -> Self {
        // Implementation with libm::erff
    }
    
    pub fn erfc(&self) -> Self {
        // Implementation with libm::erfcf
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_functions() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);

        // Test sinh, cosh, tanh
        let sinh_result = tensor.sinh();
        let cosh_result = tensor.cosh();
        let tanh_result = tensor.tanh();

        assert_eq!(sinh_result.as_slice().unwrap()[0], 0.0);
        assert_eq!(cosh_result.as_slice().unwrap()[0], 1.0);
        assert_eq!(tanh_result.as_slice().unwrap()[0], 0.0);

        // Test that sinh(-x) = -sinh(x)
        assert!((sinh_result.as_slice().unwrap()[1] + sinh_result.as_slice().unwrap()[2]).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_trigonometric_functions() {
        let tensor = Tensor::from_vec(vec![0.0, 0.5, -0.5], vec![3]);

        let asin_result = tensor.asin();
        let acos_result = tensor.acos();
        let atan_result = tensor.atan();

        assert_eq!(asin_result.as_slice().unwrap()[0], 0.0);
        assert!((acos_result.as_slice().unwrap()[0] - std::f64::consts::PI / 2.0).abs() < 1e-10);
        assert_eq!(atan_result.as_slice().unwrap()[0], 0.0);
    }

    #[test]
    fn test_rounding_functions() {
        let tensor = Tensor::from_vec(vec![1.2, 2.7, -1.3, -2.8], vec![4]);

        let floor_result = tensor.floor();
        let ceil_result = tensor.ceil();
        let round_result = tensor.round();
        let trunc_result = tensor.trunc();

        assert_eq!(floor_result.as_slice().unwrap(), &[1.0, 2.0, -2.0, -3.0]);
        assert_eq!(ceil_result.as_slice().unwrap(), &[2.0, 3.0, -1.0, -2.0]);
        assert_eq!(round_result.as_slice().unwrap(), &[1.0, 3.0, -1.0, -3.0]);
        assert_eq!(trunc_result.as_slice().unwrap(), &[1.0, 2.0, -1.0, -2.0]);
    }

    #[test]
    fn test_power_functions() {
        let tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![4]);

        let sqrt_result = tensor.sqrt();
        let square_result = tensor.square();
        let pow_result = tensor.pow(0.5);

        assert_eq!(sqrt_result.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(square_result.as_slice().unwrap(), &[1.0, 16.0, 81.0, 256.0]);
        assert_eq!(pow_result.as_slice().unwrap(), sqrt_result.as_slice().unwrap());
    }

    #[test]
    fn test_comparison_functions() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![2.0, 1.0, 3.0], vec![3]);

        let max_result = a.maximum(&b).unwrap();
        let min_result = a.minimum(&b).unwrap();

        assert_eq!(max_result.as_slice().unwrap(), &[2.0, 2.0, 3.0]);
        assert_eq!(min_result.as_slice().unwrap(), &[1.0, 1.0, 3.0]);
    }

    #[test]
    fn test_clamp_and_sign() {
        let tensor = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5]);

        let clamped = tensor.clamp(-1.0, 1.0);
        let sign_result = tensor.sign();

        assert_eq!(clamped.as_slice().unwrap(), &[-1.0, -0.5, 0.0, 0.5, 1.0]);
        assert_eq!(sign_result.as_slice().unwrap(), &[-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_atan2() {
        let y = Tensor::from_vec(vec![1.0, 1.0, -1.0, -1.0], vec![4]);
        let x = Tensor::from_vec(vec![1.0, -1.0, 1.0, -1.0], vec![4]);

        let atan2_result = y.atan2(&x).unwrap();
        let expected = vec![
            std::f64::consts::PI / 4.0,
            3.0 * std::f64::consts::PI / 4.0,
            -std::f64::consts::PI / 4.0,
            -3.0 * std::f64::consts::PI / 4.0,
        ];

        let result_slice = atan2_result.as_slice().unwrap();
        for (i, &expected_val) in expected.iter().enumerate() {
            assert!((result_slice[i] - expected_val).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lerp() {
        let a = Tensor::from_vec(vec![0.0, 2.0, 4.0], vec![3]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 40.0], vec![3]);

        let lerp_result = a.lerp(&b, 0.5).unwrap();
        assert_eq!(lerp_result.as_slice().unwrap(), &[5.0, 11.0, 22.0]);

        let lerp_result_0 = a.lerp(&b, 0.0).unwrap();
        assert_eq!(lerp_result_0.as_slice().unwrap(), a.as_slice().unwrap());

        let lerp_result_1 = a.lerp(&b, 1.0).unwrap();
        assert_eq!(lerp_result_1.as_slice().unwrap(), b.as_slice().unwrap());
    }

    // Special function tests are disabled for now
    // 特殊関数テストは現在無効化されています
    /*
    #[test]
    fn test_special_functions_f64() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        let gamma_result = tensor.gamma();
        let erf_result = tensor.erf();

        // Gamma(1) = 1, Gamma(2) = 1, Gamma(3) = 2
        assert!((gamma_result.as_slice().unwrap()[0] - 1.0).abs() < 1e-10);
        assert!((gamma_result.as_slice().unwrap()[1] - 1.0).abs() < 1e-10);
        assert!((gamma_result.as_slice().unwrap()[2] - 2.0).abs() < 1e-10);

        // erf(0) = 0, and erf is monotonic
        assert!(erf_result.as_slice().unwrap()[0] > 0.8); // erf(1) ≈ 0.84
    }
    */
}