//! Mathematical functions for tensors
//! テンソルの数学関数
//!
//! This module provides element-wise mathematical functions including
//! exponential, logarithmic, trigonometric, and power functions.
//! このモジュールは指数、対数、三角関数、べき乗関数を含む要素ごとの数学関数を提供します。

use crate::tensor::Tensor;
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Element-wise exponential function
    /// 要素ごとの指数関数
    pub fn exp(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.exp());
        Tensor::new(result_data)
    }

    /// Element-wise natural logarithm
    /// 要素ごとの自然対数
    pub fn ln(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.ln());
        Tensor::new(result_data)
    }

    /// Element-wise base-10 logarithm
    /// 要素ごとの常用対数
    pub fn log10(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.log10());
        Tensor::new(result_data)
    }

    /// Element-wise logarithm with specified base
    /// 指定底での要素ごとの対数
    pub fn log(&self, base: T) -> Self {
        let result_data = self.as_array().mapv(|x| x.log(base));
        Tensor::new(result_data)
    }

    /// Element-wise sine function
    /// 要素ごとの正弦関数
    pub fn sin(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.sin());
        Tensor::new(result_data)
    }

    /// Element-wise cosine function
    /// 要素ごとの余弦関数
    pub fn cos(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.cos());
        Tensor::new(result_data)
    }

    /// Element-wise tangent function
    /// 要素ごとの正接関数
    pub fn tan(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.tan());
        Tensor::new(result_data)
    }

    /// Element-wise arcsine function
    /// 要素ごとの逆正弦関数
    pub fn asin(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.asin());
        Tensor::new(result_data)
    }

    /// Element-wise arccosine function
    /// 要素ごとの逆余弦関数
    pub fn acos(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.acos());
        Tensor::new(result_data)
    }

    /// Element-wise arctangent function
    /// 要素ごとの逆正接関数
    pub fn atan(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.atan());
        Tensor::new(result_data)
    }

    /// Element-wise hyperbolic sine function
    /// 要素ごとの双曲線正弦関数
    pub fn sinh(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.sinh());
        Tensor::new(result_data)
    }

    /// Element-wise hyperbolic cosine function
    /// 要素ごとの双曲線余弦関数
    pub fn cosh(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.cosh());
        Tensor::new(result_data)
    }

    /// Element-wise hyperbolic tangent function
    /// 要素ごとの双曲線正接関数
    pub fn tanh(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.tanh());
        Tensor::new(result_data)
    }

    /// Element-wise square root function
    /// 要素ごとの平方根関数
    pub fn sqrt(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.sqrt());
        Tensor::new(result_data)
    }

    /// Element-wise square root function (v2)
    /// 要素ごとの平方根関数（v2版）
    pub fn sqrt_v2(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.sqrt()).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Element-wise cube root function
    /// 要素ごとの立方根関数
    pub fn cbrt(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.cbrt());
        Tensor::new(result_data)
    }

    /// Element-wise absolute value
    /// 要素ごとの絶対値
    pub fn abs(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.abs());
        Tensor::new(result_data)
    }

    /// Element-wise sign function (-1, 0, or 1)
    /// 要素ごとの符号関数（-1、0、または1）
    pub fn sign(&self) -> Self {
        let result_data = self.as_array().mapv(|x| {
            if x > T::zero() {
                T::one()
            } else if x < T::zero() {
                -T::one()
            } else {
                T::zero()
            }
        });
        Tensor::new(result_data)
    }

    /// Element-wise power function
    /// 要素ごとのべき乗関数
    pub fn pow(&self, exponent: T) -> Self {
        let result_data = self.as_array().mapv(|x| x.powf(exponent));
        Tensor::new(result_data)
    }

    /// Element-wise power function with tensor exponents
    /// テンソル指数による要素ごとのべき乗関数
    pub fn pow_tensor(&self, exponent: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != exponent.shape() {
            return Err("Shape mismatch for power operation".to_string());
        }
        
        let self_data = self.as_slice().unwrap();
        let exp_data = exponent.as_slice().unwrap();
        
        let result_data: Vec<T> = self_data.iter()
            .zip(exp_data.iter())
            .map(|(&base, &exp)| base.powf(exp))
            .collect();
        
        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }

    /// Element-wise square function (x²)
    /// 要素ごとの2乗関数（x²）
    pub fn square(&self) -> Self {
        self.pow(T::from(2.0).unwrap())
    }

    /// Element-wise reciprocal function (1/x)
    /// 要素ごとの逆数関数（1/x）
    pub fn reciprocal(&self) -> Self {
        let result_data = self.as_array().mapv(|x| T::one() / x);
        Tensor::new(result_data)
    }

    /// Element-wise floor function
    /// 要素ごとの床関数
    pub fn floor(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.floor());
        Tensor::new(result_data)
    }

    /// Element-wise ceiling function
    /// 要素ごとの天井関数
    pub fn ceil(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.ceil());
        Tensor::new(result_data)
    }

    /// Element-wise round function
    /// 要素ごとの丸め関数
    pub fn round(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.round());
        Tensor::new(result_data)
    }

    /// Element-wise truncation function (towards zero)
    /// 要素ごとの切り捨て関数（ゼロ方向）
    pub fn trunc(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.trunc());
        Tensor::new(result_data)
    }

    /// Element-wise fractional part function
    /// 要素ごとの小数部分関数
    pub fn fract(&self) -> Self {
        let result_data = self.as_array().mapv(|x| x.fract());
        Tensor::new(result_data)
    }

    /// Element-wise clamp function (clip values to range [min, max])
    /// 要素ごとのクランプ関数（値を[min, max]範囲にクリップ）
    pub fn clamp(&self, min: T, max: T) -> Self {
        let result_data = self.as_array().mapv(|x| {
            if x < min { min } else if x > max { max } else { x }
        });
        Tensor::new(result_data)
    }

    /// Element-wise minimum with scalar
    /// スカラーとの要素ごと最小値
    pub fn min_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array().mapv(|x| x.min(scalar));
        Tensor::new(result_data)
    }

    /// Element-wise maximum with scalar
    /// スカラーとの要素ごと最大値
    pub fn max_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array().mapv(|x| x.max(scalar));
        Tensor::new(result_data)
    }

    /// Check if all elements are finite (not NaN or infinite)
    /// 全要素が有限かチェック（NaNや無限大でない）
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|&x| x.is_finite())
    }

    /// Check if any element is NaN
    /// いずれかの要素がNaNかチェック
    pub fn has_nan(&self) -> bool {
        self.data.iter().any(|&x| x.is_nan())
    }

    /// Check if any element is infinite
    /// いずれかの要素が無限大かチェック
    pub fn has_inf(&self) -> bool {
        self.data.iter().any(|&x| x.is_infinite())
    }

    /// Replace NaN values with specified value
    /// NaN値を指定値に置換
    pub fn nan_to_num(&self, nan_replacement: T) -> Self {
        let result_data = self.as_array().mapv(|x| {
            if x.is_nan() { nan_replacement } else { x }
        });
        Tensor::new(result_data)
    }

    /// Replace infinite values with specified values
    /// 無限大値を指定値に置換
    pub fn inf_to_num(&self, pos_inf: T, neg_inf: T) -> Self {
        let result_data = self.as_array().mapv(|x| {
            if x.is_infinite() {
                if x > T::zero() { pos_inf } else { neg_inf }
            } else {
                x
            }
        });
        Tensor::new(result_data)
    }

    /// Normalize tensor values to [0, 1] range
    /// テンソル値を[0, 1]範囲に正規化
    pub fn normalize_0_1(&self) -> Result<Self, String> {
        let min_val = self.min().ok_or("Cannot normalize empty tensor")?;
        let max_val = self.max().ok_or("Cannot normalize empty tensor")?;
        
        if min_val == max_val {
            return Ok(Tensor::zeros(self.shape()));
        }
        
        let range = max_val - min_val;
        let result_data = self.as_array().mapv(|x| (x - min_val) / range);
        Ok(Tensor::new(result_data))
    }

    /// Standardize tensor values (zero mean, unit variance)
    /// テンソル値を標準化（平均ゼロ、単位分散）
    pub fn standardize(&self) -> Self {
        let mean = self.mean();
        let std = self.std();
        
        if std == T::zero() {
            // If standard deviation is zero, return zero tensor
            return Tensor::zeros(self.shape());
        }
        
        let result_data = self.as_array().mapv(|x| (x - mean) / std);
        Tensor::new(result_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_ln() {
        let tensor = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], vec![3]);
        
        let exp_result = tensor.exp();
        let expected_exp = vec![1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2)];
        
        let exp_data = exp_result.as_slice().unwrap();
        for (i, &expected) in expected_exp.iter().enumerate() {
            assert!((exp_data[i] - expected).abs() < 1e-6);
        }
        
        // Test ln as inverse of exp
        let ln_result = exp_result.ln();
        let ln_data = ln_result.as_slice().unwrap();
        let original = tensor.as_slice().unwrap();
        
        for (i, &orig) in original.iter().enumerate() {
            assert!((ln_data[i] - orig).abs() < 1e-6);
        }
    }

    #[test]
    fn test_trigonometric_functions() {
        let tensor = Tensor::from_vec(vec![0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI], vec![3]);
        
        let sin_result = tensor.sin();
        let sin_data = sin_result.as_slice().unwrap();
        
        assert!((sin_data[0] - 0.0).abs() < 1e-6);
        assert!((sin_data[1] - 1.0).abs() < 1e-6);
        assert!(sin_data[2].abs() < 1e-6);
        
        let cos_result = tensor.cos();
        let cos_data = cos_result.as_slice().unwrap();
        
        assert!((cos_data[0] - 1.0).abs() < 1e-6);
        assert!(cos_data[1].abs() < 1e-6);
        assert!((cos_data[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_power_functions() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        
        let square_result = tensor.square();
        assert_eq!(square_result.as_slice().unwrap(), &[1.0f32, 4.0, 9.0, 16.0]);
        
        let pow_result = tensor.pow(3.0);
        assert_eq!(pow_result.as_slice().unwrap(), &[1.0f32, 8.0, 27.0, 64.0]);
        
        let sqrt_result = square_result.sqrt();
        let sqrt_data = sqrt_result.as_slice().unwrap();
        let original = tensor.as_slice().unwrap();
        
        for (i, &orig) in original.iter().enumerate() {
            assert!((sqrt_data[i] - orig).abs() < 1e-6);
        }
    }

    #[test]
    fn test_abs_sign() {
        let tensor = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5]);
        
        let abs_result = tensor.abs();
        assert_eq!(abs_result.as_slice().unwrap(), &[2.0f32, 1.0, 0.0, 1.0, 2.0]);
        
        let sign_result = tensor.sign();
        assert_eq!(sign_result.as_slice().unwrap(), &[-1.0f32, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_clamp() {
        let tensor = Tensor::from_vec(vec![-5.0f32, -1.0, 0.0, 3.0, 10.0], vec![5]);
        
        let clamped = tensor.clamp(-2.0, 5.0);
        assert_eq!(clamped.as_slice().unwrap(), &[-2.0f32, -1.0, 0.0, 3.0, 5.0]);
    }

    #[test]
    fn test_rounding_functions() {
        let tensor = Tensor::from_vec(vec![-2.7f32, -1.3, 0.0, 1.7, 2.3], vec![5]);
        
        let floor_result = tensor.floor();
        assert_eq!(floor_result.as_slice().unwrap(), &[-3.0f32, -2.0, 0.0, 1.0, 2.0]);
        
        let ceil_result = tensor.ceil();
        assert_eq!(ceil_result.as_slice().unwrap(), &[-2.0f32, -1.0, 0.0, 2.0, 3.0]);
        
        let round_result = tensor.round();
        assert_eq!(round_result.as_slice().unwrap(), &[-3.0f32, -1.0, 0.0, 2.0, 2.0]);
    }

    #[test]
    fn test_pow_tensor() {
        let base = Tensor::from_vec(vec![2.0f32, 3.0, 4.0], vec![3]);
        let exponent = Tensor::from_vec(vec![1.0f32, 2.0, 0.5], vec![3]);
        
        let result = base.pow_tensor(&exponent).unwrap();
        let expected = vec![2.0f32, 9.0, 2.0]; // 2^1, 3^2, 4^0.5
        let result_data = result.as_slice().unwrap();
        
        for (i, &expected_val) in expected.iter().enumerate() {
            assert!((result_data[i] - expected_val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        
        let normalized = tensor.normalize_0_1().unwrap();
        let normalized_data = normalized.as_slice().unwrap();
        
        assert_eq!(normalized_data[0], 0.0); // min should be 0
        assert_eq!(normalized_data[4], 1.0); // max should be 1
        assert_eq!(normalized_data[2], 0.5); // middle should be 0.5
    }

    #[test]
    fn test_standardize() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        
        let standardized = tensor.standardize();
        
        // After standardization, mean should be close to 0 and std close to 1
        let standardized_mean = standardized.mean();
        let standardized_std = standardized.std();
        
        assert!(standardized_mean.abs() < 1e-6);
        assert!((standardized_std - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nan_inf_handling() {
        let tensor = Tensor::from_vec(
            vec![1.0f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY], 
            vec![4]
        );
        
        assert!(!tensor.is_finite());
        assert!(tensor.has_nan());
        assert!(tensor.has_inf());
        
        let nan_replaced = tensor.nan_to_num(0.0);
        let nan_data = nan_replaced.as_slice().unwrap();
        assert_eq!(nan_data[1], 0.0);
        
        let inf_replaced = tensor.inf_to_num(1000.0, -1000.0);
        let inf_data = inf_replaced.as_slice().unwrap();
        assert_eq!(inf_data[2], 1000.0);
        assert_eq!(inf_data[3], -1000.0);
    }
}