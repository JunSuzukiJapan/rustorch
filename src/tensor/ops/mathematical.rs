//! Mathematical functions for tensors
//! テンソル用数学関数

use super::super::core::Tensor;
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Apply exponential function element-wise
    /// 要素ごとに指数関数を適用
    pub fn exp(&self) -> Self {
        self.map(|x| x.exp())
    }

    /// Apply natural logarithm element-wise
    /// 要素ごとに自然対数を適用
    pub fn ln(&self) -> Self {
        self.map(|x| x.ln())
    }

    /// Apply sine function element-wise
    /// 要素ごとにサイン関数を適用
    pub fn sin(&self) -> Self {
        self.map(|x| x.sin())
    }

    /// Apply cosine function element-wise
    /// 要素ごとにコサイン関数を適用
    pub fn cos(&self) -> Self {
        self.map(|x| x.cos())
    }

    /// Apply tangent function element-wise
    /// 要素ごとにタンジェント関数を適用
    pub fn tan(&self) -> Self {
        self.map(|x| x.tan())
    }

    /// Apply square root element-wise
    /// 要素ごとに平方根を適用
    pub fn sqrt(&self) -> Self {
        self.map(|x| x.sqrt())
    }

    /// Apply absolute value element-wise
    /// 要素ごとに絶対値を適用
    pub fn abs(&self) -> Self {
        self.map(|x| x.abs())
    }

    /// Power operation element-wise
    /// 要素ごとのべき乗演算
    pub fn pow(&self, exponent: T) -> Self {
        self.map(|x| x.powf(exponent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mathematical_functions() {
        let tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2]);

        let sqrt_result = tensor.sqrt();
        let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(sqrt_result.as_slice().unwrap(), expected);

        let exp_result = tensor.exp();
        assert_eq!(exp_result.shape(), tensor.shape());

        let ln_result = tensor.ln();
        assert_eq!(ln_result.shape(), tensor.shape());
    }

    #[test]
    fn test_trigonometric_functions() {
        let tensor = Tensor::from_vec(vec![0.0, std::f32::consts::PI / 2.0], vec![2]);

        let sin_result = tensor.sin();
        let cos_result = tensor.cos();

        assert!((sin_result.as_slice().unwrap()[0] - 0.0).abs() < 1e-6);
        assert!((sin_result.as_slice().unwrap()[1] - 1.0).abs() < 1e-6);
        assert!((cos_result.as_slice().unwrap()[0] - 1.0).abs() < 1e-6);
        assert!((cos_result.as_slice().unwrap()[1]).abs() < 1e-6);
    }
}
