/// Mathematical operations for tensors
/// テンソルの数学演算

use ndarray::Zip;
use num_traits::Float;
use crate::tensor::Tensor;

impl<T: Float + 'static + Send + Sync> Tensor<T> {
    /// Element-wise sine function
    /// 要素ごとのサイン関数
    pub fn sin(&self) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.apply_math_function_f32(|x| x.sin())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            self.apply_math_function_f64(|x| x.sin())
        } else {
            self.apply_math_function_generic(|x| x.sin())
        }
    }

    /// Element-wise cosine function
    /// 要素ごとのコサイン関数
    pub fn cos(&self) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.apply_math_function_f32(|x| x.cos())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            self.apply_math_function_f64(|x| x.cos())
        } else {
            self.apply_math_function_generic(|x| x.cos())
        }
    }

    /// Element-wise tangent function
    /// 要素ごとのタンジェント関数
    pub fn tan(&self) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.apply_math_function_f32(|x| x.tan())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            self.apply_math_function_f64(|x| x.tan())
        } else {
            self.apply_math_function_generic(|x| x.tan())
        }
    }

    /// Element-wise exponential function
    /// 要素ごとの指数関数
    pub fn exp(&self) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.apply_math_function_f32(|x| x.exp())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            self.apply_math_function_f64(|x| x.exp())
        } else {
            self.apply_math_function_generic(|x| x.exp())
        }
    }

    /// Element-wise natural logarithm
    /// 要素ごとの自然対数
    pub fn log(&self) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.apply_math_function_f32(|x| x.ln())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            self.apply_math_function_f64(|x| x.ln())
        } else {
            self.apply_math_function_generic(|x| x.ln())
        }
    }

    /// Element-wise logarithm base 10
    /// 要素ごとの常用対数
    pub fn log10(&self) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.apply_math_function_f32(|x| x.log10())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            self.apply_math_function_f64(|x| x.log10())
        } else {
            self.apply_math_function_generic(|x| x.log10())
        }
    }

    /// Element-wise square root
    /// 要素ごとの平方根
    pub fn sqrt(&self) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.apply_math_function_f32(|x| x.sqrt())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            self.apply_math_function_f64(|x| x.sqrt())
        } else {
            self.apply_math_function_generic(|x| x.sqrt())
        }
    }

    /// Element-wise power function
    /// 要素ごとのべき乗関数
    pub fn pow(&self, exponent: T) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let exp_f32 = unsafe { std::mem::transmute_copy::<T, f32>(&exponent) };
            self.apply_math_function_f32(|x| x.powf(exp_f32))
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let exp_f64 = unsafe { std::mem::transmute_copy::<T, f64>(&exponent) };
            self.apply_math_function_f64(|x| x.powf(exp_f64))
        } else {
            self.apply_math_function_generic(|x| x.powf(exponent))
        }
    }

    /// Element-wise absolute value
    /// 要素ごとの絶対値
    pub fn abs(&self) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.apply_math_function_f32(|x| x.abs())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            self.apply_math_function_f64(|x| x.abs())
        } else {
            self.apply_math_function_generic(|x| x.abs())
        }
    }

    /// Element-wise sigmoid function
    /// 要素ごとのシグモイド関数
    pub fn sigmoid(&self) -> Self {
        self.apply_math_function_generic(|x| T::one() / (T::one() + (-x).exp()))
    }

    /// Element-wise tanh function
    /// 要素ごとのタンジェントハイパボリック関数
    pub fn tanh(&self) -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.apply_math_function_f32(|x| x.tanh())
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            self.apply_math_function_f64(|x| x.tanh())
        } else {
            self.apply_math_function_generic(|x| x.tanh())
        }
    }

    /// Apply math function for f32 with potential SIMD optimization
    /// f32用の数学関数を適用（SIMD最適化の可能性あり）
    fn apply_math_function_f32<F>(&self, func: F) -> Self
    where
        F: Fn(f32) -> f32 + Sync + Send,
    {
        let mut result = self.clone();
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            if let Some(data_slice) = result.data.as_slice_mut() {
                // Safe cast for f32 type
                let f32_slice = unsafe { 
                    std::slice::from_raw_parts_mut(
                        data_slice.as_mut_ptr() as *mut f32,
                        data_slice.len()
                    )
                };
                
                // Apply function element-wise (can be optimized with SIMD later)
                for element in f32_slice.iter_mut() {
                    *element = func(*element);
                }
            }
        }
        result
    }

    /// Apply math function for f64 with potential SIMD optimization
    /// f64用の数学関数を適用（SIMD最適化の可能性あり）
    fn apply_math_function_f64<F>(&self, func: F) -> Self
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        let mut result = self.clone();
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            if let Some(data_slice) = result.data.as_slice_mut() {
                // Safe cast for f64 type
                let f64_slice = unsafe { 
                    std::slice::from_raw_parts_mut(
                        data_slice.as_mut_ptr() as *mut f64,
                        data_slice.len()
                    )
                };
                
                // Apply function element-wise (can be optimized with SIMD later)
                for element in f64_slice.iter_mut() {
                    *element = func(*element);
                }
            }
        }
        result
    }

    /// Apply a mathematical function generically
    /// 汎用的な数学関数の適用
    fn apply_math_function_generic<F>(&self, func: F) -> Self
    where
        F: Fn(T) -> T + Sync + Send,
    {
        let mut result = self.data.clone();
        result.par_mapv_inplace(func);
        Tensor::new(result)
    }

    /// Element-wise maximum with another tensor
    /// 他のテンソルとの要素ごとの最大値
    pub fn max_elementwise(&self, other: &Self) -> Self {
        let mut result = self.data.clone();
        Zip::from(&mut result)
            .and(&other.data)
            .par_for_each(|a, &b| {
                if b > *a {
                    *a = b;
                }
            });
        Tensor::new(result)
    }

    /// Element-wise minimum with another tensor
    /// 他のテンソルとの要素ごとの最小値
    pub fn min_elementwise(&self, other: &Self) -> Self {
        let mut result = self.data.clone();
        Zip::from(&mut result)
            .and(&other.data)
            .par_for_each(|a, &b| {
                if b < *a {
                    *a = b;
                }
            });
        Tensor::new(result)
    }

    /// Clamp values between min and max
    /// 値を最小値と最大値の間にクランプ
    pub fn clamp(&self, min_val: T, max_val: T) -> Self {
        self.apply_math_function_generic(|x| {
            if x < min_val {
                min_val
            } else if x > max_val {
                max_val
            } else {
                x
            }
        })
    }

    /// Round to nearest integer
    /// 最近傍の整数に丸める
    pub fn round(&self) -> Self {
        self.apply_math_function_generic(|x| x.round())
    }

    /// Floor function
    /// 床関数
    pub fn floor(&self) -> Self {
        self.apply_math_function_generic(|x| x.floor())
    }

    /// Ceiling function
    /// 天井関数
    pub fn ceil(&self) -> Self {
        self.apply_math_function_generic(|x| x.ceil())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_basic_math_functions() {
        let tensor = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.14159], vec![4]);
        
        // Test sin
        let sin_result = tensor.sin();
        assert_abs_diff_eq!(sin_result.data[[0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sin_result.data[[1]], 1.0f32.sin(), epsilon = 1e-6);
        
        // Test cos
        let cos_result = tensor.cos();
        assert_abs_diff_eq!(cos_result.data[[0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cos_result.data[[1]], 1.0f32.cos(), epsilon = 1e-6);
        
        // Test exp
        let exp_result = tensor.exp();
        assert_abs_diff_eq!(exp_result.data[[0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(exp_result.data[[1]], std::f32::consts::E, epsilon = 1e-6);
        
        // Test log
        let log_tensor = Tensor::from_vec(vec![1.0f32, std::f32::consts::E, 10.0], vec![3]);
        let log_result = log_tensor.log();
        assert_abs_diff_eq!(log_result.data[[0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(log_result.data[[1]], 1.0, epsilon = 1e-6);
        
        // Test sqrt
        let sqrt_tensor = Tensor::from_vec(vec![0.0f32, 1.0, 4.0, 9.0], vec![4]);
        let sqrt_result = sqrt_tensor.sqrt();
        assert_abs_diff_eq!(sqrt_result.data[[0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sqrt_result.data[[1]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sqrt_result.data[[2]], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sqrt_result.data[[3]], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_power_function() {
        let tensor = Tensor::from_vec(vec![2.0f32, 3.0, 4.0], vec![3]);
        let pow_result = tensor.pow(2.0);
        
        assert_abs_diff_eq!(pow_result.data[[0]], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pow_result.data[[1]], 9.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pow_result.data[[2]], 16.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_function() {
        let tensor = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], vec![3]);
        let sigmoid_result = tensor.sigmoid();
        
        assert_abs_diff_eq!(sigmoid_result.data[[0]], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(sigmoid_result.data[[1]], 1.0 / (1.0 + (-1.0f32).exp()), epsilon = 1e-6);
        assert_abs_diff_eq!(sigmoid_result.data[[2]], 1.0 / (1.0 + 1.0f32.exp()), epsilon = 1e-6);
    }

    #[test]
    fn test_elementwise_operations() {
        let tensor1 = Tensor::from_vec(vec![1.0f32, 5.0, 3.0], vec![3]);
        let tensor2 = Tensor::from_vec(vec![4.0f32, 2.0, 6.0], vec![3]);
        
        // Test max
        let max_result = tensor1.max_elementwise(&tensor2);
        assert_abs_diff_eq!(max_result.data[[0]], 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(max_result.data[[1]], 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(max_result.data[[2]], 6.0, epsilon = 1e-6);
        
        // Test min
        let min_result = tensor1.min_elementwise(&tensor2);
        assert_abs_diff_eq!(min_result.data[[0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(min_result.data[[1]], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(min_result.data[[2]], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_clamp_function() {
        let tensor = Tensor::from_vec(vec![-2.0f32, 0.5, 3.0, 10.0], vec![4]);
        let clamped = tensor.clamp(0.0, 5.0);
        
        assert_abs_diff_eq!(clamped.data[[0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(clamped.data[[1]], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(clamped.data[[2]], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(clamped.data[[3]], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_rounding_functions() {
        let tensor = Tensor::from_vec(vec![1.2f32, 1.7, -1.3, -1.8], vec![4]);
        
        // Test round
        let rounded = tensor.round();
        assert_abs_diff_eq!(rounded.data[[0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(rounded.data[[1]], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(rounded.data[[2]], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(rounded.data[[3]], -2.0, epsilon = 1e-6);
        
        // Test floor
        let floored = tensor.floor();
        assert_abs_diff_eq!(floored.data[[0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(floored.data[[1]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(floored.data[[2]], -2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(floored.data[[3]], -2.0, epsilon = 1e-6);
        
        // Test ceil
        let ceiled = tensor.ceil();
        assert_abs_diff_eq!(ceiled.data[[0]], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(ceiled.data[[1]], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(ceiled.data[[2]], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(ceiled.data[[3]], -1.0, epsilon = 1e-6);
    }
}
