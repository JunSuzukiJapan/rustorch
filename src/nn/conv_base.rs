//! Base traits and utilities for convolution layers
//! 畳み込み層の基底トレイトとユーティリティ

use crate::autograd::Variable;
use crate::error::RusTorchError;
use crate::tensor::Tensor;
use num_traits::Float;
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::fmt::Debug;
use std::ops::Add;

/// Common convolution parameters and initialization
/// 共通の畳み込みパラメータと初期化
pub trait ConvolutionBase<T: Float + Send + Sync> {
    /// Get input channels
    /// 入力チャンネル数を取得
    fn in_channels(&self) -> usize;

    /// Get output channels
    /// 出力チャンネル数を取得
    fn out_channels(&self) -> usize;

    /// Get groups for grouped convolution
    /// グループ畳み込みのグループ数を取得
    fn groups(&self) -> usize;

    /// Get kernel dimensions
    /// カーネル次元を取得
    fn kernel_dims(&self) -> Vec<usize>;

    /// Calculate fan-in for weight initialization
    /// 重み初期化のためのファンイン計算
    fn calculate_fan_in(&self) -> usize {
        let kernel_size: usize = self.kernel_dims().iter().product();
        (self.in_channels() / self.groups()) * kernel_size
    }

    /// Initialize weights using Kaiming uniform distribution
    /// Kaiming uniform分布を使用した重み初期化
    fn init_weights(&self, weight_shape: Vec<usize>) -> Vec<T>
    where
        T: From<f32> + Copy,
    {
        let fan_in = self.calculate_fan_in();
        let bound = (6.0 / fan_in as f32).sqrt();
        let weight_size = weight_shape.iter().product::<usize>();

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, bound).unwrap();

        (0..weight_size)
            .map(|_| <T as From<f32>>::from(normal.sample(&mut rng)))
            .collect()
    }

    /// Calculate number of parameters
    /// パラメータ数を計算
    fn num_parameters(&self, has_bias: bool) -> usize {
        let kernel_size: usize = self.kernel_dims().iter().product();
        let weight_params =
            self.out_channels() * (self.in_channels() / self.groups()) * kernel_size;
        let bias_params = if has_bias { self.out_channels() } else { 0 };
        weight_params + bias_params
    }
}

/// Common pooling operations
/// 共通プーリング操作
pub trait PoolingBase<T: Float + Send + Sync> {
    /// Get output size
    /// 出力サイズを取得
    fn output_size(&self) -> Vec<usize>;

    /// Calculate pooling parameters for adaptive pooling
    /// 適応的プーリングのパラメータ計算
    fn calculate_adaptive_params(&self, input_size: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
        let output_size = self.output_size();
        let mut kernel_sizes = Vec::new();
        let mut strides = Vec::new();

        for (input_dim, output_dim) in input_size.iter().zip(output_size.iter()) {
            let kernel = input_dim.div_ceil(*output_dim);
            let stride = input_dim / output_dim;
            kernel_sizes.push(kernel);
            strides.push(stride);
        }

        (kernel_sizes, strides)
    }
}

/// Result type for neural network operations (統一済み)
/// ニューラルネットワーク操作の結果型 (統一済み)
pub type NNResult<T> = crate::error::RusTorchResult<T>;

// Error types for neural network operations
// RusTorchError enum removed - now using unified RusTorchError system
// RusTorchErrorエナム削除 - 統一RusTorchErrorシステムを使用

/// Validation utilities for neural network layers
/// ニューラルネットワーク層の検証ユーティリティ
pub struct Validator;

impl Validator {
    /// Validate convolution parameters
    /// 畳み込みパラメータの検証
    pub fn validate_conv_params(
        in_channels: usize,
        out_channels: usize,
        kernel_size: &[usize],
        stride: &[usize],
        _padding: &[usize],
        dilation: &[usize],
        groups: usize,
    ) -> Result<(), RusTorchError> {
        if in_channels == 0 || out_channels == 0 {
            return Err(RusTorchError::InvalidDimensions(
                "Input and output channels must be positive".to_string(),
            ));
        }

        if in_channels % groups != 0 {
            return Err(RusTorchError::InvalidDimensions(
                "Input channels must be divisible by groups".to_string(),
            ));
        }

        if out_channels % groups != 0 {
            return Err(RusTorchError::InvalidDimensions(
                "Output channels must be divisible by groups".to_string(),
            ));
        }

        for &k in kernel_size {
            if k == 0 {
                return Err(RusTorchError::InvalidDimensions(
                    "Kernel size must be positive".to_string(),
                ));
            }
        }

        for &s in stride {
            if s == 0 {
                return Err(RusTorchError::InvalidDimensions(
                    "Stride must be positive".to_string(),
                ));
            }
        }

        for &d in dilation {
            if d == 0 {
                return Err(RusTorchError::InvalidDimensions(
                    "Dilation must be positive".to_string(),
                ));
            }
        }

        if groups == 0 {
            return Err(RusTorchError::InvalidDimensions(
                "Groups must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate pooling parameters
    /// プーリングパラメータの検証
    pub fn validate_pool_params(
        kernel_size: &[usize],
        stride: &[usize],
        _padding: &[usize],
    ) -> Result<(), RusTorchError> {
        for &k in kernel_size {
            if k == 0 {
                return Err(RusTorchError::InvalidDimensions(
                    "Kernel size must be positive".to_string(),
                ));
            }
        }

        for &s in stride {
            if s == 0 {
                return Err(RusTorchError::InvalidDimensions(
                    "Stride must be positive".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Validate output size for adaptive pooling
    /// 適応的プーリングの出力サイズ検証
    pub fn validate_adaptive_output_size(output_size: &[usize]) -> Result<(), RusTorchError> {
        for &size in output_size {
            if size == 0 {
                return Err(RusTorchError::InvalidDimensions(
                    "Output size must be positive".to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Common operations for convolution layers
/// 畳み込み層用共通操作
pub struct ConvOps;

impl ConvOps {
    /// Initialize weights and bias for convolution layers
    /// 畳み込み層用の重みとバイアスを初期化
    pub fn init_conv_params<
        T: Float
            + Send
            + Sync
            + Debug
            + 'static
            + From<f32>
            + Copy
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    >(
        input_size: usize,
        output_size: usize,
        kernel_size: usize,
        groups: usize,
        use_bias: bool,
    ) -> (Variable<T>, Option<Variable<T>>) {
        let fan_in = (input_size / groups) * kernel_size;
        let bound = (6.0 / fan_in as f32).sqrt();

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, bound).unwrap();

        // Initialize weights
        let weight_data: Vec<T> = (0..output_size * input_size * kernel_size / groups)
            .map(|_| num_traits::cast(normal.sample(&mut rng) as f64).unwrap_or(T::zero()))
            .collect();
        let weight = Variable::new(
            Tensor::from_vec(
                weight_data,
                vec![output_size, input_size / groups, kernel_size],
            ),
            true,
        );

        // Initialize bias if needed
        let bias = if use_bias {
            let bias_data = vec![T::zero(); output_size];
            Some(Variable::new(
                Tensor::from_vec(bias_data, vec![output_size]),
                true,
            ))
        } else {
            None
        };

        (weight, bias)
    }

    /// Calculate output size for 1D convolution
    /// 1D畳み込みの出力サイズを計算
    pub fn calc_output_size_1d(
        input_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> usize {
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    }

    /// Calculate output size for 2D convolution
    /// 2D畳み込みの出力サイズを計算
    pub fn calc_output_size_2d(
        input_size: (usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> (usize, usize) {
        let out_h =
            Self::calc_output_size_1d(input_size.0, kernel_size.0, stride.0, padding.0, dilation.0);
        let out_w =
            Self::calc_output_size_1d(input_size.1, kernel_size.1, stride.1, padding.1, dilation.1);
        (out_h, out_w)
    }

    /// Calculate output size for 3D convolution
    /// 3D畳み込みの出力サイズを計算
    pub fn calc_output_size_3d(
        input_size: (usize, usize, usize),
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> (usize, usize, usize) {
        let out_d =
            Self::calc_output_size_1d(input_size.0, kernel_size.0, stride.0, padding.0, dilation.0);
        let out_h =
            Self::calc_output_size_1d(input_size.1, kernel_size.1, stride.1, padding.1, dilation.1);
        let out_w =
            Self::calc_output_size_1d(input_size.2, kernel_size.2, stride.2, padding.2, dilation.2);
        (out_d, out_h, out_w)
    }

    /// Linear transformation: input @ weight^T + bias
    /// 線形変換: input @ weight^T + bias
    pub fn linear_transform<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        input: &Variable<T>,
        weight: &Variable<T>,
        bias: Option<&Variable<T>>,
    ) -> Variable<T> {
        let output = Self::matmul_variables(input, &Self::transpose_variable(weight));

        match bias {
            Some(b) => Self::add_variables(&output, b),
            None => output,
        }
    }

    /// Matrix multiplication for variables
    /// Variable用の行列乗算
    pub fn matmul_variables<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        a: &Variable<T>,
        b: &Variable<T>,
    ) -> Variable<T> {
        let a_binding = a.data();
        let a_data = a_binding.read().unwrap();
        let b_binding = b.data();
        let b_data = b_binding.read().unwrap();
        let result_data = a_data.matmul(&*b_data).expect("MatMul failed");
        Variable::new(result_data, a.requires_grad() || b.requires_grad())
    }

    /// Addition for variables
    /// Variable用の加算
    pub fn add_variables<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        a: &Variable<T>,
        b: &Variable<T>,
    ) -> Variable<T> {
        let a_binding = a.data();
        let a_data = a_binding.read().unwrap();
        let b_binding = b.data();
        let b_data = b_binding.read().unwrap();
        let result_data = (&*a_data).add(&*b_data);
        Variable::new(result_data, a.requires_grad() || b.requires_grad())
    }

    /// Transpose for variables
    /// Variable用の転置
    pub fn transpose_variable<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        var: &Variable<T>,
    ) -> Variable<T> {
        let var_binding = var.data();
        let var_data = var_binding.read().unwrap();
        let transposed_data = var_data.transpose().expect("Transpose failed");
        Variable::new(transposed_data, var.requires_grad())
    }
}

/// Common parameter collection for convolution layers
/// 畳み込み層用共通パラメータ収集
pub fn collect_conv_parameters<
    T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    weight: &Variable<T>,
    bias: &Option<Variable<T>>,
) -> Vec<Variable<T>> {
    let mut params = vec![weight.clone()];

    if let Some(ref bias_var) = bias {
        params.push(bias_var.clone());
    }

    params
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestConv {
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        groups: usize,
    }

    impl<T: Float + Send + Sync> ConvolutionBase<T> for TestConv {
        fn in_channels(&self) -> usize {
            self.in_channels
        }
        fn out_channels(&self) -> usize {
            self.out_channels
        }
        fn groups(&self) -> usize {
            self.groups
        }
        fn kernel_dims(&self) -> Vec<usize> {
            vec![self.kernel_size.0, self.kernel_size.1]
        }
    }

    #[test]
    fn test_fan_in_calculation() {
        let conv = TestConv {
            in_channels: 64,
            out_channels: 32,
            kernel_size: (3, 3),
            groups: 1,
        };

        assert_eq!(
            <TestConv as ConvolutionBase<f32>>::calculate_fan_in(&conv),
            64 * 3 * 3
        );
    }

    #[test]
    fn test_parameter_count() {
        let conv = TestConv {
            in_channels: 64,
            out_channels: 32,
            kernel_size: (3, 3),
            groups: 1,
        };

        // Weight: 32 * 64 * 3 * 3 = 18432
        // Bias: 32
        // Total: 18464
        assert_eq!(
            <TestConv as ConvolutionBase<f32>>::num_parameters(&conv, true),
            18464
        );
        assert_eq!(
            <TestConv as ConvolutionBase<f32>>::num_parameters(&conv, false),
            18432
        );
    }

    #[test]
    fn test_weight_initialization() {
        let conv = TestConv {
            in_channels: 16,
            out_channels: 32,
            kernel_size: (3, 3),
            groups: 1,
        };

        let weights: Vec<f32> =
            <TestConv as ConvolutionBase<f32>>::init_weights(&conv, vec![32, 16, 3, 3]);
        assert_eq!(weights.len(), 32 * 16 * 3 * 3);
    }

    #[test]
    fn test_parameter_validation() {
        // Valid parameters
        assert!(
            Validator::validate_conv_params(64, 32, &[3, 3], &[1, 1], &[1, 1], &[1, 1], 1).is_ok()
        );

        // Invalid: zero channels
        assert!(
            Validator::validate_conv_params(0, 32, &[3, 3], &[1, 1], &[1, 1], &[1, 1], 1).is_err()
        );

        // Invalid: channels not divisible by groups
        assert!(
            Validator::validate_conv_params(64, 32, &[3, 3], &[1, 1], &[1, 1], &[1, 1], 3).is_err()
        );

        // Invalid: zero kernel size
        assert!(
            Validator::validate_conv_params(64, 32, &[0, 3], &[1, 1], &[1, 1], &[1, 1], 1).is_err()
        );
    }

    #[test]
    fn test_adaptive_pooling_params() {
        struct TestPool {
            output_size: Vec<usize>,
        }

        impl<T: Float + Send + Sync> PoolingBase<T> for TestPool {
            fn output_size(&self) -> Vec<usize> {
                self.output_size.clone()
            }
        }

        let pool = TestPool {
            output_size: vec![7, 7],
        };
        let (kernel_sizes, strides) =
            <TestPool as PoolingBase<f32>>::calculate_adaptive_params(&pool, vec![224, 224]);

        // For 224x224 -> 7x7: kernel ~= 32, stride = 32
        assert_eq!(kernel_sizes, vec![32, 32]);
        assert_eq!(strides, vec![32, 32]);
    }
}
