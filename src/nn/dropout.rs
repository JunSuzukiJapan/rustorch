//! Dropout layer implementation for regularization
//! 正則化のためのDropoutレイヤー実装

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use rand::Rng;
use std::fmt::Debug;
use std::iter::Sum;
use std::sync::{Arc, RwLock};

/// Dropout layer for regularization during training
/// 訓練中の正則化用Dropoutレイヤー
///
/// During training, randomly sets elements to zero with probability p.
/// During evaluation, scales inputs by (1-p) to maintain expected value.
/// 訓練中、確率pで要素をランダムにゼロに設定します。
/// 評価中、期待値を維持するために入力を(1-p)でスケールします。
#[derive(Debug)]
pub struct Dropout<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> {
    /// Dropout probability (0.0 to 1.0)
    /// ドロップアウト確率（0.0から1.0）
    p: T,

    /// Training mode flag
    /// 訓練モードフラグ
    training: Arc<RwLock<bool>>,

    /// Inplace operation flag
    /// インプレース演算フラグ
    inplace: bool,
}

impl<T> Dropout<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    /// Creates a new Dropout layer
    /// 新しいDropoutレイヤーを作成します
    ///
    /// # Arguments
    /// * `p` - Probability of an element to be zeroed (0.0 to 1.0)
    /// * `inplace` - If true, will do this operation in-place
    ///
    /// # 引数
    /// * `p` - 要素がゼロになる確率（0.0から1.0）
    /// * `inplace` - trueの場合、この演算をインプレースで行います
    pub fn new(p: T, inplace: bool) -> Self {
        if p < T::zero() || p > T::one() {
            panic!(
                "Dropout probability must be between 0.0 and 1.0, got: {:?}",
                p
            );
        }

        Dropout {
            p,
            training: Arc::new(RwLock::new(true)),
            inplace,
        }
    }

    /// Sets the layer to training mode
    /// レイヤーを訓練モードに設定します
    pub fn train(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = true;
        }
    }

    /// Sets the layer to evaluation mode
    /// レイヤーを評価モードに設定します
    pub fn eval(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = false;
        }
    }

    /// Returns whether the layer is in training mode
    /// レイヤーが訓練モードかどうかを返します
    pub fn is_training(&self) -> bool {
        self.training
            .read()
            .unwrap_or_else(|_| panic!("Failed to read training mode"))
            .clone()
    }

    /// Forward pass of the Dropout layer
    /// Dropoutレイヤーの順伝播
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        if self.is_training() {
            // Training mode: apply dropout
            self.apply_dropout(input)
        } else {
            // Evaluation mode: no dropout, just return input
            input.clone()
        }
    }

    /// Apply dropout during training
    /// 訓練中にドロップアウトを適用
    fn apply_dropout(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();

        // Generate dropout mask
        let mask = self.generate_dropout_mask(input_data.shape());

        // Apply mask and scale
        let output_data = self.apply_mask_and_scale(&input_data, &mask);

        if input.requires_grad() {
            Variable::new(output_data, true)
        } else {
            Variable::new(output_data, false)
        }
    }

    /// Generate a random dropout mask
    /// ランダムなドロップアウトマスクを生成
    fn generate_dropout_mask(&self, shape: &[usize]) -> Tensor<T> {
        let total_elements: usize = shape.iter().product();
        let mut rng = rand::thread_rng();

        let mask_data: Vec<T> = (0..total_elements)
            .map(|_| {
                let random_val: f32 = rng.gen();
                let _alpha_prime = -T::from(1.7580993408473766).unwrap()
                    * (T::from(1.0).unwrap() * T::from(1.0).unwrap())
                    + T::from(1.0).unwrap();
                if T::from(random_val).unwrap() < self.p {
                    T::zero() // Drop this element
                } else {
                    T::one() // Keep this element
                }
            })
            .collect();

        Tensor::from_vec(mask_data, shape.to_vec())
    }

    /// Apply mask and scale to maintain expected value
    /// 期待値を維持するためにマスクとスケールを適用
    fn apply_mask_and_scale(&self, input: &Tensor<T>, mask: &Tensor<T>) -> Tensor<T> {
        // Scale factor to maintain expected value: 1 / (1 - p)
        let scale_factor = T::one() / (T::one() - self.p);

        let input_data = input.as_array();
        let mask_data = mask.as_array();

        let result_data: Vec<T> = input_data
            .iter()
            .zip(mask_data.iter())
            .map(|(&x, &m)| x * m * scale_factor)
            .collect();

        Tensor::from_vec(result_data, input.shape().to_vec())
    }

    /// Returns the dropout probability
    /// ドロップアウト確率を返します
    pub fn p(&self) -> T {
        self.p
    }

    /// Returns whether inplace operation is enabled
    /// インプレース演算が有効かどうかを返します
    pub fn inplace(&self) -> bool {
        self.inplace
    }
}

impl<T> Module<T> for Dropout<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        vec![] // Dropout has no learnable parameters
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Functional dropout interface
/// 関数型ドロップアウトインターフェース
///
/// Applies dropout to input during training mode
/// 訓練モード中に入力にドロップアウトを適用します
pub fn dropout<
    T: Float
        + Send
        + Sync
        + 'static
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + Copy
        + Debug
        + Default
        + ScalarOperand
        + Sum
        + std::fmt::Display,
>(
    input: &Variable<T>,
    p: T,
    training: bool,
) -> Variable<T> {
    if training && p > T::zero() {
        let dropout_layer = Dropout::new(p, false);
        if training {
            dropout_layer.train();
        } else {
            dropout_layer.eval();
        }
        dropout_layer.forward(input)
    } else {
        input.clone()
    }
}

/// AlphaDropout for SELU networks (maintains mean and variance)
/// SELUネットワーク用のAlphaDropout（平均と分散を維持）
///
/// AlphaDropout maintains the self-normalizing property of SELU networks
/// AlphaDropoutはSELUネットワークの自己正規化特性を維持します
#[derive(Debug)]
pub struct AlphaDropout<T: Float + Send + Sync> {
    /// Dropout probability (0.0 to 1.0)
    /// ドロップアウト確率（0.0から1.0）
    p: T,

    /// Training mode flag
    /// 訓練モードフラグ
    training: Arc<RwLock<bool>>,
}

impl<T> AlphaDropout<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    /// Creates a new AlphaDropout layer
    /// 新しいAlphaDropoutレイヤーを作成します
    pub fn new(p: T, _inplace: bool) -> Self {
        if p < T::zero() || p > T::one() {
            panic!(
                "AlphaDropout probability must be between 0.0 and 1.0, got: {:?}",
                p
            );
        }

        AlphaDropout {
            p,
            training: Arc::new(RwLock::new(true)),
        }
    }

    /// Sets the layer to training mode
    /// レイヤーを訓練モードに設定します
    pub fn train(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = true;
        }
    }

    /// Sets the layer to evaluation mode
    /// レイヤーを評価モードに設定します
    pub fn eval(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = false;
        }
    }

    /// Returns whether the layer is in training mode
    /// レイヤーが訓練モードかどうかを返します
    pub fn is_training(&self) -> bool {
        self.training
            .read()
            .unwrap_or_else(|_| panic!("Failed to read training mode"))
            .clone()
    }

    /// Forward pass of the AlphaDropout layer
    /// AlphaDropoutレイヤーの順伝播
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        if self.is_training() {
            self.apply_alpha_dropout(input)
        } else {
            input.clone()
        }
    }

    /// Apply alpha dropout (maintains mean=0, variance=1)
    /// アルファドロップアウトを適用（平均=0、分散=1を維持）
    fn apply_alpha_dropout(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();

        // AlphaDropout parameters for SELU
        let alpha = T::from(-1.7580993408473766f32).unwrap(); // Negative saturation value for SELU
        let keep_prob = T::one() - self.p;

        // Calculate scaling parameters to maintain mean=0, var=1
        let a = T::one() / (keep_prob + alpha.powi(2) * self.p * (T::one() - keep_prob)).sqrt();
        let b = -a * alpha * self.p;

        // Generate random mask
        let mask = self.generate_alpha_mask(input_data.shape(), keep_prob);

        // Apply alpha dropout transformation
        let output_data = self.apply_alpha_transformation(&input_data, &mask, a, b, alpha);

        if input.requires_grad() {
            Variable::new(output_data, true)
        } else {
            Variable::new(output_data, false)
        }
    }

    /// Generate alpha dropout mask
    /// アルファドロップアウトマスクを生成
    fn generate_alpha_mask(&self, shape: &[usize], keep_prob: T) -> Tensor<T> {
        let total_elements: usize = shape.iter().product();
        let mut rng = rand::thread_rng();

        let mask_data: Vec<T> = (0..total_elements)
            .map(|_| {
                let random_val: f32 = rng.gen();
                if T::from(random_val).unwrap() < keep_prob {
                    T::one() // Keep
                } else {
                    T::zero() // Drop
                }
            })
            .collect();

        Tensor::from_vec(mask_data, shape.to_vec())
    }

    /// Apply alpha dropout transformation
    /// アルファドロップアウト変換を適用
    fn apply_alpha_transformation(
        &self,
        input: &Tensor<T>,
        mask: &Tensor<T>,
        a: T,
        b: T,
        alpha: T,
    ) -> Tensor<T> {
        let input_data = input.as_array();
        let mask_data = mask.as_array();

        let result_data: Vec<T> = input_data
            .iter()
            .zip(mask_data.iter())
            .map(|(&x, &m)| {
                if m > T::zero() {
                    // Keep: scale and shift
                    a * x + b
                } else {
                    // Drop: replace with alpha value and transform
                    a * alpha + b
                }
            })
            .collect();

        Tensor::from_vec(result_data, input.shape().to_vec())
    }

    /// Returns the dropout probability
    /// ドロップアウト確率を返します
    pub fn p(&self) -> T {
        self.p
    }
}

impl<T> Module<T> for AlphaDropout<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        vec![] // AlphaDropout has no learnable parameters
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dropout_creation() {
        let dropout = Dropout::<f32>::new(0.5, false);
        assert_abs_diff_eq!(dropout.p(), 0.5, epsilon = 1e-6);
        assert!(dropout.is_training());
        assert!(!dropout.inplace());
    }

    #[test]
    fn test_dropout_eval_mode() {
        let dropout = Dropout::<f32>::new(0.5, false);

        // Switch to evaluation mode
        dropout.eval();
        assert!(!dropout.is_training());

        // In eval mode, input should pass through unchanged
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]), false);

        let output = dropout.forward(&input);
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();

        // Should be identical in eval mode
        for (input_val, output_val) in input_data
            .as_array()
            .iter()
            .zip(output_data.as_array().iter())
        {
            assert_abs_diff_eq!(*input_val, *output_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dropout_training_mode() {
        let dropout = Dropout::<f32>::new(0.5, false);
        dropout.train();

        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]), false);

        let output = dropout.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();

        // In training mode with p=0.5, roughly half should be zero (scaled up)
        // This is probabilistic, so we just check the output is different
        let output_shape = output_data.shape();
        assert_eq!(output_shape, &[4]);
    }

    #[test]
    fn test_dropout_with_gradients() {
        let dropout = Dropout::<f32>::new(0.3, false);

        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]), true);

        let output = dropout.forward(&input);
        assert!(output.requires_grad());
    }

    #[test]
    fn test_functional_dropout() {
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]), false);

        // Training mode
        let output_train = dropout(&input, 0.5, true);
        assert_eq!(output_train.data().read().unwrap().shape(), &[3]);

        // Eval mode - should be identical to input
        let output_eval = dropout(&input, 0.5, false);
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let output_binding = output_eval.data();
        let output_data = output_binding.read().unwrap();

        for (input_val, output_val) in input_data
            .as_array()
            .iter()
            .zip(output_data.as_array().iter())
        {
            assert_abs_diff_eq!(*input_val, *output_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_alpha_dropout_creation() {
        let alpha_dropout = AlphaDropout::<f32>::new(0.1, false);
        assert_abs_diff_eq!(alpha_dropout.p(), 0.1, epsilon = 1e-6);
        assert!(alpha_dropout.is_training());
    }

    #[test]
    fn test_alpha_dropout_forward() {
        let alpha_dropout = AlphaDropout::<f32>::new(0.1, false);

        let input = Variable::new(Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], vec![4]), false);

        let output = alpha_dropout.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();

        // Output should have same shape
        assert_eq!(output_data.shape(), &[4]);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be between 0.0 and 1.0")]
    fn test_dropout_invalid_probability() {
        let _dropout = Dropout::<f32>::new(1.5, false);
    }

    #[test]
    fn test_dropout_zero_probability() {
        let dropout = Dropout::<f32>::new(0.0, false);
        dropout.train();

        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]), false);

        let output = dropout.forward(&input);
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();

        // With p=0.0, nothing should be dropped
        for (input_val, output_val) in input_data
            .as_array()
            .iter()
            .zip(output_data.as_array().iter())
        {
            assert_abs_diff_eq!(*input_val, *output_val, epsilon = 1e-6);
        }
    }
}
