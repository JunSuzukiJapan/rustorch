use crate::distributions::{Distribution, DistributionTrait, DistributionUtils};
use crate::error::{RusTorchError, RusTorchResult};
/// Bernoulli Distribution - torch.distributions.Bernoulli compatible
/// ベルヌーイ分布 - torch.distributions.Bernoulli互換
///
/// Implements the Bernoulli distribution parameterized by probability p
/// or log-odds (logits).
/// 確率pまたは対数オッズ（ロジット）でパラメータ化されたベルヌーイ分布を実装
use crate::tensor::Tensor;
use num_traits::Float;
use std::marker::PhantomData;

/// Bernoulli Distribution
/// ベルヌーイ分布
///
/// The Bernoulli distribution is defined by:
/// - Probability parameter p ∈ [0, 1], or
/// - Logits parameter (log-odds)
///
/// PMF: P(X = 1) = p, P(X = 0) = 1 - p
#[derive(Debug, Clone)]
pub struct Bernoulli<T: Float> {
    /// Probability parameter (p) - optional
    /// 確率パラメータ (p) - オプション
    pub probs: Option<Tensor<T>>,

    /// Logits parameter (log-odds) - optional
    /// ロジットパラメータ（対数オッズ）- オプション
    pub logits: Option<Tensor<T>>,

    /// Base distribution properties
    /// 基本分布特性
    pub base: Distribution,

    /// Phantom data for type parameter
    /// 型パラメータ用のファントムデータ
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> Bernoulli<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    /// Create a new Bernoulli distribution from probability
    /// 確率からベルヌーイ分布を作成
    ///
    /// # Arguments
    /// * `probs` - Probability tensor (values in [0, 1])
    /// * `validate_args` - Whether to validate parameters
    pub fn from_probs(probs: Tensor<T>, validate_args: bool) -> RusTorchResult<Self> {
        if validate_args {
            DistributionUtils::validate_probability(&probs)?;
        }

        let batch_shape = probs.shape().to_vec();
        let event_shape = vec![]; // Bernoulli is univariate

        Ok(Self {
            probs: Some(probs),
            logits: None,
            base: Distribution::new(batch_shape, event_shape, validate_args),
            _phantom: PhantomData,
        })
    }

    /// Create a new Bernoulli distribution from logits
    /// ロジットからベルヌーイ分布を作成
    ///
    /// # Arguments
    /// * `logits` - Log-odds tensor (any real values)
    /// * `validate_args` - Whether to validate parameters
    pub fn from_logits(logits: Tensor<T>, validate_args: bool) -> RusTorchResult<Self> {
        let batch_shape = logits.shape().to_vec();
        let event_shape = vec![]; // Bernoulli is univariate

        Ok(Self {
            probs: None,
            logits: Some(logits),
            base: Distribution::new(batch_shape, event_shape, validate_args),
            _phantom: PhantomData,
        })
    }

    /// Create Bernoulli distribution with scalar probability
    /// スカラー確率でベルヌーイ分布を作成
    pub fn from_scalar_prob(prob: T, validate_args: bool) -> RusTorchResult<Self> {
        let probs_tensor = Tensor::from_vec(vec![prob], vec![]);
        Self::from_probs(probs_tensor, validate_args)
    }

    /// Create fair coin (p = 0.5)
    /// 公正なコイン（p = 0.5）
    pub fn fair_coin(validate_args: bool) -> RusTorchResult<Self> {
        Self::from_scalar_prob(T::from(0.5).unwrap(), validate_args)
    }

    /// Get probabilities (convert from logits if necessary)
    /// 確率を取得（必要に応じてロジットから変換）
    pub fn get_probs(&self) -> RusTorchResult<Tensor<T>> {
        match (&self.probs, &self.logits) {
            (Some(probs), _) => Ok(probs.clone()),
            (None, Some(logits)) => {
                // Convert logits to probabilities using sigmoid: p = 1 / (1 + exp(-logits))
                let logits_data = logits.data.as_slice().unwrap();
                let probs_data: Vec<T> = logits_data
                    .iter()
                    .map(|&logit| T::one() / (T::one() + (-logit).exp()))
                    .collect();
                Ok(Tensor::from_vec(probs_data, logits.shape().to_vec()))
            }
            _ => Err(RusTorchError::invalid_parameter(
                "Either probs or logits must be specified",
            )),
        }
    }

    /// Get logits (convert from probs if necessary)
    /// ロジットを取得（必要に応じて確率から変換）
    pub fn get_logits(&self) -> RusTorchResult<Tensor<T>> {
        match (&self.logits, &self.probs) {
            (Some(logits), _) => Ok(logits.clone()),
            (None, Some(probs)) => {
                // Convert probs to logits: logit = log(p / (1 - p))
                let probs_data = probs.data.as_slice().unwrap();
                let logits_data: Vec<T> = probs_data
                    .iter()
                    .map(|&p| {
                        if p == T::zero() {
                            T::from(-1e10).unwrap() // Very negative for p=0
                        } else if p == T::one() {
                            T::from(1e10).unwrap() // Very positive for p=1
                        } else {
                            (p / (T::one() - p)).ln()
                        }
                    })
                    .collect();
                Ok(Tensor::from_vec(logits_data, probs.shape().to_vec()))
            }
            _ => Err(RusTorchError::invalid_parameter(
                "Either probs or logits must be specified",
            )),
        }
    }

    /// Binary cross entropy for Bernoulli distributions
    /// ベルヌーイ分布のバイナリクロスエントロピー
    pub fn binary_cross_entropy(&self, target: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        let probs = self.get_probs()?;
        let probs_data = probs.data.as_slice().unwrap();
        let target_data = target.data.as_slice().unwrap();

        if probs_data.len() != target_data.len() {
            return Err(RusTorchError::shape_mismatch(probs.shape(), target.shape()));
        }

        // BCE = -[y*log(p) + (1-y)*log(1-p)]
        let bce_data: Vec<T> = probs_data
            .iter()
            .zip(target_data.iter())
            .map(|(&p, &y)| {
                let p_clamped = p
                    .max(T::from(1e-10).unwrap())
                    .min(T::from(1.0 - 1e-10).unwrap());
                -(y * p_clamped.ln() + (T::one() - y) * (T::one() - p_clamped).ln())
            })
            .collect();

        Ok(Tensor::from_vec(bce_data, target.shape().to_vec()))
    }
}

impl<T: Float + 'static> DistributionTrait<T> for Bernoulli<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    fn sample(&self, shape: Option<&[usize]>) -> RusTorchResult<Tensor<T>> {
        let sample_shape = self.base.expand_shape(shape);
        let probs = self.get_probs()?;

        // Generate uniform random samples
        let uniform_samples = DistributionUtils::random_uniform::<T>(&sample_shape);
        let uniform_data = uniform_samples.data.as_slice().unwrap();
        let probs_data = probs.data.as_slice().unwrap();

        // Sample by comparing uniform samples with probabilities
        let result_data: Vec<T> = uniform_data
            .iter()
            .zip(probs_data.iter().cycle())
            .map(|(&u, &p)| if u < p { T::one() } else { T::zero() })
            .collect();

        Ok(Tensor::from_vec(result_data, sample_shape))
    }

    fn log_prob(&self, value: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        let logits = self.get_logits()?;
        let logits_data = logits.data.as_slice().unwrap();
        let value_data = value.data.as_slice().unwrap();

        if logits_data.len() != value_data.len() {
            return Err(RusTorchError::shape_mismatch(logits.shape(), value.shape()));
        }

        // log P(x) = x * logits - log(1 + exp(logits))
        // Use log-sum-exp trick for numerical stability
        let result_data: Vec<T> = logits_data
            .iter()
            .zip(value_data.iter())
            .map(|(&logit, &x)| {
                let max_val = logit.max(T::zero());
                x * logit - max_val - ((-max_val).exp() + (logit - max_val).exp()).ln()
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn cdf(&self, value: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        let probs = self.get_probs()?;
        let probs_data = probs.data.as_slice().unwrap();
        let value_data = value.data.as_slice().unwrap();

        // CDF is step function: 0 for x < 0, 1-p for 0 <= x < 1, 1 for x >= 1
        let result_data: Vec<T> = value_data
            .iter()
            .zip(probs_data.iter().cycle())
            .map(|(&x, &p)| {
                if x < T::zero() {
                    T::zero()
                } else if x < T::one() {
                    T::one() - p
                } else {
                    T::one()
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn icdf(&self, value: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        let probs = self.get_probs()?;
        let probs_data = probs.data.as_slice().unwrap();
        let value_data = value.data.as_slice().unwrap();

        // Inverse CDF for discrete distribution
        let result_data: Vec<T> = value_data
            .iter()
            .zip(probs_data.iter().cycle())
            .map(|(&q, &p)| {
                if q <= T::one() - p {
                    T::zero()
                } else {
                    T::one()
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn mean(&self) -> RusTorchResult<Tensor<T>> {
        // Mean of Bernoulli is p
        self.get_probs()
    }

    fn variance(&self) -> RusTorchResult<Tensor<T>> {
        // Variance of Bernoulli is p(1-p)
        let probs = self.get_probs()?;
        let probs_data = probs.data.as_slice().unwrap();
        let var_data: Vec<T> = probs_data.iter().map(|&p| p * (T::one() - p)).collect();
        Ok(Tensor::from_vec(var_data, probs.shape().to_vec()))
    }

    fn entropy(&self) -> RusTorchResult<Tensor<T>> {
        // Entropy = -p*log(p) - (1-p)*log(1-p)
        let probs = self.get_probs()?;
        let probs_data = probs.data.as_slice().unwrap();

        let result_data: Vec<T> = probs_data
            .iter()
            .map(|&p| {
                if p == T::zero() || p == T::one() {
                    T::zero() // Entropy is 0 for deterministic outcomes
                } else {
                    -(p * p.ln() + (T::one() - p) * (T::one() - p).ln())
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, probs.shape().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bernoulli_from_probs() {
        let probs = Tensor::from_vec(vec![0.3f32, 0.7, 0.5], vec![3]);
        let bernoulli = Bernoulli::from_probs(probs, true).unwrap();

        let retrieved_probs = bernoulli.get_probs().unwrap();
        let data = retrieved_probs.data.as_slice().unwrap();
        assert_eq!(data, &[0.3, 0.7, 0.5]);
    }

    #[test]
    fn test_bernoulli_from_logits() {
        let logits = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], vec![3]);
        let bernoulli = Bernoulli::from_logits(logits, true).unwrap();

        let probs = bernoulli.get_probs().unwrap();
        let probs_data = probs.data.as_slice().unwrap();

        // logit = 0 → p = 0.5
        assert_abs_diff_eq!(probs_data[0], 0.5, epsilon = 1e-6);

        // logit = 1 → p = σ(1) = 1/(1+e^(-1)) ≈ 0.731
        assert_abs_diff_eq!(probs_data[1], 0.731, epsilon = 0.01);

        // logit = -1 → p = σ(-1) ≈ 0.269
        assert_abs_diff_eq!(probs_data[2], 0.269, epsilon = 0.01);
    }

    #[test]
    fn test_fair_coin() {
        let coin = Bernoulli::<f32>::fair_coin(true).unwrap();
        let probs = coin.get_probs().unwrap();
        assert_eq!(probs.data.as_slice().unwrap()[0], 0.5);
    }

    #[test]
    fn test_bernoulli_sampling() {
        let bernoulli = Bernoulli::<f32>::fair_coin(true).unwrap();
        let samples = bernoulli.sample(Some(&[1000])).unwrap();

        assert_eq!(samples.shape(), &[1000]);

        let data = samples.data.as_slice().unwrap();
        for &val in data {
            assert!(val == 0.0 || val == 1.0); // Only 0 or 1
        }

        // Check approximate proportion for fair coin
        // Use relaxed tolerance for statistical variance
        let sum: f32 = data.iter().sum();
        let proportion = sum / data.len() as f32;
        assert_abs_diff_eq!(proportion, 0.5, epsilon = 0.15);
    }

    #[test]
    fn test_bernoulli_log_prob() {
        let probs = Tensor::from_vec(vec![0.5f32, 0.5f32], vec![2]); // Match input shape
        let bernoulli = Bernoulli::from_probs(probs, true).unwrap();
        let values = Tensor::from_vec(vec![0.0f32, 1.0], vec![2]);

        let log_probs = bernoulli.log_prob(&values).unwrap();
        let log_prob_data = log_probs.data.as_slice().unwrap();

        // For fair coin, log P(0) = log P(1) = log(0.5) ≈ -0.693
        assert_abs_diff_eq!(log_prob_data[0], -0.693, epsilon = 0.01);
        assert_abs_diff_eq!(log_prob_data[1], -0.693, epsilon = 0.01);
    }

    #[test]
    fn test_bernoulli_mean_variance() {
        let probs = Tensor::from_vec(vec![0.3f32], vec![1]);
        let bernoulli = Bernoulli::from_probs(probs, true).unwrap();

        let mean = bernoulli.mean().unwrap();
        let variance = bernoulli.variance().unwrap();

        assert_eq!(mean.data.as_slice().unwrap()[0], 0.3);
        assert_abs_diff_eq!(
            variance.data.as_slice().unwrap()[0],
            0.3 * 0.7,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_invalid_probability() {
        let invalid_probs = Tensor::from_vec(vec![-0.1f32], vec![1]);
        assert!(Bernoulli::from_probs(invalid_probs, true).is_err());

        let invalid_probs2 = Tensor::from_vec(vec![1.5f32], vec![1]);
        assert!(Bernoulli::from_probs(invalid_probs2, true).is_err());
    }

    #[test]
    fn test_binary_cross_entropy() {
        let probs = Tensor::from_vec(vec![0.8f32, 0.2], vec![2]);
        let bernoulli = Bernoulli::from_probs(probs, true).unwrap();

        let targets = Tensor::from_vec(vec![1.0f32, 0.0], vec![2]);
        let bce = bernoulli.binary_cross_entropy(&targets).unwrap();
        let bce_data = bce.data.as_slice().unwrap();

        // BCE for perfect prediction should be low
        assert!(bce_data[0] < 0.5); // p=0.8, target=1
        assert!(bce_data[1] < 0.5); // p=0.2, target=0
    }
}
