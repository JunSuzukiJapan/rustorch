use crate::distributions::{Distribution, DistributionError, DistributionTrait, DistributionUtils};
/// Exponential Distribution - torch.distributions.Exponential compatible
/// 指数分布 - torch.distributions.Exponential互換
///
/// Implements the exponential distribution with rate parameter λ.
/// レートパラメータλを持つ指数分布を実装
use crate::tensor::Tensor;
use num_traits::Float;
use std::marker::PhantomData;

/// Exponential Distribution
/// 指数分布
///
/// The exponential distribution is defined by:
/// - rate (λ): rate parameter (must be positive)
///
/// PDF: f(x) = λ * exp(-λx) for x ≥ 0, 0 otherwise
#[derive(Debug, Clone)]
pub struct Exponential<T: Float> {
    /// Rate parameter (λ)
    /// レートパラメータ (λ)
    pub rate: Tensor<T>,

    /// Base distribution properties
    /// 基本分布特性
    pub base: Distribution,

    /// Phantom data for type parameter
    /// 型パラメータ用のファントムデータ
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> Exponential<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    /// Create a new Exponential distribution
    /// 新しい指数分布を作成
    ///
    /// # Arguments
    /// * `rate` - Rate (λ) parameter tensor
    /// * `validate_args` - Whether to validate parameters
    pub fn new(rate: Tensor<T>, validate_args: bool) -> Result<Self, DistributionError> {
        if validate_args {
            DistributionUtils::validate_positive(&rate, "rate")?;
        }

        let batch_shape = rate.shape().to_vec();
        let event_shape = vec![]; // Exponential is a univariate distribution

        Ok(Self {
            rate,
            base: Distribution::new(batch_shape, event_shape, validate_args),
            _phantom: PhantomData,
        })
    }

    /// Create Exponential distribution with scalar rate
    /// スカラーレートで指数分布を作成
    pub fn from_scalar(rate: T, validate_args: bool) -> Result<Self, DistributionError> {
        let rate_tensor = Tensor::from_vec(vec![rate], vec![]);
        Self::new(rate_tensor, validate_args)
    }

    /// Standard exponential distribution (λ=1)
    /// 標準指数分布 (λ=1)
    pub fn standard(validate_args: bool) -> Result<Self, DistributionError> {
        Self::from_scalar(T::one(), validate_args)
    }

    /// Create exponential distribution from scale parameter (scale = 1/rate)
    /// スケールパラメータから指数分布を作成 (scale = 1/rate)
    pub fn from_scale(scale: Tensor<T>, validate_args: bool) -> Result<Self, DistributionError> {
        if validate_args {
            DistributionUtils::validate_positive(&scale, "scale")?;
        }

        // Convert scale to rate: rate = 1/scale
        let scale_data = scale.data.as_slice().unwrap();
        let rate_data: Vec<T> = scale_data.iter().map(|&s| T::one() / s).collect();
        let rate_tensor = Tensor::from_vec(rate_data, scale.shape().to_vec());

        Self::new(rate_tensor, false) // validation already done
    }

    /// Create exponential distribution from scalar scale parameter
    /// スカラースケールパラメータから指数分布を作成
    pub fn from_scale_scalar(scale: T, validate_args: bool) -> Result<Self, DistributionError> {
        let scale_tensor = Tensor::from_vec(vec![scale], vec![]);
        Self::from_scale(scale_tensor, validate_args)
    }

    /// Get the scale parameter (1/rate)
    /// スケールパラメータ（1/rate）を取得
    pub fn scale(&self) -> Tensor<T> {
        let rate_data = self.rate.data.as_slice().unwrap();
        let scale_data: Vec<T> = rate_data.iter().map(|&r| T::one() / r).collect();
        Tensor::from_vec(scale_data, self.rate.shape().to_vec())
    }
}

impl<T: Float + 'static> DistributionTrait<T> for Exponential<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    fn sample(&self, shape: Option<&[usize]>) -> Result<Tensor<T>, DistributionError> {
        let sample_shape = self.base.expand_shape(shape);

        // Use inverse transform sampling: X = -ln(1-U)/λ where U ~ Uniform(0,1)
        // For numerical stability, use X = -ln(U)/λ since 1-U has same distribution as U
        let uniform_samples = DistributionUtils::random_uniform::<T>(&sample_shape);
        let uniform_data = uniform_samples.data.as_slice().unwrap();
        let rate_data = self.rate.data.as_slice().unwrap();

        let result_data: Vec<T> = uniform_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&u, &lambda)| {
                // Handle edge case where u might be exactly 0
                let safe_u = if u <= T::zero() {
                    T::from(1e-10).unwrap()
                } else {
                    u
                };
                -safe_u.ln() / lambda
            })
            .collect();

        Ok(Tensor::from_vec(result_data, sample_shape))
    }

    fn log_prob(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        let value_data = value.data.as_slice().unwrap();
        let rate_data = self.rate.data.as_slice().unwrap();

        let neg_inf = T::neg_infinity();
        let result_data: Vec<T> = value_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&x, &lambda)| {
                if x < T::zero() {
                    neg_inf
                } else {
                    // log p(x) = log(λ) - λx
                    lambda.ln() - lambda * x
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn cdf(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        let value_data = value.data.as_slice().unwrap();
        let rate_data = self.rate.data.as_slice().unwrap();

        let result_data: Vec<T> = value_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&x, &lambda)| {
                if x < T::zero() {
                    T::zero()
                } else {
                    // CDF = 1 - exp(-λx)
                    T::one() - (-lambda * x).exp()
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn icdf(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        let value_data = value.data.as_slice().unwrap();
        let rate_data = self.rate.data.as_slice().unwrap();

        let result_data: Vec<T> = value_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&p, &lambda)| {
                if p <= T::zero() {
                    T::zero()
                } else if p >= T::one() {
                    T::infinity()
                } else {
                    // ICDF = -ln(1-p)/λ
                    -(T::one() - p).ln() / lambda
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn mean(&self) -> Result<Tensor<T>, DistributionError> {
        // Mean = 1/λ
        Ok(self.scale())
    }

    fn variance(&self) -> Result<Tensor<T>, DistributionError> {
        // Variance = 1/λ²
        let rate_data = self.rate.data.as_slice().unwrap();
        let var_data: Vec<T> = rate_data
            .iter()
            .map(|&lambda| T::one() / (lambda * lambda))
            .collect();

        Ok(Tensor::from_vec(var_data, self.rate.shape().to_vec()))
    }

    fn entropy(&self) -> Result<Tensor<T>, DistributionError> {
        // Entropy = 1 - log(λ)
        let rate_data = self.rate.data.as_slice().unwrap();
        let entropy_data: Vec<T> = rate_data
            .iter()
            .map(|&lambda| T::one() - lambda.ln())
            .collect();

        Ok(Tensor::from_vec(entropy_data, self.rate.shape().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_exponential_creation() {
        let rate = Tensor::from_vec([2.0f32].to_vec(), vec![1]);

        let exp_dist = Exponential::new(rate, true).unwrap();
        assert_eq!(exp_dist.base.batch_shape, vec![1]);
        assert_eq!(exp_dist.base.event_shape, Vec::<usize>::new());
    }

    #[test]
    fn test_standard_exponential() {
        let exp_dist = Exponential::<f32>::standard(true).unwrap();
        let mean = exp_dist.mean().unwrap();
        let var = exp_dist.variance().unwrap();

        // For λ=1: mean = 1, variance = 1
        assert_abs_diff_eq!(mean.data.as_slice().unwrap()[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(var.data.as_slice().unwrap()[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_exponential_from_scale() {
        let exp_dist = Exponential::<f32>::from_scale_scalar(2.0, true).unwrap();
        let mean = exp_dist.mean().unwrap();
        let var = exp_dist.variance().unwrap();

        // For scale=2 (rate=0.5): mean = 2, variance = 4
        assert_abs_diff_eq!(mean.data.as_slice().unwrap()[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(var.data.as_slice().unwrap()[0], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_exponential_sampling() {
        let exp_dist = Exponential::<f32>::from_scalar(1.0, true).unwrap();
        let samples = exp_dist.sample(Some(&[1000])).unwrap();

        assert_eq!(samples.shape(), &[1000]);

        // Check all samples are non-negative
        let data = samples.data.as_slice().unwrap();
        for &x in data {
            assert!(x >= 0.0);
        }

        // Basic statistical test - for λ=1, mean should be ≈ 1
        // Use relaxed tolerance for statistical variance
        let sample_mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert_abs_diff_eq!(sample_mean, 1.0, epsilon = 0.15);
    }

    #[test]
    fn test_exponential_log_prob() {
        let exp_dist = Exponential::<f32>::from_scalar(2.0, true).unwrap();
        let values = Tensor::from_vec([-1.0f32, 0.0, 1.0, 2.0].to_vec(), vec![4]);

        let log_probs = exp_dist.log_prob(&values).unwrap();
        let log_prob_data = log_probs.data.as_slice().unwrap();

        // For λ=2: log p(x) = log(2) - 2x for x ≥ 0
        assert_eq!(log_prob_data[0], f32::NEG_INFINITY); // x = -1 < 0
        assert_abs_diff_eq!(log_prob_data[1], 2.0f32.ln(), epsilon = 1e-6); // x = 0
        assert_abs_diff_eq!(log_prob_data[2], 2.0f32.ln() - 2.0, epsilon = 1e-6); // x = 1
        assert_abs_diff_eq!(log_prob_data[3], 2.0f32.ln() - 4.0, epsilon = 1e-6);
        // x = 2
    }

    #[test]
    fn test_exponential_cdf() {
        let exp_dist = Exponential::<f32>::from_scalar(1.0, true).unwrap();
        let values = Tensor::from_vec([-1.0f32, 0.0, 1.0, f32::INFINITY].to_vec(), vec![4]);

        let cdf_vals = exp_dist.cdf(&values).unwrap();
        let cdf_data = cdf_vals.data.as_slice().unwrap();

        // For λ=1: CDF = 1 - exp(-x) for x ≥ 0
        assert_eq!(cdf_data[0], 0.0); // x = -1 < 0
        assert_eq!(cdf_data[1], 0.0); // x = 0
        assert_abs_diff_eq!(cdf_data[2], 1.0 - (-1.0f32).exp(), epsilon = 1e-6); // x = 1
        assert_eq!(cdf_data[3], 1.0); // x = ∞
    }

    #[test]
    fn test_exponential_icdf() {
        let exp_dist = Exponential::<f32>::from_scalar(2.0, true).unwrap();
        let probs = Tensor::from_vec([0.0f32, 0.5, 0.75, 1.0].to_vec(), vec![4]);

        let icdf_vals = exp_dist.icdf(&probs).unwrap();
        let icdf_data = icdf_vals.data.as_slice().unwrap();

        // For λ=2: ICDF = -ln(1-p)/2
        assert_eq!(icdf_data[0], 0.0); // p = 0
        assert_abs_diff_eq!(icdf_data[1], -0.5f32.ln() / 2.0, epsilon = 1e-6); // p = 0.5
        assert_abs_diff_eq!(icdf_data[2], -0.25f32.ln() / 2.0, epsilon = 1e-6); // p = 0.75
        assert_eq!(icdf_data[3], f32::INFINITY); // p = 1.0
    }

    #[test]
    fn test_exponential_properties() {
        let rate = 3.0f32;
        let exp_dist = Exponential::from_scalar(rate, true).unwrap();

        let mean = exp_dist.mean().unwrap();
        let var = exp_dist.variance().unwrap();
        let entropy = exp_dist.entropy().unwrap();

        // For rate λ: mean = 1/λ, variance = 1/λ², entropy = 1 - log(λ)
        assert_abs_diff_eq!(mean.data.as_slice().unwrap()[0], 1.0 / rate, epsilon = 1e-6);
        assert_abs_diff_eq!(
            var.data.as_slice().unwrap()[0],
            1.0 / (rate * rate),
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            entropy.data.as_slice().unwrap()[0],
            1.0 - rate.ln(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_memoryless_property() {
        // Test the memoryless property: P(X > s+t | X > s) = P(X > t)
        let exp_dist = Exponential::<f32>::from_scalar(1.0, true).unwrap();

        let s = 2.0f32;
        let t = 1.0f32;

        let s_tensor = Tensor::from_vec([s].to_vec(), vec![1]);
        let t_tensor = Tensor::from_vec([t].to_vec(), vec![1]);
        let st_tensor = Tensor::from_vec([s + t].to_vec(), vec![1]);

        let cdf_s = exp_dist.cdf(&s_tensor).unwrap();
        let cdf_t = exp_dist.cdf(&t_tensor).unwrap();
        let cdf_st = exp_dist.cdf(&st_tensor).unwrap();

        let prob_gt_s = 1.0 - cdf_s.data.as_slice().unwrap()[0];
        let prob_gt_t = 1.0 - cdf_t.data.as_slice().unwrap()[0];
        let prob_gt_st = 1.0 - cdf_st.data.as_slice().unwrap()[0];

        // P(X > s+t | X > s) = P(X > s+t) / P(X > s) should equal P(X > t)
        let conditional_prob = prob_gt_st / prob_gt_s;
        assert_abs_diff_eq!(conditional_prob, prob_gt_t, epsilon = 1e-6);
    }

    #[test]
    fn test_invalid_rate() {
        // Test negative and zero rates
        assert!(Exponential::<f32>::from_scalar(-1.0, true).is_err());
        assert!(Exponential::<f32>::from_scalar(0.0, true).is_err());
    }
}
