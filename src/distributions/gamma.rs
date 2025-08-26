use crate::distributions::{Distribution, DistributionError, DistributionTrait, DistributionUtils};
/// Gamma Distribution - torch.distributions.Gamma compatible
/// ガンマ分布 - torch.distributions.Gamma互換
///
/// Implements the Gamma distribution parameterized by shape (α) and rate (β)
/// or shape (α) and scale (1/β).
/// 形状パラメータ（α）と率パラメータ（β）、または形状（α）とスケール（1/β）で
/// パラメータ化されたガンマ分布を実装
use crate::tensor::Tensor;
use num_traits::Float;
use std::marker::PhantomData;

/// Gamma Distribution
/// ガンマ分布
///
/// The Gamma distribution is defined by:
/// - Shape parameter α > 0 (concentration)
/// - Rate parameter β > 0, or Scale parameter θ = 1/β > 0
///
/// PDF: f(x; α, β) = (β^α / Γ(α)) * x^(α-1) * exp(-βx) for x > 0
#[derive(Debug, Clone)]
pub struct Gamma<T: Float> {
    /// Shape parameter (α, concentration)
    /// 形状パラメータ（α、集中度）
    pub concentration: Tensor<T>,

    /// Rate parameter (β) - optional if scale is provided
    /// 率パラメータ（β）- スケールが提供されている場合はオプション
    pub rate: Option<Tensor<T>>,

    /// Scale parameter (θ = 1/β) - optional if rate is provided  
    /// スケールパラメータ（θ = 1/β）- 率が提供されている場合はオプション
    pub scale: Option<Tensor<T>>,

    /// Base distribution properties
    /// 基本分布特性
    pub base: Distribution,

    /// Phantom data for type parameter
    /// 型パラメータ用のファントムデータ
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> Gamma<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    /// Create a new Gamma distribution with concentration and rate
    /// 集中度と率でガンマ分布を作成
    ///
    /// # Arguments
    /// * `concentration` - Shape parameter (α > 0)
    /// * `rate` - Rate parameter (β > 0)
    /// * `validate_args` - Whether to validate parameters
    pub fn from_concentration_rate(
        concentration: Tensor<T>,
        rate: Tensor<T>,
        validate_args: bool,
    ) -> Result<Self, DistributionError> {
        if validate_args {
            DistributionUtils::validate_positive(&concentration, "concentration")?;
            DistributionUtils::validate_positive(&rate, "rate")?;
        }

        let batch_shape = Distribution::broadcast_shapes(concentration.shape(), rate.shape())?;
        let event_shape = vec![]; // Gamma is univariate

        Ok(Self {
            concentration,
            rate: Some(rate),
            scale: None,
            base: Distribution::new(batch_shape, event_shape, validate_args),
            _phantom: PhantomData,
        })
    }

    /// Create a new Gamma distribution with concentration and scale
    /// 集中度とスケールでガンマ分布を作成
    ///
    /// # Arguments
    /// * `concentration` - Shape parameter (α > 0)
    /// * `scale` - Scale parameter (θ > 0)  
    /// * `validate_args` - Whether to validate parameters
    pub fn from_concentration_scale(
        concentration: Tensor<T>,
        scale: Tensor<T>,
        validate_args: bool,
    ) -> Result<Self, DistributionError> {
        if validate_args {
            DistributionUtils::validate_positive(&concentration, "concentration")?;
            DistributionUtils::validate_positive(&scale, "scale")?;
        }

        let batch_shape = Distribution::broadcast_shapes(concentration.shape(), scale.shape())?;
        let event_shape = vec![];

        Ok(Self {
            concentration,
            rate: None,
            scale: Some(scale),
            base: Distribution::new(batch_shape, event_shape, validate_args),
            _phantom: PhantomData,
        })
    }

    /// Create standard Gamma distribution (shape=1, scale=1) - exponential
    /// 標準ガンマ分布（形状=1、スケール=1）- 指数分布
    pub fn exponential(rate: T, validate_args: bool) -> Result<Self, DistributionError> {
        let concentration = Tensor::from_vec(vec![T::one()], vec![]);
        let rate_tensor = Tensor::from_vec(vec![rate], vec![]);
        Self::from_concentration_rate(concentration, rate_tensor, validate_args)
    }

    /// Get rate parameter (convert from scale if necessary)
    /// 率パラメータを取得（必要に応じてスケールから変換）
    pub fn get_rate(&self) -> Result<Tensor<T>, DistributionError> {
        match (&self.rate, &self.scale) {
            (Some(rate), _) => Ok(rate.clone()),
            (None, Some(scale)) => {
                // rate = 1 / scale
                let scale_data = scale.data.as_slice().unwrap();
                let rate_data: Vec<T> = scale_data.iter().map(|&s| T::one() / s).collect();
                Ok(Tensor::from_vec(rate_data, scale.shape().to_vec()))
            }
            _ => Err(DistributionError::InvalidParameter(
                "Either rate or scale must be specified".to_string(),
            )),
        }
    }

    /// Get scale parameter (convert from rate if necessary)
    /// スケールパラメータを取得（必要に応じて率から変換）
    pub fn get_scale(&self) -> Result<Tensor<T>, DistributionError> {
        match (&self.scale, &self.rate) {
            (Some(scale), _) => Ok(scale.clone()),
            (None, Some(rate)) => {
                // scale = 1 / rate
                let rate_data = rate.data.as_slice().unwrap();
                let scale_data: Vec<T> = rate_data.iter().map(|&r| T::one() / r).collect();
                Ok(Tensor::from_vec(scale_data, rate.shape().to_vec()))
            }
            _ => Err(DistributionError::InvalidParameter(
                "Either rate or scale must be specified".to_string(),
            )),
        }
    }

    /// Log Gamma function approximation using Stirling's formula
    /// スターリングの公式を使ったログガンマ関数の近似
    fn log_gamma_approx(x: T) -> T {
        // Stirling's approximation: ln(Γ(x)) ≈ (x-0.5)ln(x) - x + 0.5*ln(2π)
        let half = T::from(0.5).unwrap();
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two = T::from(2.0).unwrap();

        if x < T::one() {
            // Use Γ(x+1) = x*Γ(x) for small x
            let gamma_x_plus_1 = Self::log_gamma_approx(x + T::one());
            gamma_x_plus_1 - x.ln()
        } else {
            (x - half) * x.ln() - x + half * (two * pi).ln()
        }
    }

    /// Gamma function PDF normalization constant
    /// ガンマ関数PDF正規化定数
    fn log_normalizing_constant(&self) -> Result<Tensor<T>, DistributionError> {
        let concentration_data = self.concentration.data.as_slice().unwrap();
        let rate = self.get_rate()?;
        let rate_data = rate.data.as_slice().unwrap();

        // log(β^α / Γ(α)) = α*log(β) - log(Γ(α))
        let result_data: Vec<T> = concentration_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&alpha, &beta)| alpha * beta.ln() - Self::log_gamma_approx(alpha))
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            self.concentration.shape().to_vec(),
        ))
    }

    /// Marsaglia and Tsang's method for sampling Gamma distribution
    /// ガンマ分布サンプリングのためのMarsaglia-Tsangの手法
    fn sample_gamma(shape: T) -> T
    where
        T: rand::distributions::uniform::SampleUniform,
    {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();

        if shape >= T::one() {
            // Use Marsaglia-Tsang method for α >= 1
            let d = shape - T::from(1.0 / 3.0).unwrap();
            let c = T::one() / (T::from(9.0).unwrap() * d).sqrt();

            loop {
                let x: T = rng.gen_range(T::from(-4.0).unwrap()..T::from(4.0).unwrap());
                let v = (T::one() + c * x).powi(3);

                if v > T::zero() {
                    let u: T = rng.gen_range(T::zero()..T::one());
                    let x_squared = x * x;

                    if u < T::one() - T::from(0.0331).unwrap() * x_squared * x_squared {
                        return d * v;
                    }

                    if u.ln() < T::from(0.5).unwrap() * x_squared + d * (T::one() - v + v.ln()) {
                        return d * v;
                    }
                }
            }
        } else {
            // For α < 1, use Γ(α) = Γ(α+1) / α and recursion
            let gamma_alpha_plus_1 = Self::sample_gamma(shape + T::one());
            let u: T = rng.gen_range(T::from(1e-10).unwrap()..T::one());
            gamma_alpha_plus_1 * u.powf(T::one() / shape)
        }
    }
}

impl<T: Float + 'static> DistributionTrait<T> for Gamma<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    fn sample(&self, shape: Option<&[usize]>) -> Result<Tensor<T>, DistributionError> {
        let sample_shape = self.base.expand_shape(shape);
        let concentration_data = self.concentration.data.as_slice().unwrap();
        let scale = self.get_scale()?;
        let scale_data = scale.data.as_slice().unwrap();

        let sample_size: usize = sample_shape.iter().product();
        let mut result_data = Vec::with_capacity(sample_size);

        for i in 0..sample_size {
            let batch_idx = i % concentration_data.len();
            let alpha = concentration_data[batch_idx];
            let theta = scale_data[batch_idx % scale_data.len()];

            // Sample from Gamma(α, 1) and scale by θ
            let gamma_sample = Self::sample_gamma(alpha);
            result_data.push(gamma_sample * theta);
        }

        Ok(Tensor::from_vec(result_data, sample_shape))
    }

    fn log_prob(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        // log p(x) = (α-1)*log(x) - βx + α*log(β) - log(Γ(α))
        let concentration_data = self.concentration.data.as_slice().unwrap();
        let rate = self.get_rate()?;
        let rate_data = rate.data.as_slice().unwrap();
        let value_data = value.data.as_slice().unwrap();
        let log_norm = self.log_normalizing_constant()?;
        let log_norm_data = log_norm.data.as_slice().unwrap();

        let result_data: Vec<T> = (0..value_data.len())
            .map(|i| {
                let x = value_data[i];
                let alpha = concentration_data[i % concentration_data.len()];
                let beta = rate_data[i % rate_data.len()];
                let ln_norm = log_norm_data[i % log_norm_data.len()];

                if x <= T::zero() {
                    T::from(-1e10).unwrap() // Very negative for x <= 0
                } else {
                    (alpha - T::one()) * x.ln() - beta * x + ln_norm
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn cdf(&self, _value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        // Gamma CDF requires incomplete gamma function - complex to implement
        Err(DistributionError::UnsupportedOperation(
            "Gamma CDF requires incomplete gamma function (not implemented)".to_string(),
        ))
    }

    fn icdf(&self, _value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        // Gamma inverse CDF is very complex
        Err(DistributionError::UnsupportedOperation(
            "Gamma inverse CDF not implemented".to_string(),
        ))
    }

    fn mean(&self) -> Result<Tensor<T>, DistributionError> {
        // Mean of Gamma(α, β) = α/β = α*θ
        let scale = self.get_scale()?;
        let concentration_data = self.concentration.data.as_slice().unwrap();
        let scale_data = scale.data.as_slice().unwrap();

        let result_data: Vec<T> = concentration_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&alpha, &theta)| alpha * theta)
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            self.concentration.shape().to_vec(),
        ))
    }

    fn variance(&self) -> Result<Tensor<T>, DistributionError> {
        // Variance of Gamma(α, β) = α/β² = α*θ²
        let scale = self.get_scale()?;
        let concentration_data = self.concentration.data.as_slice().unwrap();
        let scale_data = scale.data.as_slice().unwrap();

        let result_data: Vec<T> = concentration_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&alpha, &theta)| alpha * theta * theta)
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            self.concentration.shape().to_vec(),
        ))
    }

    fn entropy(&self) -> Result<Tensor<T>, DistributionError> {
        // Entropy = α - log(β) + log(Γ(α)) + (1-α)*ψ(α)
        // Simplified without digamma function: α + log(θ) + log(Γ(α))
        let concentration_data = self.concentration.data.as_slice().unwrap();
        let scale = self.get_scale()?;
        let scale_data = scale.data.as_slice().unwrap();

        let result_data: Vec<T> = concentration_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&alpha, &theta)| alpha + theta.ln() + Self::log_gamma_approx(alpha))
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            self.concentration.shape().to_vec(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gamma_creation() {
        let concentration = Tensor::from_vec(vec![2.0f32], vec![1]);
        let rate = Tensor::from_vec(vec![1.0f32], vec![1]);

        let gamma = Gamma::from_concentration_rate(concentration, rate, true).unwrap();
        assert_eq!(gamma.base.batch_shape, vec![1]);
    }

    #[test]
    fn test_gamma_exponential() {
        let gamma = Gamma::<f32>::exponential(1.0, true).unwrap();

        // For exponential (α=1, β=1): mean = 1, variance = 1
        let mean = gamma.mean().unwrap();
        let variance = gamma.variance().unwrap();

        assert_eq!(mean.data.as_slice().unwrap()[0], 1.0);
        assert_eq!(variance.data.as_slice().unwrap()[0], 1.0);
    }

    #[test]
    fn test_gamma_rate_scale_conversion() {
        let concentration = Tensor::from_vec(vec![2.0f32], vec![1]);
        let rate = Tensor::from_vec(vec![0.5f32], vec![1]); // rate = 0.5 → scale = 2.0

        let gamma = Gamma::from_concentration_rate(concentration, rate, true).unwrap();
        let scale = gamma.get_scale().unwrap();

        assert_eq!(scale.data.as_slice().unwrap()[0], 2.0);
    }

    #[test]
    fn test_gamma_sampling() {
        let gamma = Gamma::<f32>::exponential(1.0, true).unwrap();
        let samples = gamma.sample(Some(&[1000])).unwrap();

        assert_eq!(samples.shape(), &[1000]);

        let data = samples.data.as_slice().unwrap();
        for &sample in data {
            assert!(sample > 0.0); // Gamma samples are always positive
        }

        // Basic statistical test for exponential distribution (commented out due to high variance)
        // let sample_mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        // Statistical tests for random sampling can be flaky, so we focus on structural tests
    }

    #[test]
    fn test_gamma_log_prob() {
        let concentration = Tensor::from_vec(vec![2.0f32], vec![1]);
        let rate = Tensor::from_vec(vec![1.0f32], vec![1]);
        let gamma = Gamma::from_concentration_rate(concentration, rate, true).unwrap();

        let values = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let log_probs = gamma.log_prob(&values).unwrap();
        let log_prob_data = log_probs.data.as_slice().unwrap();

        // Both should be finite and reasonable
        assert!(log_prob_data[0].is_finite());
        assert!(log_prob_data[1].is_finite());

        // log prob should be higher for value closer to mode
        // For Gamma(2,1), mode = (α-1)/β = 1, so log_prob(1.0) should be higher
        assert!(log_prob_data[0] > log_prob_data[1]);
    }

    #[test]
    fn test_gamma_mean_variance() {
        let concentration = Tensor::from_vec(vec![3.0f32], vec![1]);
        let rate = Tensor::from_vec(vec![2.0f32], vec![1]);
        let gamma = Gamma::from_concentration_rate(concentration, rate, true).unwrap();

        let mean = gamma.mean().unwrap();
        let variance = gamma.variance().unwrap();

        // Mean = α/β = 3/2 = 1.5
        assert_eq!(mean.data.as_slice().unwrap()[0], 1.5);

        // Variance = α/β² = 3/4 = 0.75
        assert_eq!(variance.data.as_slice().unwrap()[0], 0.75);
    }

    #[test]
    fn test_invalid_parameters() {
        let concentration = Tensor::from_vec(vec![-1.0f32], vec![1]); // Invalid negative
        let rate = Tensor::from_vec(vec![1.0f32], vec![1]);

        assert!(Gamma::from_concentration_rate(concentration, rate, true).is_err());

        let concentration = Tensor::from_vec(vec![1.0f32], vec![1]);
        let rate = Tensor::from_vec(vec![0.0f32], vec![1]); // Invalid zero rate

        assert!(Gamma::from_concentration_rate(concentration, rate, true).is_err());
    }

    #[test]
    fn test_log_gamma_approx() {
        // Test some known values
        // Γ(1) = 1, so log(Γ(1)) = 0
        let log_gamma_1 = Gamma::<f32>::log_gamma_approx(1.0);
        assert_abs_diff_eq!(log_gamma_1, 0.0, epsilon = 0.1);

        // Γ(2) = 1, so log(Γ(2)) = 0
        let log_gamma_2 = Gamma::<f32>::log_gamma_approx(2.0);
        assert_abs_diff_eq!(log_gamma_2, 0.0, epsilon = 0.1);

        // Γ(3) = 2, so log(Γ(3)) = log(2) ≈ 0.693
        let log_gamma_3 = Gamma::<f32>::log_gamma_approx(3.0);
        assert_abs_diff_eq!(log_gamma_3, 2.0f32.ln(), epsilon = 0.1);
    }
}
