use crate::distributions::{Distribution, DistributionTrait, DistributionUtils};
use crate::error::{RusTorchError, RusTorchResult};
/// Beta Distribution - torch.distributions.Beta compatible
/// ベータ分布 - torch.distributions.Beta互換
///
/// Implements the Beta distribution with concentration parameters α and β.
/// 濃度パラメータαとβを持つベータ分布を実装
use crate::tensor::Tensor;
use num_traits::Float;
use std::marker::PhantomData;

/// Beta Distribution
/// ベータ分布
///
/// The Beta distribution is defined by:
/// - concentration1 (α): first shape parameter (must be positive)
/// - concentration0 (β): second shape parameter (must be positive)
///
/// PDF: f(x) = (Γ(α+β) / (Γ(α)Γ(β))) * x^(α-1) * (1-x)^(β-1) for x ∈ \[0,1\]
#[derive(Debug, Clone)]
pub struct Beta<T: Float> {
    /// First concentration parameter (α)
    /// 第一濃度パラメータ (α)
    pub concentration1: Tensor<T>,

    /// Second concentration parameter (β)
    /// 第二濃度パラメータ (β)
    pub concentration0: Tensor<T>,

    /// Base distribution properties
    /// 基本分布特性
    pub base: Distribution,

    /// Phantom data for type parameter
    /// 型パラメータ用のファントムデータ
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> Beta<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    /// Create a new Beta distribution
    /// 新しいベータ分布を作成
    ///
    /// # Arguments
    /// * `concentration1` - First concentration (α) parameter tensor
    /// * `concentration0` - Second concentration (β) parameter tensor
    /// * `validate_args` - Whether to validate parameters
    pub fn new(
        concentration1: Tensor<T>,
        concentration0: Tensor<T>,
        validate_args: bool,
    ) -> RusTorchResult<Self> {
        if validate_args {
            DistributionUtils::validate_positive(&concentration1, "concentration1")?;
            DistributionUtils::validate_positive(&concentration0, "concentration0")?;
        }

        // Determine batch shape from broadcasting concentration parameters
        let batch_shape =
            Distribution::broadcast_shapes(concentration1.shape(), concentration0.shape())?;
        let event_shape = vec![]; // Beta is a univariate distribution

        Ok(Self {
            concentration1,
            concentration0,
            base: Distribution::new(batch_shape, event_shape, validate_args),
            _phantom: PhantomData,
        })
    }

    /// Create Beta distribution with scalar parameters
    /// スカラーパラメータでベータ分布を作成
    pub fn from_scalars(alpha: T, beta: T, validate_args: bool) -> RusTorchResult<Self> {
        let alpha_tensor = Tensor::from_vec(vec![alpha], vec![]);
        let beta_tensor = Tensor::from_vec(vec![beta], vec![]);
        Self::new(alpha_tensor, beta_tensor, validate_args)
    }

    /// Uniform Beta distribution (α=1, β=1) - equivalent to Uniform(0,1)
    /// 一様ベータ分布 (α=1, β=1) - Uniform(0,1)と等価
    pub fn uniform(validate_args: bool) -> RusTorchResult<Self> {
        Self::from_scalars(T::one(), T::one(), validate_args)
    }

    /// Symmetric Beta distribution with equal parameters (α=β)
    /// 等しいパラメータを持つ対称ベータ分布 (α=β)
    pub fn symmetric(concentration: T, validate_args: bool) -> RusTorchResult<Self> {
        Self::from_scalars(concentration, concentration, validate_args)
    }

    /// Log-beta function: log(B(α, β)) = log(Γ(α)) + log(Γ(β)) - log(Γ(α+β))
    /// ログベータ関数: log(B(α, β)) = log(Γ(α)) + log(Γ(β)) - log(Γ(α+β))
    fn log_beta(&self) -> RusTorchResult<Tensor<T>> {
        let alpha_data = self.concentration1.data.as_slice().unwrap();
        let beta_data = self.concentration0.data.as_slice().unwrap();

        let result_data: Vec<T> = alpha_data
            .iter()
            .zip(beta_data.iter().cycle())
            .map(|(&a, &b)| {
                // Use Stirling's approximation for log-gamma
                Self::log_gamma_stirling(a) + Self::log_gamma_stirling(b)
                    - Self::log_gamma_stirling(a + b)
            })
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            self.concentration1.shape().to_vec(),
        ))
    }

    /// Log-gamma function using Stirling's approximation
    /// スターリング近似を使ったログガンマ関数
    fn log_gamma_stirling(x: T) -> T {
        if x < T::one() {
            // For x < 1, use recurrence relation: Γ(x) = Γ(x+1) / x
            Self::log_gamma_stirling(x + T::one()) - x.ln()
        } else if x < T::from(12.0).unwrap() {
            // Use series expansion for better accuracy
            let mut result = T::zero();
            let mut term = x;

            // Use recurrence to get to x >= 12
            while term < T::from(12.0).unwrap() {
                result = result - term.ln();
                term = term + T::one();
            }

            result + Self::stirling_approx(term)
        } else {
            Self::stirling_approx(x)
        }
    }

    /// Stirling's approximation for log(Γ(x))
    /// log(Γ(x))のスターリング近似
    fn stirling_approx(x: T) -> T {
        let half = T::from(0.5).unwrap();
        let two = T::from(2.0).unwrap();
        let pi = T::from(std::f64::consts::PI).unwrap();
        let twelve = T::from(12.0).unwrap();

        // Stirling: log(Γ(x)) ≈ (x - 0.5) * log(x) - x + 0.5 * log(2π) + 1/(12x)
        (x - half) * x.ln() - x + half * (two * pi).ln() + T::one() / (twelve * x)
    }

    /// Digamma function (ψ) - derivative of log-gamma
    /// ディガンマ関数 (ψ) - ログガンマの導関数
    fn digamma(x: T) -> T {
        if x < T::from(6.0).unwrap() {
            // For small x, use recurrence relation
            Self::digamma(x + T::one()) - T::one() / x
        } else {
            // Asymptotic expansion for large x
            let inv_x = T::one() / x;
            let inv_x_sq = inv_x * inv_x;

            x.ln() - T::from(0.5).unwrap() * inv_x - inv_x_sq / T::from(12.0).unwrap()
                + inv_x_sq * inv_x_sq / T::from(120.0).unwrap()
        }
    }

    /// Sample using rejection sampling (for general case)
    /// 棄却サンプリングを使用してサンプル生成（一般的なケース）
    fn sample_rejection(&self, shape: &[usize]) -> RusTorchResult<Tensor<T>> {
        let total_size: usize = shape.iter().product();
        let mut samples = Vec::with_capacity(total_size);

        let alpha_data = self.concentration1.data.as_slice().unwrap();
        let beta_data = self.concentration0.data.as_slice().unwrap();

        for i in 0..total_size {
            let alpha = alpha_data[i % alpha_data.len()];
            let beta = beta_data[i % beta_data.len()];

            // For beta distribution, we can use gamma sampling: X = G1/(G1+G2)
            // where G1 ~ Gamma(α,1) and G2 ~ Gamma(β,1)
            let g1 = Self::sample_gamma_marsaglia(alpha);
            let g2 = Self::sample_gamma_marsaglia(beta);

            samples.push(g1 / (g1 + g2));
        }

        Ok(Tensor::from_vec(samples, shape.to_vec()))
    }

    /// Sample from Gamma distribution using Marsaglia-Tsang method
    /// マルサグリア・ツァング法を使ってガンマ分布からサンプル
    fn sample_gamma_marsaglia(shape: T) -> T {
        if shape < T::one() {
            // For shape < 1, use transformation: if Y ~ Gamma(shape+1), then Y * U^(1/shape) ~ Gamma(shape)
            let y = Self::sample_gamma_marsaglia(shape + T::one());
            let u = DistributionUtils::random_uniform_scalar::<T>();
            y * u.powf(T::one() / shape)
        } else {
            // Marsaglia-Tsang algorithm for shape >= 1
            #[allow(clippy::many_single_char_names)]
            // Mathematical variables d, c, x, v, u are conventional in statistical algorithms
            {
                let d = shape - T::from(1.0 / 3.0).unwrap();
                let c = T::one() / (T::from(9.0).unwrap() * d).sqrt();

                loop {
                    let mut x = DistributionUtils::random_normal_scalar::<T>();
                    let v = (T::one() + c * x).powi(3);

                    if v <= T::zero() {
                        continue;
                    }

                    x = x * x;
                    let u = DistributionUtils::random_uniform_scalar::<T>();

                    if u < T::one() - T::from(0.0331).unwrap() * x * x {
                        return d * v;
                    }

                    if u.ln() < T::from(0.5).unwrap() * x + d * (T::one() - v + v.ln()) {
                        return d * v;
                    }
                }
            }
        }
    }
}

impl<T: Float + 'static> DistributionTrait<T> for Beta<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    fn sample(&self, shape: Option<&[usize]>) -> RusTorchResult<Tensor<T>> {
        let sample_shape = self.base.expand_shape(shape);
        self.sample_rejection(&sample_shape)
    }

    fn log_prob(&self, value: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        let value_data = value.data.as_slice().unwrap();
        let alpha_data = self.concentration1.data.as_slice().unwrap();
        let beta_data = self.concentration0.data.as_slice().unwrap();

        let log_beta_vals = self.log_beta()?;
        let log_beta_data = log_beta_vals.data.as_slice().unwrap();

        let neg_inf = T::neg_infinity();
        let result_data: Vec<T> = value_data
            .iter()
            .zip(alpha_data.iter().cycle())
            .zip(beta_data.iter().cycle())
            .zip(log_beta_data.iter().cycle())
            .map(|(((&v, &a), &b), &lb)| {
                if v <= T::zero() || v >= T::one() {
                    neg_inf
                } else {
                    // log p(x) = (α-1)*log(x) + (β-1)*log(1-x) - log(B(α,β))
                    (a - T::one()) * v.ln() + (b - T::one()) * (T::one() - v).ln() - lb
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn cdf(&self, value: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        let value_data = value.data.as_slice().unwrap();
        let alpha_data = self.concentration1.data.as_slice().unwrap();
        let beta_data = self.concentration0.data.as_slice().unwrap();

        let result_data: Vec<T> = value_data
            .iter()
            .zip(alpha_data.iter().cycle())
            .zip(beta_data.iter().cycle())
            .map(|((&v, &a), &b)| {
                if v <= T::zero() {
                    T::zero()
                } else if v >= T::one() {
                    T::one()
                } else {
                    // Use incomplete beta function approximation
                    Self::incomplete_beta_approx(v, a, b)
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn icdf(&self, _value: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        // ICDF for Beta distribution requires numerical methods
        Err(RusTorchError::UnsupportedOperation(
            "ICDF for Beta distribution not implemented - requires numerical methods",
        ))
    }

    fn mean(&self) -> RusTorchResult<Tensor<T>> {
        // Mean = α / (α + β)
        let alpha_data = self.concentration1.data.as_slice().unwrap();
        let beta_data = self.concentration0.data.as_slice().unwrap();

        let mean_data: Vec<T> = alpha_data
            .iter()
            .zip(beta_data.iter().cycle())
            .map(|(&a, &b)| a / (a + b))
            .collect();

        Ok(Tensor::from_vec(
            mean_data,
            self.concentration1.shape().to_vec(),
        ))
    }

    fn variance(&self) -> RusTorchResult<Tensor<T>> {
        // Variance = (α*β) / ((α+β)²*(α+β+1))
        let alpha_data = self.concentration1.data.as_slice().unwrap();
        let beta_data = self.concentration0.data.as_slice().unwrap();

        let var_data: Vec<T> = alpha_data
            .iter()
            .zip(beta_data.iter().cycle())
            .map(|(&a, &b)| {
                let sum = a + b;
                (a * b) / (sum * sum * (sum + T::one()))
            })
            .collect();

        Ok(Tensor::from_vec(
            var_data,
            self.concentration1.shape().to_vec(),
        ))
    }

    fn entropy(&self) -> RusTorchResult<Tensor<T>> {
        let alpha_data = self.concentration1.data.as_slice().unwrap();
        let beta_data = self.concentration0.data.as_slice().unwrap();
        let log_beta_vals = self.log_beta()?;
        let log_beta_data = log_beta_vals.data.as_slice().unwrap();

        let result_data: Vec<T> = alpha_data
            .iter()
            .zip(beta_data.iter().cycle())
            .zip(log_beta_data.iter().cycle())
            .map(|((&a, &b), &lb)| {
                let sum = a + b;
                // Entropy = log(B(α,β)) - (α-1)*ψ(α) - (β-1)*ψ(β) + (α+β-2)*ψ(α+β)
                lb - (a - T::one()) * Self::digamma(a) - (b - T::one()) * Self::digamma(b)
                    + (sum - T::from(2.0).unwrap()) * Self::digamma(sum)
            })
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            self.concentration1.shape().to_vec(),
        ))
    }
}

impl<T: Float + 'static> Beta<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    /// Incomplete beta function approximation
    /// 不完全ベータ関数近似
    fn incomplete_beta_approx(x: T, a: T, b: T) -> T {
        // Simple continued fraction approximation for incomplete beta
        let mut cf = T::one();
        let mut d = T::one() / (T::one() - (a + b) * x / (a + T::one()));
        cf = cf * d;

        for m in 1..50 {
            let m_f = T::from(m).unwrap();
            let two_m = T::from(2 * m).unwrap();

            let num = m_f * (b - m_f) * x / ((a + two_m - T::one()) * (a + two_m));
            d = T::one() / (T::one() + num * d);
            cf = cf * d;

            let num2 = -(a + m_f) * (a + b + m_f) * x / ((a + two_m) * (a + two_m + T::one()));
            d = T::one() / (T::one() + num2 * d);
            cf = cf * d;
        }

        let beta_approx =
            x.powf(a) * (T::one() - x).powf(b) / (a * Self::beta_function_approx(a, b));
        beta_approx * cf
    }

    /// Beta function approximation
    /// ベータ関数近似
    fn beta_function_approx(a: T, b: T) -> T {
        (Self::log_gamma_stirling(a) + Self::log_gamma_stirling(b)
            - Self::log_gamma_stirling(a + b))
        .exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_beta_creation() {
        let alpha = Tensor::from_vec([2.0f32].to_vec(), vec![1]);
        let beta = Tensor::from_vec([3.0f32].to_vec(), vec![1]);

        let distribution = Beta::new(alpha, beta, true).unwrap();
        assert_eq!(distribution.base.batch_shape, vec![1]);
        assert_eq!(distribution.base.event_shape, Vec::<usize>::new());
    }

    #[test]
    fn test_uniform_beta() {
        let beta = Beta::<f32>::uniform(true).unwrap();
        let mean = beta.mean().unwrap();
        let var = beta.variance().unwrap();

        // For Beta(1,1): mean = 1/2, variance = 1/12
        assert_abs_diff_eq!(mean.data.as_slice().unwrap()[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(var.data.as_slice().unwrap()[0], 1.0 / 12.0, epsilon = 1e-6);
    }

    #[test]
    fn test_beta_sampling() {
        let beta = Beta::<f32>::from_scalars(2.0, 5.0, true).unwrap();
        let samples = beta.sample(Some(&[1000])).unwrap();

        assert_eq!(samples.shape(), &[1000]);

        // Check all samples are in [0, 1] range
        let data = samples.data.as_slice().unwrap();
        for &x in data {
            assert!(x > 0.0 && x < 1.0);
        }

        // Basic statistical test - mean should be α/(α+β) = 2/7 ≈ 0.286
        let sample_mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert_abs_diff_eq!(sample_mean, 2.0 / 7.0, epsilon = 0.05);
    }

    #[test]
    fn test_beta_mean_variance() {
        let beta = Beta::<f32>::from_scalars(3.0, 7.0, true).unwrap();

        let mean = beta.mean().unwrap();
        let var = beta.variance().unwrap();

        // Mean = α/(α+β) = 3/10 = 0.3
        assert_abs_diff_eq!(mean.data.as_slice().unwrap()[0], 0.3, epsilon = 1e-6);

        // Variance = αβ/((α+β)²(α+β+1)) = 3*7/(10²*11) = 21/1100 = 0.019...
        let expected_var = 3.0 * 7.0 / (10.0 * 10.0 * 11.0);
        assert_abs_diff_eq!(
            var.data.as_slice().unwrap()[0],
            expected_var,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_beta_log_prob() {
        let beta = Beta::<f32>::uniform(true).unwrap(); // Beta(1,1) = Uniform(0,1)
        let values = Tensor::from_vec([-0.1f32, 0.5, 1.1].to_vec(), vec![3]);

        let log_probs = beta.log_prob(&values).unwrap();
        let log_prob_data = log_probs.data.as_slice().unwrap();

        // For Beta(1,1), log p(x) = 0 for x ∈ (0,1), -∞ otherwise
        assert_eq!(log_prob_data[0], f32::NEG_INFINITY); // x = -0.1 < 0
        assert_abs_diff_eq!(log_prob_data[1], 0.0, epsilon = 1e-5); // x = 0.5
        assert_eq!(log_prob_data[2], f32::NEG_INFINITY); // x = 1.1 > 1
    }

    #[test]
    fn test_symmetric_beta() {
        let beta = Beta::<f32>::symmetric(2.0, true).unwrap();
        let mean = beta.mean().unwrap();

        // For symmetric Beta(a,a), mean = a/(2a) = 1/2
        assert_abs_diff_eq!(mean.data.as_slice().unwrap()[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_invalid_parameters() {
        // Test negative concentration parameters
        assert!(Beta::<f32>::from_scalars(-1.0, 2.0, true).is_err());
        assert!(Beta::<f32>::from_scalars(2.0, -1.0, true).is_err());
        assert!(Beta::<f32>::from_scalars(0.0, 1.0, true).is_err());
    }
}
