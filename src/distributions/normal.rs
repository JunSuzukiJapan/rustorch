use crate::distributions::{Distribution, DistributionError, DistributionTrait, DistributionUtils};
/// Normal (Gaussian) Distribution - torch.distributions.Normal compatible
/// 正規（ガウス）分布 - torch.distributions.Normal互換
///
/// Implements the normal distribution with mean μ and standard deviation σ.
/// 平均μと標準偏差σを持つ正規分布を実装
use crate::tensor::Tensor;
use num_traits::Float;
use std::marker::PhantomData;

/// Normal Distribution
/// 正規分布
///
/// The normal distribution is defined by:
/// - Mean (μ): location parameter
/// - Standard deviation (σ): scale parameter (must be positive)
///
/// PDF: f(x) = (1 / (σ√(2π))) * exp(-0.5 * ((x - μ) / σ)²)
#[derive(Debug, Clone)]
pub struct Normal<T: Float> {
    /// Mean parameter (μ)
    /// 平均パラメータ (μ)
    pub loc: Tensor<T>,

    /// Standard deviation parameter (σ)  
    /// 標準偏差パラメータ (σ)
    pub scale: Tensor<T>,

    /// Base distribution properties
    /// 基本分布特性
    pub base: Distribution,

    /// Phantom data for type parameter
    /// 型パラメータ用のファントムデータ
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> Normal<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    /// Create a new Normal distribution
    /// 新しい正規分布を作成
    ///
    /// # Arguments
    /// * `loc` - Mean (μ) parameter tensor
    /// * `scale` - Standard deviation (σ) parameter tensor  
    /// * `validate_args` - Whether to validate parameters
    pub fn new(
        loc: Tensor<T>,
        scale: Tensor<T>,
        validate_args: bool,
    ) -> Result<Self, DistributionError> {
        if validate_args {
            DistributionUtils::validate_positive(&scale, "scale")?;
        }

        // Determine batch shape from broadcasting loc and scale
        let batch_shape = Distribution::broadcast_shapes(loc.shape(), scale.shape())?;
        let event_shape = vec![]; // Normal is a univariate distribution

        Ok(Self {
            loc,
            scale,
            base: Distribution::new(batch_shape, event_shape, validate_args),
            _phantom: PhantomData,
        })
    }

    /// Create Normal distribution with scalar parameters
    /// スカラーパラメータで正規分布を作成
    pub fn from_scalars(loc: T, scale: T, validate_args: bool) -> Result<Self, DistributionError> {
        let loc_tensor = Tensor::from_vec(vec![loc], vec![]);
        let scale_tensor = Tensor::from_vec(vec![scale], vec![]);
        Self::new(loc_tensor, scale_tensor, validate_args)
    }

    /// Standard normal distribution (μ=0, σ=1)
    /// 標準正規分布 (μ=0, σ=1)
    pub fn standard(validate_args: bool) -> Result<Self, DistributionError> {
        Self::from_scalars(T::zero(), T::one(), validate_args)
    }

    /// Compute the log of the normalization constant
    /// 正規化定数の対数を計算
    fn log_normalizing_constant(&self) -> Result<Tensor<T>, DistributionError> {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two = T::from(2.0).unwrap();

        // log(σ√(2π)) = log(σ) + 0.5 * log(2π)
        let scale_data = self.scale.data.as_slice().unwrap();
        let log_scale: Vec<T> = scale_data.iter().map(|&s| s.ln()).collect();

        let half_log_2pi = T::from(0.5).unwrap() * (two * pi).ln();
        let result_data: Vec<T> = log_scale.iter().map(|&ls| ls + half_log_2pi).collect();

        Ok(Tensor::from_vec(result_data, self.scale.shape().to_vec()))
    }

    /// Compute standardized values (z-scores)
    /// 標準化された値（zスコア）を計算
    fn standardize(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        // z = (x - μ) / σ
        let loc_data = self.loc.data.as_slice().unwrap();
        let scale_data = self.scale.data.as_slice().unwrap();
        let value_data = value.data.as_slice().unwrap();

        let result_data: Vec<T> = value_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&v, &l), &s)| (v - l) / s)
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    /// Error function approximation for CDF calculation
    /// CDF計算のための誤差関数近似
    fn erf_approx(x: T) -> T {
        // Abramowitz and Stegun approximation
        let a1 = T::from(0.254_829_592).unwrap();
        let a2 = T::from(-0.284_496_736).unwrap();
        let a3 = T::from(1.421_413_741).unwrap();
        let a4 = T::from(-1.453_152_027).unwrap();
        let a5 = T::from(1.061_405_429).unwrap();
        let p = T::from(0.327_591_1).unwrap();

        let sign = if x >= T::zero() {
            T::one()
        } else {
            T::from(-1.0).unwrap()
        };
        let x_abs = x.abs();

        // A&S formula 7.1.26
        let t = T::one() / (T::one() + p * x_abs);
        let y = T::one()
            - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();

        sign * y
    }

    /// Inverse error function approximation
    /// 逆誤差関数近似
    fn erf_inv_approx(x: T) -> T {
        // Beasley-Springer-Moro approximation for inverse erf
        let a = T::from(0.147).unwrap();
        let two = T::from(2.0).unwrap();
        let pi = T::from(std::f64::consts::PI).unwrap();

        let ln_term = (T::one() - x * x).ln();
        let first_term = two / (pi * a) + ln_term / two;

        let sqrt_term = (first_term * first_term - ln_term / a).sqrt();
        let sign = if x >= T::zero() {
            T::one()
        } else {
            T::from(-1.0).unwrap()
        };

        sign * (sqrt_term - first_term).sqrt()
    }
}

impl<T: Float + 'static> DistributionTrait<T> for Normal<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    fn sample(&self, shape: Option<&[usize]>) -> Result<Tensor<T>, DistributionError> {
        let sample_shape = self.base.expand_shape(shape);

        // Generate standard normal samples using Box-Muller
        let standard_normal = DistributionUtils::random_normal::<T>(&sample_shape);

        // Transform to desired distribution: X = μ + σ * Z
        let standard_data = standard_normal.data.as_slice().unwrap();
        let loc_data = self.loc.data.as_slice().unwrap();
        let scale_data = self.scale.data.as_slice().unwrap();

        let result_data: Vec<T> = standard_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&z, &l), &s)| l + s * z)
            .collect();

        Ok(Tensor::from_vec(result_data, sample_shape))
    }

    fn log_prob(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        // log p(x) = -0.5 * ((x - μ) / σ)² - log(σ√(2π))
        let standardized = self.standardize(value)?;
        let log_norm = self.log_normalizing_constant()?;

        let std_data = standardized.data.as_slice().unwrap();
        let log_norm_data = log_norm.data.as_slice().unwrap();

        let half = T::from(-0.5).unwrap();
        let result_data: Vec<T> = std_data
            .iter()
            .zip(log_norm_data.iter().cycle())
            .map(|(&z, &ln)| half * z * z - ln)
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn cdf(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        // CDF = 0.5 * (1 + erf((x - μ) / (σ√2)))
        let sqrt_2 = T::from(2.0_f64.sqrt()).unwrap();
        let half = T::from(0.5).unwrap();

        let standardized = self.standardize(value)?;
        let std_data = standardized.data.as_slice().unwrap();

        let result_data: Vec<T> = std_data
            .iter()
            .map(|&z| half * (T::one() + Self::erf_approx(z / sqrt_2)))
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn icdf(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        // ICDF = μ + σ * √2 * erf⁻¹(2p - 1)
        let sqrt_2 = T::from(2.0_f64.sqrt()).unwrap();
        let two = T::from(2.0).unwrap();

        let value_data = value.data.as_slice().unwrap();
        let loc_data = self.loc.data.as_slice().unwrap();
        let scale_data = self.scale.data.as_slice().unwrap();

        let result_data: Vec<T> = value_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&p, &l), &s)| {
                let erf_input = two * p - T::one();
                l + s * sqrt_2 * Self::erf_inv_approx(erf_input)
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn mean(&self) -> Result<Tensor<T>, DistributionError> {
        Ok(self.loc.clone())
    }

    fn variance(&self) -> Result<Tensor<T>, DistributionError> {
        let scale_data = self.scale.data.as_slice().unwrap();
        let var_data: Vec<T> = scale_data.iter().map(|&s| s * s).collect();
        Ok(Tensor::from_vec(var_data, self.scale.shape().to_vec()))
    }

    fn entropy(&self) -> Result<Tensor<T>, DistributionError> {
        // Entropy = 0.5 * log(2πe * σ²) = log(σ) + 0.5 * log(2πe)
        let pi = T::from(std::f64::consts::PI).unwrap();
        let e = T::from(std::f64::consts::E).unwrap();
        let two = T::from(2.0).unwrap();
        let half = T::from(0.5).unwrap();

        let scale_data = self.scale.data.as_slice().unwrap();
        let constant = half * (two * pi * e).ln();

        let result_data: Vec<T> = scale_data.iter().map(|&s| s.ln() + constant).collect();

        Ok(Tensor::from_vec(result_data, self.scale.shape().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_normal_creation() {
        let loc = Tensor::from_vec(vec![0.0f32], vec![1]);
        let scale = Tensor::from_vec(vec![1.0f32], vec![1]);

        let normal = Normal::new(loc, scale, true).unwrap();
        assert_eq!(normal.base.batch_shape, vec![1]);
        assert_eq!(normal.base.event_shape, Vec::<usize>::new());
    }

    #[test]
    fn test_standard_normal() {
        let normal = Normal::<f32>::standard(true).unwrap();
        let mean = normal.mean().unwrap();
        let var = normal.variance().unwrap();

        assert_eq!(mean.data.as_slice().unwrap()[0], 0.0);
        assert_eq!(var.data.as_slice().unwrap()[0], 1.0);
    }

    #[test]
    fn test_normal_sampling() {
        let normal = Normal::<f32>::standard(true).unwrap();
        let samples = normal.sample(Some(&[1000])).unwrap();

        assert_eq!(samples.shape(), &[1000]);

        // Basic statistical test
        let data = samples.data.as_slice().unwrap();
        let sample_mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let sample_var: f32 =
            data.iter().map(|&x| (x - sample_mean).powi(2)).sum::<f32>() / (data.len() - 1) as f32;

        // Should be approximately 0 and 1 for large samples
        // Use more relaxed tolerances to account for statistical variance
        assert_abs_diff_eq!(sample_mean, 0.0, epsilon = 0.15);
        assert_abs_diff_eq!(sample_var, 1.0, epsilon = 0.2);
    }

    #[test]
    fn test_normal_log_prob() {
        let normal = Normal::<f32>::standard(true).unwrap();
        let values = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], vec![3]);

        let log_probs = normal.log_prob(&values).unwrap();
        let log_prob_data = log_probs.data.as_slice().unwrap();

        // For standard normal, log p(0) = -0.5 * log(2π) ≈ -0.9189
        assert_abs_diff_eq!(log_prob_data[0], -0.9189f32, epsilon = 0.01);

        // Symmetric around 0
        assert_abs_diff_eq!(log_prob_data[1], log_prob_data[2], epsilon = 1e-6);
    }

    #[test]
    fn test_normal_cdf() {
        let normal = Normal::<f32>::standard(true).unwrap();
        let values = Tensor::from_vec(vec![0.0f32, 1.96, -1.96], vec![3]);

        let cdf_vals = normal.cdf(&values).unwrap();
        let cdf_data = cdf_vals.data.as_slice().unwrap();

        // CDF(0) = 0.5 for standard normal
        assert_abs_diff_eq!(cdf_data[0], 0.5, epsilon = 0.01);

        // CDF(1.96) ≈ 0.975
        assert_abs_diff_eq!(cdf_data[1], 0.975, epsilon = 0.01);

        // Symmetric around 0.5
        assert_abs_diff_eq!(cdf_data[2], 1.0 - cdf_data[1], epsilon = 0.01);
    }

    #[test]
    fn test_normal_with_parameters() {
        let loc = Tensor::from_vec(vec![2.0f32], vec![1]);
        let scale = Tensor::from_vec(vec![3.0f32], vec![1]);
        let normal = Normal::new(loc, scale, true).unwrap();

        let mean = normal.mean().unwrap();
        let var = normal.variance().unwrap();

        assert_eq!(mean.data.as_slice().unwrap()[0], 2.0);
        assert_eq!(var.data.as_slice().unwrap()[0], 9.0); // σ² = 3² = 9
    }

    #[test]
    fn test_invalid_scale() {
        let loc = Tensor::from_vec(vec![0.0f32], vec![1]);
        let scale = Tensor::from_vec(vec![-1.0f32], vec![1]); // Invalid negative scale

        assert!(Normal::new(loc, scale, true).is_err());
    }
}
