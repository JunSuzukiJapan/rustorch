use crate::distributions::{Distribution, DistributionError, DistributionTrait, DistributionUtils};
/// Uniform Distribution - torch.distributions.Uniform compatible
/// 一様分布 - torch.distributions.Uniform互換
///
/// Implements the continuous uniform distribution over the interval [low, high).
/// 区間[low, high)上の連続一様分布を実装
use crate::tensor::Tensor;
use num_traits::Float;
use std::marker::PhantomData;

/// Uniform Distribution
/// 一様分布
///
/// The uniform distribution is defined by:
/// - Low (a): lower bound (inclusive)
/// - High (b): upper bound (exclusive)
///
/// PDF: f(x) = 1 / (b - a) for a ≤ x < b, 0 otherwise
#[derive(Debug, Clone)]
pub struct Uniform<T: Float> {
    /// Lower bound parameter (a)
    /// 下限パラメータ (a)
    pub low: Tensor<T>,

    /// Upper bound parameter (b)
    /// 上限パラメータ (b)
    pub high: Tensor<T>,

    /// Base distribution properties
    /// 基本分布特性
    pub base: Distribution,

    /// Phantom data for type parameter
    /// 型パラメータ用のファントムデータ
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> Uniform<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    /// Create a new Uniform distribution
    /// 新しい一様分布を作成
    ///
    /// # Arguments
    /// * `low` - Lower bound (a) parameter tensor
    /// * `high` - Upper bound (b) parameter tensor
    /// * `validate_args` - Whether to validate parameters
    pub fn new(
        low: Tensor<T>,
        high: Tensor<T>,
        validate_args: bool,
    ) -> Result<Self, DistributionError> {
        if validate_args {
            // Validate low < high
            let low_data = low.data.as_slice().unwrap();
            let high_data = high.data.as_slice().unwrap();

            for (i, (&l, &h)) in low_data.iter().zip(high_data.iter().cycle()).enumerate() {
                if l >= h {
                    return Err(DistributionError::InvalidParameter(format!(
                        "low must be less than high, got low[{}] = {}, high[{}] = {}",
                        i, l, i, h
                    )));
                }
            }
        }

        // Determine batch shape from broadcasting low and high
        let batch_shape = Distribution::broadcast_shapes(low.shape(), high.shape())?;
        let event_shape = vec![]; // Uniform is a univariate distribution

        Ok(Self {
            low,
            high,
            base: Distribution::new(batch_shape, event_shape, validate_args),
            _phantom: PhantomData,
        })
    }

    /// Create Uniform distribution with scalar parameters
    /// スカラーパラメータで一様分布を作成
    pub fn from_scalars(low: T, high: T, validate_args: bool) -> Result<Self, DistributionError> {
        let low_tensor = Tensor::from_vec(vec![low], vec![]);
        let high_tensor = Tensor::from_vec(vec![high], vec![]);
        Self::new(low_tensor, high_tensor, validate_args)
    }

    /// Standard uniform distribution on [0, 1)
    /// [0, 1)上の標準一様分布
    pub fn standard(validate_args: bool) -> Result<Self, DistributionError> {
        Self::from_scalars(T::zero(), T::one(), validate_args)
    }

    /// Symmetric uniform distribution around zero [-a, a)
    /// ゼロ周りの対称一様分布 [-a, a)
    pub fn symmetric(half_width: T, validate_args: bool) -> Result<Self, DistributionError> {
        let neg_width = T::zero() - half_width;
        Self::from_scalars(neg_width, half_width, validate_args)
    }

    /// Compute the range (high - low)
    /// 範囲（high - low）を計算
    fn range(&self) -> Tensor<T> {
        let low_data = self.low.data.as_slice().unwrap();
        let high_data = self.high.data.as_slice().unwrap();

        let range_data: Vec<T> = low_data
            .iter()
            .zip(high_data.iter().cycle())
            .map(|(&l, &h)| h - l)
            .collect();

        Tensor::from_vec(range_data, self.low.shape().to_vec())
    }
}

impl<T: Float + 'static> DistributionTrait<T> for Uniform<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    fn sample(&self, shape: Option<&[usize]>) -> Result<Tensor<T>, DistributionError> {
        let sample_shape = self.base.expand_shape(shape);

        // Generate uniform samples on [0, 1)
        let uniform_01 = DistributionUtils::random_uniform::<T>(&sample_shape);

        // Transform to [low, high): X = low + (high - low) * U
        let uniform_data = uniform_01.data.as_slice().unwrap();
        let low_data = self.low.data.as_slice().unwrap();
        let high_data = self.high.data.as_slice().unwrap();

        let result_data: Vec<T> = uniform_data
            .iter()
            .zip(low_data.iter().cycle())
            .zip(high_data.iter().cycle())
            .map(|((&u, &l), &h)| l + (h - l) * u)
            .collect();

        Ok(Tensor::from_vec(result_data, sample_shape))
    }

    fn log_prob(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        let value_data = value.data.as_slice().unwrap();
        let low_data = self.low.data.as_slice().unwrap();
        let high_data = self.high.data.as_slice().unwrap();

        let neg_inf = T::neg_infinity();
        let result_data: Vec<T> = value_data
            .iter()
            .zip(low_data.iter().cycle())
            .zip(high_data.iter().cycle())
            .map(|((&v, &l), &h)| {
                if v >= l && v < h {
                    // log(1 / (high - low)) = -log(high - low)
                    -(h - l).ln()
                } else {
                    neg_inf
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn cdf(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        let value_data = value.data.as_slice().unwrap();
        let low_data = self.low.data.as_slice().unwrap();
        let high_data = self.high.data.as_slice().unwrap();

        let result_data: Vec<T> = value_data
            .iter()
            .zip(low_data.iter().cycle())
            .zip(high_data.iter().cycle())
            .map(|((&v, &l), &h)| {
                if v < l {
                    T::zero()
                } else if v >= h {
                    T::one()
                } else {
                    // CDF = (x - low) / (high - low)
                    (v - l) / (h - l)
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn icdf(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        // ICDF = low + (high - low) * p
        let value_data = value.data.as_slice().unwrap();
        let low_data = self.low.data.as_slice().unwrap();
        let high_data = self.high.data.as_slice().unwrap();

        let result_data: Vec<T> = value_data
            .iter()
            .zip(low_data.iter().cycle())
            .zip(high_data.iter().cycle())
            .map(|((&p, &l), &h)| l + (h - l) * p)
            .collect();

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn mean(&self) -> Result<Tensor<T>, DistributionError> {
        // Mean = (low + high) / 2
        let low_data = self.low.data.as_slice().unwrap();
        let high_data = self.high.data.as_slice().unwrap();
        let half = T::from(0.5).unwrap();

        let mean_data: Vec<T> = low_data
            .iter()
            .zip(high_data.iter().cycle())
            .map(|(&l, &h)| half * (l + h))
            .collect();

        Ok(Tensor::from_vec(mean_data, self.low.shape().to_vec()))
    }

    fn variance(&self) -> Result<Tensor<T>, DistributionError> {
        // Variance = (high - low)² / 12
        let range = self.range();
        let range_data = range.data.as_slice().unwrap();
        let twelve = T::from(12.0).unwrap();

        let var_data: Vec<T> = range_data.iter().map(|&r| (r * r) / twelve).collect();

        Ok(Tensor::from_vec(var_data, range.shape().to_vec()))
    }

    fn entropy(&self) -> Result<Tensor<T>, DistributionError> {
        // Entropy = log(high - low)
        let range = self.range();
        let range_data = range.data.as_slice().unwrap();

        let entropy_data: Vec<T> = range_data.iter().map(|&r| r.ln()).collect();

        Ok(Tensor::from_vec(entropy_data, range.shape().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_uniform_creation() {
        let low = Tensor::from_vec(vec![0.0f32], vec![1]);
        let high = Tensor::from_vec(vec![1.0f32], vec![1]);

        let uniform = Uniform::new(low, high, true).unwrap();
        assert_eq!(uniform.base.batch_shape, vec![1]);
        assert_eq!(uniform.base.event_shape, vec![] as Vec<usize>);
    }

    #[test]
    fn test_standard_uniform() {
        let uniform = Uniform::<f32>::standard(true).unwrap();
        let mean = uniform.mean().unwrap();
        let var = uniform.variance().unwrap();

        assert_abs_diff_eq!(mean.data.as_slice().unwrap()[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(var.data.as_slice().unwrap()[0], 1.0 / 12.0, epsilon = 1e-6);
    }

    #[test]
    fn test_uniform_sampling() {
        let uniform = Uniform::<f32>::from_scalars(2.0, 5.0, true).unwrap();
        let samples = uniform.sample(Some(&[1000])).unwrap();

        assert_eq!(samples.shape(), &[1000]);

        // Check all samples are in [2, 5) range
        let data = samples.data.as_slice().unwrap();
        for &x in data {
            assert!(x >= 2.0 && x < 5.0);
        }

        // Basic statistical test
        // Use relaxed tolerance for statistical variance
        let sample_mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert_abs_diff_eq!(sample_mean, 3.5, epsilon = 0.15); // Should be (2+5)/2 = 3.5
    }

    #[test]
    fn test_uniform_log_prob() {
        let uniform = Uniform::<f32>::from_scalars(1.0, 3.0, true).unwrap();
        let values = Tensor::from_vec(vec![0.5f32, 1.5, 2.0, 3.5], vec![4]);

        let log_probs = uniform.log_prob(&values).unwrap();
        let log_prob_data = log_probs.data.as_slice().unwrap();

        // For uniform on [1, 3), log p(x) = -log(2) ≈ -0.693 for x in [1, 3)
        assert_eq!(log_prob_data[0], f32::NEG_INFINITY); // x = 0.5 < 1
        assert_abs_diff_eq!(log_prob_data[1], -(2.0f32).ln(), epsilon = 1e-6); // x = 1.5
        assert_abs_diff_eq!(log_prob_data[2], -(2.0f32).ln(), epsilon = 1e-6); // x = 2.0
        assert_eq!(log_prob_data[3], f32::NEG_INFINITY); // x = 3.5 >= 3
    }

    #[test]
    fn test_uniform_cdf() {
        let uniform = Uniform::<f32>::from_scalars(2.0, 6.0, true).unwrap();
        let values = Tensor::from_vec(vec![1.0f32, 2.0, 4.0, 6.0, 7.0], vec![5]);

        let cdf_vals = uniform.cdf(&values).unwrap();
        let cdf_data = cdf_vals.data.as_slice().unwrap();

        assert_eq!(cdf_data[0], 0.0); // x = 1 < 2
        assert_eq!(cdf_data[1], 0.0); // x = 2 = low
        assert_abs_diff_eq!(cdf_data[2], 0.5, epsilon = 1e-6); // x = 4, CDF = (4-2)/(6-2) = 0.5
        assert_eq!(cdf_data[3], 1.0); // x = 6 = high
        assert_eq!(cdf_data[4], 1.0); // x = 7 > 6
    }

    #[test]
    fn test_uniform_icdf() {
        let uniform = Uniform::<f32>::from_scalars(1.0, 5.0, true).unwrap();
        let probs = Tensor::from_vec(vec![0.0f32, 0.25, 0.5, 0.75, 1.0], vec![5]);

        let icdf_vals = uniform.icdf(&probs).unwrap();
        let icdf_data = icdf_vals.data.as_slice().unwrap();

        assert_abs_diff_eq!(icdf_data[0], 1.0, epsilon = 1e-6); // p = 0 -> low
        assert_abs_diff_eq!(icdf_data[1], 2.0, epsilon = 1e-6); // p = 0.25 -> 1 + 4*0.25 = 2
        assert_abs_diff_eq!(icdf_data[2], 3.0, epsilon = 1e-6); // p = 0.5 -> 1 + 4*0.5 = 3
        assert_abs_diff_eq!(icdf_data[3], 4.0, epsilon = 1e-6); // p = 0.75 -> 1 + 4*0.75 = 4
        assert_abs_diff_eq!(icdf_data[4], 5.0, epsilon = 1e-6); // p = 1.0 -> high
    }

    #[test]
    fn test_symmetric_uniform() {
        let uniform = Uniform::<f32>::symmetric(2.5, true).unwrap();
        let mean = uniform.mean().unwrap();

        assert_abs_diff_eq!(mean.data.as_slice().unwrap()[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_invalid_parameters() {
        // Test low >= high
        assert!(Uniform::<f32>::from_scalars(3.0, 2.0, true).is_err());
        assert!(Uniform::<f32>::from_scalars(1.0, 1.0, true).is_err());
    }
}
