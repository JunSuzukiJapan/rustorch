/// Bernoulli distribution implementation
pub mod bernoulli;
/// Beta distribution implementation
pub mod beta;
/// Categorical distribution implementation  
pub mod categorical;
/// Statistical Distributions - torch.distributions.* compatible
/// 統計分布 - torch.distributions.*互換
///
/// This module provides PyTorch-compatible statistical distribution implementations
/// including probability density functions, sampling, and parameter estimation.
/// PyTorch互換の統計分布実装を提供し、確率密度関数、サンプリング、パラメータ推定を含む
pub mod distribution;
/// Exponential distribution implementation
pub mod exponential;
/// Gamma distribution implementation
pub mod gamma;
/// Normal (Gaussian) distribution implementation
pub mod normal;
/// Uniform distribution implementation
pub mod uniform;

// Re-export main types
pub use bernoulli::Bernoulli;
pub use beta::Beta;
pub use categorical::Categorical;
pub use distribution::Distribution;
use crate::error::{RusTorchError, RusTorchResult};
pub use exponential::Exponential;
pub use gamma::Gamma;
pub use normal::Normal;
pub use uniform::Uniform;

use crate::tensor::Tensor;
use num_traits::Float;

/// Distribution trait for all statistical distributions
/// 全ての統計分布のための分布トレイト
pub trait DistributionTrait<T: Float + 'static> {
    /// Sample from the distribution
    /// 分布からサンプルを生成
    fn sample(&self, shape: Option<&[usize]>) -> RusTorchResult<Tensor<T>>;

    /// Sample with replacement
    /// 復元抽出でサンプル
    fn sample_n(&self, n: usize) -> RusTorchResult<Tensor<T>> {
        self.sample(Some(&[n]))
    }

    /// Compute log probability density/mass function
    /// 対数確率密度/質量関数を計算
    fn log_prob(&self, value: &Tensor<T>) -> RusTorchResult<Tensor<T>>;

    /// Compute probability density/mass function
    /// 確率密度/質量関数を計算
    fn prob(&self, value: &Tensor<T>) -> RusTorchResult<Tensor<T>> {
        let log_p = self.log_prob(value)?;
        // Apply exp to convert log prob to prob
        let data = log_p.data.as_slice().unwrap();
        let prob_data: Vec<T> = data.iter().map(|&x| x.exp()).collect();
        Ok(Tensor::from_vec(prob_data, log_p.shape().to_vec()))
    }

    /// Compute cumulative distribution function
    /// 累積分布関数を計算
    fn cdf(&self, value: &Tensor<T>) -> RusTorchResult<Tensor<T>>;

    /// Compute inverse cumulative distribution function (quantile function)
    /// 逆累積分布関数（分位点関数）を計算
    fn icdf(&self, value: &Tensor<T>) -> RusTorchResult<Tensor<T>>;

    /// Get distribution mean
    /// 分布の平均を取得
    fn mean(&self) -> RusTorchResult<Tensor<T>>;

    /// Get distribution variance
    /// 分布の分散を取得
    fn variance(&self) -> RusTorchResult<Tensor<T>>;

    /// Get distribution standard deviation
    /// 分布の標準偏差を取得
    fn stddev(&self) -> RusTorchResult<Tensor<T>> {
        let var = self.variance()?;
        let data = var.data.as_slice().unwrap();
        let stddev_data: Vec<T> = data.iter().map(|&x| x.sqrt()).collect();
        Ok(Tensor::from_vec(stddev_data, var.shape().to_vec()))
    }

    /// Get distribution entropy
    /// 分布のエントロピーを取得
    fn entropy(&self) -> RusTorchResult<Tensor<T>>;
}

/// Validation utilities for distributions
/// 分布の検証ユーティリティ
pub struct DistributionUtils;

impl DistributionUtils {
    /// Validate probability parameter (0 <= p <= 1)
    /// 確率パラメータの検証 (0 <= p <= 1)
    pub fn validate_probability<T: Float + std::fmt::Display>(
        p: &Tensor<T>,
    ) -> RusTorchResult<()> {
        let data = p.data.as_slice().unwrap();
        for &val in data {
            if val < T::zero() || val > T::one() {
                return Err(RusTorchError::invalid_parameter(&format!(
                    "Probability must be in [0, 1], got {}",
                    val
                )));
            }
        }
        Ok(())
    }

    /// Validate positive parameter (> 0)
    /// 正のパラメータの検証 (> 0)
    pub fn validate_positive<T: Float + std::fmt::Display>(
        param: &Tensor<T>,
        name: &str,
    ) -> RusTorchResult<()> {
        let data = param.data.as_slice().unwrap();
        for &val in data {
            if val <= T::zero() {
                return Err(RusTorchError::invalid_parameter(&format!(
                    "{} must be positive, got {}",
                    name, val
                )));
            }
        }
        Ok(())
    }

    /// Validate non-negative parameter (>= 0)
    /// 非負パラメータの検証 (>= 0)
    pub fn validate_non_negative<T: Float + std::fmt::Display>(
        param: &Tensor<T>,
        name: &str,
    ) -> RusTorchResult<()> {
        let data = param.data.as_slice().unwrap();
        for &val in data {
            if val < T::zero() {
                return Err(RusTorchError::invalid_parameter(&format!(
                    "{} must be non-negative, got {}",
                    name, val
                )));
            }
        }
        Ok(())
    }

    /// Generate random uniform samples
    /// ランダムな一様サンプルを生成
    pub fn random_uniform<T>(shape: &[usize]) -> Tensor<T>
    where
        T: Float + 'static + rand::distributions::uniform::SampleUniform,
    {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<T> = (0..size)
            .map(|_| rng.gen_range(T::zero()..T::one()))
            .collect();
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Generate random normal samples using Box-Muller transform
    /// Box-Muller変換を使用してランダムな正規サンプルを生成
    pub fn random_normal<T>(shape: &[usize]) -> Tensor<T>
    where
        T: Float + 'static + rand::distributions::uniform::SampleUniform,
    {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        // Box-Muller transform for normal distribution
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two = T::from(2.0).unwrap();

        for _ in 0..size.div_ceil(2) {
            let u1: T = rng.gen_range(T::from(1e-10).unwrap()..T::one());
            let u2: T = rng.gen_range(T::zero()..T::one());

            let z0 = (-two * u1.ln()).sqrt() * (two * pi * u2).cos();
            let z1 = (-two * u1.ln()).sqrt() * (two * pi * u2).sin();

            data.push(z0);
            if data.len() < size {
                data.push(z1);
            }
        }

        data.truncate(size);
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Generate single random uniform scalar
    /// 単一のランダムな一様スカラーを生成
    pub fn random_uniform_scalar<T>() -> T
    where
        T: Float + 'static + rand::distributions::uniform::SampleUniform,
    {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();
        rng.gen_range(T::zero()..T::one())
    }

    /// Generate single random normal scalar
    /// 単一のランダムな正規スカラーを生成
    pub fn random_normal_scalar<T>() -> T
    where
        T: Float + 'static + rand::distributions::uniform::SampleUniform,
    {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();

        // Box-Muller transform
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two = T::from(2.0).unwrap();

        let u1: T = rng.gen_range(T::from(1e-10).unwrap()..T::one());
        let u2: T = rng.gen_range(T::zero()..T::one());

        (-two * u1.ln()).sqrt() * (two * pi * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_utils() {
        // Test probability validation
        let valid_prob = Tensor::from_vec([0.0f32, 0.5, 1.0].to_vec(), vec![3]);
        assert!(DistributionUtils::validate_probability(&valid_prob).is_ok());

        let invalid_prob = Tensor::from_vec([0.0f32, 0.5, 1.5].to_vec(), vec![3]);
        assert!(DistributionUtils::validate_probability(&invalid_prob).is_err());

        // Test positive validation
        let positive = Tensor::from_vec([0.1f32, 1.0, 2.0].to_vec(), vec![3]);
        assert!(DistributionUtils::validate_positive(&positive, "param").is_ok());

        let non_positive = Tensor::from_vec([0.0f32, 1.0, 2.0].to_vec(), vec![3]);
        assert!(DistributionUtils::validate_positive(&non_positive, "param").is_err());
    }

    #[test]
    fn test_random_generators() {
        // Test uniform random generation
        let uniform = DistributionUtils::random_uniform::<f32>(&[10, 5]);
        assert_eq!(uniform.shape(), &[10, 5]);

        let data = uniform.data.as_slice().unwrap();
        for &val in data {
            assert!(val >= 0.0 && val <= 1.0);
        }

        // Test normal random generation
        let normal = DistributionUtils::random_normal::<f32>(&[100]);
        assert_eq!(normal.shape(), &[100]);

        // Basic statistical test - mean should be close to 0
        let data = normal.data.as_slice().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.5); // Should be close to 0 for large sample
    }
}
