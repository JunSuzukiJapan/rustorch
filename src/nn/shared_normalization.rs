//! Shared normalization function traits and implementations
//! 共通正規化関数トレイトと実装

use num_traits::Float;

/// Common normalization trait for both regular and WASM implementations
/// 通常実装とWASM実装の両方用の共通正規化トレイト
pub trait NormalizationFunction<T> {
    type Error;
    type Input;
    type Output;

    /// Batch normalization
    /// バッチ正規化
    fn batch_norm(
        &self,
        input: Self::Input,
        gamma: &[T],
        beta: &[T],
        running_mean: &mut [T],
        running_var: &mut [T],
        momentum: T,
        epsilon: T,
        training: bool,
    ) -> Result<Self::Output, Self::Error>;

    /// Layer normalization
    /// レイヤー正規化
    fn layer_norm(
        &self,
        input: Self::Input,
        gamma: &[T],
        beta: &[T],
        epsilon: T,
    ) -> Result<Self::Output, Self::Error>;

    /// Group normalization
    /// グループ正規化
    fn group_norm(
        &self,
        input: Self::Input,
        gamma: &[T],
        beta: &[T],
        num_groups: usize,
        epsilon: T,
    ) -> Result<Self::Output, Self::Error>;
}

/// Shared normalization implementations
/// 共通正規化実装
pub mod shared_normalizations {
    use super::*;

    /// Compute mean for normalization
    /// 正規化用の平均を計算
    pub fn compute_mean<T: Float>(data: &[T]) -> T {
        if data.is_empty() {
            return T::zero();
        }

        data.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(data.len()).unwrap_or(T::one())
    }

    /// Compute variance for normalization
    /// 正規化用の分散を計算
    pub fn compute_variance<T: Float>(data: &[T], mean: T) -> T {
        if data.is_empty() {
            return T::zero();
        }

        let sum_sq_diff = data
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x);

        sum_sq_diff / T::from(data.len()).unwrap_or(T::one())
    }

    /// Apply normalization transformation: (x - mean) / sqrt(var + epsilon) * gamma + beta
    /// 正規化変換を適用: (x - mean) / sqrt(var + epsilon) * gamma + beta
    pub fn apply_normalization<T: Float>(
        data: &[T],
        mean: T,
        variance: T,
        gamma: &[T],
        beta: &[T],
        epsilon: T,
    ) -> Result<Vec<T>, &'static str> {
        if gamma.len() != beta.len() {
            return Err("Gamma and beta must have same length");
        }

        let inv_std = T::one() / (variance + epsilon).sqrt();
        let mut result = Vec::with_capacity(data.len());

        for (i, &x) in data.iter().enumerate() {
            let gamma_idx = i % gamma.len();
            let beta_idx = i % beta.len();

            let normalized = (x - mean) * inv_std;
            let transformed = normalized * gamma[gamma_idx] + beta[beta_idx];
            result.push(transformed);
        }

        Ok(result)
    }

    /// Simple batch normalization for 1D data
    /// 1Dデータ用の簡単なバッチ正規化
    pub fn batch_norm_1d<T: Float>(
        input: &[T],
        gamma: &[T],
        beta: &[T],
        running_mean: &mut [T],
        running_var: &mut [T],
        momentum: T,
        epsilon: T,
        training: bool,
    ) -> Result<Vec<T>, &'static str> {
        if training {
            // Training mode: compute statistics from current batch
            let batch_mean = compute_mean(input);
            let batch_var = compute_variance(input, batch_mean);

            // Update running statistics
            for (rm, bm) in running_mean.iter_mut().zip(std::iter::repeat(batch_mean)) {
                *rm = (T::one() - momentum) * (*rm) + momentum * bm;
            }
            for (rv, bv) in running_var.iter_mut().zip(std::iter::repeat(batch_var)) {
                *rv = (T::one() - momentum) * (*rv) + momentum * bv;
            }

            apply_normalization(input, batch_mean, batch_var, gamma, beta, epsilon)
        } else {
            // Inference mode: use running statistics
            let mean = running_mean.get(0).copied().unwrap_or(T::zero());
            let var = running_var.get(0).copied().unwrap_or(T::one());

            apply_normalization(input, mean, var, gamma, beta, epsilon)
        }
    }

    /// Layer normalization implementation
    /// レイヤー正規化実装
    pub fn layer_norm_1d<T: Float>(
        input: &[T],
        gamma: &[T],
        beta: &[T],
        epsilon: T,
    ) -> Result<Vec<T>, &'static str> {
        let mean = compute_mean(input);
        let variance = compute_variance(input, mean);

        apply_normalization(input, mean, variance, gamma, beta, epsilon)
    }
}
