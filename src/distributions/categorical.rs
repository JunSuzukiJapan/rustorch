use crate::distributions::{Distribution, DistributionError, DistributionTrait, DistributionUtils};
/// Categorical Distribution - torch.distributions.Categorical compatible
/// カテゴリカル分布 - torch.distributions.Categorical互換
///
/// Implements the categorical distribution over a finite set of categories,
/// parameterized by probabilities or log-probabilities (logits).
/// 有限個のカテゴリ上のカテゴリカル分布を、確率または対数確率（ロジット）で
/// パラメータ化して実装
use crate::tensor::Tensor;
use num_traits::Float;
use std::marker::PhantomData;

/// Categorical Distribution
/// カテゴリカル分布
///
/// The categorical distribution is defined by:
/// - Probability parameters p_i for each category i, where Σp_i = 1
/// - Or logits parameters (log-probabilities before normalization)
///
/// PMF: P(X = k) = p_k for k ∈ {0, 1, ..., K-1}
#[derive(Debug, Clone)]
pub struct Categorical<T: Float> {
    /// Probability parameters - optional
    /// 確率パラメータ - オプション
    pub probs: Option<Tensor<T>>,

    /// Logits parameters (unnormalized log probabilities) - optional
    /// ロジットパラメータ（正規化前対数確率）- オプション
    pub logits: Option<Tensor<T>>,

    /// Number of categories
    /// カテゴリ数
    pub num_categories: usize,

    /// Base distribution properties
    /// 基本分布特性
    pub base: Distribution,

    /// Phantom data for type parameter
    /// 型パラメータ用のファントムデータ
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> Categorical<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    /// Create a new Categorical distribution from probabilities
    /// 確率からカテゴリカル分布を作成
    ///
    /// # Arguments
    /// * `probs` - Probability tensor with shape [..., num_categories]
    /// * `validate_args` - Whether to validate parameters
    pub fn from_probs(probs: Tensor<T>, validate_args: bool) -> Result<Self, DistributionError> {
        if validate_args {
            DistributionUtils::validate_probability(&probs)?;
            Self::validate_probabilities_sum(&probs)?;
        }

        let shape = probs.shape();
        if shape.is_empty() {
            return Err(DistributionError::InvalidParameter(
                "Probs tensor cannot be empty".to_string(),
            ));
        }

        let num_categories = shape[shape.len() - 1];
        let batch_shape = if shape.len() > 1 {
            shape[..shape.len() - 1].to_vec()
        } else {
            vec![]
        };
        let event_shape = vec![]; // Categorical outputs single category index

        Ok(Self {
            probs: Some(probs),
            logits: None,
            num_categories,
            base: Distribution::new(batch_shape, event_shape, validate_args),
            _phantom: PhantomData,
        })
    }

    /// Create a new Categorical distribution from logits
    /// ロジットからカテゴリカル分布を作成
    ///
    /// # Arguments
    /// * `logits` - Logits tensor with shape [..., num_categories]
    /// * `validate_args` - Whether to validate parameters
    pub fn from_logits(logits: Tensor<T>, validate_args: bool) -> Result<Self, DistributionError> {
        let shape = logits.shape();
        if shape.is_empty() {
            return Err(DistributionError::InvalidParameter(
                "Logits tensor cannot be empty".to_string(),
            ));
        }

        let num_categories = shape[shape.len() - 1];
        let batch_shape = if shape.len() > 1 {
            shape[..shape.len() - 1].to_vec()
        } else {
            vec![]
        };
        let event_shape = vec![];

        Ok(Self {
            probs: None,
            logits: Some(logits),
            num_categories,
            base: Distribution::new(batch_shape, event_shape, validate_args),
            _phantom: PhantomData,
        })
    }

    /// Create uniform categorical distribution
    /// 一様カテゴリカル分布を作成
    pub fn uniform(num_categories: usize, validate_args: bool) -> Result<Self, DistributionError> {
        if num_categories == 0 {
            return Err(DistributionError::InvalidParameter(
                "Number of categories must be positive".to_string(),
            ));
        }

        let uniform_prob = T::one() / T::from(num_categories).unwrap();
        let probs_data = vec![uniform_prob; num_categories];
        let probs = Tensor::from_vec(probs_data, vec![num_categories]);

        Self::from_probs(probs, validate_args)
    }

    /// Validate that probabilities sum to 1 (within tolerance)
    /// 確率の合計が1であることを検証（許容範囲内）
    fn validate_probabilities_sum(probs: &Tensor<T>) -> Result<(), DistributionError> {
        let shape = probs.shape();
        let num_categories = shape[shape.len() - 1];
        let batch_size: usize = shape[..shape.len() - 1].iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        let data = probs.data.as_slice().unwrap();
        let tolerance = T::from(1e-6).unwrap();

        for batch in 0..batch_size {
            let start_idx = batch * num_categories;
            let mut sum = T::zero();
            for &val in &data[start_idx..start_idx + num_categories] {
                sum = sum + val;
            }

            if (sum - T::one()).abs() > tolerance {
                return Err(DistributionError::InvalidParameter(format!(
                    "Probabilities must sum to 1.0, got {}",
                    sum
                )));
            }
        }

        Ok(())
    }

    /// Get probabilities (convert from logits if necessary)
    /// 確率を取得（必要に応じてロジットから変換）
    pub fn get_probs(&self) -> Result<Tensor<T>, DistributionError> {
        match (&self.probs, &self.logits) {
            (Some(probs), _) => Ok(probs.clone()),
            (None, Some(logits)) => {
                // Convert logits to probabilities using softmax
                self.softmax(logits)
            }
            _ => Err(DistributionError::InvalidParameter(
                "Either probs or logits must be specified".to_string(),
            )),
        }
    }

    /// Get logits (convert from probs if necessary)  
    /// ロジットを取得（必要に応じて確率から変換）
    pub fn get_logits(&self) -> Result<Tensor<T>, DistributionError> {
        match (&self.logits, &self.probs) {
            (Some(logits), _) => Ok(logits.clone()),
            (None, Some(probs)) => {
                // Convert probs to logits: logit = log(prob)
                let probs_data = probs.data.as_slice().unwrap();
                let logits_data: Vec<T> = probs_data
                    .iter()
                    .map(|&p| {
                        if p > T::zero() {
                            p.ln()
                        } else {
                            T::from(-1e10).unwrap()
                        }
                    })
                    .collect();
                Ok(Tensor::from_vec(logits_data, probs.shape().to_vec()))
            }
            _ => Err(DistributionError::InvalidParameter(
                "Either probs or logits must be specified".to_string(),
            )),
        }
    }

    /// Softmax function to convert logits to probabilities
    /// ロジットを確率に変換するソフトマックス関数
    fn softmax(&self, logits: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        let shape = logits.shape();
        let num_categories = shape[shape.len() - 1];
        let batch_size: usize = shape[..shape.len() - 1].iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        let data = logits.data.as_slice().unwrap();
        let mut result_data = Vec::with_capacity(data.len());

        for batch in 0..batch_size {
            let start_idx = batch * num_categories;
            let batch_logits = &data[start_idx..start_idx + num_categories];

            // Find max for numerical stability
            let max_logit = batch_logits.iter().fold(batch_logits[0], |a, &b| a.max(b));

            // Compute exp(logit - max)
            let exp_logits: Vec<T> = batch_logits
                .iter()
                .map(|&logit| (logit - max_logit).exp())
                .collect();

            // Compute sum and normalize
            let mut sum_exp = T::zero();
            for &val in exp_logits.iter() {
                sum_exp = sum_exp + val;
            }
            let probs: Vec<T> = exp_logits
                .iter()
                .map(|&exp_logit| exp_logit / sum_exp)
                .collect();

            result_data.extend(probs);
        }

        Ok(Tensor::from_vec(result_data, shape.to_vec()))
    }

    /// Compute cross entropy loss
    /// クロスエントロピー損失を計算
    pub fn cross_entropy(&self, target: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        let log_probs = self.log_prob(target)?;
        let log_prob_data = log_probs.data.as_slice().unwrap();
        let ce_data: Vec<T> = log_prob_data.iter().map(|&lp| -lp).collect();
        Ok(Tensor::from_vec(ce_data, log_probs.shape().to_vec()))
    }
}

impl<T: Float + 'static> DistributionTrait<T> for Categorical<T>
where
    T: rand::distributions::uniform::SampleUniform + num_traits::FromPrimitive + std::fmt::Display,
{
    fn sample(&self, shape: Option<&[usize]>) -> Result<Tensor<T>, DistributionError> {
        let sample_shape = self.base.expand_shape(shape);
        let probs = self.get_probs()?;

        let probs_data = probs.data.as_slice().unwrap();
        let batch_size: usize = self.base.batch_shape.iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };
        let sample_size: usize = sample_shape.iter().product();

        // Generate uniform samples for inverse transform sampling
        let uniform_samples = DistributionUtils::random_uniform::<T>(&sample_shape);
        let uniform_data = uniform_samples.data.as_slice().unwrap();

        let mut result_data = Vec::with_capacity(sample_size);

        for (i, &u) in uniform_data.iter().enumerate().take(sample_size) {
            let batch_idx = i % batch_size;
            let batch_start = batch_idx * self.num_categories;
            let batch_probs = &probs_data[batch_start..batch_start + self.num_categories];

            // Inverse transform sampling using cumulative probabilities
            let mut cumulative = T::zero();
            let mut category = 0;

            for (cat, &prob) in batch_probs.iter().enumerate() {
                cumulative = cumulative + prob;
                if u <= cumulative {
                    category = cat;
                    break;
                }
            }

            result_data.push(T::from(category).unwrap());
        }

        Ok(Tensor::from_vec(result_data, sample_shape))
    }

    fn log_prob(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        let probs = self.get_probs()?;
        let probs_data = probs.data.as_slice().unwrap();
        let value_data = value.data.as_slice().unwrap();

        let batch_size: usize = self.base.batch_shape.iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        let mut result_data = Vec::with_capacity(value_data.len());

        for (i, &category_float) in value_data.iter().enumerate() {
            let category = category_float.to_usize().unwrap_or(0);

            if category >= self.num_categories {
                return Err(DistributionError::InvalidParameter(format!(
                    "Category {} out of range [0, {})",
                    category, self.num_categories
                )));
            }

            let batch_idx = i % batch_size;
            let prob_idx = batch_idx * self.num_categories + category;
            let prob = probs_data[prob_idx];

            let log_prob = if prob > T::zero() {
                prob.ln()
            } else {
                T::from(-1e10).unwrap()
            };
            result_data.push(log_prob);
        }

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn cdf(&self, value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        let probs = self.get_probs()?;
        let probs_data = probs.data.as_slice().unwrap();
        let value_data = value.data.as_slice().unwrap();

        let batch_size: usize = self.base.batch_shape.iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        let mut result_data = Vec::with_capacity(value_data.len());

        for (i, &category_float) in value_data.iter().enumerate() {
            let category = category_float.to_usize().unwrap_or(0);
            let batch_idx = i % batch_size;
            let batch_start = batch_idx * self.num_categories;
            let batch_probs = &probs_data[batch_start..batch_start + self.num_categories];

            // CDF = sum of probabilities up to and including the category
            let mut cdf_val = T::zero();
            for &val in &batch_probs[..=category.min(self.num_categories - 1)] {
                cdf_val = cdf_val + val;
            }
            result_data.push(cdf_val);
        }

        Ok(Tensor::from_vec(result_data, value.shape().to_vec()))
    }

    fn icdf(&self, _value: &Tensor<T>) -> Result<Tensor<T>, DistributionError> {
        Err(DistributionError::UnsupportedOperation(
            "Inverse CDF not well-defined for discrete distributions".to_string(),
        ))
    }

    fn mean(&self) -> Result<Tensor<T>, DistributionError> {
        // Mean of categorical distribution = Σ(k * p_k) for k=0,1,...,K-1
        let probs = self.get_probs()?;
        let probs_data = probs.data.as_slice().unwrap();

        let batch_size: usize = self.base.batch_shape.iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        let result_data: Vec<_> = (0..batch_size)
            .map(|batch| {
                let batch_start = batch * self.num_categories;
                let batch_probs = &probs_data[batch_start..batch_start + self.num_categories];

                let mut mean = T::zero();
                for (k, &p) in batch_probs.iter().enumerate() {
                    mean = mean + T::from(k).unwrap() * p;
                }

                mean
            })
            .collect();

        let result_shape = if self.base.batch_shape.is_empty() {
            vec![1]
        } else {
            self.base.batch_shape.clone()
        };

        Ok(Tensor::from_vec(result_data, result_shape))
    }

    fn variance(&self) -> Result<Tensor<T>, DistributionError> {
        // Variance = E[X²] - E[X]²
        let probs = self.get_probs()?;
        let probs_data = probs.data.as_slice().unwrap();
        let mean_tensor = self.mean()?;
        let mean_data = mean_tensor.data.as_slice().unwrap();

        let batch_size: usize = self.base.batch_shape.iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        let result_data: Vec<_> = (0..batch_size)
            .map(|batch| {
                let batch_start = batch * self.num_categories;
                let batch_probs = &probs_data[batch_start..batch_start + self.num_categories];
                let mean_val = mean_data[batch];

                // E[X²] = Σ(k² * p_k)
                let mut second_moment = T::zero();
                for (k, &p) in batch_probs.iter().enumerate() {
                    let k_float = T::from(k).unwrap();
                    second_moment = second_moment + k_float * k_float * p;
                }

                second_moment - mean_val * mean_val
            })
            .collect();

        let result_shape = if self.base.batch_shape.is_empty() {
            vec![1]
        } else {
            self.base.batch_shape.clone()
        };

        Ok(Tensor::from_vec(result_data, result_shape))
    }

    fn entropy(&self) -> Result<Tensor<T>, DistributionError> {
        // Entropy = -Σ(p_k * log(p_k))
        let probs = self.get_probs()?;
        let probs_data = probs.data.as_slice().unwrap();

        let batch_size: usize = self.base.batch_shape.iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        let mut result_data = Vec::with_capacity(batch_size);

        for batch in 0..batch_size {
            let batch_start = batch * self.num_categories;
            let batch_probs = &probs_data[batch_start..batch_start + self.num_categories];

            let mut entropy = T::zero();
            for &p in batch_probs.iter() {
                if p > T::zero() {
                    entropy = entropy + (-p * p.ln());
                }
            }

            result_data.push(entropy);
        }

        let result_shape = if self.base.batch_shape.is_empty() {
            vec![1]
        } else {
            self.base.batch_shape.clone()
        };

        Ok(Tensor::from_vec(result_data, result_shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_categorical_from_probs() {
        let probs = Tensor::from_vec(vec![0.2f32, 0.3, 0.5], vec![3]);
        let categorical = Categorical::from_probs(probs, true).unwrap();

        assert_eq!(categorical.num_categories, 3);

        let retrieved_probs = categorical.get_probs().unwrap();
        let data = retrieved_probs.data.as_slice().unwrap();
        assert_abs_diff_eq!(data[0], 0.2, epsilon = 1e-6);
        assert_abs_diff_eq!(data[1], 0.3, epsilon = 1e-6);
        assert_abs_diff_eq!(data[2], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_categorical_from_logits() {
        let logits = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], vec![3]);
        let categorical = Categorical::from_logits(logits, true).unwrap();

        let probs = categorical.get_probs().unwrap();
        let probs_data = probs.data.as_slice().unwrap();

        // Check that probabilities sum to 1
        let sum: f32 = probs_data.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

        // Higher logit should give higher probability
        assert!(probs_data[2] > probs_data[1]);
        assert!(probs_data[1] > probs_data[0]);
    }

    #[test]
    fn test_uniform_categorical() {
        let uniform = Categorical::<f32>::uniform(4, true).unwrap();
        let probs = uniform.get_probs().unwrap();
        let probs_data = probs.data.as_slice().unwrap();

        for &prob in probs_data {
            assert_abs_diff_eq!(prob, 0.25, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_categorical_sampling() {
        let categorical = Categorical::<f32>::uniform(3, true).unwrap();
        let samples = categorical.sample(Some(&[1000])).unwrap();

        assert_eq!(samples.shape(), &[1000]);

        let data = samples.data.as_slice().unwrap();
        let mut counts = [0; 3];

        for &sample in data {
            let category = sample as usize;
            if category < 3 {
                counts[category] += 1;
            }
        }

        // For uniform distribution, counts should be approximately equal
        // Use relaxed tolerance for statistical variance
        for count in counts {
            let proportion = count as f32 / 1000.0;
            assert_abs_diff_eq!(proportion, 1.0 / 3.0, epsilon = 0.15);
        }
    }

    #[test]
    fn test_categorical_log_prob() {
        let probs = Tensor::from_vec(vec![0.1f32, 0.6, 0.3], vec![3]);
        let categorical = Categorical::from_probs(probs, true).unwrap();

        let values = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], vec![3]);
        let log_probs = categorical.log_prob(&values).unwrap();
        let log_prob_data = log_probs.data.as_slice().unwrap();

        // log P(X=0) = log(0.1)
        assert_abs_diff_eq!(log_prob_data[0], 0.1f32.ln(), epsilon = 1e-6);
        // log P(X=1) = log(0.6)
        assert_abs_diff_eq!(log_prob_data[1], 0.6f32.ln(), epsilon = 1e-6);
        // log P(X=2) = log(0.3)
        assert_abs_diff_eq!(log_prob_data[2], 0.3f32.ln(), epsilon = 1e-6);
    }

    #[test]
    fn test_categorical_mean_variance() {
        let probs = Tensor::from_vec(vec![0.1f32, 0.2, 0.7], vec![3]);
        let categorical = Categorical::from_probs(probs, true).unwrap();

        let mean = categorical.mean().unwrap();
        let mean_val = mean.data.as_slice().unwrap()[0];

        // Mean = 0*0.1 + 1*0.2 + 2*0.7 = 0.2 + 1.4 = 1.6
        assert_abs_diff_eq!(mean_val, 1.6, epsilon = 1e-6);

        let variance = categorical.variance().unwrap();
        let var_val = variance.data.as_slice().unwrap()[0];

        // E[X²] = 0²*0.1 + 1²*0.2 + 2²*0.7 = 0.2 + 2.8 = 3.0
        // Var = E[X²] - E[X]² = 3.0 - 1.6² = 3.0 - 2.56 = 0.44
        assert_abs_diff_eq!(var_val, 0.44, epsilon = 1e-6);
    }

    #[test]
    fn test_invalid_probabilities() {
        // Probabilities don't sum to 1
        let invalid_probs = Tensor::from_vec(vec![0.2f32, 0.3, 0.6], vec![3]);
        assert!(Categorical::from_probs(invalid_probs, true).is_err());

        // Negative probability
        let negative_probs = Tensor::from_vec(vec![-0.1f32, 0.6, 0.5], vec![3]);
        assert!(Categorical::from_probs(negative_probs, true).is_err());
    }

    #[test]
    fn test_cross_entropy() {
        let probs = Tensor::from_vec(vec![0.7f32, 0.2, 0.1], vec![3]);
        let categorical = Categorical::from_probs(probs, true).unwrap();

        // Perfect prediction (category 0 has highest prob)
        let target = Tensor::from_vec(vec![0.0f32], vec![1]);
        let ce = categorical.cross_entropy(&target).unwrap();
        let ce_val = ce.data.as_slice().unwrap()[0];

        // CE = -log(0.7)
        assert_abs_diff_eq!(ce_val, -0.7f32.ln(), epsilon = 1e-6);
    }
}
