// Sampling strategies for text generation

use anyhow::Result;
use rustorch::tensor::Tensor;

/// Sampling configuration for text generation
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: f64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingConfig {
    /// Create greedy sampling (always select most likely token)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: Some(1),
            top_p: None,
            repetition_penalty: 1.0,
        }
    }

    /// Create sampling with top-k filtering
    pub fn top_k(k: usize, temperature: f64) -> Self {
        Self {
            temperature,
            top_k: Some(k),
            top_p: None,
            repetition_penalty: 1.0,
        }
    }

    /// Create sampling with nucleus (top-p) filtering
    pub fn nucleus(p: f64, temperature: f64) -> Self {
        Self {
            temperature,
            top_k: None,
            top_p: Some(p),
            repetition_penalty: 1.0,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.temperature < 0.0 {
            anyhow::bail!("Temperature must be non-negative");
        }

        if let Some(k) = self.top_k {
            if k == 0 {
                anyhow::bail!("top_k must be positive");
            }
        }

        if let Some(p) = self.top_p {
            if !(0.0..=1.0).contains(&p) {
                anyhow::bail!("top_p must be between 0.0 and 1.0");
            }
        }

        if self.repetition_penalty < 0.0 {
            anyhow::bail!("Repetition penalty must be non-negative");
        }

        Ok(())
    }
}

/// Sample next token from logits
pub fn sample_token(
    logits: &Tensor<f64>,
    config: &SamplingConfig,
    _previous_tokens: &[u32],
) -> Result<u32> {
    config.validate()?;

    // For now, return a dummy token (will be implemented with actual sampling)
    // TODO: Implement temperature scaling
    // TODO: Implement top-k filtering
    // TODO: Implement top-p (nucleus) filtering
    // TODO: Implement repetition penalty
    // TODO: Implement multinomial sampling

    tracing::debug!(
        "Sampling with config: temp={}, top_k={:?}, top_p={:?}",
        config.temperature,
        config.top_k,
        config.top_p
    );

    // Placeholder: return token 0
    // In real implementation, this would:
    // 1. Apply temperature scaling to logits
    // 2. Apply top-k/top-p filtering
    // 3. Apply repetition penalty
    // 4. Sample from the filtered distribution
    let vocab_size = logits.shape()[logits.shape().len() - 1];

    // Return middle token as placeholder
    Ok((vocab_size / 2) as u32)
}

/// Softmax function for converting logits to probabilities
pub fn softmax(logits: &[f64]) -> Result<Vec<f64>> {
    // Find max for numerical stability
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute exp(x - max)
    let exp_values: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

    // Compute sum of exponentials
    let sum: f64 = exp_values.iter().sum();

    // Normalize
    let probs: Vec<f64> = exp_values.iter().map(|&x| x / sum).collect();

    Ok(probs)
}

/// Apply temperature scaling to logits (returns Vec for sampling operations)
pub fn apply_temperature_to_vec(logits: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if temperature <= 0.0 {
        anyhow::bail!("Temperature must be positive");
    }
    Ok(logits.iter().map(|&x| x / temperature).collect())
}

/// Apply temperature scaling to logits
pub fn apply_temperature(logits: &Tensor<f64>, temperature: f64) -> Tensor<f64> {
    if temperature == 0.0 || temperature == 1.0 {
        return logits.clone();
    }

    // Scale logits by temperature
    let scaled_data: Vec<f64> = logits.data.iter().map(|&x| x / temperature).collect();
    Tensor::from_vec(scaled_data, logits.size().to_vec())
}

/// Apply top-k sampling: keep only top-k highest probabilities
pub fn apply_top_k_to_probs(probs: &[f64], k: usize) -> Result<Vec<f64>> {
    if k == 0 || k >= probs.len() {
        return Ok(probs.to_vec());
    }

    // Create indices with probabilities
    let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

    // Sort by probability (descending)
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Keep only top-k, zero out the rest
    let mut filtered = vec![0.0; probs.len()];
    for (idx, prob) in indexed.iter().take(k) {
        filtered[*idx] = *prob;
    }

    // Renormalize
    let sum: f64 = filtered.iter().sum();
    if sum > 0.0 {
        for p in filtered.iter_mut() {
            *p /= sum;
        }
    }

    Ok(filtered)
}

/// Apply top-k filtering to logits
pub fn apply_top_k(logits: &Tensor<f64>, k: usize) -> Tensor<f64> {
    if k == 0 {
        return logits.clone();
    }

    tracing::debug!("Applying top-k filtering with k={}", k);

    // Extract logits as vec, apply filtering, convert back
    let logits_vec: Vec<f64> = logits.data.iter().copied().collect();
    match softmax(&logits_vec).and_then(|probs| apply_top_k_to_probs(&probs, k)) {
        Ok(filtered) => Tensor::from_vec(filtered, logits.size().to_vec()),
        Err(_) => logits.clone(),
    }
}

/// Apply top-p (nucleus) sampling: keep smallest set of top probabilities that sum to >= p
pub fn apply_top_p_to_probs(probs: &[f64], p: f64) -> Result<Vec<f64>> {
    if p <= 0.0 || p >= 1.0 {
        return Ok(probs.to_vec());
    }

    // Create indices with probabilities
    let mut indexed: Vec<(usize, f64)> = probs
        .iter()
        .enumerate()
        .map(|(i, &prob)| (i, prob))
        .collect();

    // Sort by probability (descending)
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find cutoff: smallest set where cumulative prob >= p
    let mut cumulative = 0.0;
    let mut cutoff = indexed.len();
    for (i, (_idx, prob)) in indexed.iter().enumerate() {
        cumulative += prob;
        if cumulative >= p {
            cutoff = i + 1;
            break;
        }
    }

    // Keep only top-p tokens, zero out the rest
    let mut filtered = vec![0.0; probs.len()];
    for (idx, prob) in indexed.iter().take(cutoff) {
        filtered[*idx] = *prob;
    }

    // Renormalize
    let sum: f64 = filtered.iter().sum();
    if sum > 0.0 {
        for prob in filtered.iter_mut() {
            *prob /= sum;
        }
    }

    Ok(filtered)
}

/// Apply top-p (nucleus) filtering to logits
pub fn apply_top_p(logits: &Tensor<f64>, p: f64) -> Tensor<f64> {
    if !(0.0..=1.0).contains(&p) {
        return logits.clone();
    }

    tracing::debug!("Applying top-p filtering with p={}", p);

    // Extract logits as vec, apply filtering, convert back
    let logits_vec: Vec<f64> = logits.data.iter().copied().collect();
    match softmax(&logits_vec).and_then(|probs| apply_top_p_to_probs(&probs, p)) {
        Ok(filtered) => Tensor::from_vec(filtered, logits.size().to_vec()),
        Err(_) => logits.clone(),
    }
}

/// Sample from a probability distribution using multinomial sampling
pub fn multinomial_sample(probs: &[f64]) -> Result<usize> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let random: f64 = rng.gen();

    let mut cumulative = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if random < cumulative {
            return Ok(i);
        }
    }

    // Fallback: return last index
    Ok(probs.len() - 1)
}

/// Apply repetition penalty to logits
pub fn apply_repetition_penalty(
    logits: &Tensor<f64>,
    previous_tokens: &[u32],
    penalty: f64,
) -> Tensor<f64> {
    if penalty == 1.0 || previous_tokens.is_empty() {
        return logits.clone();
    }

    // TODO: Implement actual repetition penalty
    // For each token in previous_tokens:
    //   if logits[token] > 0: logits[token] /= penalty
    //   else: logits[token] *= penalty
    // For now, just return the original logits
    tracing::debug!(
        "Applying repetition penalty={} for {} tokens",
        penalty,
        previous_tokens.len()
    );
    logits.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, None);
        assert_eq!(config.top_p, None);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_greedy_config() {
        let config = SamplingConfig::greedy();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, Some(1));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_top_k_config() {
        let config = SamplingConfig::top_k(50, 0.8);
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_k, Some(50));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_nucleus_config() {
        let config = SamplingConfig::nucleus(0.9, 0.7);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, Some(0.9));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_temperature() {
        let config = SamplingConfig {
            temperature: -1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_top_k() {
        let config = SamplingConfig {
            top_k: Some(0),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_top_p() {
        let config = SamplingConfig {
            top_p: Some(1.5),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sample_token_basic() {
        let logits = Tensor::<f64>::zeros(&[1, 100]);
        let config = SamplingConfig::default();
        let tokens: Vec<u32> = vec![];

        let result = sample_token(&logits, &config, &tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits).unwrap();

        // Check sum equals 1.0
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check probabilities are in ascending order
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_apply_top_k_to_probs() {
        let probs = vec![0.1, 0.4, 0.3, 0.2];
        let filtered = apply_top_k_to_probs(&probs, 2).unwrap();

        // Only top-2 should be non-zero
        let non_zero_count = filtered.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(non_zero_count, 2);

        // Sum should be 1.0
        let sum: f64 = filtered.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_top_p_to_probs() {
        let probs = vec![0.5, 0.3, 0.15, 0.05];
        let filtered = apply_top_p_to_probs(&probs, 0.8).unwrap();

        // Should keep smallest set that sums to >= 0.8
        let sum: f64 = filtered.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multinomial_sample() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];

        // Should return valid index
        for _ in 0..100 {
            let idx = multinomial_sample(&probs).unwrap();
            assert!(idx < 4);
        }
    }

    #[test]
    fn test_apply_temperature() {
        let logits = Tensor::<f64>::zeros(&[10]);
        let scaled = apply_temperature(&logits, 0.8);
        assert_eq!(scaled.shape(), logits.shape());
    }

    #[test]
    fn test_apply_top_k() {
        let logits = Tensor::<f64>::zeros(&[100]);
        let filtered = apply_top_k(&logits, 10);
        assert_eq!(filtered.shape(), logits.shape());
    }

    #[test]
    fn test_apply_top_p() {
        let logits = Tensor::<f64>::zeros(&[100]);
        let filtered = apply_top_p(&logits, 0.9);
        assert_eq!(filtered.shape(), logits.shape());
    }

    #[test]
    fn test_apply_repetition_penalty() {
        let logits = Tensor::<f64>::zeros(&[100]);
        let tokens = vec![1, 5, 10];
        let penalized = apply_repetition_penalty(&logits, &tokens, 1.2);
        assert_eq!(penalized.shape(), logits.shape());
    }
}
