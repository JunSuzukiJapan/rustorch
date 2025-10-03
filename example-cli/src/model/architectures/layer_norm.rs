// Layer Normalization implementation
use anyhow::Result;
use rustorch::prelude::Tensor;

/// Layer Normalization
/// Normalizes across features for each sample independently
#[derive(Debug, Clone)]
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: f64,
    gamma: Option<Tensor<f64>>, // Scale parameter
    beta: Option<Tensor<f64>>,  // Shift parameter
    elementwise_affine: bool,
}

impl LayerNorm {
    /// Create a new LayerNorm layer
    pub fn new(normalized_shape: Vec<usize>, eps: f64, elementwise_affine: bool) -> Self {
        Self {
            normalized_shape,
            eps,
            gamma: None,
            beta: None,
            elementwise_affine,
        }
    }

    /// Initialize learnable parameters
    pub fn init_parameters(&mut self) -> Result<()> {
        if self.elementwise_affine {
            // Initialize gamma (scale) to ones
            let size: usize = self.normalized_shape.iter().product();
            let gamma_data = vec![1.0; size];
            self.gamma = Some(Tensor::from_vec(gamma_data, self.normalized_shape.clone()));

            // Initialize beta (shift) to zeros
            let beta_data = vec![0.0; size];
            self.beta = Some(Tensor::from_vec(beta_data, self.normalized_shape.clone()));
        }

        Ok(())
    }

    /// Forward pass
    /// Input shape: [batch_size, ..., normalized_shape]
    /// Output shape: Same as input
    pub fn forward(&self, input: &Tensor<f64>) -> Result<Tensor<f64>> {
        // Calculate mean across normalized dimensions
        let mean = Self::calculate_mean(input)?;

        // Calculate variance
        let variance = Self::calculate_variance(input, &mean)?;

        // Normalize: (x - mean) / sqrt(variance + eps)
        let normalized = Self::normalize(input, &mean, &variance, self.eps)?;

        // Apply affine transformation if enabled
        if self.elementwise_affine {
            if let (Some(gamma), Some(beta)) = (&self.gamma, &self.beta) {
                // y = gamma * normalized + beta
                return Self::apply_affine(&normalized, gamma, beta);
            }
        }

        Ok(normalized)
    }

    /// Calculate mean across features
    fn calculate_mean(input: &Tensor<f64>) -> Result<Tensor<f64>> {
        let data = input.data;
        let shape = input.size();

        // For now, calculate mean across last dimension
        // In a full implementation, would handle arbitrary normalized_shape
        if shape.is_empty() {
            return Ok(Tensor::from_vec(vec![0.0], vec![1]));
        }

        let last_dim = shape[shape.len() - 1];
        let batch_size = data.len() / last_dim;

        let mut mean_data = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let start = i * last_dim;
            let end = start + last_dim;
            let sum: f64 = data[start..end].iter().sum();
            mean_data.push(sum / last_dim as f64);
        }

        let mut mean_shape = shape.to_vec();
        mean_shape[mean_shape.len() - 1] = 1;

        Ok(Tensor::from_vec(mean_data, mean_shape))
    }

    /// Calculate variance across features
    fn calculate_variance(input: &Tensor<f64>, mean: &Tensor<f64>) -> Result<Tensor<f64>> {
        let data = input.data;
        let mean_data = mean.data;
        let shape = input.size();

        if shape.is_empty() {
            return Ok(Tensor::from_vec(vec![0.0], vec![1]));
        }

        let last_dim = shape[shape.len() - 1];
        let batch_size = data.len() / last_dim;

        let mut var_data = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let start = i * last_dim;
            let end = start + last_dim;
            let mean_val = mean_data[i];

            let variance: f64 = data[start..end]
                .iter()
                .map(|&x| {
                    let diff = x - mean_val;
                    diff * diff
                })
                .sum::<f64>()
                / last_dim as f64;

            var_data.push(variance);
        }

        let mut var_shape = shape.to_vec();
        var_shape[var_shape.len() - 1] = 1;

        Ok(Tensor::from_vec(var_data, var_shape))
    }

    /// Normalize input
    fn normalize(
        input: &Tensor<f64>,
        mean: &Tensor<f64>,
        variance: &Tensor<f64>,
        eps: f64,
    ) -> Result<Tensor<f64>> {
        let data = input.data;
        let mean_data = mean.data;
        let var_data = variance.data;
        let shape = input.size();

        if shape.is_empty() {
            return Ok(input.clone());
        }

        let last_dim = shape[shape.len() - 1];
        let batch_size = data.len() / last_dim;

        let mut normalized_data = Vec::with_capacity(data.len());

        for i in 0..batch_size {
            let start = i * last_dim;
            let end = start + last_dim;
            let mean_val = mean_data[i];
            let std_val = (var_data[i] + eps).sqrt();

            for &x in &data[start..end] {
                normalized_data.push((x - mean_val) / std_val);
            }
        }

        Ok(Tensor::from_vec(normalized_data, shape.to_vec()))
    }

    /// Apply affine transformation
    fn apply_affine(
        normalized: &Tensor<f64>,
        gamma: &Tensor<f64>,
        beta: &Tensor<f64>,
    ) -> Result<Tensor<f64>> {
        let norm_data = normalized.data;
        let gamma_data = gamma.data;
        let beta_data = beta.data;

        let result_data: Vec<f64> = norm_data
            .iter()
            .zip(gamma_data.iter().cycle())
            .zip(beta_data.iter().cycle())
            .map(|((&n, &g), &b)| g * n + b)
            .collect();

        Ok(Tensor::from_vec(result_data, normalized.size().to_vec()))
    }

    /// Get learnable parameters
    pub fn parameters(&self) -> Vec<&Tensor<f64>> {
        let mut params = Vec::new();
        if let Some(ref gamma) = self.gamma {
            params.push(gamma);
        }
        if let Some(ref beta) = self.beta {
            params.push(beta);
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let ln = LayerNorm::new(vec![128], 1e-5, true);
        assert_eq!(ln.normalized_shape, vec![128]);
        assert_eq!(ln.eps, 1e-5);
        assert!(ln.elementwise_affine);
    }

    #[test]
    fn test_layer_norm_init() {
        let mut ln = LayerNorm::new(vec![4], 1e-5, true);
        ln.init_parameters().unwrap();
        assert!(ln.gamma.is_some());
        assert!(ln.beta.is_some());
    }

    #[test]
    fn test_layer_norm_forward() {
        let mut ln = LayerNorm::new(vec![4], 1e-5, false);
        ln.init_parameters().unwrap();

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let output = ln.forward(&input).unwrap();

        assert_eq!(output.size(), &[1, 4]);

        // Check that mean is approximately 0
        let out_data = output.data;
        let mean: f64 = out_data.iter().sum::<f64>() / out_data.len() as f64;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_layer_norm_with_affine() {
        let mut ln = LayerNorm::new(vec![4], 1e-5, true);
        ln.init_parameters().unwrap();

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let output = ln.forward(&input).unwrap();

        assert_eq!(output.size(), &[1, 4]);
    }
}
