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
        let mean = self.calculate_mean(input)?;

        // Calculate variance
        let variance = self.calculate_variance(input, &mean)?;

        // Normalize: (x - mean) / sqrt(variance + eps)
        let normalized = self.normalize(input, &mean, &variance)?;

        // Apply affine transformation if enabled
        if self.elementwise_affine {
            if let (Some(gamma), Some(beta)) = (&self.gamma, &self.beta) {
                // y = gamma * normalized + beta
                return self.apply_affine(&normalized, gamma, beta);
            }
        }

        Ok(normalized)
    }

    /// Calculate mean across features
    fn calculate_mean(&self, input: &Tensor<f64>) -> Result<Tensor<f64>> {
        // Simplified: calculate global mean for now
        // Full implementation would use RusTorch reduction operations
        let input_data = input.as_slice().ok_or_else(|| anyhow::anyhow!("Failed to get input data"))?;
        let total_elements = input_data.len();
        if total_elements == 0 {
            return Ok(Tensor::from_vec(vec![0.0], vec![1]));
        }

        let sum: f64 = input_data.iter().sum();
        let mean = sum / total_elements as f64;

        Ok(Tensor::from_vec(vec![mean], vec![1]))
    }

    /// Calculate variance across features
    fn calculate_variance(&self, input: &Tensor<f64>, mean: &Tensor<f64>) -> Result<Tensor<f64>> {
        // Simplified: calculate global variance for now
        let input_data = input.as_slice().ok_or_else(|| anyhow::anyhow!("Failed to get input data"))?;
        let total_elements = input_data.len();
        if total_elements == 0 {
            return Ok(Tensor::from_vec(vec![0.0], vec![1]));
        }

        let mean_data = mean.as_slice().ok_or_else(|| anyhow::anyhow!("Failed to get mean data"))?;
        let mean_val = mean_data[0];
        let variance: f64 = input_data
            .iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .sum::<f64>()
            / total_elements as f64;

        Ok(Tensor::from_vec(vec![variance], vec![1]))
    }

    /// Normalize input
    fn normalize(
        &self,
        input: &Tensor<f64>,
        mean: &Tensor<f64>,
        variance: &Tensor<f64>,
    ) -> Result<Tensor<f64>> {
        // Simplified: global normalization
        let shape = input.size();
        if shape.is_empty() {
            return Ok(input.clone());
        }

        let mean_val = mean.data[[0]];
        let std_val = (variance.data[[0]] + self.eps).sqrt();

        let normalized_data: Vec<f64> = input
            .data
            .iter()
            .map(|&x| (x - mean_val) / std_val)
            .collect();

        Ok(Tensor::from_vec(normalized_data, shape.to_vec()))
    }

    /// Apply affine transformation
    fn apply_affine(
        &self,
        normalized: &Tensor<f64>,
        gamma: &Tensor<f64>,
        beta: &Tensor<f64>,
    ) -> Result<Tensor<f64>> {
        // Use RusTorch operations for affine transformation
        // result = gamma * normalized + beta
        let scaled = normalized * gamma;
        Ok(&scaled + beta)
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
