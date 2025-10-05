// Feed-Forward Network implementation
use anyhow::Result;
use rustorch::prelude::Tensor;

/// Position-wise Feed-Forward Network
/// FFN(x) = max(0, xW1 + b1)W2 + b2
/// Typically expands to 4*d_model in the hidden layer
#[derive(Debug, Clone)]
pub struct FeedForward {
    d_model: usize,
    d_ff: usize, // Hidden dimension (typically 4 * d_model)

    // Layer 1: d_model -> d_ff
    w1: Option<Tensor<f64>>,
    b1: Option<Tensor<f64>>,

    // Layer 2: d_ff -> d_model
    w2: Option<Tensor<f64>>,
    b2: Option<Tensor<f64>>,

    dropout: f64,
}

impl FeedForward {
    /// Create a new FeedForward layer
    pub fn new(d_model: usize, d_ff: usize, dropout: f64) -> Self {
        Self {
            d_model,
            d_ff,
            w1: None,
            b1: None,
            w2: None,
            b2: None,
            dropout,
        }
    }

    /// Initialize weights with Xavier/Glorot initialization
    pub fn init_weights(&mut self) -> Result<()> {
        // Xavier initialization
        let std1 = (2.0 / (self.d_model + self.d_ff) as f64).sqrt();
        let std2 = (2.0 / (self.d_ff + self.d_model) as f64).sqrt();

        // Layer 1: d_model -> d_ff
        self.w1 = Some(Self::xavier_init(self.d_ff, self.d_model, std1)?);
        self.b1 = Some(Tensor::from_vec(vec![0.0; self.d_ff], vec![self.d_ff]));

        // Layer 2: d_ff -> d_model
        self.w2 = Some(Self::xavier_init(self.d_model, self.d_ff, std2)?);
        self.b2 = Some(Tensor::from_vec(
            vec![0.0; self.d_model],
            vec![self.d_model],
        ));

        Ok(())
    }

    /// Xavier/Glorot initialization
    fn xavier_init(rows: usize, cols: usize, std: f64) -> Result<Tensor<f64>> {
        let size = rows * cols;
        let mut data = Vec::with_capacity(size);

        for i in 0..size {
            // Simple pseudo-random initialization
            let val = ((i as f64 * 54321.0).sin() * std).rem_euclid(std);
            data.push(val - std / 2.0);
        }

        Ok(Tensor::from_vec(data, vec![rows, cols]))
    }

    /// Forward pass
    /// Input shape: [batch_size, seq_len, d_model]
    /// Output shape: [batch_size, seq_len, d_model]
    pub fn forward(&self, input: &Tensor<f64>) -> Result<Tensor<f64>> {
        // First linear layer: x @ W1^T + b1
        let hidden =
            self.linear_with_bias(input, self.w1.as_ref().unwrap(), self.b1.as_ref().unwrap())?;

        // ReLU activation: max(0, x)
        let activated = self.relu(&hidden)?;

        // Dropout (simplified - no actual dropout during inference)
        let dropped = if self.dropout > 0.0 {
            self.apply_dropout(&activated, self.dropout)?
        } else {
            activated
        };

        // Second linear layer: x @ W2^T + b2
        self.linear_with_bias(
            &dropped,
            self.w2.as_ref().unwrap(),
            self.b2.as_ref().unwrap(),
        )
    }

    /// Linear transformation with bias: input @ weight^T + bias
    fn linear_with_bias(
        &self,
        input: &Tensor<f64>,
        weight: &Tensor<f64>,
        bias: &Tensor<f64>,
    ) -> Result<Tensor<f64>> {
        let input_data = input
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Failed to get input data"))?;
        let weight_data = weight
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Failed to get weight data"))?;
        let bias_data = bias
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Failed to get bias data"))?;
        let input_shape = input.size();
        let weight_shape = weight.size();

        if input_shape.is_empty() || weight_shape.len() != 2 {
            anyhow::bail!("Invalid shapes for linear layer");
        }

        let d_in = input_shape[input_shape.len() - 1];
        let d_out = weight_shape[0];

        if weight_shape[1] != d_in {
            anyhow::bail!(
                "Weight dimension mismatch: expected {}, got {}",
                d_in,
                weight_shape[1]
            );
        }

        if bias_data.len() != d_out {
            anyhow::bail!(
                "Bias dimension mismatch: expected {}, got {}",
                d_out,
                bias_data.len()
            );
        }

        // Calculate batch size
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();

        let mut output_data = Vec::with_capacity(batch_size * d_out);

        for b in 0..batch_size {
            for i in 0..d_out {
                let mut sum = bias_data[i]; // Add bias
                for j in 0..d_in {
                    let input_idx = b * d_in + j;
                    let weight_idx = i * d_in + j;
                    sum += input_data[input_idx] * weight_data[weight_idx];
                }
                output_data.push(sum);
            }
        }

        let mut output_shape = input_shape.to_vec();
        let last_idx = output_shape.len() - 1;
        output_shape[last_idx] = d_out;

        Ok(Tensor::from_vec(output_data, output_shape))
    }

    /// ReLU activation: max(0, x)
    fn relu(&self, input: &Tensor<f64>) -> Result<Tensor<f64>> {
        let data = input
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Failed to get input data"))?;
        let activated: Vec<f64> = data.iter().map(|&x| x.max(0.0)).collect();

        Ok(Tensor::from_vec(activated, input.size().to_vec()))
    }

    /// Apply dropout (simplified - for training)
    /// During inference, this is typically a no-op
    fn apply_dropout(&self, input: &Tensor<f64>, _rate: f64) -> Result<Tensor<f64>> {
        // For now, just return input (dropout is only during training)
        Ok(input.clone())
    }

    /// GELU activation (alternative to ReLU, used in GPT-2 and modern models)
    /// GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
    #[allow(dead_code)]
    fn gelu(&self, input: &Tensor<f64>) -> Result<Tensor<f64>> {
        let data = input
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Failed to get input data"))?;
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();

        let activated: Vec<f64> = data
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();

        Ok(Tensor::from_vec(activated, input.size().to_vec()))
    }

    /// Get learnable parameters
    pub fn parameters(&self) -> Vec<&Tensor<f64>> {
        vec![
            self.w1.as_ref().unwrap(),
            self.b1.as_ref().unwrap(),
            self.w2.as_ref().unwrap(),
            self.b2.as_ref().unwrap(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward_creation() {
        let ff = FeedForward::new(512, 2048, 0.1);
        assert_eq!(ff.d_model, 512);
        assert_eq!(ff.d_ff, 2048);
        assert_eq!(ff.dropout, 0.1);
    }

    #[test]
    fn test_weight_initialization() {
        let mut ff = FeedForward::new(256, 1024, 0.0);
        ff.init_weights().unwrap();

        assert!(ff.w1.is_some());
        assert!(ff.b1.is_some());
        assert!(ff.w2.is_some());
        assert!(ff.b2.is_some());

        let w1 = ff.w1.as_ref().unwrap();
        assert_eq!(w1.size(), &[1024, 256]);
    }

    #[test]
    fn test_forward() {
        let mut ff = FeedForward::new(64, 256, 0.0);
        ff.init_weights().unwrap();

        // Input: [batch=2, seq_len=10, d_model=64]
        let input = Tensor::from_vec(vec![0.1; 2 * 10 * 64], vec![2, 10, 64]);

        let output = ff.forward(&input).unwrap();
        assert_eq!(output.size(), &[2, 10, 64]);
    }

    #[test]
    fn test_relu() {
        let ff = FeedForward::new(128, 512, 0.0);

        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let output = ff.relu(&input).unwrap();

        let expected = vec![0.0, 0.0, 1.0, 2.0];
        assert_eq!(output.data.as_slice().unwrap(), &expected[..]);
    }

    #[test]
    fn test_gelu() {
        let ff = FeedForward::new(128, 512, 0.0);

        let input = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);
        let output = ff.gelu(&input).unwrap();

        // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert_eq!(output.size(), &[3]);
        assert!(output.data[0].abs() < 0.1); // GELU(0) ≈ 0
    }

    #[test]
    fn test_linear_with_bias() {
        let ff = FeedForward::new(4, 8, 0.0);

        let input = Tensor::from_vec(vec![1.0; 4], vec![1, 4]);
        let weight = Tensor::from_vec(vec![0.1; 32], vec![8, 4]);
        let bias = Tensor::from_vec(vec![0.5; 8], vec![8]);

        let output = ff.linear_with_bias(&input, &weight, &bias).unwrap();
        assert_eq!(output.size(), &[1, 8]);
    }
}
