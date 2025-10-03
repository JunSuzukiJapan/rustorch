// Multi-Head Attention implementation
use anyhow::Result;
use rustorch::prelude::Tensor;

/// Multi-Head Attention layer
/// Implements scaled dot-product attention with multiple heads
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    d_model: usize,
    num_heads: usize,
    d_k: usize, // Dimension per head (d_model / num_heads)

    // Linear projection weights
    w_q: Option<Tensor<f64>>, // Query projection
    w_k: Option<Tensor<f64>>, // Key projection
    w_v: Option<Tensor<f64>>, // Value projection
    w_o: Option<Tensor<f64>>, // Output projection
}

impl MultiHeadAttention {
    /// Create a new MultiHeadAttention layer
    pub fn new(d_model: usize, num_heads: usize) -> Result<Self> {
        if d_model % num_heads != 0 {
            anyhow::bail!(
                "d_model ({}) must be divisible by num_heads ({})",
                d_model,
                num_heads
            );
        }

        let d_k = d_model / num_heads;

        Ok(Self {
            d_model,
            num_heads,
            d_k,
            w_q: None,
            w_k: None,
            w_v: None,
            w_o: None,
        })
    }

    /// Initialize weight matrices with Xavier/Glorot initialization
    pub fn init_weights(&mut self) -> Result<()> {
        let std = (2.0 / (self.d_model as f64)).sqrt();

        // Initialize query, key, value projection weights
        self.w_q = Some(Self::xavier_init(self.d_model, self.d_model, std)?);
        self.w_k = Some(Self::xavier_init(self.d_model, self.d_model, std)?);
        self.w_v = Some(Self::xavier_init(self.d_model, self.d_model, std)?);
        self.w_o = Some(Self::xavier_init(self.d_model, self.d_model, std)?);

        Ok(())
    }

    /// Xavier/Glorot initialization
    fn xavier_init(rows: usize, cols: usize, std: f64) -> Result<Tensor<f64>> {
        // Initialize with small random values (simplified - real implementation would use proper RNG)
        let size = rows * cols;
        let mut data = Vec::with_capacity(size);

        for i in 0..size {
            // Simple pseudo-random initialization
            let val = ((i as f64 * 12345.0).sin() * std).rem_euclid(std);
            data.push(val - std / 2.0);
        }

        Ok(Tensor::from_vec(data, vec![rows, cols]))
    }

    /// Forward pass with self-attention
    /// Input shape: [batch_size, seq_len, d_model]
    /// Output shape: [batch_size, seq_len, d_model]
    pub fn forward(&self, input: &Tensor<f64>, mask: Option<&Tensor<f64>>) -> Result<Tensor<f64>> {
        self.forward_with_kv(input, input, input, mask)
    }

    /// Forward pass with separate key-value pairs (for cross-attention)
    /// q: Query tensor [batch_size, seq_len_q, d_model]
    /// k: Key tensor [batch_size, seq_len_k, d_model]
    /// v: Value tensor [batch_size, seq_len_v, d_model]
    pub fn forward_with_kv(
        &self,
        q: &Tensor<f64>,
        k: &Tensor<f64>,
        v: &Tensor<f64>,
        mask: Option<&Tensor<f64>>,
    ) -> Result<Tensor<f64>> {
        let q_shape = q.size();
        if q_shape.len() != 3 {
            anyhow::bail!(
                "Expected query shape [batch, seq_len, d_model], got {:?}",
                q_shape
            );
        }

        let batch_size = q_shape[0];
        let seq_len_q = q_shape[1];
        let seq_len_k = k.size()[1];

        // Linear projections
        let q_proj = self.linear_projection(q, self.w_q.as_ref().unwrap())?;
        let k_proj = self.linear_projection(k, self.w_k.as_ref().unwrap())?;
        let v_proj = self.linear_projection(v, self.w_v.as_ref().unwrap())?;

        // Reshape for multi-head: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        let q_heads = self.split_heads(&q_proj, batch_size, seq_len_q)?;
        let k_heads = self.split_heads(&k_proj, batch_size, seq_len_k)?;
        let v_heads = self.split_heads(&v_proj, batch_size, seq_len_k)?;

        // Scaled dot-product attention for each head
        let attention_output =
            self.scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask)?;

        // Concatenate heads: [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        let concat = self.concatenate_heads(&attention_output, batch_size, seq_len_q)?;

        // Final output projection
        self.linear_projection(&concat, self.w_o.as_ref().unwrap())
    }

    /// Linear projection: input @ weight^T
    fn linear_projection(&self, input: &Tensor<f64>, weight: &Tensor<f64>) -> Result<Tensor<f64>> {
        let input_data = input.data;
        let weight_data = weight.data;
        let input_shape = input.size();
        let weight_shape = weight.size();

        if input_shape.is_empty() || weight_shape.len() != 2 {
            anyhow::bail!("Invalid shapes for linear projection");
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

        // Calculate batch size
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();

        let mut output_data = Vec::with_capacity(batch_size * d_out);

        for b in 0..batch_size {
            for i in 0..d_out {
                let mut sum = 0.0;
                for j in 0..d_in {
                    let input_idx = b * d_in + j;
                    let weight_idx = i * d_in + j;
                    sum += input_data[input_idx] * weight_data[weight_idx];
                }
                output_data.push(sum);
            }
        }

        let mut output_shape = input_shape.to_vec();
        output_shape[output_shape.len() - 1] = d_out;

        Ok(Tensor::from_vec(output_data, output_shape))
    }

    /// Split into multiple heads
    fn split_heads(
        &self,
        input: &Tensor<f64>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor<f64>> {
        let input_data = input.data;

        // Reshape from [batch, seq_len, d_model] to [batch, seq_len, num_heads, d_k]
        // Then permute to [batch, num_heads, seq_len, d_k]
        let mut output_data = Vec::with_capacity(input_data.len());

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for s in 0..seq_len {
                    for d in 0..self.d_k {
                        let input_idx =
                            b * seq_len * self.d_model + s * self.d_model + h * self.d_k + d;
                        output_data.push(input_data[input_idx]);
                    }
                }
            }
        }

        Ok(Tensor::from_vec(
            output_data,
            vec![batch_size, self.num_heads, seq_len, self.d_k],
        ))
    }

    /// Concatenate multi-head outputs
    fn concatenate_heads(
        &self,
        input: &Tensor<f64>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor<f64>> {
        let input_data = input.data;

        // Reshape from [batch, num_heads, seq_len, d_k] to [batch, seq_len, d_model]
        let mut output_data = Vec::with_capacity(input_data.len());

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.d_k {
                        let input_idx = b * self.num_heads * seq_len * self.d_k
                            + h * seq_len * self.d_k
                            + s * self.d_k
                            + d;
                        output_data.push(input_data[input_idx]);
                    }
                }
            }
        }

        Ok(Tensor::from_vec(
            output_data,
            vec![batch_size, seq_len, self.d_model],
        ))
    }

    /// Scaled dot-product attention
    /// Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    fn scaled_dot_product_attention(
        &self,
        q: &Tensor<f64>,
        k: &Tensor<f64>,
        v: &Tensor<f64>,
        _mask: Option<&Tensor<f64>>,
    ) -> Result<Tensor<f64>> {
        let q_data = q.data;
        let k_data = k.data;
        let v_data = v.data;
        let q_shape = q.size();

        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_len_q = q_shape[2];
        let seq_len_k = k.size()[2];
        let d_k = q_shape[3];

        let scale = 1.0 / (d_k as f64).sqrt();

        // Compute Q @ K^T / sqrt(d_k)
        let mut scores = Vec::with_capacity(batch_size * num_heads * seq_len_q * seq_len_k);

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_q {
                    for j in 0..seq_len_k {
                        let mut score = 0.0;
                        for d in 0..d_k {
                            let q_idx =
                                b * num_heads * seq_len_q * d_k + h * seq_len_q * d_k + i * d_k + d;
                            let k_idx =
                                b * num_heads * seq_len_k * d_k + h * seq_len_k * d_k + j * d_k + d;
                            score += q_data[q_idx] * k_data[k_idx];
                        }
                        scores.push(score * scale);
                    }
                }
            }
        }

        // Apply softmax
        let attention_weights =
            self.softmax_2d(&scores, batch_size * num_heads, seq_len_q, seq_len_k)?;

        // Apply attention weights to V
        let mut output_data = Vec::with_capacity(batch_size * num_heads * seq_len_q * d_k);

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_q {
                    for d in 0..d_k {
                        let mut value = 0.0;
                        for j in 0..seq_len_k {
                            let weight_idx =
                                (b * num_heads + h) * seq_len_q * seq_len_k + i * seq_len_k + j;
                            let v_idx =
                                b * num_heads * seq_len_k * d_k + h * seq_len_k * d_k + j * d_k + d;
                            value += attention_weights[weight_idx] * v_data[v_idx];
                        }
                        output_data.push(value);
                    }
                }
            }
        }

        Ok(Tensor::from_vec(
            output_data,
            vec![batch_size, num_heads, seq_len_q, d_k],
        ))
    }

    /// Softmax over last dimension (2D version for efficiency)
    fn softmax_2d(
        &self,
        input: &[f64],
        batch_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f64>> {
        let mut output = Vec::with_capacity(input.len());

        for b in 0..batch_size {
            for i in 0..rows {
                let start = b * rows * cols + i * cols;
                let end = start + cols;
                let row = &input[start..end];

                // Find max for numerical stability
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                // Compute exp(x - max)
                let exp_values: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();

                // Compute sum of exponentials
                let sum: f64 = exp_values.iter().sum();

                // Normalize
                for exp_val in exp_values {
                    output.push(exp_val / sum);
                }
            }
        }

        Ok(output)
    }

    /// Get learnable parameters
    pub fn parameters(&self) -> Vec<&Tensor<f64>> {
        vec![
            self.w_q.as_ref().unwrap(),
            self.w_k.as_ref().unwrap(),
            self.w_v.as_ref().unwrap(),
            self.w_o.as_ref().unwrap(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention_creation() {
        let mha = MultiHeadAttention::new(512, 8).unwrap();
        assert_eq!(mha.d_model, 512);
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.d_k, 64);
    }

    #[test]
    fn test_multi_head_attention_invalid_heads() {
        let result = MultiHeadAttention::new(512, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_initialization() {
        let mut mha = MultiHeadAttention::new(256, 4).unwrap();
        mha.init_weights().unwrap();

        assert!(mha.w_q.is_some());
        assert!(mha.w_k.is_some());
        assert!(mha.w_v.is_some());
        assert!(mha.w_o.is_some());
    }

    #[test]
    fn test_forward() {
        let mut mha = MultiHeadAttention::new(64, 4).unwrap();
        mha.init_weights().unwrap();

        // Input: [batch=2, seq_len=10, d_model=64]
        let input = Tensor::from_vec(vec![0.1; 2 * 10 * 64], vec![2, 10, 64]);

        let output = mha.forward(&input, None).unwrap();
        assert_eq!(output.size(), &[2, 10, 64]);
    }

    #[test]
    fn test_linear_projection() {
        let mha = MultiHeadAttention::new(128, 8).unwrap();

        let input = Tensor::from_vec(vec![1.0; 128], vec![1, 128]);
        let weight = Tensor::from_vec(vec![0.1; 128 * 128], vec![128, 128]);

        let output = mha.linear_projection(&input, &weight).unwrap();
        assert_eq!(output.size(), &[1, 128]);
    }
}
