// Positional Encoding implementation
use anyhow::Result;
use rustorch::prelude::Tensor;

/// Positional Encoding using sinusoidal functions
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    d_model: usize,
    max_len: usize,
    encoding: Option<Tensor<f64>>,
}

impl PositionalEncoding {
    /// Create a new PositionalEncoding
    pub fn new(d_model: usize, max_len: usize) -> Self {
        Self {
            d_model,
            max_len,
            encoding: None,
        }
    }

    /// Initialize positional encoding table
    pub fn init(&mut self) -> Result<()> {
        let mut pe_data = Vec::with_capacity(self.max_len * self.d_model);

        for pos in 0..self.max_len {
            for i in 0..self.d_model {
                let angle = Self::calculate_angle(pos, i, self.d_model);

                // Even indices: sin, Odd indices: cos
                let value = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };

                pe_data.push(value);
            }
        }

        self.encoding = Some(Tensor::from_vec(
            pe_data,
            vec![self.max_len, self.d_model],
        ));

        Ok(())
    }

    /// Calculate angle for positional encoding
    fn calculate_angle(pos: usize, i: usize, d_model: usize) -> f64 {
        let pos_f = pos as f64;
        let i_f = (i / 2) as f64; // Integer division to pair sin/cos

        // pos / 10000^(2i/d_model)
        let div_term = 10000_f64.powf(2.0 * i_f / d_model as f64);
        pos_f / div_term
    }

    /// Add positional encoding to input embeddings
    /// Input shape: [batch_size, seq_len, d_model]
    /// Output shape: [batch_size, seq_len, d_model]
    pub fn forward(&self, input: &Tensor<f64>) -> Result<Tensor<f64>> {
        let input_data = input.data;
        let input_shape = input.size();

        if input_shape.len() != 3 {
            anyhow::bail!(
                "Expected input shape [batch_size, seq_len, d_model], got {:?}",
                input_shape
            );
        }

        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let d_model = input_shape[2];

        if d_model != self.d_model {
            anyhow::bail!(
                "Input d_model {} doesn't match positional encoding d_model {}",
                d_model,
                self.d_model
            );
        }

        if seq_len > self.max_len {
            anyhow::bail!(
                "Sequence length {} exceeds max_len {}",
                seq_len,
                self.max_len
            );
        }

        let encoding = self
            .encoding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Positional encoding not initialized"))?;
        let pe_data = encoding.data;

        // Add positional encoding to input
        let mut output_data = Vec::with_capacity(input_data.len());

        for b in 0..batch_size {
            for pos in 0..seq_len {
                for i in 0..d_model {
                    let input_idx = b * seq_len * d_model + pos * d_model + i;
                    let pe_idx = pos * d_model + i;

                    output_data.push(input_data[input_idx] + pe_data[pe_idx]);
                }
            }
        }

        Ok(Tensor::from_vec(output_data, input_shape.to_vec()))
    }

    /// Get encoding for a specific position range
    pub fn get_encoding(&self, start: usize, end: usize) -> Result<Tensor<f64>> {
        let encoding = self
            .encoding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Positional encoding not initialized"))?;

        if end > self.max_len {
            anyhow::bail!("End position {} exceeds max_len {}", end, self.max_len);
        }

        let pe_data = encoding.data;
        let length = end - start;

        let mut output_data = Vec::with_capacity(length * self.d_model);
        for pos in start..end {
            for i in 0..self.d_model {
                let idx = pos * self.d_model + i;
                output_data.push(pe_data[idx]);
            }
        }

        Ok(Tensor::from_vec(output_data, vec![length, self.d_model]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positional_encoding_creation() {
        let pe = PositionalEncoding::new(512, 2048);
        assert_eq!(pe.d_model, 512);
        assert_eq!(pe.max_len, 2048);
    }

    #[test]
    fn test_positional_encoding_init() {
        let mut pe = PositionalEncoding::new(128, 512);
        pe.init().unwrap();
        assert!(pe.encoding.is_some());

        let encoding = pe.encoding.as_ref().unwrap();
        assert_eq!(encoding.size(), &[512, 128]);
    }

    #[test]
    fn test_angle_calculation() {
        let angle = PositionalEncoding::calculate_angle(0, 0, 512);
        assert_eq!(angle, 0.0);

        let angle = PositionalEncoding::calculate_angle(1, 0, 512);
        assert!(angle > 0.0);
    }

    #[test]
    fn test_forward() {
        let mut pe = PositionalEncoding::new(64, 128);
        pe.init().unwrap();

        // Create input tensor: [batch_size=2, seq_len=10, d_model=64]
        let input_data = vec![0.5; 2 * 10 * 64];
        let input = Tensor::from_vec(input_data, vec![2, 10, 64]);

        let output = pe.forward(&input).unwrap();
        assert_eq!(output.size(), &[2, 10, 64]);
    }

    #[test]
    fn test_get_encoding() {
        let mut pe = PositionalEncoding::new(32, 64);
        pe.init().unwrap();

        let encoding = pe.get_encoding(0, 10).unwrap();
        assert_eq!(encoding.size(), &[10, 32]);
    }

    #[test]
    fn test_forward_wrong_dimensions() {
        let mut pe = PositionalEncoding::new(64, 128);
        pe.init().unwrap();

        // Wrong shape: only 2D
        let input = Tensor::from_vec(vec![0.5; 64], vec![64]);
        assert!(pe.forward(&input).is_err());
    }

    #[test]
    fn test_forward_wrong_d_model() {
        let mut pe = PositionalEncoding::new(64, 128);
        pe.init().unwrap();

        // Wrong d_model
        let input = Tensor::from_vec(vec![0.5; 2 * 10 * 32], vec![2, 10, 32]);
        assert!(pe.forward(&input).is_err());
    }
}
