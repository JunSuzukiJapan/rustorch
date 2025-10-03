// GPT (Generative Pre-trained Transformer) model implementation
use super::{FeedForward, LayerNorm, MultiHeadAttention, PositionalEncoding};
use anyhow::Result;
use rustorch::prelude::Tensor;

/// GPT Transformer Block
/// Consists of: Multi-Head Attention + LayerNorm + FeedForward + LayerNorm
#[derive(Debug, Clone)]
pub struct GPTBlock {
    attention: MultiHeadAttention,
    ln1: LayerNorm,
    feed_forward: FeedForward,
    ln2: LayerNorm,
}

impl GPTBlock {
    /// Create a new GPT transformer block
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout: f64) -> Result<Self> {
        let mut ln1 = LayerNorm::new(vec![d_model], 1e-5, true);
        ln1.init_parameters()?;

        let mut ln2 = LayerNorm::new(vec![d_model], 1e-5, true);
        ln2.init_parameters()?;

        Ok(Self {
            attention: MultiHeadAttention::new(d_model, num_heads)?,
            ln1,
            feed_forward: FeedForward::new(d_model, d_ff, dropout),
            ln2,
        })
    }

    /// Initialize all weights
    pub fn init_weights(&mut self) -> Result<()> {
        self.attention.init_weights()?;
        self.feed_forward.init_weights()?;
        Ok(())
    }

    /// Forward pass
    /// Input shape: [batch_size, seq_len, d_model]
    /// Output shape: [batch_size, seq_len, d_model]
    pub fn forward(&self, input: &Tensor<f64>, mask: Option<&Tensor<f64>>) -> Result<Tensor<f64>> {
        // Pre-Layer Norm + Multi-Head Attention + Residual
        let ln1_out = self.ln1.forward(input)?;
        let attn_out = self.attention.forward(&ln1_out, mask)?;
        let residual1 = self.add_tensors(input, &attn_out)?;

        // Pre-Layer Norm + FeedForward + Residual
        let ln2_out = self.ln2.forward(&residual1)?;
        let ff_out = self.feed_forward.forward(&ln2_out)?;
        let residual2 = self.add_tensors(&residual1, &ff_out)?;

        Ok(residual2)
    }

    /// Add two tensors (residual connection)
    fn add_tensors(&self, a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>> {
        // Use RusTorch element-wise addition operator
        Ok(a + b)
    }

    /// Get all learnable parameters
    pub fn parameters(&self) -> Vec<&Tensor<f64>> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.ln1.parameters());
        params.extend(self.feed_forward.parameters());
        params.extend(self.ln2.parameters());
        params
    }
}

/// GPT Model Configuration
#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocabulary size
            d_model: 768,      // GPT-2 Small
            num_layers: 12,
            num_heads: 12,
            d_ff: 3072, // 4 * d_model
            max_seq_len: 1024,
            dropout: 0.1,
        }
    }
}

/// GPT Model
pub struct GPTModel {
    config: GPTConfig,

    // Token embedding: [vocab_size, d_model]
    token_embedding: Option<Tensor<f64>>,

    // Positional encoding
    pos_encoding: PositionalEncoding,

    // Transformer blocks
    blocks: Vec<GPTBlock>,

    // Final layer normalization
    ln_final: LayerNorm,

    // Output projection: [d_model, vocab_size]
    output_projection: Option<Tensor<f64>>,
}

impl GPTModel {
    /// Create a new GPT model with given configuration
    pub fn new(config: GPTConfig) -> Result<Self> {
        let mut pos_encoding = PositionalEncoding::new(config.d_model, config.max_seq_len);
        pos_encoding.init()?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(GPTBlock::new(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.dropout,
            )?);
        }

        let mut ln_final = LayerNorm::new(vec![config.d_model], 1e-5, true);
        ln_final.init_parameters()?;

        Ok(Self {
            config,
            token_embedding: None,
            pos_encoding,
            blocks,
            ln_final,
            output_projection: None,
        })
    }

    /// Initialize all weights
    pub fn init_weights(&mut self) -> Result<()> {
        // Initialize token embedding
        let embed_size = self.config.vocab_size * self.config.d_model;
        let std = (1.0 / self.config.d_model as f64).sqrt();
        let mut embed_data = Vec::with_capacity(embed_size);

        for i in 0..embed_size {
            let val = ((i as f64 * 98765.0).sin() * std).rem_euclid(std);
            embed_data.push(val - std / 2.0);
        }

        self.token_embedding = Some(Tensor::from_vec(
            embed_data,
            vec![self.config.vocab_size, self.config.d_model],
        ));

        // Initialize transformer blocks
        for block in &mut self.blocks {
            block.init_weights()?;
        }

        // Initialize output projection (tied with token embedding)
        // In practice, many models share weights between embedding and output
        let proj_size = self.config.d_model * self.config.vocab_size;
        let mut proj_data = Vec::with_capacity(proj_size);

        for i in 0..proj_size {
            let val = ((i as f64 * 45678.0).sin() * std).rem_euclid(std);
            proj_data.push(val - std / 2.0);
        }

        self.output_projection = Some(Tensor::from_vec(
            proj_data,
            vec![self.config.d_model, self.config.vocab_size],
        ));

        Ok(())
    }

    /// Forward pass
    /// Input: token_ids [batch_size, seq_len]
    /// Output: logits [batch_size, seq_len, vocab_size]
    pub fn forward(&self, token_ids: &[usize], batch_size: usize, seq_len: usize) -> Result<Tensor<f64>> {
        // Embedding lookup
        let embedded = self.embed_tokens(token_ids, batch_size, seq_len)?;

        // Add positional encoding
        let pos_encoded = self.pos_encoding.forward(&embedded)?;

        // Pass through transformer blocks
        let mut hidden = pos_encoded;
        for block in &self.blocks {
            hidden = block.forward(&hidden, None)?;
        }

        // Final layer normalization
        let normalized = self.ln_final.forward(&hidden)?;

        // Project to vocabulary size
        self.project_to_vocab(&normalized)
    }

    /// Embed token IDs
    fn embed_tokens(&self, token_ids: &[usize], batch_size: usize, seq_len: usize) -> Result<Tensor<f64>> {
        let embedding = self
            .token_embedding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Token embedding not initialized"))?;

        let d_model = self.config.d_model;

        let mut output_data = Vec::with_capacity(batch_size * seq_len * d_model);

        for &token_id in token_ids {
            if token_id >= self.config.vocab_size {
                anyhow::bail!("Token ID {} out of vocabulary range", token_id);
            }

            // Extract embedding row for this token
            // Access each element using 2D indexing
            for d in 0..d_model {
                let value = embedding.data[[token_id, d]];
                output_data.push(value);
            }
        }

        Ok(Tensor::from_vec(
            output_data,
            vec![batch_size, seq_len, d_model],
        ))
    }

    /// Project hidden states to vocabulary logits
    fn project_to_vocab(&self, hidden: &Tensor<f64>) -> Result<Tensor<f64>> {
        let projection = self
            .output_projection
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Output projection not initialized"))?;

        // Use RusTorch matmul operation
        hidden.matmul(projection).map_err(|e| anyhow::anyhow!("Matmul failed: {}", e))
    }

    /// Get model configuration
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    /// Get all learnable parameters
    pub fn parameters(&self) -> Vec<&Tensor<f64>> {
        let mut params = Vec::new();

        if let Some(ref embed) = self.token_embedding {
            params.push(embed);
        }

        for block in &self.blocks {
            params.extend(block.parameters());
        }

        params.extend(self.ln_final.parameters());

        if let Some(ref proj) = self.output_projection {
            params.push(proj);
        }

        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_block_creation() {
        let block = GPTBlock::new(768, 12, 3072, 0.1).unwrap();
        assert_eq!(block.attention.d_model, 768);
    }

    #[test]
    fn test_gpt_block_forward() {
        let mut block = GPTBlock::new(256, 4, 1024, 0.0).unwrap();
        block.init_weights().unwrap();

        let input = Tensor::from_vec(vec![0.1; 2 * 10 * 256], vec![2, 10, 256]);
        let output = block.forward(&input, None).unwrap();

        assert_eq!(output.size(), &[2, 10, 256]);
    }

    #[test]
    fn test_gpt_config_default() {
        let config = GPTConfig::default();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.d_model, 768);
        assert_eq!(config.num_layers, 12);
    }

    #[test]
    fn test_gpt_model_creation() {
        let config = GPTConfig {
            vocab_size: 1000,
            d_model: 128,
            num_layers: 2,
            num_heads: 4,
            d_ff: 512,
            max_seq_len: 256,
            dropout: 0.1,
        };

        let model = GPTModel::new(config).unwrap();
        assert_eq!(model.blocks.len(), 2);
    }

    #[test]
    fn test_gpt_model_forward() {
        let config = GPTConfig {
            vocab_size: 100,
            d_model: 64,
            num_layers: 2,
            num_heads: 4,
            d_ff: 256,
            max_seq_len: 128,
            dropout: 0.0,
        };

        let mut model = GPTModel::new(config).unwrap();
        model.init_weights().unwrap();

        // Input: [batch=1, seq_len=5], token_ids = [1, 2, 3, 4, 5]
        let token_ids = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&token_ids, 1, 5).unwrap();

        assert_eq!(logits.size(), &[1, 5, 100]);
    }
}
