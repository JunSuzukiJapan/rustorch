use anyhow::Result;
use rustorch::autograd::Variable;
use rustorch::nn::{activation::gelu, Embedding, Linear, MultiheadAttention};
use rustorch::tensor::Tensor;
use std::fmt::Debug;

/// Configuration for Transformer decoder model
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub dropout: f64,
    pub layer_norm_eps: f64,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 512,
            num_layers: 6,
            num_heads: 8,
            intermediate_size: 2048,
            max_position_embeddings: 2048,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
}

/// Decoder-only Transformer model using RusTorch components
pub struct TransformerModel {
    config: TransformerConfig,
    #[allow(dead_code)]
    token_embedding: Embedding<f64>,
    #[allow(dead_code)]
    position_embedding: Embedding<f64>,
    #[allow(dead_code)]
    layers: Vec<DecoderLayer>,
    #[allow(dead_code)]
    output_projection: Linear<f64>,
}

impl TransformerModel {
    pub fn new(config: TransformerConfig) -> Result<Self> {
        tracing::info!("Creating Transformer model with config: {:?}", config);

        // Token embedding using RusTorch Embedding layer
        let token_embedding = Embedding::new(
            config.vocab_size,
            config.hidden_size,
            None, // no padding_idx
            None, // no max_norm
            None, // not frozen
        );

        // Position embedding
        let position_embedding = Embedding::new(
            config.max_position_embeddings,
            config.hidden_size,
            None,
            None,
            None,
        );

        // Create decoder layers
        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(DecoderLayer::new(
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
                config.dropout,
            )?);
        }

        // Output projection to vocabulary
        let output_projection = Linear::new(config.hidden_size, config.vocab_size);

        Ok(Self {
            config,
            token_embedding,
            position_embedding,
            layers,
            output_projection,
        })
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Forward pass through the transformer
    pub fn forward(&self, input_ids: &Tensor<f64>) -> Result<Tensor<f64>> {
        tracing::debug!("Forward pass with input shape: {:?}", input_ids.shape());

        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // Get token embeddings
        let token_embeds = self
            .token_embedding
            .forward(&Variable::new(input_ids.clone(), false));
        let token_tensor = token_embeds.data().read().unwrap().clone();

        // Get position IDs [0, 1, 2, ..., seq_len-1]
        // Repeat for each batch
        let mut position_ids = Vec::with_capacity(batch_size * seq_len);
        for _ in 0..batch_size {
            for i in 0..seq_len {
                position_ids.push(i as f64);
            }
        }
        let position_ids_tensor = Tensor::from_vec(position_ids, vec![batch_size, seq_len]);

        // Get position embeddings
        let position_embeds = self
            .position_embedding
            .forward(&Variable::new(position_ids_tensor, false));
        let position_tensor = position_embeds.data().read().unwrap().clone();

        // Add token and position embeddings
        let mut hidden_states = token_tensor + position_tensor;

        // Pass through decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            tracing::trace!("Processing layer {}/{}", i + 1, self.config.num_layers);
            hidden_states = layer.forward(&hidden_states, None)?;
        }

        // Project to vocabulary
        let logits = self
            .output_projection
            .forward(&Variable::new(hidden_states, false));
        let logits_tensor = logits.data().read().unwrap().clone();

        Ok(logits_tensor)
    }
}

/// Decoder layer with self-attention and feedforward
pub struct DecoderLayer {
    #[allow(dead_code)]
    hidden_size: usize,
    #[allow(dead_code)]
    self_attn: MultiheadAttention<f64>,
    #[allow(dead_code)]
    ffn: FeedForward,
    // Note: Layer normalization will be added when RusTorch provides LayerNorm
}

impl DecoderLayer {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        dropout: f64,
    ) -> Result<Self> {
        // Create self-attention layer using RusTorch MultiheadAttention
        let self_attn = MultiheadAttention::new(
            hidden_size,
            num_heads,
            Some(dropout),
            Some(true), // use bias
            None,       // kdim = hidden_size
            None,       // vdim = hidden_size
            Some(true), // batch_first = true
        )?;

        // Create feedforward network
        let ffn = FeedForward::new(hidden_size, intermediate_size, dropout);

        Ok(Self {
            hidden_size,
            self_attn,
            ffn,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<f64>,
        _attention_mask: Option<&Tensor<f64>>,
    ) -> Result<Tensor<f64>> {
        // Self-attention with residual connection
        let hidden_var = Variable::new(hidden_states.clone(), false);

        // Apply self-attention (query = key = value = hidden_states)
        // Note: RusTorch MultiheadAttention.forward has 7 parameters
        let (attn_output, _attn_weights) = self.self_attn.forward(
            &hidden_var,
            &hidden_var,
            &hidden_var,
            None,        // key_padding_mask
            Some(false), // need_weights
            None,        // attn_mask
            None,        // average_attn_weights
        )?;

        // Residual connection
        let attn_tensor = attn_output.data().read().unwrap().clone();
        let hidden_states = hidden_states.clone() + attn_tensor;

        // Feedforward with residual connection
        let ffn_input = Variable::new(hidden_states.clone(), false);
        let ffn_output = self.ffn.forward(&ffn_input)?;
        let ffn_tensor = ffn_output.data().read().unwrap().clone();
        let output = hidden_states + ffn_tensor;

        Ok(output)
    }
}

/// Feedforward network (MLP) using RusTorch Linear layers
pub struct FeedForward {
    fc1: Linear<f64>,
    fc2: Linear<f64>,
    #[allow(dead_code)]
    dropout: f64,
}

impl FeedForward {
    pub fn new(hidden_size: usize, intermediate_size: usize, dropout: f64) -> Self {
        // Use RusTorch Linear layers
        let fc1 = Linear::new(hidden_size, intermediate_size);
        let fc2 = Linear::new(intermediate_size, hidden_size);

        Self { fc1, fc2, dropout }
    }

    pub fn forward(&self, x: &Variable<f64>) -> Result<Variable<f64>> {
        // x -> fc1 -> GELU -> fc2
        let hidden = self.fc1.forward(x);

        // Apply GELU activation
        let activated = gelu(&hidden);

        // Note: Dropout is not applied during inference
        // Training mode would require dropout here

        let output = self.fc2.forward(&activated);

        // Note: Dropout is not applied during inference
        // Training mode would require dropout here

        Ok(output)
    }
}

/// KV Cache for efficient generation
pub struct KVCache {
    num_layers: usize,
    k_cache: Vec<Option<Tensor<f64>>>,
    v_cache: Vec<Option<Tensor<f64>>>,
    seq_len: usize,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            k_cache: vec![None; num_layers],
            v_cache: vec![None; num_layers],
            seq_len: 0,
        }
    }

    pub fn update(&mut self, layer_idx: usize, k: Tensor<f64>, v: Tensor<f64>) -> Result<()> {
        if layer_idx >= self.num_layers {
            anyhow::bail!("Layer index {} out of range", layer_idx);
        }

        self.k_cache[layer_idx] = Some(k);
        self.v_cache[layer_idx] = Some(v);
        self.seq_len += 1;

        Ok(())
    }

    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor<f64>, &Tensor<f64>)> {
        if layer_idx >= self.num_layers {
            return None;
        }

        match (&self.k_cache[layer_idx], &self.v_cache[layer_idx]) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    pub fn clear(&mut self) {
        self.k_cache = vec![None; self.num_layers];
        self.v_cache = vec![None; self.num_layers];
        self.seq_len = 0;
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_config_default() {
        let config = TransformerConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.num_heads, 8);
    }

    #[test]
    fn test_transformer_model_creation() {
        let config = TransformerConfig::default();
        let model = TransformerModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_decoder_layer_creation() {
        let layer = DecoderLayer::new(512, 8, 2048, 0.1);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_feedforward_creation() {
        let ff = FeedForward::new(512, 2048, 0.1);
        assert_eq!(ff.dropout, 0.1);
    }

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new(6);
        assert_eq!(cache.num_layers, 6);
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_update() {
        let mut cache = KVCache::new(2);
        let k = Tensor::<f64>::zeros(&[1, 4, 64]);
        let v = Tensor::<f64>::zeros(&[1, 4, 64]);

        let result = cache.update(0, k, v);
        assert!(result.is_ok());
        assert!(cache.get(0).is_some());
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KVCache::new(2);
        let k = Tensor::<f64>::zeros(&[1, 4, 64]);
        let v = Tensor::<f64>::zeros(&[1, 4, 64]);

        cache.update(0, k, v).unwrap();
        cache.clear();

        assert_eq!(cache.seq_len(), 0);
        assert!(cache.get(0).is_none());
    }
}
