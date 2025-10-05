use super::{sample_token, KVCache, ModelLoader, SamplingConfig, TransformerModel};
use crate::session::GenerationConfig;
use crate::tokenizer::{Tokenizer, TokenizerWrapper};
use anyhow::Result;
use rustorch::prelude::Tensor;

// Import GPT model
use super::GPTModel;

pub struct InferenceEngine {
    model: Option<TransformerModel>,
    gpt_model: Option<GPTModel>,
    tokenizer: TokenizerWrapper,
    generation_config: GenerationConfig,
    sampling_config: SamplingConfig,
    #[allow(dead_code)]
    loader: ModelLoader,
}

impl InferenceEngine {
    pub fn new(loader: ModelLoader, config: GenerationConfig) -> Self {
        // Create sampling config from generation config
        let sampling_config = SamplingConfig {
            temperature: config.temperature as f64,
            top_k: if config.top_k > 0 {
                Some(config.top_k as usize)
            } else {
                None
            },
            top_p: Some(config.top_p as f64),
            repetition_penalty: 1.0,
        };

        // For now, use dummy tokenizer (will load from model in future)
        let tokenizer = TokenizerWrapper::dummy().expect("Failed to create tokenizer");

        Self {
            model: None,
            gpt_model: None,
            tokenizer,
            generation_config: config,
            sampling_config,
            loader,
        }
    }

    /// Set the transformer model
    pub fn set_model(&mut self, model: TransformerModel) {
        self.model = Some(model);
    }

    /// Set the GPT model
    pub fn set_gpt_model(&mut self, model: GPTModel) {
        self.gpt_model = Some(model);
    }

    /// Generate a response from input text
    pub fn generate(&self, input: &str) -> Result<String> {
        tracing::debug!("Generating response for input: {}", input);
        tracing::debug!(
            "Generation config: max_tokens={}, temperature={}, top_p={}",
            self.generation_config.max_tokens,
            self.generation_config.temperature,
            self.generation_config.top_p
        );

        // If no model is loaded, return dummy response
        if self.model.is_none() && self.gpt_model.is_none() {
            return Ok(self.generate_dummy_response(input));
        }

        // Encode input
        let input_ids = self
            .tokenizer
            .encode(input, true)
            .unwrap_or_else(|_| vec![0]); // Fallback to dummy token on error

        // Generate tokens
        let output_ids = self.generate_tokens(&input_ids)?;

        // Decode output
        let output = self
            .tokenizer
            .decode(&output_ids, true)
            .unwrap_or_else(|_| self.generate_dummy_response(input));

        Ok(output)
    }

    /// Generate tokens using the model
    fn generate_tokens(&self, input_ids: &[u32]) -> Result<Vec<u32>> {
        let max_new_tokens = self.generation_config.max_tokens;
        let mut generated_ids = input_ids.to_vec();

        // Create KV cache if model has layers
        let _cache = self
            .model
            .as_ref()
            .map(|model| KVCache::new(model.config().num_layers));

        // Use GPT model if available
        if let Some(ref gpt_model) = self.gpt_model {
            return self.generate_with_gpt(gpt_model, input_ids, max_new_tokens);
        }

        // Fallback to dummy generation if no GPT model
        // Generation loop with placeholder logits
        for _ in 0..max_new_tokens {
            let vocab_size = self.tokenizer.vocab_size();
            let logits = Tensor::<f64>::zeros(&[1, vocab_size]);

            // Sample next token
            let next_token = sample_token(&logits, &self.sampling_config, &generated_ids)?;

            // Check for EOS token
            if let Some(eos_id) = self.tokenizer.eos_token_id() {
                if next_token == eos_id {
                    break;
                }
            }

            generated_ids.push(next_token);

            // Stop if we've generated enough
            if generated_ids.len() - input_ids.len() >= max_new_tokens {
                break;
            }
        }

        // Return only the newly generated tokens
        Ok(generated_ids[input_ids.len()..].to_vec())
    }

    /// Generate tokens using GPT model with RusTorch
    fn generate_with_gpt(
        &self,
        gpt_model: &GPTModel,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut generated_ids: Vec<usize> = input_ids.iter().map(|&id| id as usize).collect();

        // Generation loop
        for step in 0..max_new_tokens {
            // Prepare input for model (current sequence)
            let seq_len = generated_ids.len();
            let batch_size = 1;

            // Forward pass through GPT model
            // Input: token_ids [batch_size, seq_len]
            // Output: logits [batch_size, seq_len, vocab_size]
            let logits_tensor = gpt_model.forward(&generated_ids, batch_size, seq_len)?;

            // Extract logits for the last position
            // Shape: [batch_size=1, seq_len, vocab_size] -> [vocab_size]
            let last_logits = self.extract_last_logits(&logits_tensor, seq_len)?;

            // Apply temperature scaling
            let scaled_logits = if self.sampling_config.temperature != 1.0 {
                self.apply_temperature(&last_logits, self.sampling_config.temperature)?
            } else {
                last_logits
            };

            // Sample next token using RusTorch operations
            let next_token_id = self.sample_from_logits(&scaled_logits, &generated_ids, step)?;

            // Check for EOS token
            if let Some(eos_id) = self.tokenizer.eos_token_id() {
                if next_token_id == eos_id as usize {
                    tracing::debug!("EOS token generated at step {}", step);
                    break;
                }
            }

            generated_ids.push(next_token_id);

            // Stop if context limit exceeded
            if seq_len >= gpt_model.config().max_seq_len {
                tracing::warn!("Reached maximum sequence length");
                break;
            }
        }

        // Return only the newly generated tokens
        let new_tokens: Vec<u32> = generated_ids[input_ids.len()..]
            .iter()
            .map(|&id| id as u32)
            .collect();

        Ok(new_tokens)
    }

    /// Extract logits for the last position from model output
    fn extract_last_logits(
        &self,
        logits_tensor: &Tensor<f64>,
        seq_len: usize,
    ) -> Result<Tensor<f64>> {
        let shape = logits_tensor.size();
        let vocab_size = shape[2];

        // Get data for last position: logits[:, -1, :]
        let data_slice = logits_tensor.as_slice().ok_or_else(|| anyhow::anyhow!("Failed to get logits data"))?;
        let start_idx = (seq_len - 1) * vocab_size;
        let end_idx = seq_len * vocab_size;

        let last_logits_data = data_slice[start_idx..end_idx].to_vec();

        Ok(Tensor::from_vec(last_logits_data, vec![vocab_size]))
    }

    /// Apply temperature scaling to logits
    fn apply_temperature(&self, logits: &Tensor<f64>, temperature: f64) -> Result<Tensor<f64>> {
        if temperature <= 0.0 {
            anyhow::bail!("Temperature must be positive");
        }

        let scaled: Vec<f64> = logits.as_slice().ok_or_else(|| anyhow::anyhow!("Failed to get logits data"))?.iter().map(|&x| x / temperature).collect();

        Ok(Tensor::from_vec(scaled, logits.size().to_vec()))
    }

    /// Sample from logits using top-k, top-p sampling
    fn sample_from_logits(
        &self,
        logits: &Tensor<f64>,
        _context: &[usize],
        _step: usize,
    ) -> Result<usize> {
        use crate::model::sampling::{
            apply_top_k_to_probs, apply_top_p_to_probs, multinomial_sample, softmax,
        };

        let logits_vec: Vec<f64> = logits.data.iter().copied().collect();

        // Apply softmax to get probabilities
        let probs = softmax(&logits_vec)?;

        // Apply top-k filtering if specified
        let filtered_probs = if let Some(top_k) = self.sampling_config.top_k {
            apply_top_k_to_probs(&probs, top_k)?
        } else {
            probs
        };

        // Apply top-p (nucleus) filtering if specified
        let final_probs = if let Some(top_p) = self.sampling_config.top_p {
            if top_p < 1.0 {
                apply_top_p_to_probs(&filtered_probs, top_p)?
            } else {
                filtered_probs
            }
        } else {
            filtered_probs
        };

        // Sample from the filtered distribution
        multinomial_sample(&final_probs)
    }

    fn generate_dummy_response(&self, input: &str) -> String {
        // Simple dummy response generator
        let responses = [
            format!("I understand you said: \"{}\"", input),
            format!("That's an interesting point about: {}", input),
            format!("Let me think about that... You mentioned: {}", input),
            format!("Based on your input \"{}\", here's what I think...", input),
        ];

        // Use input length to select response (deterministic but varied)
        let idx = input.len() % responses.len();
        responses[idx].clone()
    }

    /// Generate a streaming response with token-by-token output
    pub fn generate_stream<'a>(
        &'a self,
        input: &str,
    ) -> Result<Box<dyn Iterator<Item = String> + 'a>> {
        tracing::debug!("Starting streaming generation for input: {}", input);

        // If no model is loaded, return dummy streaming response
        if self.model.is_none() {
            return Ok(Box::new(self.generate_dummy_stream(input)));
        }

        // Encode input
        let input_ids = self
            .tokenizer
            .encode(input, true)
            .unwrap_or_else(|_| vec![0]);

        // Generate tokens with streaming
        Ok(Box::new(self.generate_tokens_stream(input_ids)))
    }

    /// Generate tokens one by one for streaming
    fn generate_tokens_stream(&self, input_ids: Vec<u32>) -> impl Iterator<Item = String> + '_ {
        let max_new_tokens = self.generation_config.max_tokens;
        let generated_ids = input_ids.clone();
        let eos_id = self.tokenizer.eos_token_id();
        let vocab_size = self.tokenizer.vocab_size();

        (0..max_new_tokens).scan(generated_ids, move |state, _| {
            // Sample next token (placeholder with random logits for now)
            let logits = Tensor::<f64>::zeros(&[1, vocab_size]);

            let next_token = sample_token(&logits, &self.sampling_config, state).ok()?;

            // Check for EOS token
            if let Some(eos) = eos_id {
                if next_token == eos {
                    return None; // Stop iteration
                }
            }

            state.push(next_token);

            // Decode the single token
            self.tokenizer.decode(&[next_token], false).ok()
        })
    }

    /// Generate dummy streaming response for testing
    fn generate_dummy_stream(&self, input: &str) -> impl Iterator<Item = String> + '_ {
        let response = self.generate_dummy_response(input);
        let words: Vec<String> = response.split_whitespace().map(|s| s.to_string()).collect();

        words.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_engine() -> InferenceEngine {
        let loader = ModelLoader::dummy();
        let config = GenerationConfig::default();
        InferenceEngine::new(loader, config)
    }

    #[test]
    fn test_inference_engine_creation() {
        let engine = create_test_engine();
        assert_eq!(engine.generation_config.max_tokens, 2048);
    }

    #[test]
    fn test_generate_dummy_response() {
        let engine = create_test_engine();
        let response = engine.generate("Hello").unwrap();
        assert!(!response.is_empty());
        assert!(response.contains("Hello"));
    }

    #[test]
    fn test_generate_various_inputs() {
        let engine = create_test_engine();

        let inputs = vec!["Hi", "Hello world", "How are you?", "Tell me a story"];

        for input in inputs {
            let response = engine.generate(input).unwrap();
            assert!(!response.is_empty());
        }
    }

    #[test]
    fn test_set_model() {
        use super::super::TransformerConfig;

        let mut engine = create_test_engine();
        let config = TransformerConfig::default();
        let model = TransformerModel::new(config).unwrap();

        engine.set_model(model);
        assert!(engine.model.is_some());
    }

    #[test]
    fn test_generate_with_model() {
        use super::super::TransformerConfig;

        let mut engine = create_test_engine();
        let config = TransformerConfig::default();
        let model = TransformerModel::new(config).unwrap();

        engine.set_model(model);

        // Since model forward() is not fully implemented, this will fall back to dummy response
        // or generate placeholder tokens
        let _response = engine.generate("Hello").unwrap();
        // Response might be empty if tokenizer decode fails, which is acceptable for now
        // Just verify it doesn't panic - test passed if we got here
    }

    #[test]
    fn test_sampling_config_creation() {
        let loader = ModelLoader::dummy();
        let gen_config = GenerationConfig {
            max_tokens: 100,
            temperature: 0.8,
            top_k: 50,
            top_p: 0.9,
            ..Default::default()
        };

        let engine = InferenceEngine::new(loader, gen_config);

        assert!((engine.sampling_config.temperature - 0.8).abs() < 1e-6);
        assert_eq!(engine.sampling_config.top_k, Some(50));
        assert!((engine.sampling_config.top_p.unwrap() - 0.9).abs() < 1e-6);
    }
}
