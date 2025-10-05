//! GPT model implementation for hybrid_f32 (native f32 precision)
//! hybrid_f32用GPTモデル実装（ネイティブf32精度）

use crate::hybrid_f32::error::{F32Error, F32Result};
use crate::hybrid_f32::tensor::F32Tensor;
use crate::formats::gguf::{GGUFLoader, ModelParams};
use std::collections::HashMap;
use std::path::Path;

/// GPT model configuration
/// GPTモデル設定
#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub dropout: f32,
}

impl GPTConfig {
    /// Create config from GGUF model parameters
    /// GGUFモデルパラメータから設定を作成
    pub fn from_model_params(params: &ModelParams) -> Self {
        Self {
            vocab_size: params.vocab_size as usize,
            d_model: params.hidden_size as usize,
            num_layers: params.num_layers as usize,
            num_heads: params.num_heads as usize,
            d_ff: (params.hidden_size * 4) as usize,
            max_seq_len: params.context_length as usize,
            dropout: 0.1,
        }
    }
}

/// Device type for GPU acceleration
/// GPU加速用デバイスタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU computation
    Cpu,
    /// Metal GPU (macOS)
    Metal,
    /// CoreML Neural Engine (macOS)
    CoreML,
    /// Hybrid Metal + CoreML
    Hybrid,
}

/// GPT model with native f32 precision for GPU acceleration
/// GPU加速用ネイティブf32精度GPTモデル
pub struct F32GPTModel {
    config: GPTConfig,
    weights: HashMap<String, F32Tensor>,
    device_type: DeviceType,
}

impl F32GPTModel {
    /// Create a new GPT model with CPU backend
    /// CPUバックエンドで新しいGPTモデルを作成
    pub fn new(config: GPTConfig) -> F32Result<Self> {
        Self::with_device(config, DeviceType::Cpu)
    }

    /// Create a new GPT model with specified device
    /// 指定デバイスで新しいGPTモデルを作成
    pub fn with_device(config: GPTConfig, device_type: DeviceType) -> F32Result<Self> {
        eprintln!("🚀 Creating F32GPTModel with {:?} device", device_type);
        eprintln!("   Precision: native f32 (optimized for GPU)");

        Ok(Self {
            config,
            weights: HashMap::new(),
            device_type,
        })
    }

    /// Load GPT model from GGUF file with Metal GPU support
    /// Metal GPUサポート付きでGGUFファイルからGPTモデルを読み込み
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> F32Result<Self> {
        Self::from_gguf_with_device(path, DeviceType::Metal)
    }

    /// Load GPT model from GGUF file with specified device
    /// 指定デバイスでGGUFファイルからGPTモデルを読み込み
    pub fn from_gguf_with_device<P: AsRef<Path>>(
        path: P,
        device_type: DeviceType,
    ) -> F32Result<Self> {
        let loader = GGUFLoader::from_file(path)
            .map_err(|e| F32Error::device_error(format!("Failed to load GGUF: {}", e)))?;

        // Extract model parameters
        let params = loader
            .get_model_params()
            .map_err(|e| F32Error::device_error(format!("Failed to get model params: {}", e)))?;

        let config = GPTConfig::from_model_params(&params);

        // Create model with device
        let mut model = Self::with_device(config, device_type)?;

        eprintln!("📊 Loading GPT model weights as f32");
        eprintln!("   Device: {:?}", device_type);
        eprintln!("   Vocab size: {}", model.config.vocab_size);
        eprintln!("   Layers: {}", model.config.num_layers);
        eprintln!("   d_model: {}", model.config.d_model);

        // Load weights as f32
        let tensor_names = loader.tensor_names();
        let mut loaded_count = 0;

        for name in tensor_names.iter() {
            // Load tensor as f64, then convert to f32
            match loader.load_tensor(name) {
                Ok(tensor_f64) => {
                    // Convert f64 tensor to f32
                    if let Some(data_slice) = tensor_f64.as_array().as_slice() {
                        let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
                        let shape = tensor_f64.shape().to_vec();

                        // Create F32Tensor from converted data
                        let f32_tensor = F32Tensor::from_vec(f32_data, &shape)
                            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create F32Tensor: {}", e)))?;

                        model.weights.insert(name.to_string(), f32_tensor);
                        loaded_count += 1;
                    }
                }
                Err(e) => {
                    // Show which tensors fail to load (only first 5)
                    if loaded_count < 5 {
                        eprintln!("⚠️  Failed to load tensor '{}': {}", name, e);
                    }
                    continue;
                }
            }
        }

        eprintln!("✅ Loaded {} weights as f32", loaded_count);

        // Debug: Print first 10 weight names
        eprintln!("📝 Sample weight names:");
        for (i, name) in model.weights.keys().take(10).enumerate() {
            eprintln!("   {}: {}", i + 1, name);
        }

        Ok(model)
    }

    /// Get model configuration
    /// モデル設定を取得
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    /// Get device type
    /// デバイスタイプを取得
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Get a weight tensor by name
    /// 名前で重みテンソルを取得
    pub fn get_weight(&self, name: &str) -> Option<&F32Tensor> {
        self.weights.get(name)
    }

    /// List all weight names
    /// すべての重み名をリスト
    pub fn weight_names(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }

    /// Forward pass through the model
    /// モデルの順伝播
    pub fn forward(&self, input_ids: &[usize]) -> F32Result<F32Tensor> {
        let batch_size = 1;
        let seq_len = input_ids.len();

        // Step 1: Token Embedding
        let mut hidden_states = self.get_embeddings(input_ids)?;

        // Step 2: Process through transformer layers
        let num_layers_to_process = self.config.num_layers;
        for layer_idx in 0..num_layers_to_process {

            // Pre-attention LayerNorm
            let normed_hidden = self.apply_layer_norm(hidden_states.clone(), layer_idx)?;

            // Self-Attention with residual connection (placeholder - just returns input)
            let attention_out = self.apply_attention(normed_hidden, layer_idx)?;
            hidden_states = self.add_tensors(&hidden_states, &attention_out)?;

            // Pre-FFN LayerNorm
            let normed_hidden = self.apply_ffn_layer_norm(hidden_states.clone(), layer_idx)?;

            // Feed-Forward Network with residual connection (placeholder - just returns input)
            let ffn_out = self.apply_ffn(normed_hidden, layer_idx)?;
            hidden_states = self.add_tensors(&hidden_states, &ffn_out)?;
        }

        // Step 3: Final layer norm and projection to vocabulary
        hidden_states = self.apply_final_layer_norm(hidden_states)?;
        let logits = self.project_to_vocab(hidden_states)?;

        Ok(logits)
    }

    /// Get token embeddings
    fn get_embeddings(&self, input_ids: &[usize]) -> F32Result<F32Tensor> {
        // Try multiple possible embedding weight names
        let embed_weight = self.weights.get("token_embd.weight")
            .or_else(|| self.weights.get("model.embed_tokens.weight"))
            .or_else(|| self.weights.get("tok_embeddings.weight"))
            .or_else(|| self.weights.get("transformer.wte.weight"))
            .or_else(|| self.weights.get("embeddings.weight"))
            .ok_or_else(|| F32Error::device_error("Embedding weight not found"))?;

        let seq_len = input_ids.len();
        let d_model = self.config.d_model;
        let mut embedding_data = Vec::with_capacity(seq_len * d_model);

        if let Some(embed_slice) = embed_weight.data.as_slice() {
            for &token_id in input_ids {
                let token_idx = token_id.min(self.config.vocab_size - 1);
                let start = token_idx * d_model;
                let end = start + d_model;
                embedding_data.extend_from_slice(&embed_slice[start..end]);
            }
        } else {
            return Err(F32Error::device_error("Failed to access embedding weight data"));
        }

        F32Tensor::from_vec(embedding_data, &[1, seq_len, d_model])
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create embedding tensor: {}", e)))
    }

    /// Apply LayerNorm using GPU acceleration if available (for attention)
    /// Apply LayerNorm with given weight key prefix
    fn apply_layer_norm_by_key(&self, input: F32Tensor, weight_key: &str, bias_key: &str) -> F32Result<F32Tensor> {
        let gamma = self.weights.get(weight_key)
            .ok_or_else(|| F32Error::device_error(format!("LayerNorm weight not found: {}", weight_key)))?;

        // Beta (bias) is optional - Llama models use RMSNorm without bias
        let beta = self.weights.get(bias_key);

        self.layer_norm_impl(&input, gamma, beta)
    }

    /// Apply LayerNorm for attention
    fn apply_layer_norm(&self, input: F32Tensor, layer_idx: usize) -> F32Result<F32Tensor> {
        self.apply_layer_norm_by_key(
            input,
            &format!("blk.{}.attn_norm.weight", layer_idx),
            &format!("blk.{}.attn_norm.bias", layer_idx)
        )
    }

    /// Apply LayerNorm for FFN
    fn apply_ffn_layer_norm(&self, input: F32Tensor, layer_idx: usize) -> F32Result<F32Tensor> {
        self.apply_layer_norm_by_key(
            input,
            &format!("blk.{}.ffn_norm.weight", layer_idx),
            &format!("blk.{}.ffn_norm.bias", layer_idx)
        )
    }

    /// Add two tensors element-wise (for residual connections)
    fn add_tensors(&self, a: &F32Tensor, b: &F32Tensor) -> F32Result<F32Tensor> {
        let a_shape = a.data.shape();
        let b_shape = b.data.shape();

        if a_shape != b_shape {
            return Err(F32Error::shape_mismatch(format!(
                "Tensor shapes don't match: {:?} vs {:?}",
                a_shape, b_shape
            )));
        }

        let a_slice = a.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access tensor A"))?;
        let b_slice = b.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access tensor B"))?;

        let result: Vec<f32> = a_slice.iter()
            .zip(b_slice.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        F32Tensor::from_vec(result, &a_shape.to_vec())
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create result tensor: {}", e)))
    }

    /// Apply final LayerNorm
    fn apply_final_layer_norm(&self, input: F32Tensor) -> F32Result<F32Tensor> {
        // Try multiple possible weight names
        let gamma = self.weights.get("output_norm.weight")
            .or_else(|| self.weights.get("model.norm.weight"))
            .ok_or_else(|| F32Error::device_error("Final LayerNorm weight not found"))?;

        let beta = self.weights.get("output_norm.bias")
            .or_else(|| self.weights.get("model.norm.bias"));

        self.layer_norm_impl(&input, gamma, beta)
    }

    /// LayerNorm implementation with Metal GPU acceleration
    fn layer_norm_impl(&self, input: &F32Tensor, gamma: &F32Tensor, beta: Option<&F32Tensor>) -> F32Result<F32Tensor> {
        let input_shape = input.data.shape();
        if input_shape.len() != 3 {
            return Err(F32Error::dimension_error(3, input_shape.len()));
        }

        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let features = input_shape[2];

        // Use Metal GPU acceleration for LayerNorm (f32 kernel)
        #[cfg(feature = "metal")]
        if self.device_type == DeviceType::Metal || self.device_type == DeviceType::Hybrid {
            return self.metal_layer_norm(input, gamma, beta, batch_size, seq_len, features);
        }

        self.cpu_layer_norm(input, gamma, beta, batch_size, seq_len, features)
    }

    /// Helper to get beta slice, creating zero vector for RMSNorm if None
    #[inline]
    fn get_beta_slice<'a>(
        beta: Option<&'a F32Tensor>,
        zero_beta: &'a mut Vec<f32>,
        features: usize,
    ) -> F32Result<&'a [f32]> {
        if let Some(b) = beta {
            b.data.as_slice()
                .ok_or_else(|| F32Error::device_error("Failed to access beta data"))
        } else {
            *zero_beta = vec![0.0f32; features];
            Ok(&zero_beta[..])
        }
    }

    #[cfg(feature = "metal")]
    fn metal_layer_norm(
        &self,
        input: &F32Tensor,
        gamma: &F32Tensor,
        beta: Option<&F32Tensor>,
        batch_size: usize,
        seq_len: usize,
        features: usize,
    ) -> F32Result<F32Tensor> {
        use crate::gpu::metal_kernels::metal_layer_norm_f32;

        let input_slice = input.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access input data"))?;
        let gamma_slice = gamma.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access gamma data"))?;

        let mut zero_beta = Vec::new();
        let beta_slice = Self::get_beta_slice(beta, &mut zero_beta, features)?;

        let mut output = vec![0.0f32; input_slice.len()];

        metal_layer_norm_f32(
            input_slice,
            &mut output,
            gamma_slice,
            beta_slice,
            batch_size,
            seq_len,
            features,
            1e-5,
        )
        .map_err(|e| F32Error::device_error(format!("Metal LayerNorm failed: {}", e)))?;

        F32Tensor::from_vec(output, &[batch_size, seq_len, features])
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create output tensor: {}", e)))
    }

    fn cpu_layer_norm(
        &self,
        input: &F32Tensor,
        gamma: &F32Tensor,
        beta: Option<&F32Tensor>,
        batch_size: usize,
        seq_len: usize,
        features: usize,
    ) -> F32Result<F32Tensor> {
        let input_slice = input.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access input data"))?;
        let gamma_slice = gamma.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access gamma data"))?;

        let mut zero_beta = Vec::new();
        let beta_slice = Self::get_beta_slice(beta, &mut zero_beta, features)?;

        let mut output = vec![0.0f32; input_slice.len()];
        let eps = 1e-5f32;

        for b in 0..batch_size {
            for s in 0..seq_len {
                let offset = (b * seq_len + s) * features;
                let sum: f32 = input_slice[offset..offset + features].iter().sum();
                let mean = sum / features as f32;
                let var_sum: f32 = input_slice[offset..offset + features]
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum();
                let variance = var_sum / features as f32;
                let std = (variance + eps).sqrt();

                for f in 0..features {
                    let normalized = (input_slice[offset + f] - mean) / std;
                    output[offset + f] = gamma_slice[f] * normalized + beta_slice[f];
                }
            }
        }

        F32Tensor::from_vec(output, &[batch_size, seq_len, features])
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create output tensor: {}", e)))
    }

    /// Apply Grouped Query Attention (GQA)
    /// グループ化クエリ注意機構（GQA）を適用
    fn apply_attention(&self, input: F32Tensor, layer_idx: usize) -> F32Result<F32Tensor> {
        let batch_size = input.data.shape()[0];
        let seq_len = input.data.shape()[1];
        let d_model = input.data.shape()[2];

        // Load Q, K, V, output weight matrices
        let q_weight = self.weights.get(&format!("blk.{}.attn_q.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Q weight not found for layer {}", layer_idx)))?;
        let k_weight = self.weights.get(&format!("blk.{}.attn_k.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("K weight not found for layer {}", layer_idx)))?;
        let v_weight = self.weights.get(&format!("blk.{}.attn_v.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("V weight not found for layer {}", layer_idx)))?;
        let o_weight = self.weights.get(&format!("blk.{}.attn_output.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Output weight not found for layer {}", layer_idx)))?;

        let input_slice = input.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access input data"))?;

        // Get weight dimensions
        let q_shape = q_weight.data.shape();
        let k_shape = k_weight.data.shape();
        let v_shape = v_weight.data.shape();

        // GGUF weight format: [input_dim, output_dim] (transposed from PyTorch)
        let q_dim = q_shape[1]; // Output dimension for Q
        let k_dim = k_shape[1]; // Output dimension for K (256 for GQA)
        let v_dim = v_shape[1]; // Output dimension for V (256 for GQA)

        // Project Q, K, V using transposed weights
        let q = self.linear_projection_transposed(input_slice, q_weight, batch_size, seq_len, d_model, q_dim)?;
        let k = self.linear_projection_transposed(input_slice, k_weight, batch_size, seq_len, d_model, k_dim)?;
        let v = self.linear_projection_transposed(input_slice, v_weight, batch_size, seq_len, d_model, v_dim)?;

        // Head dimensions
        let head_dim = 64; // Standard head dimension for Llama
        let num_q_heads = q_dim / head_dim; // 32 heads
        let num_kv_heads = k_dim / head_dim; // 4 heads for GQA
        let num_groups = num_q_heads / num_kv_heads; // 8 Q heads share 1 KV head

        // Step 2: Compute GQA attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attention_output = vec![0.0f32; batch_size * seq_len * q_dim];

        for b in 0..batch_size {
            for h in 0..num_q_heads {
                let kv_head = h / num_groups; // Find corresponding KV head

                for i in 0..seq_len {
                    // Compute attention scores
                    let mut scores = vec![0.0f32; seq_len];
                    let mut max_score = f32::NEG_INFINITY;

                    for j in 0..=i {
                        let q_offset = (b * seq_len + i) * q_dim + h * head_dim;
                        let k_offset = (b * seq_len + j) * k_dim + kv_head * head_dim;

                        let mut dot_product = 0.0f32;
                        for d in 0..head_dim {
                            dot_product += q[q_offset + d] * k[k_offset + d];
                        }
                        let score = dot_product * scale;
                        scores[j] = score;
                        max_score = max_score.max(score);
                    }

                    // Apply softmax
                    let mut sum = 0.0f32;
                    for j in 0..=i {
                        scores[j] = (scores[j] - max_score).exp();
                        sum += scores[j];
                    }
                    for j in 0..=i {
                        scores[j] /= sum;
                    }

                    // Apply attention to V
                    for d in 0..head_dim {
                        let mut weighted_sum = 0.0f32;
                        for j in 0..=i {
                            let v_offset = (b * seq_len + j) * v_dim + kv_head * head_dim;
                            weighted_sum += scores[j] * v[v_offset + d];
                        }
                        let output_offset = (b * seq_len + i) * q_dim + h * head_dim;
                        attention_output[output_offset + d] = weighted_sum;
                    }
                }
            }
        }

        // Step 3: Output projection (GGUF format: transposed)
        let o_shape = o_weight.data.shape();
        let o_output_dim = o_shape[1]; // Should be d_model (2048)
        let output = self.linear_projection_transposed(&attention_output, o_weight, batch_size, seq_len, q_dim, o_output_dim)?;

        F32Tensor::from_vec(output, &[batch_size, seq_len, d_model])
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create attention output: {}", e)))
    }

    /// Linear projection for GGUF format weights
    ///
    /// GGUF stores all weights as [input_dim, output_dim] (transposed from PyTorch [output_dim, input_dim]).
    /// This performs: output = input @ weight where weight is [input_dim, output_dim]
    ///
    /// # Arguments
    /// * `input` - Input tensor data [batch_size, seq_len, input_dim]
    /// * `weight` - Weight tensor [input_dim, output_dim]
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `input_dim` - Input dimension
    /// * `output_dim` - Output dimension
    fn linear_projection_transposed(
        &self,
        input: &[f32],
        weight: &F32Tensor,
        batch_size: usize,
        seq_len: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> F32Result<Vec<f32>> {
        let weight_slice = weight.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access weight data"))?;

        // Weight shape: [input_dim, output_dim] (transposed from normal)
        let weight_shape = weight.data.shape();
        let expected_input_dim = weight_shape[0];
        let expected_output_dim = weight_shape[1];

        if expected_input_dim != input_dim || expected_output_dim != output_dim {
            return Err(F32Error::shape_mismatch(format!(
                "Weight shape [{}, {}] doesn't match expected [{}, {}]",
                expected_input_dim, expected_output_dim, input_dim, output_dim
            )));
        }

        let mut output = vec![0.0f32; batch_size * seq_len * output_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                let input_offset = (b * seq_len + s) * input_dim;
                let output_offset = (b * seq_len + s) * output_dim;

                for o in 0..output_dim {
                    let mut sum = 0.0f32;
                    for d in 0..input_dim {
                        // Transposed access: weight[d][o] instead of weight[o][d]
                        sum += input[input_offset + d] * weight_slice[d * output_dim + o];
                    }
                    output[output_offset + o] = sum;
                }
            }
        }

        Ok(output)
    }

    /// Apply Feed-Forward Network with SwiGLU activation
    /// SwiGLU活性化関数付きフィードフォワードネットワークを適用
    fn apply_ffn(&self, input: F32Tensor, layer_idx: usize) -> F32Result<F32Tensor> {
        let batch_size = input.data.shape()[0];
        let seq_len = input.data.shape()[1];
        let d_model = input.data.shape()[2];

        // Load FFN weights
        let gate_weight = self.weights.get(&format!("blk.{}.ffn_gate.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("FFN gate weight not found for layer {}", layer_idx)))?;
        let up_weight = self.weights.get(&format!("blk.{}.ffn_up.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("FFN up weight not found for layer {}", layer_idx)))?;
        let down_weight = self.weights.get(&format!("blk.{}.ffn_down.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("FFN down weight not found for layer {}", layer_idx)))?;

        let input_slice = input.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access input data"))?;

        // Get FFN weight shapes (GGUF format: [input_dim, output_dim])
        let gate_shape = gate_weight.data.shape();
        let down_shape = down_weight.data.shape();
        let d_ff = gate_shape[1]; // Intermediate FFN dimension

        // Step 1: Gate projection (GGUF format: transposed)
        let gate_proj = self.linear_projection_transposed(input_slice, gate_weight, batch_size, seq_len, d_model, d_ff)?;

        // Step 2: Up projection (GGUF format: transposed)
        let up_proj = self.linear_projection_transposed(input_slice, up_weight, batch_size, seq_len, d_model, d_ff)?;

        // Step 3: Apply SwiGLU activation: gate_proj * silu(up_proj)
        // SiLU (Swish): x * sigmoid(x)
        let mut swiglu_output = vec![0.0f32; batch_size * seq_len * d_ff];
        for i in 0..(batch_size * seq_len * d_ff) {
            let x = up_proj[i];
            let sigmoid_x = 1.0 / (1.0 + (-x).exp());
            let silu_x = x * sigmoid_x;
            swiglu_output[i] = gate_proj[i] * silu_x;
        }

        // Step 4: Down projection back to d_model (GGUF format: transposed)
        let down_output_dim = down_shape[1]; // GGUF: [d_ff, d_model]
        let output = self.linear_projection_transposed(&swiglu_output, down_weight, batch_size, seq_len, d_ff, down_output_dim)?;

        F32Tensor::from_vec(output, &[batch_size, seq_len, down_output_dim])
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create FFN output: {}", e)))
    }

    /// Project hidden states to vocabulary logits
    fn project_to_vocab(&self, hidden_states: F32Tensor) -> F32Result<F32Tensor> {
        let output_weight = self.weights.get("output.weight")
            .or_else(|| self.weights.get("lm_head.weight"))
            .or_else(|| self.weights.get("token_embd.weight"))
            .ok_or_else(|| F32Error::device_error("Output projection weight not found"))?;

        let hidden_shape = hidden_states.data.shape();
        let batch_size = hidden_shape[0];
        let seq_len = hidden_shape[1];
        let d_model = hidden_shape[2];

        // GGUF format: [d_model, vocab_size]
        let weight_shape = output_weight.data.shape();
        let vocab_size = weight_shape[1];

        let hidden_slice = hidden_states.data.as_slice()
            .ok_or_else(|| F32Error::device_error("Failed to access hidden states"))?;

        // Use transposed projection for GGUF format
        let logits_vec = self.linear_projection_transposed(
            hidden_slice,
            output_weight,
            batch_size,
            seq_len,
            d_model,
            vocab_size
        )?;

        F32Tensor::from_vec(logits_vec, &[batch_size, seq_len, vocab_size])
            .map_err(|e| F32Error::shape_mismatch(format!("Failed to create logits tensor: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_config_creation() {
        let config = GPTConfig {
            vocab_size: 50257,
            d_model: 768,
            num_layers: 12,
            num_heads: 12,
            d_ff: 3072,
            max_seq_len: 1024,
            dropout: 0.1,
        };

        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.num_layers, 12);
    }

    #[test]
    fn test_f32_gpt_model_creation() {
        let config = GPTConfig {
            vocab_size: 1000,
            d_model: 128,
            num_layers: 2,
            num_heads: 4,
            d_ff: 512,
            max_seq_len: 256,
            dropout: 0.0,
        };

        let model = F32GPTModel::new(config).unwrap();
        assert_eq!(model.config().vocab_size, 1000);
        assert_eq!(model.device_type(), DeviceType::Cpu);
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_f32_gpt_model_with_metal() {
        let config = GPTConfig {
            vocab_size: 1000,
            d_model: 128,
            num_layers: 2,
            num_heads: 4,
            d_ff: 512,
            max_seq_len: 256,
            dropout: 0.0,
        };

        let model = F32GPTModel::with_device(config, DeviceType::Metal).unwrap();
        assert_eq!(model.device_type(), DeviceType::Metal);
    }
}
