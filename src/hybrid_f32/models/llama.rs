//! Llama model implementation for hybrid_f32 (native f32 precision)
//! hybrid_f32用Llamaモデル実装（ネイティブf32精度）

use crate::hybrid_f32::error::{F32Error, F32Result};
use crate::hybrid_f32::tensor::F32Tensor;
use crate::formats::gguf::{GGUFLoader, ModelParams};
use std::collections::HashMap;
use std::path::Path;

// Re-export DeviceType from gpt module to avoid duplication
pub use super::gpt::DeviceType;

/// Llama model configuration
/// Llamaモデル設定
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,      // d_model
    pub num_layers: usize,
    pub num_heads: usize,         // num_attention_heads
    pub num_kv_heads: usize,      // num_key_value_heads (for GQA)
    pub intermediate_size: usize, // FFN hidden size
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,          // RoPE base frequency
}

impl LlamaConfig {
    /// Create config from GGUF model parameters
    /// GGUFモデルパラメータから設定を作成
    pub fn from_model_params(params: &ModelParams) -> Self {
        let hidden_size = params.hidden_size as usize;
        let num_heads = params.num_heads as usize;

        // Llama-2 uses GQA with num_kv_heads = num_heads (for 7B model)
        // Llama-2 7Bはnum_kv_heads = num_headsのGQAを使用
        let num_kv_heads = num_heads;

        Self {
            vocab_size: params.vocab_size as usize,
            hidden_size,
            num_layers: params.num_layers as usize,
            num_heads,
            num_kv_heads,
            intermediate_size: hidden_size * 4,  // Standard FFN expansion
            max_seq_len: params.context_length as usize,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }

    /// Get head dimension
    /// ヘッド次元を取得
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

/// KV cache for a single layer with GQA support
/// GQA対応の単一レイヤーKVキャッシュ
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// Cached keys: [batch_size, cached_seq_len, num_kv_heads, head_dim]
    pub keys: Vec<f32>,
    /// Cached values: [batch_size, cached_seq_len, num_kv_heads, head_dim]
    pub values: Vec<f32>,
    /// Number of cached tokens
    pub cached_len: usize,
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            cached_len: 0,
        }
    }
}

/// Llama model with native f32 precision for GPU acceleration
/// GPU加速用ネイティブf32精度Llamaモデル
pub struct F32LlamaModel {
    config: LlamaConfig,
    weights: HashMap<String, F32Tensor>,
    device_type: DeviceType,
    /// KV cache for each layer [num_layers]
    kv_cache: Vec<LayerKVCache>,
    /// Precomputed RoPE frequencies
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl F32LlamaModel {
    /// Create a new Llama model with CPU backend
    /// CPUバックエンドで新しいLlamaモデルを作成
    pub fn new(config: LlamaConfig) -> F32Result<Self> {
        Self::with_device(config, DeviceType::Cpu)
    }

    /// Create a new Llama model with specified device
    /// 指定デバイスでLlamaモデルを作成
    pub fn with_device(config: LlamaConfig, device_type: DeviceType) -> F32Result<Self> {
        eprintln!("🚀 Creating F32LlamaModel with {:?} device", device_type);
        eprintln!("   Precision: native f32 (optimized for GPU)");
        eprintln!("   Architecture: Llama-2 with RoPE + RMSNorm + SwiGLU");

        // Initialize KV cache for each layer
        let kv_cache = (0..config.num_layers)
            .map(|_| LayerKVCache::new())
            .collect();

        // Precompute RoPE frequencies
        let (rope_cos, rope_sin) = Self::precompute_rope_frequencies(&config);

        Ok(Self {
            config,
            weights: HashMap::new(),
            device_type,
            kv_cache,
            rope_cos,
            rope_sin,
        })
    }

    /// Precompute RoPE (Rotary Position Embedding) frequencies
    /// RoPE周波数を事前計算
    fn precompute_rope_frequencies(config: &LlamaConfig) -> (Vec<f32>, Vec<f32>) {
        let head_dim = config.head_dim();
        let max_seq_len = config.max_seq_len;
        let theta = config.rope_theta;

        let mut cos_values = Vec::with_capacity(max_seq_len * head_dim);
        let mut sin_values = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            for i in 0..(head_dim / 2) {
                let freq = 1.0 / theta.powf(2.0 * (i as f32) / (head_dim as f32));
                let angle = (pos as f32) * freq;
                cos_values.push(angle.cos());
                sin_values.push(angle.sin());
            }
        }

        (cos_values, sin_values)
    }

    /// Load Llama model from GGUF file with Metal GPU support
    /// Metal GPUサポート付きでGGUFファイルからLlamaモデルを読み込み
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> F32Result<Self> {
        Self::from_gguf_with_device(path, DeviceType::Metal)
    }

    /// Load Llama model from GGUF file with specified device
    /// 指定デバイスでGGUFファイルからLlamaモデルを読み込み
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

        let config = LlamaConfig::from_model_params(&params);

        // Create model with device
        let mut model = Self::with_device(config, device_type)?;

        eprintln!("📊 Loading Llama model weights as f32");
        eprintln!("   Device: {:?}", device_type);
        eprintln!("   Vocab size: {}", model.config.vocab_size);
        eprintln!("   Layers: {}", model.config.num_layers);
        eprintln!("   Hidden size: {}", model.config.hidden_size);
        eprintln!("   Num heads: {}", model.config.num_heads);
        eprintln!("   Num KV heads: {} (GQA)", model.config.num_kv_heads);

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

                        // Debug: log ALL weight names and shapes to identify LM head
                        eprintln!("WEIGHT_DEBUG: {} {:?}", name, shape);

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

        eprintln!("✅ Loaded {}/{} weights successfully", loaded_count, tensor_names.len());

        Ok(model)
    }

    /// Get model configuration
    /// モデル設定を取得
    pub fn config(&self) -> &LlamaConfig {
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

    /// Clear KV cache
    /// KVキャッシュをクリア
    pub fn clear_kv_cache(&mut self) {
        for cache in &mut self.kv_cache {
            cache.keys.clear();
            cache.values.clear();
            cache.cached_len = 0;
        }
    }

    /// RMSNorm (Root Mean Square Layer Normalization)
    /// RMSNorm（二乗平均平方根正規化）
    fn rms_norm(&self, x: &F32Tensor, weight: &F32Tensor) -> F32Result<F32Tensor> {
        // RMS = sqrt(mean(x^2) + eps)
        // output = (x / RMS) * weight

        let eps = self.config.rms_norm_eps;
        let x_data = x.as_slice();
        let weight_data = weight.as_slice();
        let shape = x.shape();

        // Calculate RMS over last dimension
        let last_dim = shape[shape.len() - 1];
        let batch_size = x_data.len() / last_dim;

        let mut output = Vec::with_capacity(x_data.len());

        for i in 0..batch_size {
            let start = i * last_dim;
            let end = start + last_dim;
            let slice = &x_data[start..end];

            // Calculate RMS
            let sum_sq: f32 = slice.iter().map(|&v| v * v).sum();
            let rms = (sum_sq / (last_dim as f32) + eps).sqrt();

            // Normalize and scale
            for j in 0..last_dim {
                output.push((slice[j] / rms) * weight_data[j]);
            }
        }

        F32Tensor::from_vec(output, shape)
            .map_err(|e| F32Error::device_error(format!("RMSNorm failed: {}", e)))
    }

    /// SwiGLU activation: SiLU(gate) * up
    /// SwiGLU活性化関数: SiLU(gate) * up
    fn swiglu(&self, gate: &F32Tensor, up: &F32Tensor) -> F32Result<F32Tensor> {
        let gate_data = gate.as_slice();
        let up_data = up.as_slice();

        if gate_data.len() != up_data.len() {
            return Err(F32Error::shape_mismatch(format!(
                "SwiGLU shape mismatch: gate {} vs up {}",
                gate_data.len(),
                up_data.len()
            )));
        }

        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let output: Vec<f32> = gate_data.iter().zip(up_data.iter())
            .map(|(&g, &u)| {
                let silu = g / (1.0 + (-g).exp());
                silu * u
            })
            .collect();

        F32Tensor::from_vec(output, gate.shape())
            .map_err(|e| F32Error::device_error(format!("SwiGLU failed: {}", e)))
    }

    /// Apply RoPE (Rotary Position Embedding)
    /// RoPE（回転位置埋め込み）を適用
    fn apply_rope(&self, x: &F32Tensor, start_position: usize) -> F32Result<F32Tensor> {
        let shape = x.shape();
        let seq_len = shape[0];
        let total_dim = shape[1];
        let head_dim = self.config.head_dim();
        let num_heads = total_dim / head_dim;
        let x_data = x.as_slice();

        let mut output = Vec::with_capacity(x_data.len());

        // Apply rotation for each token in sequence
        for token_idx in 0..seq_len {
            let position = start_position + token_idx;

            // For each head of this token
            for head_idx in 0..num_heads {
                let head_offset = token_idx * total_dim + head_idx * head_dim;
                let head_data = &x_data[head_offset..head_offset + head_dim];

                for i in 0..(head_dim / 2) {
                    let cos = self.rope_cos[position * (head_dim / 2) + i];
                    let sin = self.rope_sin[position * (head_dim / 2) + i];

                    let x0 = head_data[2 * i];
                    let x1 = head_data[2 * i + 1];

                    // Rotate: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
                    output.push(x0 * cos - x1 * sin);
                    output.push(x0 * sin + x1 * cos);
                }
            }
        }

        F32Tensor::from_vec(output, shape)
            .map_err(|e| F32Error::device_error(format!("RoPE failed: {}", e)))
    }

    /// Grouped-Query Attention (GQA)
    /// グループ化クエリアテンション
    fn grouped_query_attention(
        &self,
        q: &F32Tensor,
        k: &F32Tensor,
        v: &F32Tensor,
        cached_k: Option<&[f32]>,
        cached_v: Option<&[f32]>,
    ) -> F32Result<(F32Tensor, Vec<f32>, Vec<f32>)> {
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let num_groups = num_heads / num_kv_heads;

        // Q: [seq_len, num_heads * head_dim]
        // K, V: [seq_len, num_kv_heads * head_dim]

        let q_data = q.as_slice();
        let k_data = k.as_slice();
        let v_data = v.as_slice();

        let seq_len = q_data.len() / (num_heads * head_dim);
        let kv_seq_len = k_data.len() / (num_kv_heads * head_dim);

        // Concatenate with cached K, V if available
        let (full_k, full_v, total_kv_len) = if let (Some(ck), Some(cv)) = (cached_k, cached_v) {
            let cached_len = ck.len() / (num_kv_heads * head_dim);
            let mut new_k = ck.to_vec();
            new_k.extend_from_slice(k_data);
            let mut new_v = cv.to_vec();
            new_v.extend_from_slice(v_data);
            (new_k, new_v, cached_len + kv_seq_len)
        } else {
            (k_data.to_vec(), v_data.to_vec(), kv_seq_len)
        };

        // Attention output
        let mut output = vec![0.0f32; seq_len * num_heads * head_dim];

        // For each query position
        for q_pos in 0..seq_len {
            // For each query head
            for h in 0..num_heads {
                let kv_head = h / num_groups;  // Which KV head this Q head uses

                // Get query vector
                let q_start = q_pos * num_heads * head_dim + h * head_dim;
                let q_vec = &q_data[q_start..q_start + head_dim];

                // Compute attention scores
                let mut scores = Vec::with_capacity(total_kv_len);
                for kv_pos in 0..total_kv_len {
                    let k_start = kv_pos * num_kv_heads * head_dim + kv_head * head_dim;
                    let k_vec = &full_k[k_start..k_start + head_dim];

                    // Dot product
                    let score: f32 = q_vec.iter().zip(k_vec.iter())
                        .map(|(&q, &k)| q * k)
                        .sum();

                    scores.push(score / (head_dim as f32).sqrt());
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let attn_weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

                // Weighted sum of values
                for dim in 0..head_dim {
                    let mut weighted_sum = 0.0f32;
                    for kv_pos in 0..total_kv_len {
                        let v_start = kv_pos * num_kv_heads * head_dim + kv_head * head_dim;
                        weighted_sum += attn_weights[kv_pos] * full_v[v_start + dim];
                    }
                    output[q_pos * num_heads * head_dim + h * head_dim + dim] = weighted_sum;
                }
            }
        }

        let output_tensor = F32Tensor::from_vec(output, &[seq_len, num_heads * head_dim])
            .map_err(|e| F32Error::device_error(format!("GQA output failed: {}", e)))?;

        Ok((output_tensor, full_k, full_v))
    }

    /// Get embedding for a single token
    /// 単一トークンの埋め込みを取得
    fn get_embedding(&self, token_id: usize) -> F32Result<Vec<f32>> {
        // Try multiple possible embedding weight names
        let embed_weight = self.weights.get("token_embd.weight")
            .or_else(|| self.weights.get("model.embed_tokens.weight"))
            .or_else(|| self.weights.get("tok_embeddings.weight"))
            .or_else(|| self.weights.get("transformer.wte.weight"))
            .or_else(|| self.weights.get("embeddings.weight"))
            .ok_or_else(|| F32Error::device_error("Embedding weight not found"))?;

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let embed_data = embed_weight.as_slice();
        let embed_shape = embed_weight.shape();

        if token_id >= vocab_size {
            return Err(F32Error::device_error(format!(
                "Token ID {} out of vocab range {}",
                token_id, vocab_size
            )));
        }

        // GGUF loads embeddings as [hidden_size, vocab_size] instead of [vocab_size, hidden_size]
        // So we need to extract the token_id-th column, not row
        if embed_shape[0] == hidden_size && embed_shape[1] == vocab_size {
            // Weight is [hidden_size, vocab_size] - extract column token_id
            let mut embedding = Vec::with_capacity(hidden_size);
            for i in 0..hidden_size {
                let idx = i * vocab_size + token_id;
                if idx >= embed_data.len() {
                    return Err(F32Error::device_error(format!(
                        "Embedding index out of range: {} >= {}",
                        idx, embed_data.len()
                    )));
                }
                embedding.push(embed_data[idx]);
            }
            Ok(embedding)
        } else {
            // Standard layout [vocab_size, hidden_size] - extract row token_id
            let start = token_id * hidden_size;
            let end = start + hidden_size;

            if end > embed_data.len() {
                return Err(F32Error::device_error(format!(
                    "Embedding index out of range: {} > {}",
                    end, embed_data.len()
                )));
            }

            Ok(embed_data[start..end].to_vec())
        }
    }

    /// Llama attention layer
    /// Llamaアテンション層
    fn attention_layer(
        &mut self,
        x: &F32Tensor,
        layer_idx: usize,
        position: usize,
    ) -> F32Result<F32Tensor> {
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;

        // Get weights
        let q_weight = self.weights.get(&format!("blk.{}.attn_q.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Q weight not found for layer {}", layer_idx)))?;
        let k_weight = self.weights.get(&format!("blk.{}.attn_k.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("K weight not found for layer {}", layer_idx)))?;
        let v_weight = self.weights.get(&format!("blk.{}.attn_v.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("V weight not found for layer {}", layer_idx)))?;
        let o_weight = self.weights.get(&format!("blk.{}.attn_output.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Output weight not found for layer {}", layer_idx)))?;

        // Linear projections: Q, K, V
        let q = x.matmul(q_weight)?;
        let k = x.matmul(k_weight)?;
        let v = x.matmul(v_weight)?;

        // Apply RoPE to Q and K
        let q_rope = self.apply_rope(&q, position)?;
        let k_rope = self.apply_rope(&k, position)?;

        // Get cached K, V
        let cache = &self.kv_cache[layer_idx];
        let cached_k = if cache.cached_len > 0 { Some(cache.keys.as_slice()) } else { None };
        let cached_v = if cache.cached_len > 0 { Some(cache.values.as_slice()) } else { None };

        // Grouped-Query Attention
        let (attn_output, new_k, new_v) = self.grouped_query_attention(
            &q_rope,
            &k_rope,
            &v,
            cached_k,
            cached_v,
        )?;

        // Update KV cache with correct sequence length increment
        let seq_len = k_rope.shape()[0];
        let old_len = self.kv_cache[layer_idx].cached_len;
        self.kv_cache[layer_idx].keys = new_k.clone();
        self.kv_cache[layer_idx].values = new_v.clone();
        self.kv_cache[layer_idx].cached_len += seq_len;

        // DEBUG: Log KV cache size for first layer
        if layer_idx == 0 {
            eprintln!("🔍 [KV] layer={}, old_len={}, seq_len={}, new_len={}, keys_size={}",
                layer_idx, old_len, seq_len, self.kv_cache[layer_idx].cached_len, self.kv_cache[layer_idx].keys.len());
        }

        // Output projection
        attn_output.matmul(o_weight).map_err(|e| e.into())
    }

    /// Llama FFN layer with SwiGLU
    /// SwiGLU活性化関数を使用したLlama FFN層
    fn ffn_layer(&self, x: &F32Tensor, layer_idx: usize) -> F32Result<F32Tensor> {
        // Get weights
        let gate_weight = self.weights.get(&format!("blk.{}.ffn_gate.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Gate weight not found for layer {}", layer_idx)))?;
        let up_weight = self.weights.get(&format!("blk.{}.ffn_up.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Up weight not found for layer {}", layer_idx)))?;
        let down_weight = self.weights.get(&format!("blk.{}.ffn_down.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Down weight not found for layer {}", layer_idx)))?;

        // FFN with SwiGLU: down(SwiGLU(gate(x), up(x)))
        let gate = x.matmul(gate_weight)?;
        let up = x.matmul(up_weight)?;
        let swiglu_out = self.swiglu(&gate, &up)?;
        swiglu_out.matmul(down_weight).map_err(|e| e.into())
    }

    /// Single Llama transformer layer
    /// 単一のLlamaトランスフォーマー層
    fn transformer_layer(
        &mut self,
        x: &F32Tensor,
        layer_idx: usize,
        position: usize,
    ) -> F32Result<F32Tensor> {
        // Get normalization weights and clone to avoid borrow issues
        let attn_norm_weight = self.weights.get(&format!("blk.{}.attn_norm.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Attention norm weight not found for layer {}", layer_idx)))?.clone();
        let ffn_norm_weight = self.weights.get(&format!("blk.{}.ffn_norm.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("FFN norm weight not found for layer {}", layer_idx)))?.clone();

        // Attention block with residual connection
        let normed = self.rms_norm(x, &attn_norm_weight)?;
        let attn_out = self.attention_layer(&normed, layer_idx, position)?;
        let x = x.add(&attn_out)?;

        // FFN block with residual connection
        let normed = self.rms_norm(&x, &ffn_norm_weight)?;
        let ffn_out = self.ffn_layer(&normed, layer_idx)?;
        x.add(&ffn_out).map_err(|e| e.into())
    }

    /// Forward pass through Llama model
    /// Llamaモデルのフォワードパス
    pub fn forward(&mut self, input_ids: &[usize]) -> F32Result<F32Tensor> {
        if input_ids.is_empty() {
            return Err(F32Error::device_error("Empty input_ids"));
        }

        let seq_len = input_ids.len();
        let hidden_size = self.config.hidden_size;

        // Get embeddings for all tokens
        let mut embeddings = Vec::with_capacity(seq_len * hidden_size);
        for &token_id in input_ids {
            let emb = self.get_embedding(token_id)?;
            embeddings.extend_from_slice(&emb);
        }

        // DEBUG: Log first token embedding (first 10 elements to see pattern)
        if !embeddings.is_empty() {
            let first_10: Vec<f32> = embeddings.iter().take(10).copied().collect();
            eprintln!("🔍 [EMB] token={} embedding[0..10]={:?}", input_ids[0], first_10);
            // Count zeros in first 100 elements
            let zeros = embeddings.iter().take(100).filter(|&&x| x == 0.0).count();
            eprintln!("🔍 [EMB] zeros in first 100 elements: {}/100", zeros);
        }

        let mut x = F32Tensor::from_vec(embeddings, &[seq_len, hidden_size])?;

        // Get current position from first layer's cache (all layers should have same length)
        let current_position = self.kv_cache[0].cached_len;
        eprintln!("🔍 [FORWARD] seq_len={}, current_position={}", seq_len, current_position);

        // Pass through all transformer layers
        for layer_idx in 0..self.config.num_layers {
            x = self.transformer_layer(&x, layer_idx, current_position)?;
            
            // DEBUG: Log first layer output (first 5 elements)
            if layer_idx == 0 {
                let layer0_out: Vec<f32> = x.as_slice().iter().take(5).copied().collect();
                eprintln!("🔍 [L0] layer_0_output[0..5]={:?}", layer0_out);
            }
        }

        // Final RMSNorm
        let output_norm_weight = self.weights.get("output_norm.weight")
            .or_else(|| self.weights.get("model.norm.weight"))
            .or_else(|| self.weights.get("norm.weight"))
            .ok_or_else(|| F32Error::device_error("Output norm weight not found"))?;

        let normed = self.rms_norm(&x, output_norm_weight)?;

        // LM head (project to vocabulary)
        let lm_head_weight = self.weights.get("output.weight")
            .or_else(|| self.weights.get("lm_head.weight"))
            .or_else(|| self.weights.get("token_embd.weight"))  // Weight tying
            .ok_or_else(|| F32Error::device_error("LM head weight not found"))?;

        // Get logits for last token only
        let last_token_hidden = F32Tensor::from_vec(
            normed.as_slice()[(seq_len - 1) * hidden_size..seq_len * hidden_size].to_vec(),
            &[1, hidden_size]
        )?;

        let logits = last_token_hidden.matmul(lm_head_weight).map_err(|e: crate::error::RusTorchError| -> F32Error { e.into() })?;
        
        // DEBUG: Log top-5 logits
        let logits_slice = logits.as_slice();
        let mut indexed: Vec<(usize, f32)> = logits_slice.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<(usize, f32)> = indexed.iter().take(5).copied().collect();
        eprintln!("🔍 [LOGITS] top5={:?}", top5);
        
        Ok(logits)
    }

    /// Clear KV cache for all layers
    /// 全レイヤーのKVキャッシュをクリア
    pub fn clear_cache(&mut self) {
        for cache in &mut self.kv_cache {
            cache.keys.clear();
            cache.values.clear();
            cache.cached_len = 0;
        }
    }

    /// Load Llama model from GGUF file with custom config
    /// カスタム設定でGGUFファイルからLlamaモデルを読み込む
    pub fn from_gguf_with_config<P: AsRef<std::path::Path>>(
        path: P,
        config: LlamaConfig,
        device_type: DeviceType,
    ) -> F32Result<Self> {
        use crate::formats::gguf::GGUFLoader;

        let loader = GGUFLoader::from_file(path)
            .map_err(|e| F32Error::device_error(format!("Failed to load GGUF: {}", e)))?;

        // Create model with device
        let mut model = Self::with_device(config, device_type)?;

        eprintln!("📊 Loading Llama model weights as f32");
        eprintln!("   Device: {:?}", device_type);
        eprintln!("   Vocab size: {}", model.config.vocab_size);
        eprintln!("   Layers: {}", model.config.num_layers);
        eprintln!("   Hidden size: {}", model.config.hidden_size);
        eprintln!("   Num heads: {}", model.config.num_heads);
        eprintln!("   Num KV heads: {}", model.config.num_kv_heads);

        // Load weights as f32
        let tensor_names = loader.tensor_names();
        let mut loaded_count = 0;

        for name in tensor_names.iter() {
            // Load tensor as f64, then convert to f32
            match loader.load_tensor(name) {
                Ok(tensor_f64) => {
                    let data_f64 = &tensor_f64.data;
                    let data_f32: Vec<f32> = data_f64.iter().map(|&x| x as f32).collect();
                    let shape = tensor_f64.shape();

                    match F32Tensor::from_vec(data_f32, shape) {
                        Ok(tensor_f32) => {
                            // Debug: check key weights (embeddings, attention Q/K/V, output)
                            if name.contains("token_embd") || name.contains("attn_q") ||
                               name.contains("attn_k") || name.contains("attn_v") ||
                               name.contains("output") || name.contains("ffn_gate") {
                                eprintln!("🔍 WEIGHT: '{}' shape: {:?}", name, shape);
                            }
                            model.weights.insert(name.to_string(), tensor_f32);
                            loaded_count += 1;
                        }
                        Err(e) => {
                            eprintln!("⚠️  Failed to convert tensor '{}': {}", name, e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to load tensor '{}': {}", name, e);
                }
            }
        }

        eprintln!("✅ Loaded {}/{} weights successfully", loaded_count, tensor_names.len());

        if loaded_count == 0 {
            return Err(F32Error::device_error("No weights loaded successfully"));
        }

        Ok(model)
    }
}
