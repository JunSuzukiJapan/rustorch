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

        // Use actual num_kv_heads from GGUF metadata
        // GQA (Grouped Query Attention): TinyLlama uses 4 KV heads with 32 query heads
        // MHA (Multi-Head Attention): num_kv_heads = num_heads
        let num_kv_heads = params.num_kv_heads as usize;

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

        // DEBUG: Log RoPE parameters
        let _debug_rope = false;
        if _debug_rope && start_position == 0 && seq_len == 1 {
            eprintln!("🔍 [ROPE_INIT] start_position={}, seq_len={}, total_dim={}, head_dim={}, num_heads={}",
                start_position, seq_len, total_dim, head_dim, num_heads);
            eprintln!("🔍 [ROPE_INIT] rope_cos.len()={}, rope_sin.len()={}", self.rope_cos.len(), self.rope_sin.len());
        }

        // Apply rotation for each token in sequence
        for token_idx in 0..seq_len {
            let position = start_position + token_idx;

            // For each head of this token
            for head_idx in 0..num_heads {
                let head_offset = token_idx * total_dim + head_idx * head_dim;
                let head_data = &x_data[head_offset..head_offset + head_dim];

                for i in 0..(head_dim / 2) {
                    let rope_idx = position * (head_dim / 2) + i;

                    // DEBUG: Log first RoPE access
                    if _debug_rope && start_position == 0 && token_idx == 0 && head_idx == 0 && i < 3 {
                        eprintln!("🔍 [ROPE_IDX] position={}, i={}, rope_idx={}, head_data[{}]={}, head_data[{}]={}",
                            position, i, rope_idx, 2*i, head_data[2*i], 2*i+1, head_data[2*i+1]);
                    }

                    let cos = self.rope_cos[rope_idx];
                    let sin = self.rope_sin[rope_idx];

                    if _debug_rope && start_position == 0 && token_idx == 0 && head_idx == 0 && i < 3 {
                        eprintln!("🔍 [ROPE_VAL] i={}, cos={}, sin={}", i, cos, sin);
                    }

                    let x0 = head_data[2 * i];
                    let x1 = head_data[2 * i + 1];

                    // Rotate: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
                    let rotated_0 = x0 * cos - x1 * sin;
                    let rotated_1 = x0 * sin + x1 * cos;

                    if _debug_rope && start_position == 0 && token_idx == 0 && head_idx == 0 && i < 3 {
                        eprintln!("🔍 [ROPE_OUT] i={}, x0={}, x1={} → rotated_0={}, rotated_1={}",
                            i, x0, x1, rotated_0, rotated_1);
                    }

                    output.push(rotated_0);
                    output.push(rotated_1);
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

                // Compute attention scores with causal masking
                // Query at position q_pos can only attend to keys at positions 0..=current_kv_pos
                // where current_kv_pos = cached_len + q_pos
                let current_kv_pos = total_kv_len - seq_len + q_pos;
                let mut scores = Vec::with_capacity(current_kv_pos + 1);

                for kv_pos in 0..=current_kv_pos {
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

                // Weighted sum of values (only up to current_kv_pos due to causal masking)
                for dim in 0..head_dim {
                    let mut weighted_sum = 0.0f32;
                    for kv_pos in 0..=current_kv_pos {
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
    pub fn get_embedding(&self, token_id: usize) -> F32Result<Vec<f32>> {
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

        // CRITICAL: GGUF embedding layout
        // Shape [2048, 32000] means row-major storage:
        //   Row 0: [dim0_token0, dim0_token1, ..., dim0_token31999]  (32000 values)
        //   Row 1: [dim1_token0, dim1_token1, ..., dim1_token31999]  (32000 values)
        //   ...
        //   Row 2047: [dim2047_token0, dim2047_token1, ..., dim2047_token31999]
        //
        // To extract embedding for token_id:
        //   embedding[dim] = data[dim * vocab_size + token_id]

        let mut embedding = Vec::with_capacity(hidden_size);
        for dim in 0..hidden_size {
            let idx = dim * vocab_size + token_id;
            if idx >= embed_data.len() {
                return Err(F32Error::device_error(format!(
                    "Embedding index out of range: dim={}, token_id={}, idx={}, data_len={}",
                    dim, token_id, idx, embed_data.len()
                )));
            }
            embedding.push(embed_data[idx]);
        }

        // Debug: Show first 10 values of embedding for important tokens
        // Token 1 (BOS), Token 1724 ("What"), Token 3681 (Paris)
        if token_id == 1 || token_id == 1724 || token_id == 3681 {
            let first_10: Vec<f32> = embedding.iter().take(10).copied().collect();
            let non_zero_count = embedding.iter().filter(|&&x| x.abs() > 1e-8).count();
            eprintln!("🔍 [GET_EMB] token={} embedding[0..10]={:?} (non_zero={}/{})",
                token_id, first_10, non_zero_count, embedding.len());
        }

        Ok(embedding)
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

        // DEBUG: Log Q weight values for layer 0
        let _debug_layer0 = true; // Enable for debugging weight shapes
        if _debug_layer0 && layer_idx == 0 {
            let q_weight_first: Vec<f32> = q_weight.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [WEIGHT] layer=0 Q_weight[0..10]={:?}", q_weight_first);
            eprintln!("🔍 [WEIGHT] layer=0 Q_weight shape={:?}", q_weight.shape());
            let x_first: Vec<f32> = x.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [INPUT] layer=0 x[0..10]={:?}", x_first);
            eprintln!("🔍 [INPUT] layer=0 x shape={:?}", x.shape());
        }

        // Linear projections: Q, K, V
        let q = x.matmul(q_weight)?;
        let k = x.matmul(k_weight)?;
        let v = x.matmul(v_weight)?;

        // DEBUG: Log Q, K, V values for layer 0
        if _debug_layer0 && layer_idx == 0 {
            let q_first: Vec<f32> = q.as_slice().iter().take(10).copied().collect();
            let k_first: Vec<f32> = k.as_slice().iter().take(10).copied().collect();
            let v_first: Vec<f32> = v.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [QKV] layer=0 Q[0..10]={:?}", q_first);
            eprintln!("🔍 [QKV] layer=0 K[0..10]={:?}", k_first);
            eprintln!("🔍 [QKV] layer=0 V[0..10]={:?}", v_first);
        }

        // Apply RoPE to Q and K
        let q_rope = self.apply_rope(&q, position)?;
        let k_rope = self.apply_rope(&k, position)?;

        // DEBUG: Log after RoPE for layer 0
        if _debug_layer0 && layer_idx == 0 {
            let q_rope_first: Vec<f32> = q_rope.as_slice().iter().take(10).copied().collect();
            let k_rope_first: Vec<f32> = k_rope.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [ROPE] layer=0 Q_rope[0..10]={:?}", q_rope_first);
            eprintln!("🔍 [ROPE] layer=0 K_rope[0..10]={:?}", k_rope_first);
        }

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

        // DEBUG: Log attention output for layer 0 (disabled)
        let _debug_attn = false;
        if _debug_attn && layer_idx == 0 {
            let attn_out_vals: Vec<f32> = attn_output.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [ATTN] layer=0 grouped_attn_output[0..10]={:?}, shape={:?}", attn_out_vals, attn_output.shape());
        }

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
        let final_out = attn_output.matmul(o_weight)?;

        // DEBUG: Log final projection output for layer 0 (disabled)
        if false && layer_idx == 0 {
            let final_vals: Vec<f32> = final_out.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [ATTN] layer=0 after_output_proj[0..10]={:?}", final_vals);
        }

        Ok(final_out)
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

        // DEBUG: Log FFN intermediate values for layer 0 (disabled)
        let _debug_ffn = false;
        if _debug_ffn && layer_idx == 0 {
            let gate_vals: Vec<f32> = gate.as_slice().iter().take(10).copied().collect();
            let up_vals: Vec<f32> = up.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [FFN] layer=0 gate[0..10]={:?}", gate_vals);
            eprintln!("🔍 [FFN] layer=0 up[0..10]={:?}", up_vals);
        }

        let swiglu_out = self.swiglu(&gate, &up)?;

        if _debug_ffn && layer_idx == 0 {
            let swiglu_vals: Vec<f32> = swiglu_out.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [FFN] layer=0 swiglu[0..10]={:?}", swiglu_vals);
        }

        let final_out = swiglu_out.matmul(down_weight)?;

        if _debug_ffn && layer_idx == 0 {
            let final_vals: Vec<f32> = final_out.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [FFN] layer=0 after_down_proj[0..10]={:?}", final_vals);
        }

        Ok(final_out)
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

        // DEBUG: Log layer 0 intermediate values (disabled to reduce overhead)
        let debug = false; // layer_idx == 0;

        // Attention block with residual connection
        let normed = self.rms_norm(x, &attn_norm_weight)?;
        if debug {
            let normed_vals: Vec<f32> = normed.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [L{}] attn_norm_out[0..10]={:?}", layer_idx, normed_vals);
        }

        let attn_out = self.attention_layer(&normed, layer_idx, position)?;
        if debug {
            let attn_vals: Vec<f32> = attn_out.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [L{}] attn_out[0..10]={:?}", layer_idx, attn_vals);
        }

        let x = x.add(&attn_out)?;
        if debug {
            let after_attn_res: Vec<f32> = x.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [L{}] after_attn_residual[0..10]={:?}", layer_idx, after_attn_res);
        }

        // FFN block with residual connection
        let normed = self.rms_norm(&x, &ffn_norm_weight)?;
        if debug {
            let ffn_norm_vals: Vec<f32> = normed.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [L{}] ffn_norm_out[0..10]={:?}", layer_idx, ffn_norm_vals);
        }

        let ffn_out = self.ffn_layer(&normed, layer_idx)?;
        if debug {
            let ffn_vals: Vec<f32> = ffn_out.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [L{}] ffn_out[0..10]={:?}", layer_idx, ffn_vals);
        }

        let result = x.add(&ffn_out)?;
        if debug {
            let final_vals: Vec<f32> = result.as_slice().iter().take(10).copied().collect();
            eprintln!("🔍 [L{}] final_out[0..10]={:?}", layer_idx, final_vals);
        }

        Ok(result)
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

        // DEBUG: Enable to debug embedding extraction
        let _debug_emb = true;
        if _debug_emb && !embeddings.is_empty() {
            let first_20: Vec<f32> = embeddings.iter().take(20).copied().collect();
            eprintln!("🔍 [EMB] first_token={} embedding[0..20]={:?}", input_ids[0], first_20);
        }

        let mut x = F32Tensor::from_vec(embeddings, &[seq_len, hidden_size])?;

        // Get current position from first layer's cache (all layers should have same length)
        let current_position = self.kv_cache[0].cached_len;

        // Strategic debug: Monitor problematic positions only
        let is_critical_position = current_position >= 12 && current_position <= 14;
        let _debug_pos = true;  // Enable for transpose debugging
        if _debug_pos {
            if is_critical_position {
                eprintln!("⚠️  [CRITICAL] position={}, seq_len={}", current_position, seq_len);
            } else {
                eprintln!("🔍 [FORWARD] seq_len={}, current_position={}", seq_len, current_position);
            }
        }

        // Pass through all transformer layers
        for layer_idx in 0..self.config.num_layers {
            x = self.transformer_layer(&x, layer_idx, current_position)?;

            // DEBUG: Log only last layer output, with special attention at critical positions
            if layer_idx == self.config.num_layers - 1 {
                let last_layer_out: Vec<f32> = x.as_slice().iter().take(10).copied().collect();
                if is_critical_position {
                    eprintln!("⚠️  [L{}] CRITICAL last_layer[0..10]={:?}", layer_idx, last_layer_out);
                } else {
                    eprintln!("🔍 [L{}] last_layer_output[0..10]={:?}", layer_idx, last_layer_out);
                }
            }
        }

        // Final RMSNorm
        let output_norm_weight = self.weights.get("output_norm.weight")
            .or_else(|| self.weights.get("model.norm.weight"))
            .or_else(|| self.weights.get("norm.weight"))
            .ok_or_else(|| F32Error::device_error("Output norm weight not found"))?;

        let normed = self.rms_norm(&x, output_norm_weight)?;

        // DEBUG: Log normed output only at critical positions
        if is_critical_position {
            let normed_out: Vec<f32> = normed.as_slice().iter().take(10).copied().collect();
            eprintln!("⚠️  [NORM] CRITICAL after_output_norm[0..10]={:?}", normed_out);
        }

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

        // DEBUG: Log LM head (always for debugging)
        let last_hidden_vals: Vec<f32> = last_token_hidden.as_slice().iter().take(10).copied().collect();
        eprintln!("🔍 [LM_HEAD] last_token_hidden[0..10]={:?}", last_hidden_vals);
        eprintln!("🔍 [LM_HEAD] last_token_hidden shape: {:?}", last_token_hidden.shape());
        eprintln!("🔍 [LM_HEAD] weight shape: {:?}", lm_head_weight.shape());

        // DEBUG: Check output.weight values for token 1552
        let lm_head_data = lm_head_weight.as_slice();
        eprintln!("🔍 [WEIGHT_CHECK] lm_head_data.len()={}", lm_head_data.len());
        if lm_head_data.len() >= 10 * 32000 + 1552 {
            let token_1552_weights_start = 1552;  // Column 1552 in row-major layout
            let mut token_1552_sample: Vec<f32> = Vec::new();
            for i in 0..10 {
                let idx = i * 32000 + token_1552_weights_start;
                token_1552_sample.push(lm_head_data[idx]);
            }
            eprintln!("🔍 [WEIGHT_CHECK] output.weight[:10, 1552]={:?}", token_1552_sample);
        } else {
            eprintln!("⚠️  [WEIGHT_CHECK] Not enough data to check token 1552");
        }

        // GGUF format: output.weight is [hidden_size, vocab_size] for matmul [1, hidden] @ [hidden, vocab]
        eprintln!("🔹 [BEFORE_MATMUL] About to call matmul");
        let logits = last_token_hidden.matmul(lm_head_weight).map_err(|e: crate::error::RusTorchError| -> F32Error { e.into() })?;
        eprintln!("🔸 [AFTER_MATMUL] Matmul completed, logits.shape={:?}", logits.shape());

        // DEBUG: Log top-5 logits only at critical positions
        if is_critical_position {
            let logits_slice = logits.as_slice();
            let mut indexed: Vec<(usize, f32)> = logits_slice.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5: Vec<(usize, f32)> = indexed.iter().take(5).copied().collect();
            eprintln!("⚠️  [LOGITS] CRITICAL top5={:?}", top5);
        }
        
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

                    // GGML format analysis confirmed:
                    // - All weights work as-is, no transpose needed
                    // - Linear layers: [2048, 256] correct for matmul
                    // - token_embd.weight [2048, 32000]: column extraction works
                    // - output.weight [2048, 32000]: correct for matmul [1,2048] @ [2048,32000] → [1,32000]

                    let needs_transpose = false;

                    if name.contains("blk.0.attn") || needs_transpose {
                        eprintln!("🔍 Loading '{}' with shape {:?} (transpose={})", name, shape, needs_transpose);
                    }

                    let final_tensor = if needs_transpose && shape.len() == 2 {
                        // Transpose output.weight and token_embd.weight for matmul compatibility
                        match F32Tensor::from_vec(data_f32, shape) {
                            Ok(t) => {
                                match t.transpose() {
                                    Ok(transposed) => {
                                        eprintln!("🔄 TRANSPOSED '{}': {:?} → {:?}", name, shape, transposed.shape());
                                        transposed
                                    },
                                    Err(e) => {
                                        eprintln!("⚠️  Transpose failed for '{}': {}", name, e);
                                        continue;
                                    }
                                }
                            },
                            Err(e) => {
                                eprintln!("⚠️  Failed to create tensor '{}': {}", name, e);
                                continue;
                            }
                        }
                    } else {
                        match F32Tensor::from_vec(data_f32, shape) {
                            Ok(t) => t,
                            Err(e) => {
                                eprintln!("⚠️  Failed to convert tensor '{}': {}", name, e);
                                continue;
                            }
                        }
                    };

                    // Debug: check key weights (embeddings, attention Q/K/V, output)
                    if name.contains("token_embd") || name.contains("attn_q") ||
                       name.contains("attn_k") || name.contains("attn_v") ||
                       name.contains("output") || name.contains("ffn_gate") {
                        eprintln!("🔍 WEIGHT: '{}' shape: {:?}", name, final_tensor.shape());
                    }
                    model.weights.insert(name.to_string(), final_tensor);
                    loaded_count += 1;
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
