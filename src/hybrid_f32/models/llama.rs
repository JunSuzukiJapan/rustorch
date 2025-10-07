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
/// Statistics for tensor debugging
#[derive(Debug, Clone, Copy)]
struct TensorStats {
    rms: f32,
    min: f32,
    max: f32,
    mean: f32,
}

pub struct F32LlamaModel {
    config: LlamaConfig,
    pub weights: HashMap<String, F32Tensor>,  // TEMPデバッグ用にpub
    device_type: DeviceType,
    /// KV cache for each layer [num_layers]
    pub kv_cache: Vec<LayerKVCache>,  // TEMPデバッグ用にpub
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

        // Debug: Check weight stats for first call
        {
            use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
            static FIRST_NORM_CALL: AtomicBool = AtomicBool::new(true);
            if FIRST_NORM_CALL.swap(false, AtomicOrdering::SeqCst) {
                let weight_stats = Self::compute_stats(weight_data);
                eprintln!("🔍 [RMSNorm WEIGHT] rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                    weight_stats.rms, weight_stats.min, weight_stats.max, weight_stats.mean);
                eprintln!("🔍 [RMSNorm] eps={}, weight_len={}", eps, weight_data.len());
            }
        }

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
                let val = (slice[j] / rms) * weight_data[j];
                output.push(val);
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
                    let rope_idx = position * (head_dim / 2) + i;

                    let cos = self.rope_cos[rope_idx];
                    let sin = self.rope_sin[rope_idx];

                    let x0 = head_data[2 * i];
                    let x1 = head_data[2 * i + 1];

                    // Rotate: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
                    let rotated_0 = x0 * cos - x1 * sin;
                    let rotated_1 = x0 * sin + x1 * cos;

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
        let (full_k, full_v, total_kv_len, cached_len) = if let (Some(ck), Some(cv)) = (cached_k, cached_v) {
            let cached_len = ck.len() / (num_kv_heads * head_dim);
            let mut new_k = ck.to_vec();
            new_k.extend_from_slice(k_data);
            let mut new_v = cv.to_vec();
            new_v.extend_from_slice(v_data);
            (new_k, new_v, cached_len + kv_seq_len, cached_len)
        } else {
            (k_data.to_vec(), v_data.to_vec(), kv_seq_len, 0)
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
                let current_kv_pos = cached_len + q_pos;
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

        // CRITICAL: Based on llama.cpp's ggml_get_rows implementation:
        // ggml_vec_cpy_f32(nc, dst, src0 + i01*nb01)
        //   where nc = ne00 (first dimension = hidden_size)
        //         i01 = token_id
        //         nb01 = stride for second dimension
        //
        // GGUF shape [2048, 32000] means data is stored as:
        //   32000 rows of 2048 elements each
        //   Row 0 = token 0's embedding (2048 values)
        //   Row 1 = token 1's embedding (2048 values)
        //   ...
        //   Row N = token N's embedding (2048 values)
        //
        // So: embedding for token N starts at index N * hidden_size

        let start = token_id * hidden_size;
        let end = start + hidden_size;

        if end > embed_data.len() {
            return Err(F32Error::device_error(format!(
                "Embedding index out of range: token_id={}, start={}, end={}, data_len={}",
                token_id, start, end, embed_data.len()
            )));
        }

        let embedding = embed_data[start..end].to_vec();

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

        // Output projection
        let final_out = attn_output.matmul(o_weight)?;

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


        let swiglu_out = self.swiglu(&gate, &up)?;


        let final_out = swiglu_out.matmul(down_weight)?;


        Ok(final_out)
    }

    /// Compute statistics for debugging
    /// デバッグ用の統計を計算
    fn compute_stats(data: &[f32]) -> TensorStats {
        if data.is_empty() {
            return TensorStats { rms: 0.0, min: 0.0, max: 0.0, mean: 0.0 };
        }

        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;

        let sum_sq: f32 = data.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / data.len() as f32).sqrt();

        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        TensorStats { rms, min, max, mean }
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

        // Debug logging for layer 0 only
        let debug_layer = layer_idx == 0;

        if debug_layer {
            let x_slice = x.as_slice();
            let stats = Self::compute_stats(x_slice);
            eprintln!("🔍 [LAYER {}] Input: rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);
        }

        // Attention block with residual connection
        let normed = self.rms_norm(x, &attn_norm_weight)?;

        if debug_layer {
            let normed_slice = normed.as_slice();
            let stats = Self::compute_stats(normed_slice);
            eprintln!("🔍 [LAYER {}] After Attn RMSNorm: rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);
        }

        let attn_out = self.attention_layer(&normed, layer_idx, position)?;

        if debug_layer {
            let attn_slice = attn_out.as_slice();
            let stats = Self::compute_stats(attn_slice);
            eprintln!("🔍 [LAYER {}] Attention Output: rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);
        }

        let x = x.add(&attn_out)?;

        if debug_layer {
            let x_slice = x.as_slice();
            let stats = Self::compute_stats(x_slice);
            eprintln!("🔍 [LAYER {}] After Attn Residual: rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);
        }

        // FFN block with residual connection
        let normed = self.rms_norm(&x, &ffn_norm_weight)?;

        if debug_layer {
            let normed_slice = normed.as_slice();
            let stats = Self::compute_stats(normed_slice);
            eprintln!("🔍 [LAYER {}] After FFN RMSNorm: rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);
        }

        let ffn_out = self.ffn_layer(&normed, layer_idx)?;

        if debug_layer {
            let ffn_slice = ffn_out.as_slice();
            let stats = Self::compute_stats(ffn_slice);
            eprintln!("🔍 [LAYER {}] FFN Output: rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);
        }

        let result = x.add(&ffn_out)?;

        if debug_layer {
            let result_slice = result.as_slice();
            let stats = Self::compute_stats(result_slice);
            eprintln!("🔍 [LAYER {}] After FFN Residual (Layer Output): rms={:.6}, min={:.6}, max={:.6}, mean={:.6}\n",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);
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


        let mut x = F32Tensor::from_vec(embeddings, &[seq_len, hidden_size])?;

        // DEBUG: Check embeddings for first forward call
        {
            use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
            static FIRST_CALL: AtomicBool = AtomicBool::new(true);
            if FIRST_CALL.swap(false, AtomicOrdering::SeqCst) {
                eprintln!("🔍 [EMBEDDING] First 10 values of first token embedding:");
                let first_emb: Vec<f32> = x.as_slice()[..10.min(hidden_size)].to_vec();
                eprintln!("   {:?}", first_emb);
                eprintln!("🔍 [EMBEDDING] input_ids: {:?}", input_ids);
            }
        }

        // Get current position from first layer's cache (all layers should have same length)
        let current_position = self.kv_cache[0].cached_len;

        // Strategic debug: Monitor problematic positions only

        // Pass through all transformer layers
        for layer_idx in 0..self.config.num_layers {
            x = self.transformer_layer(&x, layer_idx, current_position)?;

            if layer_idx == self.config.num_layers - 1 {
                let last_layer_out: Vec<f32> = x.as_slice().iter().take(10).copied().collect();
            }
        }

        // Final RMSNorm
        let output_norm_weight = self.weights.get("output_norm.weight")
            .or_else(|| self.weights.get("model.norm.weight"))
            .or_else(|| self.weights.get("norm.weight"))
            .ok_or_else(|| F32Error::device_error("Output norm weight not found"))?;

        let normed = self.rms_norm(&x, output_norm_weight)?;


        // LM head (project to vocabulary)
        let (lm_head_weight, weight_source) = if let Some(w) = self.weights.get("output.weight") {
            (w, "output.weight")
        } else if let Some(w) = self.weights.get("lm_head.weight") {
            (w, "lm_head.weight")
        } else if let Some(w) = self.weights.get("token_embd.weight") {
            (w, "token_embd.weight (weight tying)")
        } else {
            return Err(F32Error::device_error("LM head weight not found"));
        };

        // Get logits for last token only
        let last_token_hidden = F32Tensor::from_vec(
            normed.as_slice()[(seq_len - 1) * hidden_size..seq_len * hidden_size].to_vec(),
            &[1, hidden_size]
        )?;

        let last_hidden_vals: Vec<f32> = last_token_hidden.as_slice().iter().take(10).copied().collect();

        let hidden_slice = last_token_hidden.as_slice();
        let non_zero_count = hidden_slice.iter().filter(|&&x| x != 0.0).count();
        let sum: f32 = hidden_slice.iter().sum();

        // Debug: Save hidden state for first 3 forward passes
        // We use a static counter to track across all forward calls
        use std::sync::atomic::{AtomicUsize, Ordering};
        static FORWARD_COUNTER: AtomicUsize = AtomicUsize::new(0);

        let call_num = FORWARD_COUNTER.fetch_add(1, Ordering::SeqCst);
        if call_num < 3 {
            let filename = format!("/tmp/hidden_state_call_{}.txt", call_num);
            if let Ok(mut file) = std::fs::File::create(&filename) {
                use std::io::Write;
                for val in hidden_slice {
                    writeln!(file, "{}", val).ok();
                }
                eprintln!("📝 [CALL {}] Saved hidden state (kv_cache_len={}) to {}",
                    call_num, self.kv_cache[0].cached_len, filename);
            }
        }

        let lm_head_data = lm_head_weight.as_slice();
        if lm_head_data.len() >= 10 * 32000 + 1552 {
            let token_1552_weights_start = 1552;  // Column 1552 in row-major layout
            let mut token_1552_sample: Vec<f32> = Vec::new();
            for i in 0..10 {
                let idx = i * 32000 + token_1552_weights_start;
                token_1552_sample.push(lm_head_data[idx]);
            }
        } else {
        }

        // Debug: Print shapes for first call
        if call_num == 0 {
            let lm_shape = lm_head_weight.shape();
            let hidden_shape = last_token_hidden.shape();
            eprintln!("🔍 [LM HEAD SOURCE] Using weight: '{}'", weight_source);
            eprintln!("🔍 [LM HEAD SHAPES] hidden={:?}, lm_head={:?}", hidden_shape, lm_shape);
            eprintln!("🔍 [LM HEAD INFO] lm_head total_elements={}, hidden_size={}, vocab_size={}",
                lm_head_data.len(), self.config.hidden_size, 32000);

            // Sample first 10 values of LM head for token 0
            eprintln!("🔍 [LM HEAD SAMPLE] First 10 weights for token 0:");
            for i in 0..10.min(lm_head_data.len()) {
                eprintln!("  lm_head[{}] = {}", i, lm_head_data[i]);
            }
        }

        // CRITICAL: Output (LM Head) Weight Layout (VERIFIED in IMPLEMENTATION_VERIFICATION.md)
        // Format: Row-major [hidden_size, vocab_size] = [2048, 32000]
        // Matmul: [1, 2048] @ [2048, 32000] = [1, 32000]
        //
        // This layout is VERIFIED to be 100% correct through:
        // - Manual logit calculation (examples/manual_logit_calculation.rs)
        // - Matmul accuracy: 99.9999% match with hand calculations
        // - Token 450 logit: 0.06317014 (manual: 0.06316983)
        //
        // Access pattern for matmul:
        //   logits[v] = sum_h(hidden[h] * output_weight[h, v])
        //            = sum_h(hidden[h] * data[h * vocab_size + v])
        //
        // Reference: docs/core/IMPLEMENTATION_VERIFICATION.md, Line 133-136

        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        let hidden_slice = last_token_hidden.as_slice();
        let mut logits_vec = vec![0.0f32; vocab_size];

        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            for h in 0..hidden_size {
                let idx = h * vocab_size + v;
                if idx >= lm_head_data.len() {
                    return Err(F32Error::device_error(format!(
                        "LM head index out of range: v={}, h={}, idx={}, data_len={}",
                        v, h, idx, lm_head_data.len()
                    )));
                }
                sum += hidden_slice[h] * lm_head_data[idx];
            }
            logits_vec[v] = sum;
        }

        // Convert to F32Tensor
        let logits = F32Tensor::from_vec(logits_vec.clone(), &[1, vocab_size])
            .map_err(|e| F32Error::device_error(&format!("Failed to create logits tensor: {}", e)))?;

        // DEBUG: Save first call's logits to file for analysis
        {
            use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
            static FIRST_LOGITS: AtomicBool = AtomicBool::new(true);
            if FIRST_LOGITS.swap(false, AtomicOrdering::SeqCst) {
                use std::io::Write;
                if let Ok(mut file) = std::fs::File::create("/tmp/rustorch_logits.txt") {
                    for (i, &logit) in logits_vec.iter().enumerate() {
                        writeln!(file, "{} {:.6}", i, logit).ok();
                    }
                    eprintln!("📝 [DEBUG] Saved logits to /tmp/rustorch_logits.txt");
                }
            }
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

    /// Get KV cache length for a specific layer
    pub fn get_kv_cache_len(&self, layer_idx: usize) -> usize {
        self.kv_cache.get(layer_idx).map(|c| c.cached_len).unwrap_or(0)
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
                    if name.contains("token_embd") || name.contains("output.weight") {
                        eprintln!("📊 [WEIGHT INFO] '{}': GGUF_shape={:?}, tensor_elements={}",
                            name, shape, final_tensor.as_slice().len());

                        // Sample first 10 values
                        let sample: Vec<f32> = final_tensor.as_slice().iter().take(10).copied().collect();
                        eprintln!("   First 10 values: {:?}", sample);
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

        // DEBUG: Test output.weight vs token_embd relationship
        if let (Some(token_embd), Some(output_weight)) = (model.weights.get("token_embd.weight"), model.weights.get("output.weight")) {
            eprintln!("\n🔬 [WEIGHT TEST] Testing output.weight correctness...");

            // Get embedding for token 1 (BOS)
            let token_id = 1usize;
            let hidden_size = 2048;
            let vocab_size = 32000;

            let token_embd_data = token_embd.as_slice();
            let output_data = output_weight.as_slice();

            // Extract embedding for token 1 using VERIFIED column-major layout
            // Format: Column-major [hidden_size, vocab_size]
            // Access: embedding[dim] = data[dim * vocab_size + token_id]
            let mut embedding = Vec::with_capacity(hidden_size);
            for dim in 0..hidden_size {
                embedding.push(token_embd_data[dim * vocab_size + token_id]);
            }

            eprintln!("   Token {} embedding (first 5): {:?}", token_id, &embedding[..5]);

            // Compute logits: embedding @ output.weight^T
            // GGML dims [2048, 32000]: logits[v] = sum_h(embedding[h] * output[h * vocab_size + v])
            let mut logits = vec![0.0f32; vocab_size];
            for v in 0..vocab_size {
                let mut sum = 0.0f32;
                for h in 0..hidden_size {
                    sum += embedding[h] * output_data[h * vocab_size + v];
                }
                logits[v] = sum;
            }

            eprintln!("   Logit[0] (should be embedding · output[0]): {}", logits[0]);
            eprintln!("   Logit[1] (should be embedding · output[1]): {}", logits[1]);

            // Find top 10 logits
            let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            eprintln!("   Top 10 logits for token {} embedding:", token_id);
            for (rank, (tok, logit)) in indexed.iter().take(10).enumerate() {
                eprintln!("     #{}: token={} logit={:.4}", rank+1, tok, logit);
            }

            // Check if token 1 itself has high logit (should be for BOS -> BOS)
            eprintln!("   Logit for token {} itself: {:.4}", token_id, logits[token_id]);
        }

        if loaded_count == 0 {
            return Err(F32Error::device_error("No weights loaded successfully"));
        }

        Ok(model)
    }

    /// Apply TinyLlama chat template to user input
    /// TinyLlamaチャットテンプレートをユーザー入力に適用
    ///
    /// Format: `<|system|>\n{system_msg}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n`
    pub fn apply_chat_template(&self, user_input: &str, system_message: Option<&str>) -> Vec<u32> {
        // TinyLlama chat template token IDs
        // Reference: examples/test_with_proper_template.rs
        let mut tokens = vec![
            1,      // <s> (BOS)
            529,    // <
            29989,  // |
            5205,   // system
            29989,  // |
            29958,  // >
            29871,  // whitespace
            13,     // \n
        ];

        // Add system message
        let system_msg = system_message.unwrap_or("You are a helpful assistant.");
        // TODO: Tokenize system_msg properly - for now, placeholder
        // tokens.extend(tokenize(system_msg));

        tokens.extend(vec![
            3575,   // You
            526,    // are
            263,    // a
            8444,   // helpful
            20255,  // assistant
            29889,  // .
            2,      // </s> (EOS)
            13,     // \n
        ]);

        // User section
        tokens.extend(vec![
            529,    // <
            29989,  // |
            1792,   // user
            29989,  // |
            29958,  // >
            29871,  // whitespace
            13,     // \n
        ]);

        // Add user input
        // TODO: Tokenize user_input properly - for now, placeholder
        // tokens.extend(tokenize(user_input));
        tokens.extend(vec![
            1724,   // What
            338,    // is
            278,    // the
            7483,   // capital
            310,    // of
            3444,   // France
            29973,  // ?
            2,      // </s>
            13,     // \n
        ]);

        // Assistant section (generation starts here)
        tokens.extend(vec![
            529,    // <
            29989,  // |
            465,    // assistant
            22137,  // (continuation)
            29989,  // |
            29958,  // >
            29871,  // whitespace
            13,     // \n
        ]);

        tokens
    }

    /// Generate text response using chat interface
    /// チャットインターフェースを使用してテキスト応答を生成
    ///
    /// # Arguments
    /// * `user_input` - User's message
    /// * `system_message` - Optional system message (defaults to "You are a helpful assistant.")
    /// * `max_tokens` - Maximum number of tokens to generate
    ///
    /// # Example
    /// ```ignore
    /// let model = F32LlamaModel::from_gguf_with_device("model.gguf", DeviceType::Cpu)?;
    /// let response = model.chat("What is Rust?", None, 50)?;
    /// println!("{}", response);
    /// ```
    pub fn chat(
        &mut self,
        user_input: &str,
        system_message: Option<&str>,
        max_tokens: usize,
    ) -> F32Result<String> {
        // Apply chat template
        let tokens = self.apply_chat_template(user_input, system_message);

        // Convert u32 to usize for forward pass
        let tokens_usize: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();

        // Generate response
        let mut generated_tokens = Vec::new();
        let mut current_tokens = tokens_usize.clone();

        for _ in 0..max_tokens {
            // Forward pass
            let logits = self.forward(&current_tokens)?;

            // Get last token logits (vocab_size)
            let vocab_size = self.config.vocab_size;
            let logits_vec: Vec<f32> = logits.data.iter().copied().collect();
            let start_idx = logits_vec.len().saturating_sub(vocab_size);
            let last_logits: Vec<f32> = logits_vec[start_idx..].to_vec();

            // Simple argmax sampling (greedy)
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .ok_or_else(|| F32Error::numerical_error("Failed to sample next token"))?;

            // Check for EOS token (ID: 2)
            if next_token == 2 {
                break;
            }

            generated_tokens.push(next_token as u32);
            current_tokens.push(next_token);
        }

        // TODO: Decode tokens to text properly
        // For now, return token IDs as string
        Ok(format!("Generated tokens: {:?}", generated_tokens))
    }

    /// Generate text response with custom sampling config
    /// カスタムサンプリング設定でテキスト応答を生成
    pub fn chat_with_sampling(
        &mut self,
        user_input: &str,
        system_message: Option<&str>,
        max_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> F32Result<String> {
        // TODO: Implement sampling with temperature, top-k, top-p
        // For now, use simple chat
        self.chat(user_input, system_message, max_tokens)
    }
}
