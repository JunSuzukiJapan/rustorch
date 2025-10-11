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

        eprintln!("🔧 [RoPE PRECOMPUTE] head_dim={}, max_seq_len={}, theta={}",
            head_dim, max_seq_len, theta);

        let mut cos_values = Vec::with_capacity(max_seq_len * head_dim);
        let mut sin_values = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            for i in 0..(head_dim / 2) {
                let freq = 1.0 / theta.powf(2.0 * (i as f32) / (head_dim as f32));
                let angle = (pos as f32) * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                // DEBUG: First position, first 3 frequency pairs
                if pos == 0 && i < 3 {
                    eprintln!("  pos={}, i={}, freq={:.9}, angle={:.9}, cos={:.9}, sin={:.9}",
                        pos, i, freq, angle, cos_val, sin_val);
                }

                cos_values.push(cos_val);
                sin_values.push(sin_val);
            }
        }

        eprintln!("🔧 [RoPE PRECOMPUTE] Generated {} cos values, {} sin values",
            cos_values.len(), sin_values.len());
        eprintln!("🔧 [RoPE PRECOMPUTE] Index 0-9:   cos={:?}", &cos_values[0..10]);
        eprintln!("🔧 [RoPE PRECOMPUTE] Index 32-41: cos={:?}", &cos_values[32..42]);

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
            // Load tensor directly as f32 using generic loader
            // ジェネリック型ローダーを使ってf32として直接読み込む（f64への変換を回避）
            match loader.load_tensor_generic::<f32>(name) {
                Ok(f32_data) => {
                    // Get tensor info for shape
                    let tensor_info = loader.get_tensor_info(name);
                    if let Some(info) = tensor_info {
                        let original_dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
                        let ggml_type = crate::formats::gguf::GGMLType::from_u32(info.ggml_type).ok();

                        // CRITICAL: Only F32/F16 tensors need shape reversal
                        // Quantized tensors (Q8_0, Q4_K, etc.) are stored with correct dimensions
                        let shape: Vec<usize> = match ggml_type {
                            Some(crate::formats::gguf::GGMLType::F32) | Some(crate::formats::gguf::GGMLType::F16) => {
                                let mut s = original_dims.clone();
                                s.reverse();
                                s
                            }
                            _ => original_dims.clone(),
                        };

                        // Create F32Tensor from f32 data directly
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
        // EXACT port from llama.cpp ggml_compute_forward_rms_norm_f32
        // See: ggml-cpu/ops.cpp:3521-3570

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

                eprintln!("🔍 [RMSNorm WEIGHT] First 20 values:");
                eprint!("    [");
                for i in 0..20.min(weight_data.len()) {
                    eprint!("{:.9}", weight_data[i]);
                    if i < 19 { eprint!(", "); }
                }
                eprintln!("]");
            }
        }

        // Following llama.cpp EXACTLY:
        // ne00 = last dimension (hidden_size)
        // ne01 = batch dimension (seq_len)
        let ne00 = shape[shape.len() - 1];  // hidden_size
        let ne01 = x_data.len() / ne00;     // seq_len

        let mut output = vec![0.0f32; x_data.len()];

        // Debug: track first token for detailed analysis
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        static FIRST_RMSNORM: AtomicBool = AtomicBool::new(true);
        let is_first = FIRST_RMSNORM.swap(false, AtomicOrdering::SeqCst);

        for i01 in 0..ne01 {
            // Line 3545: const float * x = ...
            let x_offset = i01 * ne00;
            let x_slice = &x_data[x_offset..x_offset + ne00];

            // Line 3547-3550: Calculate sum of squares
            let mut sum: f32 = 0.0;
            for i00 in 0..ne00 {
                sum += x_slice[i00] * x_slice[i00];
            }

            // Line 3552: const float mean = sum/ne00;
            let mean = sum / (ne00 as f32);

            // Debug
            if is_first {
                let input_stats = Self::compute_stats(x_slice);
                eprintln!("🐛 [RMSNorm DEBUG] Token {}:", i01);
                eprintln!("   Input: rms={:.6}, min={:.6}, max={:.6}", input_stats.rms, input_stats.min, input_stats.max);
                eprintln!("   sum={:.6}, mean={:.6}", sum, mean);
            }

            // Line 3554-3556: Copy input to output
            let y_offset = i01 * ne00;
            output[y_offset..y_offset + ne00].copy_from_slice(x_slice);

            // Line 3561: const float scale = 1.0f/sqrtf(mean + eps);
            let scale = 1.0 / (mean + eps).sqrt();

            // Line 3566: ggml_vec_scale_f32(ne00, y, scale);
            // This is: y[i] = y[i] * scale for all i
            // NOTE: llama.cpp does NOT multiply by weight in ggml_rms_norm
            // Weight multiplication is done separately via ggml_mul
            for i00 in 0..ne00 {
                output[y_offset + i00] *= scale;
                // REMOVED: weight multiplication - should be done by caller
                // output[y_offset + i00] *= weight_data[i00];
            }

            // Debug token output
            if is_first {
                let token_output = &output[y_offset..y_offset + ne00];
                let output_stats = Self::compute_stats(token_output);
                eprintln!("   scale={:.6}", scale);
                eprintln!("   After norm: rms={:.6}, min={:.6}, max={:.6}\n",
                    output_stats.rms, output_stats.min, output_stats.max);
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

        // DEBUG: ALWAYS print RoPE call info
        eprintln!("🌀 [RoPE CALLED] seq_len={}, total_dim={}, head_dim={}, num_heads={}, start_position={}",
            seq_len, total_dim, head_dim, num_heads, start_position);
        eprintln!("🌀 [RoPE INPUT] First 10 values: {:?}", &x_data[0..10.min(x_data.len())]);

        // DEBUG: Check rope_cos/sin arrays
        if start_position == 0 {
            eprintln!("🌀 [RoPE FREQS] rope_cos.len={}, rope_sin.len={}", self.rope_cos.len(), self.rope_sin.len());
            eprintln!("🌀 [RoPE FREQS] First 10 cos values: {:?}", &self.rope_cos[0..10.min(self.rope_cos.len())]);
            eprintln!("🌀 [RoPE FREQS] First 10 sin values: {:?}", &self.rope_sin[0..10.min(self.rope_sin.len())]);
        }

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

                    // DEBUG: First and second token, first head, first 3 pairs
                    if token_idx <= 1 && head_idx == 0 && i < 3 {
                        eprintln!("🌀 [RoPE DETAIL] token={}, head={}, pair={}, pos={}, rope_idx={}",
                            token_idx, head_idx, i, position, rope_idx);
                        eprintln!("  cos={:.9}, sin={:.9}", cos, sin);
                        eprintln!("  input:  x0={:.9}, x1={:.9}", x0, x1);
                        eprintln!("  output: rot0={:.9}, rot1={:.9}", rotated_0, rotated_1);
                    }

                    output.push(rotated_0);
                    output.push(rotated_1);
                }
            }
        }

        eprintln!("🌀 [RoPE OUTPUT] First 10 values: {:?}", &output[0..10.min(output.len())]);

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

        // DEBUG: GQA call info
        eprintln!("💫 [GQA CALLED] seq_len={}, kv_seq_len={}, num_heads={}, num_kv_heads={}, head_dim={}",
            seq_len, kv_seq_len, num_heads, num_kv_heads, head_dim);

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
                // BUT: current_kv_pos must not exceed total_kv_len - 1
                let current_kv_pos = (cached_len + q_pos).min(total_kv_len - 1);
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

                // DEBUG: First query position, first head only
                if q_pos == 0 && h == 0 {
                    let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
                    eprintln!("💫 [ATTENTION] q_pos={}, head={}, kv_head={}, num_scores={}",
                        q_pos, h, kv_head, scores.len());
                    eprintln!("  Raw scores: min={:.6}, max={:.6}, first_5={:?}",
                        min_score, max_score, &scores[..scores.len().min(5)]);
                    eprintln!("  Exp scores: first_5={:?}", &exp_scores[..exp_scores.len().min(5)]);
                    eprintln!("  Sum of exp: {:.9}", sum_exp);
                }

                // 安全性チェック: sum_expが0または異常値の場合
                let attn_weights: Vec<f32> = if sum_exp <= 0.0 || !sum_exp.is_finite() {
                    eprintln!("⚠️  [ATTENTION WARNING] Invalid sum_exp: {}, max_score: {}, scores_len: {}",
                        sum_exp, max_score, scores.len());
                    eprintln!("   First 5 scores: {:?}", &scores[..scores.len().min(5)]);
                    eprintln!("   First 5 exp_scores: {:?}", &exp_scores[..exp_scores.len().min(5)]);
                    // Fallback: 均等な重み
                    let uniform_weight = 1.0 / scores.len() as f32;
                    vec![uniform_weight; scores.len()]
                } else {
                    exp_scores.iter().map(|&e| e / sum_exp).collect()
                };

                // DEBUG: First query position, first head only
                if q_pos == 0 && h == 0 {
                    let weights_sum: f32 = attn_weights.iter().sum();
                    eprintln!("  Attention weights: sum={:.9}, first_5={:?}",
                        weights_sum, &attn_weights[..attn_weights.len().min(5)]);
                }

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

        let output_tensor = F32Tensor::from_vec(output.clone(), &[seq_len, num_heads * head_dim])
            .map_err(|e| F32Error::device_error(format!("GQA output failed: {}", e)))?;

        // DEBUG: GQA output
        eprintln!("💫 [GQA OUTPUT] First 10 values: {:?}", &output[..10.min(output.len())]);

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

        // Debug: Print embedding weight info for first token
        static EMBED_DEBUG_PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !EMBED_DEBUG_PRINTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
            eprintln!("🔍 [EMBED DEBUG] token_embd.weight shape: {:?}", embed_shape);
            eprintln!("🔍 [EMBED DEBUG] hidden_size={}, vocab_size={}", hidden_size, vocab_size);
            eprintln!("🔍 [EMBED DEBUG] embed_data.len()={}", embed_data.len());
            eprintln!("🔍 [EMBED DEBUG] Formula: start = token_id * hidden_size");
        }

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

        // Debug: Show embedding for token 1 (BOS) and token 29896 ("1")
        if token_id == 1 || token_id == 29896 {
            eprintln!("🔍 [EMBED DEBUG] Token {} embedding (first 10):", token_id);
            for i in 0..10.min(embedding.len()) {
                eprintln!("  [{}] = {:.9}", i, embedding[i]);
            }
            eprintln!("  RMS = {:.9}", (embedding.iter().map(|x| x * x).sum::<f32>() / embedding.len() as f32).sqrt());
        }

        // Debug: Show first 10 values of embedding for important tokens
        // Token 1 (BOS), Token 13 (newline), Token 1724 ("What"), Token 3681 (Paris), Token 29896 ("1")
        if token_id == 1 || token_id == 13 || token_id == 1724 || token_id == 3681 || token_id == 29896 {
            let first_10: Vec<f32> = embedding.iter().take(10).copied().collect();
            let non_zero_count = embedding.iter().filter(|&&x| x.abs() > 1e-8).count();
            let emb_stats = Self::compute_stats(&embedding);
            eprintln!("🔍 [GET_EMB] token={} embedding[0..10]={:?}", token_id, first_10);
            eprintln!("           rms={:.6}, min={:.6}, max={:.6} (non_zero={}/{})",
                emb_stats.rms, emb_stats.min, emb_stats.max, non_zero_count, embedding.len());
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
        debug_layer: bool,
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

        // DEBUG: Check input to attention
        if layer_idx == 0 {
            let x_stats = Self::compute_stats(x.as_slice());
            eprintln!("🔧 [LAYER 0] attention input (after RMSNorm): rms={:.6}, max={:.6}", x_stats.rms, x_stats.max);
            eprintln!("🔍 [MATMUL DEBUG] x.shape={:?}, q_weight.shape={:?}, k_weight.shape={:?}, v_weight.shape={:?}",
                x.shape(), q_weight.shape(), k_weight.shape(), v_weight.shape());
        }

        // Linear projections: Q, K, V
        // Try WITHOUT transpose - maybe GGUF weights are already in the right layout
        let q = x.matmul(&q_weight)?;
        let k = x.matmul(&k_weight)?;
        let v = x.matmul(&v_weight)?;

        // DEBUG: Q/K/V projections before RoPE (Layer 0 only)
        if layer_idx == 0 {
            let q_stats = Self::compute_stats(q.as_slice());
            let k_stats = Self::compute_stats(k.as_slice());
            let v_stats = Self::compute_stats(v.as_slice());
            eprintln!("🎯 [LAYER 0] Q before reshape: rms={:.9}, max={:.9}, shape={:?}, first 10: {:?}",
                q_stats.rms, q_stats.max, q.shape(), &q.as_slice()[0..10]);
            eprintln!("🎯 [LAYER 0] K before reshape: rms={:.9}, max={:.9}, shape={:?}, first 10: {:?}",
                k_stats.rms, k_stats.max, k.shape(), &k.as_slice()[0..10]);
            eprintln!("🎯 [LAYER 0] V before reshape: rms={:.9}, max={:.9}, shape={:?}",
                v_stats.rms, v_stats.max, v.shape());
        }

        // PORTED FROM: llama.cpp/src/llama-model.cpp:6482-6484
        // llama.cpp reshapes Q/K/V to 3D BEFORE RoPE:
        //   Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        //   Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        //   Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        // This changes memory layout from [tokens, features] to [head_dim, n_heads, tokens]
        // For RusTorch with [tokens, features] layout, we need to keep 2D but be aware
        // that llama.cpp's subsequent operations expect [head_dim][heads][tokens] layout
        //
        // Actually, our current apply_rope already handles the 2D layout correctly
        // by iterating token-by-token, then head-by-head within each token.
        // So NO RESHAPE needed - the key is that our iteration order matches llama.cpp's expectation.

        // Apply RoPE to Q and K
        let q_rope = self.apply_rope(&q, position)?;
        let k_rope = self.apply_rope(&k, position)?;

        // DEBUG: Q/K after RoPE (Layer 0 only)
        if layer_idx == 0 {
            let q_rope_stats = Self::compute_stats(q_rope.as_slice());
            let k_rope_stats = Self::compute_stats(k_rope.as_slice());
            eprintln!("🎯 [LAYER 0] Q after RoPE: rms={:.9}, max={:.9}, first 10: {:?}",
                q_rope_stats.rms, q_rope_stats.max, &q_rope.as_slice()[0..10]);
            eprintln!("🎯 [LAYER 0] K after RoPE: rms={:.9}, max={:.9}, first 10: {:?}",
                k_rope_stats.rms, k_rope_stats.max, &k_rope.as_slice()[0..10]);
        }


        // Get cached K, V
        let cache = &self.kv_cache[layer_idx];
        // ✅ Re-enable KV cache after position fix
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
        if layer_idx == 0 {
            let o_stats = Self::compute_stats(o_weight.as_slice());
            let attn_out_stats = Self::compute_stats(attn_output.as_slice());
            eprintln!("🔧 [LAYER 0] attn_output stats (before o_proj): rms={:.6}, max={:.6}", attn_out_stats.rms, attn_out_stats.max);
            eprintln!("🔧 [LAYER 0] o_weight stats: rms={:.6}, max={:.6}", o_stats.rms, o_stats.max);
        }

        // PORTED FROM: llama.cpp - O projection - try without transpose
        let final_out = attn_output.matmul(&o_weight)?;

        if debug_layer {
            let final_stats = Self::compute_stats(final_out.as_slice());
            eprintln!("🔧 [LAYER {}] final attention output (after o_proj): rms={:.6}, max={:.6}, mean={:.9}",
                layer_idx, final_stats.rms, final_stats.max, final_stats.mean);
        }

        Ok(final_out)
    }

    /// Llama FFN layer with SwiGLU
    /// SwiGLU活性化関数を使用したLlama FFN層
    fn ffn_layer(&self, x: &F32Tensor, layer_idx: usize, debug_layer: bool) -> F32Result<F32Tensor> {
        // Get weights
        let gate_weight = self.weights.get(&format!("blk.{}.ffn_gate.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Gate weight not found for layer {}", layer_idx)))?;
        let up_weight = self.weights.get(&format!("blk.{}.ffn_up.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Up weight not found for layer {}", layer_idx)))?;
        let down_weight = self.weights.get(&format!("blk.{}.ffn_down.weight", layer_idx))
            .ok_or_else(|| F32Error::device_error(format!("Down weight not found for layer {}", layer_idx)))?;

        // DEBUG: Check weight statistics
        if debug_layer {
            let down_data = down_weight.as_slice();
            let down_stats = Self::compute_stats(down_data);
            eprintln!("🔧 [LAYER {}] FFN down_weight stats: rms={:.6}, min={:.6}, max={:.6}, mean={:.9}",
                layer_idx, down_stats.rms, down_stats.min, down_stats.max, down_stats.mean);
        }

        // FFN with SwiGLU: down(SwiGLU(gate(x), up(x)))
        // Try without transpose
        let gate = x.matmul(&gate_weight)?;
        let up = x.matmul(&up_weight)?;

        if debug_layer {
            let gate_stats = Self::compute_stats(gate.as_slice());
            let up_stats = Self::compute_stats(up.as_slice());
            eprintln!("🔧 [LAYER {}] gate projection: rms={:.6}, max={:.6}, mean={:.9}", layer_idx, gate_stats.rms, gate_stats.max, gate_stats.mean);
            eprintln!("🔧 [LAYER {}] up projection: rms={:.6}, max={:.6}, mean={:.9}", layer_idx, up_stats.rms, up_stats.max, up_stats.mean);

            // Print first 10 values for manual verification
            eprint!("🔧 [LAYER {}] gate first 10: [", layer_idx);
            for i in 0..10.min(gate.as_slice().len()) {
                if i > 0 { eprint!(", "); }
                eprint!("{:.9}", gate.as_slice()[i]);
            }
            eprintln!("]");

            eprint!("🔧 [LAYER {}] up first 10: [", layer_idx);
            for i in 0..10.min(up.as_slice().len()) {
                if i > 0 { eprint!(", "); }
                eprint!("{:.9}", up.as_slice()[i]);
            }
            eprintln!("]");

            // Also check weight statistics
            let gate_weight_stats = Self::compute_stats(gate_weight.as_slice());
            let up_weight_stats = Self::compute_stats(up_weight.as_slice());
            eprintln!("🔧 [LAYER {}] gate_weight stats: rms={:.6}, mean={:.9}", layer_idx, gate_weight_stats.rms, gate_weight_stats.mean);
            eprintln!("🔧 [LAYER {}] up_weight stats: rms={:.6}, mean={:.9}", layer_idx, up_weight_stats.rms, up_weight_stats.mean);
        }

        let swiglu_out = self.swiglu(&gate, &up)?;

        if debug_layer {
            let swiglu_stats = Self::compute_stats(swiglu_out.as_slice());
            eprintln!("🔧 [LAYER {}] SwiGLU output: rms={:.6}, max={:.6}, mean={:.9}", layer_idx, swiglu_stats.rms, swiglu_stats.max, swiglu_stats.mean);

            // Print first 10 values
            eprint!("🔧 [LAYER {}] SwiGLU first 10: [", layer_idx);
            for i in 0..10.min(swiglu_out.as_slice().len()) {
                if i > 0 { eprint!(", "); }
                eprint!("{:.9}", swiglu_out.as_slice()[i]);
            }
            eprintln!("]");
        }

        // PORTED FROM: llama.cpp - FFN down projection - try without transpose
        let final_out = swiglu_out.matmul(&down_weight)?;


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

        // Debug logging for selected layers (0, 5, 10, 15, 21)
        let debug_layer = layer_idx == 0 || layer_idx == 5 || layer_idx == 10 || layer_idx == 15 || layer_idx == 21;

        if debug_layer {
            let x_slice = x.as_slice();
            let stats = Self::compute_stats(x_slice);
            eprintln!("🔍 [LAYER {}] Input: rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);
        }

        // Attention block with residual connection
        if layer_idx == 0 {
            let attn_norm_stats = Self::compute_stats(attn_norm_weight.as_slice());
            let x_stats = Self::compute_stats(x.as_slice());
            eprintln!("🔧 [LAYER 0] Before attn RMSNorm: input rms={:.6}, max={:.6}", x_stats.rms, x_stats.max);
            eprintln!("🔧 [LAYER 0] attn_norm.weight stats: rms={:.6}, max={:.6}", attn_norm_stats.rms, attn_norm_stats.max);
        }

        // RMS Norm (normalize only, matching llama.cpp)
        let normed = self.rms_norm(x, &attn_norm_weight)?;

        if layer_idx == 0 {
            let normed_stats = Self::compute_stats(normed.as_slice());
            eprintln!("🔧 [LAYER 0] After attn RMSNorm (before weight): rms={:.6}, max={:.6}", normed_stats.rms, normed_stats.max);
            eprintln!("🔧 [LAYER 0] normed shape: {:?}, weight shape: {:?}", normed.shape(), attn_norm_weight.shape());
            eprintln!("🔧 [LAYER 0] normed.len()={}, weight.len()={}", normed.as_slice().len(), attn_norm_weight.as_slice().len());

            // Print first 10 values
            let normed_slice = normed.as_slice();
            let weight_slice = attn_norm_weight.as_slice();
            print!("🔧 [LAYER 0] First 10 normed values: ");
            for i in 0..10.min(normed_slice.len()) {
                print!("{:.6} ", normed_slice[i]);
            }
            println!();
            print!("🔧 [LAYER 0] First 10 weight values: ");
            for i in 0..10.min(weight_slice.len()) {
                print!("{:.6} ", weight_slice[i]);
            }
            println!();
        }

        // Multiply by weight (separate step, matching llama.cpp's ggml_mul)
        let normed = {
            let normed_data = normed.as_slice();
            let weight_data = attn_norm_weight.as_slice();
            let shape = normed.shape();
            let mut result = Vec::with_capacity(normed_data.len());
            for i in 0..normed_data.len() {
                result.push(normed_data[i] * weight_data[i % weight_data.len()]);
            }
            F32Tensor::from_vec(result, shape)?
        };

        if layer_idx == 0 {
            let normed_stats = Self::compute_stats(normed.as_slice());
            eprintln!("🔧 [LAYER 0] After weight multiplication: rms={:.6}, max={:.6}", normed_stats.rms, normed_stats.max);

            // Print first 10 results
            let result_slice = normed.as_slice();
            print!("🔧 [LAYER 0] First 10 result values: ");
            for i in 0..10.min(result_slice.len()) {
                print!("{:.6} ", result_slice[i]);
            }
            println!();
        }

        if debug_layer {
            let normed_slice = normed.as_slice();
            let stats = Self::compute_stats(normed_slice);
            eprintln!("🔍 [LAYER {}] After Attn RMSNorm: rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);
        }

        let attn_out = self.attention_layer(&normed, layer_idx, position, debug_layer)?;

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
        // RMS Norm (normalize only, matching llama.cpp)
        let normed = self.rms_norm(&x, &ffn_norm_weight)?;

        // Multiply by weight (separate step, matching llama.cpp's ggml_mul)
        let normed = {
            let normed_data = normed.as_slice();
            let weight_data = ffn_norm_weight.as_slice();
            let shape = normed.shape();
            let mut result = Vec::with_capacity(normed_data.len());
            for i in 0..normed_data.len() {
                result.push(normed_data[i] * weight_data[i % weight_data.len()]);
            }
            F32Tensor::from_vec(result, shape)?
        };

        if debug_layer {
            let normed_slice = normed.as_slice();
            let stats = Self::compute_stats(normed_slice);
            eprintln!("🔍 [LAYER {}] After FFN RMSNorm: rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);
        }

        let ffn_out = self.ffn_layer(&normed, layer_idx, debug_layer)?;

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
            eprintln!("🔍 [LAYER {}] After FFN Residual (Layer Output): rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
                layer_idx, stats.rms, stats.min, stats.max, stats.mean);

            // Print first 10 values for direct comparison with llama.cpp
            eprint!("🔍 [LAYER {}] First 10 values: [", layer_idx);
            for i in 0..10.min(result_slice.len()) {
                if i > 0 { eprint!(", "); }
                eprint!("{:.9}", result_slice[i]);
            }
            eprintln!("]\n");
        }

        Ok(result)
    }

    /// Forward pass through Llama model
    /// Llamaモデルのフォワードパス
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    /// * `start_position` - Starting position in the sequence (for RoPE)
    pub fn forward(&mut self, input_ids: &[usize], start_position: usize) -> F32Result<F32Tensor> {
        eprintln!("🔍 [FORWARD START] input_ids.len={}, start_position={}", input_ids.len(), start_position);
        eprintln!("🔍 [INPUT TOKENS] {:?}", input_ids);

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
                eprintln!("\n🔍 [TOKEN EMBEDDING DUMP] input_ids: {:?}", input_ids);

                // Dump embeddings for first 3 tokens
                for tok_idx in 0..3.min(seq_len) {
                    let token_id = input_ids[tok_idx];
                    let emb_start = tok_idx * hidden_size;
                    let emb_end = emb_start + hidden_size;
                    let token_emb = &x.as_slice()[emb_start..emb_end];

                    eprintln!("\n  Token {} (ID={}): First 20 values:", tok_idx, token_id);
                    eprint!("    [");
                    for i in 0..20.min(hidden_size) {
                        eprint!("{:.9}", token_emb[i]);
                        if i < 19 { eprint!(", "); }
                    }
                    eprintln!("]");

                    let stats = Self::compute_stats(token_emb);
                    eprintln!("    Stats: mean={:.9}, rms={:.9}, min={:.9}, max={:.9}",
                        stats.mean, stats.rms, stats.min, stats.max);
                }
            }
        }

        // ✅ FIX: Use explicit start_position parameter instead of KV cache length
        let current_position = start_position;

        // DEBUG: Log position for first 3 forward calls
        {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static POSITION_LOG_COUNTER: AtomicUsize = AtomicUsize::new(0);
            let log_count = POSITION_LOG_COUNTER.load(Ordering::SeqCst);
            if log_count < 3 {
                eprintln!("🔍 [POSITION] Forward call {}: seq_len={}, current_position={}, will apply RoPE positions: {}..{}",
                    log_count, seq_len, current_position, current_position, current_position + seq_len - 1);
                POSITION_LOG_COUNTER.fetch_add(1, Ordering::SeqCst);
            }
        }

        // Strategic debug: Monitor problematic positions only

        // DEBUG: Save embedding output (before any layers)
        {
            use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
            static SAVE_EMBEDDINGS: AtomicBool = AtomicBool::new(true);
            if SAVE_EMBEDDINGS.swap(false, AtomicOrdering::SeqCst) {
                // Save last token's embedding to file
                let last_token_start = (seq_len - 1) * hidden_size;
                let last_token_emb = &x.as_slice()[last_token_start..last_token_start + hidden_size];
                if let Ok(mut file) = std::fs::File::create("/tmp/rustorch_embedding.txt") {
                    use std::io::Write;
                    for val in last_token_emb {
                        let _ = writeln!(file, "{}", val);
                    }
                    eprintln!("💾 Saved embedding to /tmp/rustorch_embedding.txt");
                }
            }
        }

        // Pass through all transformer layers
        for layer_idx in 0..self.config.num_layers {
            // DEBUG: Save layer inputs (before transformer_layer)
            {
                use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
                static FORWARD_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
                let call_count = FORWARD_CALL_COUNT.load(AtomicOrdering::SeqCst);

                // Only save on first forward call and for layer 0
                if call_count == 0 && layer_idx == 0 {
                    let last_token_start = (seq_len - 1) * hidden_size;
                    let last_token_input = &x.as_slice()[last_token_start..last_token_start + hidden_size];
                    let filename = format!("/tmp/rustorch_layer_{}_input.txt", layer_idx);
                    if let Ok(mut file) = std::fs::File::create(&filename) {
                        use std::io::Write;
                        for val in last_token_input {
                            let _ = writeln!(file, "{}", val);
                        }
                        eprintln!("💾 Saved Layer {} INPUT to {}", layer_idx, filename);
                    }
                }

                if layer_idx == 0 {
                    FORWARD_CALL_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
                }
            }

            x = self.transformer_layer(&x, layer_idx, current_position)?;

            // DEBUG: Dump layer outputs for key layers (0, 10, 21)
            if layer_idx == 0 || layer_idx == 10 || layer_idx == self.config.num_layers - 1 {
                let stats = Self::compute_stats(x.as_slice());
                eprintln!("📊 [LAYER {}] Output: rms={:.6}, min={:.6}, max={:.6}, first_10={:?}",
                    layer_idx, stats.rms, stats.min, stats.max,
                    &x.as_slice()[..10.min(x.as_slice().len())]);
            }

            // DEBUG: Save layer outputs for first few layers and last token only
            {
                use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
                static FORWARD_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
                let call_count = FORWARD_CALL_COUNT.load(AtomicOrdering::SeqCst);

                // Only save on first forward call (call_count == 0)
                if call_count == 0 && (layer_idx <= 2 || layer_idx == self.config.num_layers - 1) {
                    let last_token_start = (seq_len - 1) * hidden_size;
                    let last_token_output = &x.as_slice()[last_token_start..last_token_start + hidden_size];
                    let filename = format!("/tmp/rustorch_layer_{}.txt", layer_idx);
                    if let Ok(mut file) = std::fs::File::create(&filename) {
                        use std::io::Write;
                        for val in last_token_output {
                            let _ = writeln!(file, "{}", val);
                        }
                        eprintln!("💾 Saved Layer {} output to {}", layer_idx, filename);
                    }
                }

                // Increment counter after processing layer 21
                if layer_idx == self.config.num_layers - 1 {
                    FORWARD_CALL_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
                }
            }
        }

        eprintln!("🔍 [BEFORE FINAL NORM] seq_len={}", seq_len);

        // Final RMSNorm
        let output_norm_weight = self.weights.get("output_norm.weight")
            .or_else(|| self.weights.get("model.norm.weight"))
            .or_else(|| self.weights.get("norm.weight"))
            .ok_or_else(|| F32Error::device_error("Output norm weight not found"))?;

        eprintln!("🔍 [FOUND OUTPUT NORM WEIGHT]");
        // RMS Norm (normalize only, matching llama.cpp)
        let normed = self.rms_norm(&x, output_norm_weight)?;
        eprintln!("🔍 [AFTER RMS_NORM (before weight)]");

        // Multiply by weight (separate step, matching llama.cpp's ggml_mul)
        let normed = {
            let normed_data = normed.as_slice();
            let weight_data = output_norm_weight.as_slice();
            let shape = normed.shape();
            let mut result = Vec::with_capacity(normed_data.len());
            for i in 0..normed_data.len() {
                result.push(normed_data[i] * weight_data[i % weight_data.len()]);
            }
            F32Tensor::from_vec(result, shape)?
        };
        eprintln!("🔍 [AFTER WEIGHT MULTIPLICATION]");

        // DEBUG: Always check final normalized hidden state for last token
        let normed_slice = normed.as_slice();
        let last_token_start = (seq_len - 1) * hidden_size;
        let last_token_normed = &normed_slice[last_token_start..];
        let stats = Self::compute_stats(last_token_normed);
        eprintln!("🔍 [FINAL NORM] After output_norm (last token): rms={:.6}, min={:.6}, max={:.6}, mean={:.6}",
            stats.rms, stats.min, stats.max, stats.mean);

        // Print first 10 values for llama.cpp comparison
        eprint!("🔍 [FINAL NORM] First 10 values: [");
        for i in 0..10.min(last_token_normed.len()) {
            if i > 0 { eprint!(", "); }
            eprint!("{:.9}", last_token_normed[i]);
        }
        eprintln!("]");

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

        // Debug: Dump first 20 raw values from output.weight to verify dequantization
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static FIRST_LOGIT_CALC: AtomicBool = AtomicBool::new(true);
            if FIRST_LOGIT_CALC.swap(false, Ordering::SeqCst) {
                eprintln!("\n🔍 [Q8_0 WEIGHT VALUES] First 20 raw values from output.weight:");
                for i in 0..20.min(lm_head_data.len()) {
                    eprintln!("  output.weight[{}] = {:.9}", i, lm_head_data[i]);
                }
                eprintln!("Total output.weight elements: {}", lm_head_data.len());

                eprintln!("\n🔍 [HIDDEN STATE] First 20 values of hidden state:");
                for i in 0..20.min(hidden_slice.len()) {
                    eprintln!("  hidden[{}] = {:.9}", i, hidden_slice[i]);
                }
                eprintln!("Total hidden_size: {}", hidden_slice.len());

                // Save hidden state to file
                if let Ok(mut f) = std::fs::File::create("/tmp/rustorch_hidden.txt") {
                    use std::io::Write;
                    for i in 0..20.min(hidden_slice.len()) {
                        let _ = writeln!(f, "{} {:.9}", i, hidden_slice[i]);
                    }
                }

                // Also save weights to file for comparison
                if let Ok(mut f) = std::fs::File::create("/tmp/rustorch_weights.txt") {
                    use std::io::Write;
                    for i in 0..20.min(lm_head_data.len()) {
                        let _ = writeln!(f, "{} {:.9}", i, lm_head_data[i]);
                    }
                }
            }
        }

        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            for h in 0..hidden_size {
                // CRITICAL FIX: GGML dims are REVERSED from intuition!
                // GGML original_dims=[2048, 32000] means ne[0]=2048 (fastest-changing), ne[1]=32000
                // In memory, this is stored as [vocab_size=32000, hidden_size=2048] in row-major
                // So to access output.weight[v][h], we use: v * hidden_size + h
                //
                // This matches llama.cpp's ggml_mul_mat which transposes the first argument:
                // ggml_mul_mat(output, hidden) = output^T @ hidden
                // where output is stored as [vocab_size, hidden_size]
                let idx = v * hidden_size + h;
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
            // Load tensor directly as f32 using generic loader
            // ジェネリック型ローダーを使ってf32として直接読み込む（f64への変換を回避）
            match loader.load_tensor_generic::<f32>(name) {
                Ok(data_f32) => {
                    // Get tensor info for shape
                    let tensor_info = loader.get_tensor_info(name);
                    let shape = if let Some(info) = tensor_info {
                        let original_dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
                        let ggml_type = crate::formats::gguf::GGMLType::from_u32(info.ggml_type).ok();

                        // CRITICAL: Only F32/F16 tensors need shape reversal
                        // Quantized tensors (Q8_0, Q4_K, etc.) are stored with correct dimensions
                        let s: Vec<usize> = match ggml_type {
                            Some(crate::formats::gguf::GGMLType::F32) | Some(crate::formats::gguf::GGMLType::F16) => {
                                let mut reversed = original_dims.clone();
                                reversed.reverse();
                                reversed
                            }
                            _ => original_dims.clone(),
                        };
                        s
                    } else {
                        continue;
                    };

                    // Weight transpose investigation:
                    // llama.cpp: ggml_mul_mat(W, X) = W^T @ X
                    // RusTorch: X @ W
                    // After investigation: NO TRANSPOSE NEEDED
                    // GGUF weights are stored correctly for RusTorch's tensor layout

                    let needs_transpose = false;

                    if name.contains("blk.0.attn") || needs_transpose {
                    }

                    let final_tensor = if needs_transpose && shape.len() == 2 {
                        // Transpose output.weight and token_embd.weight for matmul compatibility
                        match F32Tensor::from_vec(data_f32, &shape) {
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
                        match F32Tensor::from_vec(data_f32, &shape) {
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

        // DEBUG: Test BOTH memory layouts to determine which is correct
        if let Some(token_embd) = model.weights.get("token_embd.weight") {
            eprintln!("\n🔬 [LAYOUT TEST] Testing token_embd.weight memory layout...");

            let token_id = 1usize;
            let hidden_size = 2048;
            let vocab_size = 32000;
            let token_embd_data = token_embd.as_slice();

            // Method 1: Row-major (used by get_embedding)
            let start = token_id * hidden_size;
            let end = start + hidden_size;
            let embedding_row_major: Vec<f32> = token_embd_data[start..end].to_vec();

            // Method 2: Column-major (used by WEIGHT TEST)
            let mut embedding_col_major = Vec::with_capacity(hidden_size);
            for dim in 0..hidden_size {
                embedding_col_major.push(token_embd_data[dim * vocab_size + token_id]);
            }

            eprintln!("   Token {} Row-major [0..10]: {:?}", token_id, &embedding_row_major[..10]);
            eprintln!("   Token {} Col-major [0..10]: {:?}", token_id, &embedding_col_major[..10]);

            // Calculate statistics for both
            let row_stats = Self::compute_stats(&embedding_row_major);
            let col_stats = Self::compute_stats(&embedding_col_major);

            eprintln!("   Row-major: rms={:.6}, min={:.6}, max={:.6}", row_stats.rms, row_stats.min, row_stats.max);
            eprintln!("   Col-major: rms={:.6}, min={:.6}, max={:.6}", col_stats.rms, col_stats.min, col_stats.max);

            // Check which one matches get_embedding() output
            eprintln!("\n   ✅ get_embedding() uses Row-major layout");
            eprintln!("   ⚠️  If Col-major values look more reasonable, we have a BUG!");
        }

        // DEBUG: Test output.weight vs token_embd relationship
        if let (Some(token_embd), Some(output_weight)) = (model.weights.get("token_embd.weight"), model.weights.get("output.weight")) {
            eprintln!("\n🔬 [WEIGHT TEST] Testing output.weight correctness...");

            let token_id = 1usize;
            let hidden_size = 2048;
            let vocab_size = 32000;

            let token_embd_data = token_embd.as_slice();
            let output_data = output_weight.as_slice();

            // Use Row-major layout (CORRECTED)
            let start = token_id * hidden_size;
            let end = start + hidden_size;
            let embedding: Vec<f32> = token_embd_data[start..end].to_vec();

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

        for step in 0..max_tokens {
            // Calculate position for this step
            let start_position = current_tokens.len().saturating_sub(1);

            // Forward pass
            let logits = self.forward(&current_tokens, start_position)?;

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
