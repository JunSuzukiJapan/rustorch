//! Llama Model Architecture Implementation
//! Llamaモデルアーキテクチャ実装
//!
//! This module provides a pure Metal implementation for Llama models,
//! independent of the experimental hybrid_f32 system.

use crate::backends::DeviceType;
use crate::error::{RusTorchError, RusTorchResult};
use crate::formats::gguf::GGUFLoader;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// Configuration for Llama models
/// Llamaモデルの設定
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA - Grouped Query Attention)
    pub num_kv_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// FFN intermediate dimension
    pub intermediate_size: usize,
    /// RoPE theta value
    pub rope_theta: f32,
    /// RMS norm epsilon
    pub norm_eps: f64,
}

impl LlamaConfig {
    /// Create Llama config from GGUF ModelParams
    pub fn from_model_params(params: &crate::formats::gguf::ModelParams) -> Self {
        // Use intermediate_size from GGUF if available, otherwise calculate
        // Mistral: 14336, TinyLlama: 5632 (2048 * 2.75)
        let intermediate_size = params.intermediate_size
            .map(|s| s as usize)
            .unwrap_or_else(|| (params.hidden_size as f32 * 2.75) as usize);

        // Use RoPE frequency base from GGUF if available
        // Mistral: 1000000.0, Standard LLaMA: 10000.0
        let rope_theta = params.rope_freq_base.unwrap_or(10000.0);

        eprintln!("🔧 [LLAMA CONFIG] intermediate_size={}, rope_theta={}",
                  intermediate_size, rope_theta);

        Self {
            vocab_size: params.vocab_size as usize,
            hidden_size: params.hidden_size as usize,
            num_heads: params.num_heads as usize,
            num_kv_heads: params.num_kv_heads as usize,
            num_layers: params.num_layers as usize,
            max_seq_len: params.context_length as usize,
            intermediate_size,
            rope_theta,
            norm_eps: 1e-5,       // Default RMS norm epsilon
        }
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Get KV head dimension
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }
}

/// Llama Model with Metal GPU support
/// Metal GPU対応のLlamaモデル
#[derive(Debug)]
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub weights: HashMap<String, Tensor<f64>>,
    pub device_type: DeviceType,
    rope_cos: Vec<f32>,  // Precomputed RoPE cosine values
    rope_sin: Vec<f32>,  // Precomputed RoPE sine values
}

impl LlamaModel {
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
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                cos_values.push(cos_val);
                sin_values.push(sin_val);
            }
        }

        (cos_values, sin_values)
    }

    /// Create a new Llama model with specified config and device
    pub fn new(config: LlamaConfig, device_type: DeviceType) -> RusTorchResult<Self> {
        let (rope_cos, rope_sin) = Self::precompute_rope_frequencies(&config);

        Ok(Self {
            config,
            weights: HashMap::new(),
            device_type,
            rope_cos,
            rope_sin,
        })
    }

    /// Load Llama model from GGUF file
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> RusTorchResult<Self> {
        Self::from_gguf_with_backend(path, DeviceType::Cpu)
    }

    /// Load Llama model from GGUF with specified backend
    pub fn from_gguf_with_backend<P: AsRef<Path>>(
        path: P,
        device_type: DeviceType,
    ) -> RusTorchResult<Self> {
        let loader = GGUFLoader::from_file(path)?;

        // Extract model parameters
        let params = loader.get_model_params()?;
        let config = LlamaConfig::from_model_params(&params);

        eprintln!(
            "📊 Loading Llama model on {:?} backend (vocab={}, hidden={}, layers={}, heads={}, kv_heads={})",
            device_type, config.vocab_size, config.hidden_size, config.num_layers,
            config.num_heads, config.num_kv_heads
        );

        // Create model
        let mut model = Self::new(config, device_type)?;

        // Load weights from GGUF
        let tensor_names = loader.tensor_names();
        eprintln!("🔧 Loading {} tensors from GGUF file", tensor_names.len());

        for name in tensor_names.iter() {
            match loader.load_tensor(name) {
                Ok(tensor) => {
                    // BUGFIX 2025-10-16: Based on rustorch-cli gguf_loader_trait.rs
                    // Only token_embd, attn_q, and attn_output are transposed
                    // K/V and FFN weights are NOT transposed (used directly from GGUF)
                    let final_tensor = if *name == "token_embd.weight"
                        || name.contains("attn_q.weight")
                        || name.contains("attn_output.weight") {
                        match tensor.transpose() {
                            Ok(t) => t,
                            Err(e) => {
                                eprintln!("⚠️  Failed to transpose '{}': {}", name, e);
                                tensor
                            }
                        }
                    } else {
                        tensor
                    };
                    model.weights.insert(name.to_string(), final_tensor);
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to load tensor '{}': {}", name, e);
                }
            }
        }

        eprintln!("✅ Loaded {} tensors successfully", model.weights.len());

        Ok(model)
    }

    /// Get model configuration
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Forward pass through the model
    pub fn forward(&self, input_ids: &[usize]) -> RusTorchResult<Tensor<f64>> {
        self.forward_with_position(input_ids, 0)
    }

    /// Forward pass with position tracking
    pub fn forward_with_position(&self, input_ids: &[usize], start_position: usize) -> RusTorchResult<Tensor<f64>> {
        match self.device_type {
            #[cfg(feature = "metal")]
            DeviceType::Metal => self.forward_metal(input_ids, start_position),
            _ => self.forward_cpu(input_ids, start_position),
        }
    }

    /// CPU implementation of forward pass
    fn forward_cpu(&self, input_ids: &[usize], _start_position: usize) -> RusTorchResult<Tensor<f64>> {
        eprintln!("🖥️  Llama forward_cpu called (not yet implemented)");
        Err(RusTorchError::tensor_op("CPU implementation for Llama not yet available"))
    }

    /// Metal GPU implementation of forward pass
    #[cfg(feature = "metal")]
    fn forward_metal(&self, input_ids: &[usize], start_position: usize) -> RusTorchResult<Tensor<f64>> {
        use crate::gpu::metal_kernels::MetalKernelExecutor;

        let debug = std::env::var("RUSTORCH_DEBUG").is_ok();

        if debug {
            eprintln!("🦙 Llama forward_metal called (input_len={}, start_pos={})",
                input_ids.len(), start_position);
        }

        // Get Metal executor
        let executor_mutex = MetalKernelExecutor::get()?;
        let executor_guard = executor_mutex.lock().unwrap();
        let executor = executor_guard.as_ref()
            .ok_or_else(|| RusTorchError::tensor_op("Metal executor not initialized"))?;

        let batch_size = 1;
        let seq_len = input_ids.len();
        let d_model = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();

        // 1. Token Embedding Lookup
        let token_emb_key = "token_embd.weight";
        let token_emb_tensor = self.weights.get(token_emb_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Token embedding not found: {}", token_emb_key)))?;

        let mut embedding_data = Vec::with_capacity(seq_len * d_model);
        let emb_shape = token_emb_tensor.data.shape();
        // BUGFIX 2025-10-16: After transpose, token_embd is [vocab_size, d_model] = [32000, 2048]
        let vocab_size = emb_shape[0];   // vocab_size = 32000
        let hidden_size = emb_shape[1];  // d_model = 2048

        if debug {
            eprintln!("🔧 [EMB SHAPE DEBUG] emb_shape={:?}, hidden_size={}, vocab_size={}",
                     emb_shape, hidden_size, vocab_size);
        }

        let emb_data = token_emb_tensor.data.as_slice()
            .ok_or_else(|| RusTorchError::tensor_op("Failed to get embedding data as slice"))?;

        for (token_idx, &token_id) in input_ids.iter().enumerate() {
            if token_id >= vocab_size {
                return Err(RusTorchError::tensor_op(
                    format!("Token ID {} out of range (vocab_size={})", token_id, vocab_size)
                ));
            }

            // After transpose: [vocab_size, d_model] in row-major layout
            // Data is stored as: Token0[2048 dims], Token1[2048 dims], ..., Token31999[2048 dims]
            // For token_id, embedding starts at: token_id * hidden_size
            let emb_start = token_id * hidden_size;

            // Always output Token ID=1 embedding for debugging
            if token_id == 1 {
                eprintln!("🔧 [TOKEN=1 EMBEDDING] token_id={}, emb_start={}, first 10 values: {:?}",
                         token_id, emb_start, &emb_data[emb_start..emb_start + 10]);
            }

            for dim_idx in 0..hidden_size {
                embedding_data.push(emb_data[emb_start + dim_idx]);
            }

            // Always show Token ID=1 embedding
            if token_id == 1 {
                let current_emb_start = embedding_data.len() - hidden_size;
                let emb_slice = &embedding_data[current_emb_start..current_emb_start + 10.min(hidden_size)];
                eprintln!("🔍 [TOKEN ID=1 EMBEDDING] First 10 values: {:?}", emb_slice);
            }
        }

        // Convert to f32 for Metal operations
        let mut x_f32: Vec<f32> = embedding_data.iter().map(|&v| v as f32).collect();

        if debug {
            let mean: f32 = x_f32.iter().sum::<f32>() / x_f32.len() as f32;
            let rms = (x_f32.iter().map(|&v| v * v).sum::<f32>() / x_f32.len() as f32).sqrt();
            eprintln!("🎯 [INPUT] After embedding: mean={:.6}, rms={:.6}", mean, rms);
        }

        // 2. Process through all Transformer layers
        let num_layers = self.config.num_layers;

        if debug {
            eprintln!("🔧 Processing {} transformer layers", num_layers);
        }

        for layer_idx in 0..num_layers {
            if debug {
                eprintln!("   📍 Layer {}/{}", layer_idx + 1, num_layers);
            }

            // Pre-Attention RMS Norm
            let ln1_key = format!("blk.{}.attn_norm.weight", layer_idx);
            let ln1_weight = self.weights.get(&ln1_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("Attn norm weight not found: {}", ln1_key)))?;

            let ln1_gamma_f32: Vec<f32> = ln1_weight.data.iter().map(|&v| v as f32).collect();

            if debug && layer_idx == 0 {
                eprintln!("     🔍 [RMS NORM] ln1_gamma len={}, x_f32 len={}, seq_len={}, d_model={}",
                    ln1_gamma_f32.len(), x_f32.len(), seq_len, d_model);
            }

            let mut x_ln1 = vec![0.0f32; x_f32.len()];
            Self::rms_norm_f32(&x_f32, &ln1_gamma_f32, &mut x_ln1, seq_len, d_model, self.config.norm_eps as f32);

            if debug && layer_idx == 0 {
                eprintln!("     ✓ Pre-Attention RMS Norm complete");
                let x_ln1_rms = (x_ln1.iter().map(|&v| v * v).sum::<f32>() / x_ln1.len() as f32).sqrt();
                eprintln!("🔍 [LAYER 0 RMS NORM] After RMS Norm:");
                eprintln!("   x_ln1_rms={:.6}, x_ln1[0..10]={:?}", x_ln1_rms, &x_ln1[0..10]);
                eprintln!("   ln1_gamma[0..10]={:?}", &ln1_gamma_f32[0..10]);
            } else if debug {
                eprintln!("     ✓ Pre-Attention RMS Norm complete");
            }

            // Attention with GQA (Grouped Query Attention)
            let q_key = format!("blk.{}.attn_q.weight", layer_idx);
            let k_key = format!("blk.{}.attn_k.weight", layer_idx);
            let v_key = format!("blk.{}.attn_v.weight", layer_idx);
            let o_key = format!("blk.{}.attn_output.weight", layer_idx);

            let q_weight = self.weights.get(&q_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("Q weight not found: {}", q_key)))?;
            let k_weight = self.weights.get(&k_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("K weight not found: {}", k_key)))?;
            let v_weight = self.weights.get(&v_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("V weight not found: {}", v_key)))?;
            let o_weight = self.weights.get(&o_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("O weight not found: {}", o_key)))?;

            if debug && layer_idx == 0 {
                eprintln!("     🔍 [WEIGHT CONVERSION] Converting weights to f32...");
            }

            let q_weight_f32: Vec<f32> = q_weight.data.iter().map(|&v| v as f32).collect();

            if debug && layer_idx == 0 {
                eprintln!("        Q weight converted: len={}", q_weight_f32.len());
            }

            let k_weight_f32: Vec<f32> = k_weight.data.iter().map(|&v| v as f32).collect();

            if debug && layer_idx == 0 {
                eprintln!("        K weight converted: len={}", k_weight_f32.len());
            }

            let v_weight_f32: Vec<f32> = v_weight.data.iter().map(|&v| v as f32).collect();

            if debug && layer_idx == 0 {
                eprintln!("        V weight converted: len={}, first 5: {:?}", v_weight_f32.len(), &v_weight_f32[0..5]);
            }

            let o_weight_f32: Vec<f32> = o_weight.data.iter().map(|&v| v as f32).collect();

            if debug && layer_idx == 0 {
                eprintln!("        O weight converted: len={}", o_weight_f32.len());
            }

            // Get actual dimensions from weight shapes
            let q_shape = q_weight.data.shape();
            let k_shape = k_weight.data.shape();
            let v_shape = v_weight.data.shape();

            // BUGFIX 2025-10-16: Based on rustorch-cli - only Q is transposed, K/V are NOT
            // Q after transpose: [2048, 2048] (symmetric)
            // K/V in GGML order: [2048, 256] = [d_model, kv_dim]
            // For GGML [ne[0], ne[1]], ne[0]=d_model (innermost), ne[1]=kv_dim (output)
            let q_out_dim = q_shape[0];  // 2048
            let k_out_dim = k_shape[1];  // 256 (shape[1] because K is NOT transposed!)
            let v_out_dim = v_shape[1];  // 256 (shape[1] because V is NOT transposed!)

            if debug && layer_idx == 0 {
                eprintln!("     🔍 [WEIGHT SHAPES] Q: {:?} -> out_dim={}, K: {:?} -> out_dim={}, V: {:?} -> out_dim={}",
                    q_shape, q_out_dim, k_shape, k_out_dim, v_shape, v_out_dim);
            }

            // Q, K, V projections
            // Q, K, V射影
            let mut q_out = vec![0.0f32; seq_len * q_out_dim];
            let mut k_out = vec![0.0f32; seq_len * k_out_dim];
            let mut v_out = vec![0.0f32; seq_len * v_out_dim];

            // Each matmul is automatically wrapped in its own autoreleasepool by metal_kernels.rs
            // metal_kernels.rsが各matmulを自動的にautoreleasepoolで囲む

            // After transpose: Q weight is [2048, 2048] (symmetric)
            // matmul_metal_f32(A, B, C, m, n, k): C[m,n] = A[m,k] @ B[k,n]
            // x[seq_len,d_model] @ W[d_model,q_out_dim] = out[seq_len,q_out_dim]
            executor.matmul_metal_f32(&x_ln1, &q_weight_f32, &mut q_out, seq_len, q_out_dim, d_model)?;

            // K weight NOT transposed: [2048, 256] = [d_model, k_out_dim] in GGML order
            // x[seq_len,d_model] @ W[d_model,k_out_dim] = out[seq_len,k_out_dim]
            executor.matmul_metal_f32(&x_ln1, &k_weight_f32, &mut k_out, seq_len, k_out_dim, d_model)?;

            // V weight NOT transposed: [2048, 256] = [d_model, v_out_dim] in GGML order
            // x[seq_len,d_model] @ W[d_model,v_out_dim] = out[seq_len,v_out_dim]
            executor.matmul_metal_f32(&x_ln1, &v_weight_f32, &mut v_out, seq_len, v_out_dim, d_model)?;

            // Apply RoPE to Q and K projections
            // Q, K投影にRoPEを適用
            let head_dim = q_out_dim / self.config.num_heads;
            let num_heads = self.config.num_heads;
            let num_kv_heads = self.config.num_kv_heads;

            if debug && layer_idx == 0 {
                eprintln!("🔍 [ATTENTION LAYER 0] Starting attention computation");
                eprintln!("   q_out_dim={}, k_out_dim={}, v_out_dim={}", q_out_dim, k_out_dim, v_out_dim);
                eprintln!("   head_dim={}, num_heads={}, num_kv_heads={}", head_dim, num_heads, num_kv_heads);
            }

            // Apply RoPE to Q and K projections
            let q_proj = self.apply_rope(&q_out, seq_len, num_heads, head_dim, 0);
            let k_proj = self.apply_rope(&k_out, seq_len, num_kv_heads, head_dim, 0);

            if debug && layer_idx == 0 {
                let q_rms = (q_proj.iter().map(|&v| v * v).sum::<f32>() / q_proj.len() as f32).sqrt();
                let k_rms = (k_proj.iter().map(|&v| v * v).sum::<f32>() / k_proj.len() as f32).sqrt();
                eprintln!("   After RoPE: Q_rms={:.6}, K_rms={:.6}", q_rms, k_rms);
                eprintln!("   Q_proj[0..5]={:?}, K_proj[0..5]={:?}", &q_proj[0..5], &k_proj[0..5]);
            }

            // Compute attention: attn_out = softmax(Q @ K^T / sqrt(head_dim)) @ V
            // Attention計算: attn_out = softmax(Q @ K^T / sqrt(head_dim)) @ V
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn_out = vec![0.0f32; seq_len * d_model];

            // For Grouped Query Attention: repeat KV heads to match Q heads
            // Grouped Query Attention用: KVヘッドをQヘッドに合わせて繰り返す
            let heads_per_kv = num_heads / num_kv_heads;

            for q_head in 0..num_heads {
                let kv_head = q_head / heads_per_kv;

                for q_seq in 0..seq_len {
                    // Q vector for this head and sequence position
                    let q_offset = q_seq * (num_heads * head_dim) + q_head * head_dim;
                    let q_vec = &q_proj[q_offset..q_offset + head_dim];

                    // Debug first token, first head
                    if debug && q_head == 0 && q_seq == 0 {
                        let q_rms = (q_vec.iter().map(|&v| v * v).sum::<f32>() / head_dim as f32).sqrt();
                        eprintln!("🔍 [ATTENTION DEBUG] q_head=0, q_seq=0, kv_head={}", kv_head);
                        eprintln!("   Q_rms={:.6}, Q[0..5]={:?}", q_rms, &q_vec[0..5]);
                    }

                    // Compute attention scores for all key positions
                    let mut scores = vec![0.0f32; seq_len];
                    for k_seq in 0..=q_seq {  // Causal masking: only attend to past
                        let k_offset = k_seq * (num_kv_heads * head_dim) + kv_head * head_dim;
                        let k_vec = &k_proj[k_offset..k_offset + head_dim];

                        // Dot product: Q @ K^T
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_vec[d] * k_vec[d];
                        }
                        scores[k_seq] = score * scale;

                        // Debug first K vector
                        if debug && q_head == 0 && q_seq == 0 && k_seq == 0 {
                            let k_rms = (k_vec.iter().map(|&v| v * v).sum::<f32>() / head_dim as f32).sqrt();
                            eprintln!("   K_rms={:.6}, K[0..5]={:?}", k_rms, &k_vec[0..5]);
                            eprintln!("   Unscaled score={:.6}, Scaled score={:.6}", score / scale, score);
                        }
                    }

                    // Softmax normalization
                    let max_score = scores[0..=q_seq].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for k_seq in 0..=q_seq {
                        scores[k_seq] = (scores[k_seq] - max_score).exp();
                        exp_sum += scores[k_seq];
                    }
                    for k_seq in 0..=q_seq {
                        scores[k_seq] /= exp_sum;
                    }

                    // Debug: Show attention scores for first head of first token in layer 0
                    if debug && layer_idx == 0 && q_head == 0 && q_seq == 0 {
                        eprintln!("🔍 [LAYER 0 ATTENTION] Head 0, Token 0:");
                        eprintln!("   Raw scores (before softmax): {:?}", &scores[0..=q_seq.min(4)]);
                        eprintln!("   After softmax: {:?}", &scores[0..=q_seq.min(4)]);
                    }

                    // Weighted sum of values: attn @ V
                    let out_offset = q_seq * d_model + q_head * head_dim;
                    for k_seq in 0..=q_seq {
                        let v_offset = k_seq * (num_kv_heads * head_dim) + kv_head * head_dim;
                        let v_vec = &v_out[v_offset..v_offset + head_dim];
                        let weight = scores[k_seq];

                        for d in 0..head_dim {
                            attn_out[out_offset + d] += weight * v_vec[d];
                        }
                    }
                }
            }

            // Attention output projection
            // Attention出力射影
            let attn_proj_key = format!("blk.{}.attn_output.weight", layer_idx);
            let attn_proj_weight = self.weights.get(&attn_proj_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("Attention output weight not found: {}", attn_proj_key)))?;
            let attn_proj_weight_f32: Vec<f32> = attn_proj_weight.data.iter().map(|&v| v as f32).collect();

            if debug && layer_idx == 0 {
                let shape = attn_proj_weight.data.shape();
                eprintln!("🔍 [ATTN OUT PROJ] shape={:?}, expecting [d_model={}, d_model={}] or transposed", shape, d_model, d_model);
            }

            let mut attn_proj = vec![0.0f32; seq_len * d_model];
            // attn_output weight is [2048, 2048] = [in_features, out_features] in GGUF
            // Use regular matmul: x[1,2048] @ W[2048,2048] = out[1,2048]
            executor.matmul_metal_f32(&attn_out, &attn_proj_weight_f32, &mut attn_proj, seq_len, d_model, d_model)?;

            // Residual connection
            for i in 0..x_f32.len() {
                x_f32[i] += attn_proj[i];
            }

            if debug {
                eprintln!("     ✓ Attention complete (simplified)");
            }

            // Pre-FFN RMS Norm
            let ln2_key = format!("blk.{}.ffn_norm.weight", layer_idx);
            let ln2_weight = self.weights.get(&ln2_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("FFN norm weight not found: {}", ln2_key)))?;

            let ln2_gamma_f32: Vec<f32> = ln2_weight.data.iter().map(|&v| v as f32).collect();
            let mut x_ln2 = vec![0.0f32; x_f32.len()];
            Self::rms_norm_f32(&x_f32, &ln2_gamma_f32, &mut x_ln2, seq_len, d_model, self.config.norm_eps as f32);

            // FFN with SwiGLU
            let gate_key = format!("blk.{}.ffn_gate.weight", layer_idx);
            let up_key = format!("blk.{}.ffn_up.weight", layer_idx);
            let down_key = format!("blk.{}.ffn_down.weight", layer_idx);

            let gate_weight = self.weights.get(&gate_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("Gate weight not found: {}", gate_key)))?;
            let up_weight = self.weights.get(&up_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("Up weight not found: {}", up_key)))?;
            let down_weight = self.weights.get(&down_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("Down weight not found: {}", down_key)))?;

            let gate_weight_f32: Vec<f32> = gate_weight.data.iter().map(|&v| v as f32).collect();
            let up_weight_f32: Vec<f32> = up_weight.data.iter().map(|&v| v as f32).collect();
            let down_weight_f32: Vec<f32> = down_weight.data.iter().map(|&v| v as f32).collect();

            let intermediate_size = self.config.intermediate_size;

            // Show FFN shapes for first layer
            if layer_idx == 0 && debug {
                let gate_shape = gate_weight.data.shape();
                let up_shape = up_weight.data.shape();
                let down_shape = down_weight.data.shape();
                eprintln!("🔍 [FFN SHAPES] Layer {}: gate={:?}, up={:?}, down={:?}, d_model={}, intermediate_size={}",
                       layer_idx, gate_shape, up_shape, down_shape, d_model, intermediate_size);
            }

            // Gate and Up projections
            let mut gate_out = vec![0.0f32; seq_len * intermediate_size];
            let mut up_out = vec![0.0f32; seq_len * intermediate_size];

            // gate/up weights are [2048, 5632] = [in_features, out_features] in GGUF
            // Use regular matmul: x[1,2048] @ W[2048,5632] = out[1,5632]
            executor.matmul_metal_f32(&x_ln2, &gate_weight_f32, &mut gate_out, seq_len, intermediate_size, d_model)?;
            executor.matmul_metal_f32(&x_ln2, &up_weight_f32, &mut up_out, seq_len, intermediate_size, d_model)?;

            // SwiGLU: gate(x) * SiLU(up(x))
            for i in 0..gate_out.len() {
                let silu = up_out[i] / (1.0 + (-up_out[i]).exp()); // SiLU activation
                gate_out[i] *= silu;
            }

            // Down projection
            let mut ffn_out = vec![0.0f32; seq_len * d_model];
            // down weight is [5632, 2048] = [in_features, out_features] in GGUF
            // Use regular matmul: x[1,5632] @ W[5632,2048] = out[1,2048]
            executor.matmul_metal_f32(&gate_out, &down_weight_f32, &mut ffn_out, seq_len, d_model, intermediate_size)?;

            // Residual connection
            for i in 0..x_f32.len() {
                x_f32[i] += ffn_out[i];
            }

            if debug {
                eprintln!("     ✓ FFN complete");
            }

            // Force cleanup of Metal resources after each layer to prevent accumulation
            // 各レイヤー後にMetalリソースを強制クリーンアップして蓄積を防ぐ
            executor.force_cleanup();
        }

        // 3. Final RMS Norm
        let output_norm_key = "output_norm.weight";
        let output_norm_weight = self.weights.get(output_norm_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Output norm weight not found: {}", output_norm_key)))?;

        let output_norm_gamma_f32: Vec<f32> = output_norm_weight.data.iter().map(|&v| v as f32).collect();
        let mut x_final = vec![0.0f32; x_f32.len()];
        Self::rms_norm_f32(&x_f32, &output_norm_gamma_f32, &mut x_final, seq_len, d_model, self.config.norm_eps as f32);

        if debug {
            eprintln!("✓ Final RMS Norm complete");
        }

        // 4. Output projection to vocabulary
        if debug {
            eprintln!("🔍 [OUTPUT PROJECTION] Starting output projection...");
        }
        let output_key = "output.weight";
        let output_weight = self.weights.get(output_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Output weight not found: {}", output_key)))?;

        eprintln!("🔍 [OUTPUT WEIGHT] Found output.weight");
        let output_weight_f32: Vec<f32> = output_weight.data.iter().map(|&v| v as f32).collect();
        let output_shape = output_weight.data.shape();
        eprintln!("🔍 [OUTPUT WEIGHT] shape={:?}, shape[0]={}, shape[1]={}, len={}", output_shape, output_shape[0], output_shape[1], output_weight_f32.len());

        // BUGFIX 2025-10-16: GGML dimension order is [d_model, vocab_size] = [2048, 32000]
        // shape[0] = ne[0] = d_model (innermost)
        // shape[1] = ne[1] = vocab_size (outermost)
        let output_d_model = output_shape[0];     // d_model = 2048
        let output_vocab_size = output_shape[1];  // vocab_size = 32000

        let mut logits_f32 = vec![0.0f32; seq_len * output_vocab_size];
        // We need: logits[seq_len, vocab_size] = x_final[seq_len, d_model] @ W^T
        // W is [d_model, vocab_size] in GGML order
        // matmul_metal_f32(A, B, C, m, n, k): C[m,n] = A[m,k] @ B[k,n]
        // We want: C[seq_len, vocab_size] = A[seq_len, d_model] @ B[d_model, vocab_size]
        executor.matmul_metal_f32(&x_final, &output_weight_f32, &mut logits_f32,
                                  seq_len, output_vocab_size, output_d_model)?;

        if debug {
            eprintln!("✓ Output projection complete");
            let last_token_logits_start = (seq_len - 1) * output_vocab_size;
            eprintln!("🔍 [FINAL LOGITS] First 20:");
            for i in 0..20 {
                eprintln!("  [{}] = {:.6}", i, logits_f32[last_token_logits_start + i]);
            }
        }

        // Convert back to f64 and create tensor
        let logits_f64: Vec<f64> = logits_f32.iter().map(|&v| v as f64).collect();
        let shape = vec![batch_size, seq_len, output_vocab_size];

        use ndarray::Array;
        let logits_array = Array::from_shape_vec(
            ndarray::IxDyn(&shape),
            logits_f64
        ).map_err(|e| RusTorchError::tensor_op(format!("Failed to reshape logits: {}", e)))?;

        Ok(Tensor::new(logits_array))
    }

    #[cfg(not(feature = "metal"))]
    fn forward_metal(&self, _input_ids: &[usize], _start_position: usize) -> RusTorchResult<Tensor<f64>> {
        Err(RusTorchError::UnsupportedDevice("Metal not available".to_string()))
    }

    /// RMS Normalization (CPU implementation in f32)
    fn rms_norm_f32(
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) {
        for seq_idx in 0..seq_len {
            let offset = seq_idx * hidden_size;
            let row = &input[offset..offset + hidden_size];

            // Compute RMS (Root Mean Square)
            let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / (hidden_size as f32);
            let rms = (mean_sq + eps).sqrt();
            let scale = 1.0 / rms;

            // Normalize and scale with weight
            for i in 0..hidden_size {
                output[offset + i] = row[i] * scale * weight[i];
            }
        }
    }

    /// Apply RoPE (Rotary Position Embedding) to Q or K projections
    /// RoPE（回転位置埋め込み）をQ/K投影に適用
    fn apply_rope(
        &self,
        x: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        start_position: usize,
    ) -> Vec<f32> {
        let total_dim = num_heads * head_dim;
        let mut output = Vec::with_capacity(x.len());

        // Debug first call
        let has_nan = x.iter().any(|&v| v.is_nan());
        let has_inf = x.iter().any(|&v| v.is_infinite());
        if has_nan || has_inf {
            eprintln!("❌ [ROPE INPUT] Input contains NaN={}, Inf={}", has_nan, has_inf);
            eprintln!("   x.len()={}, seq_len={}, num_heads={}, head_dim={}", x.len(), seq_len, num_heads, head_dim);
        }

        // Apply rotation for each token in sequence
        for token_idx in 0..seq_len {
            let position = start_position + token_idx;

            // For each head of this token
            for head_idx in 0..num_heads {
                let head_offset = token_idx * total_dim + head_idx * head_dim;
                let head_data = &x[head_offset..head_offset + head_dim];

                for i in 0..(head_dim / 2) {
                    let rope_idx = position * (head_dim / 2) + i;

                    // Bounds checking for debugging
                    if rope_idx >= self.rope_cos.len() {
                        eprintln!("❌ [ROPE ERROR] rope_idx={} >= rope_cos.len()={}", rope_idx, self.rope_cos.len());
                        eprintln!("   position={}, head_dim={}, i={}, num_heads={}", position, head_dim, i, num_heads);
                        eprintln!("   token_idx={}, head_idx={}", token_idx, head_idx);
                        return output;  // Return what we have so far to avoid panic
                    }

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

        output
    }
}
