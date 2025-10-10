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
        // Calculate intermediate_size from hidden_size
        // For TinyLlama: 2048 * 2.75 = 5632
        let intermediate_size = (params.hidden_size as f32 * 2.75) as usize;

        Self {
            vocab_size: params.vocab_size as usize,
            hidden_size: params.hidden_size as usize,
            num_heads: params.num_heads as usize,
            num_kv_heads: params.num_kv_heads as usize,
            num_layers: params.num_layers as usize,
            max_seq_len: params.context_length as usize,
            intermediate_size,
            rope_theta: 10000.0,  // Default for Llama
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

/// KV Cache for attention layers
/// Attentionレイヤー用のKVキャッシュ
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cached key tensors for each layer [num_layers][max_cached_tokens, kv_dim]
    pub k_cache: Vec<Vec<f32>>,
    /// Cached value tensors for each layer [num_layers][max_cached_tokens, kv_dim]
    pub v_cache: Vec<Vec<f32>>,
    /// Number of tokens currently cached
    pub cached_tokens: usize,
    /// Maximum cache capacity
    pub max_cache_size: usize,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(num_layers: usize, max_cache_size: usize, kv_dim: usize) -> Self {
        let k_cache = vec![vec![0.0f32; max_cache_size * kv_dim]; num_layers];
        let v_cache = vec![vec![0.0f32; max_cache_size * kv_dim]; num_layers];

        Self {
            k_cache,
            v_cache,
            cached_tokens: 0,
            max_cache_size,
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cached_tokens = 0;
    }
}

/// Llama Model with Metal GPU support
/// Metal GPU対応のLlamaモデル
#[derive(Debug)]
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub weights: HashMap<String, Tensor<f64>>,
    pub device_type: DeviceType,
    /// KV cache for efficient multi-token generation
    pub kv_cache: Option<KVCache>,
}

impl LlamaModel {
    /// Create a new Llama model with specified config and device
    pub fn new(config: LlamaConfig, device_type: DeviceType) -> RusTorchResult<Self> {
        // Initialize KV cache for multi-token generation
        let kv_cache = Some(KVCache::new(
            config.num_layers,
            config.max_seq_len,
            config.kv_dim(),
        ));

        Ok(Self {
            config,
            weights: HashMap::new(),
            device_type,
            kv_cache,
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
                    model.weights.insert(name.to_string(), tensor);
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
    pub fn forward(&mut self, input_ids: &[usize]) -> RusTorchResult<Tensor<f64>> {
        self.forward_with_position(input_ids, 0)
    }

    /// Forward pass with position tracking
    pub fn forward_with_position(&mut self, input_ids: &[usize], start_position: usize) -> RusTorchResult<Tensor<f64>> {
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
    fn forward_metal(&mut self, input_ids: &[usize], start_position: usize) -> RusTorchResult<Tensor<f64>> {
        use crate::gpu::metal_kernels::MetalKernelExecutor;

        let debug = std::env::var("RUSTORCH_DEBUG").is_ok();

        eprintln!("🦙 Llama forward_metal called (input_len={}, start_pos={}, debug={})",
            input_ids.len(), start_position, debug);

        eprintln!("🔍 [METAL DEBUG] Step 1: Getting Metal executor...");
        // Get Metal executor
        let executor_mutex = MetalKernelExecutor::get()?;
        eprintln!("🔍 [METAL DEBUG] Step 2: Metal executor obtained, locking...");
        let executor_guard = executor_mutex.lock().unwrap();
        eprintln!("🔍 [METAL DEBUG] Step 3: Lock acquired, extracting executor...");
        let executor = executor_guard.as_ref()
            .ok_or_else(|| RusTorchError::tensor_op("Metal executor not initialized"))?;
        eprintln!("🔍 [METAL DEBUG] Step 4: Metal executor ready");

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
        let hidden_size = emb_shape[0];
        let vocab_size = emb_shape[1];

        let emb_data = token_emb_tensor.data.as_slice()
            .ok_or_else(|| RusTorchError::tensor_op("Failed to get embedding data as slice"))?;

        for (token_idx, &token_id) in input_ids.iter().enumerate() {
            if token_id >= vocab_size {
                return Err(RusTorchError::tensor_op(
                    format!("Token ID {} out of range (vocab_size={})", token_id, vocab_size)
                ));
            }

            let start = token_id * hidden_size;
            let end = start + hidden_size;
            embedding_data.extend_from_slice(&emb_data[start..end]);

            if token_idx < 3 && debug {
                let current_emb_start = embedding_data.len() - hidden_size;
                let emb_slice = &embedding_data[current_emb_start..current_emb_start + 10.min(hidden_size)];
                eprintln!("🔍 [EMBEDDING] Token {} (ID={}): first 10: {:?}", token_idx, token_id, emb_slice);
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
        // TEMPORARY: Only run first layer for debugging
        let num_layers = if debug { 1 } else { self.config.num_layers };

        if debug {
            eprintln!("⚠️  DEBUG MODE: Only processing {} layer(s)", num_layers);
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

            if debug {
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

            // GGUF shape: [out_features, in_features]
            let q_out_dim = q_shape[1];  // Should be 2048
            let k_out_dim = k_shape[1];  // Should be 256 (4 * 64)
            let v_out_dim = v_shape[1];  // Should be 256 (4 * 64)

            if debug && layer_idx == 0 {
                eprintln!("     🔍 [WEIGHT SHAPES] Q: {:?} -> out_dim={}, K: {:?} -> out_dim={}, V: {:?} -> out_dim={}",
                    q_shape, q_out_dim, k_shape, k_out_dim, v_shape, v_out_dim);
            }

            // Q, K, V projections
            // Q, K, V射影
            let mut q_out = vec![0.0f32; seq_len * q_out_dim];
            let mut k_out = vec![0.0f32; seq_len * k_out_dim];
            let mut v_out = vec![0.0f32; seq_len * v_out_dim];

            if debug {
                eprintln!("     🔧 [DEBUG] About to call Q, K, V projections with single autoreleasepool...");
                eprintln!("        x_ln1.len={}, q_weight_f32.len={}, q_out.len={}", x_ln1.len(), q_weight_f32.len(), q_out.len());
                eprintln!("        Dimensions: seq_len={}, d_model={}, q_out_dim={}", seq_len, d_model, q_out_dim);
            }

            // Each matmul is automatically wrapped in its own autoreleasepool by metal_kernels.rs
            // metal_kernels.rsが各matmulを自動的にautoreleasepoolで囲む

            eprintln!("🔍 [METAL DEBUG] Step 5: About to call FIRST Metal matmul (Q projection)...");
            eprintln!("     Params: seq_len={}, d_model={}, q_out_dim={}", seq_len, d_model, q_out_dim);

            executor.matmul_f32(&x_ln1, &q_weight_f32, &mut q_out, seq_len, d_model, q_out_dim)?;

            eprintln!("🔍 [METAL DEBUG] Step 6: Q projection completed successfully!");

            if debug {
                eprintln!("     ✓ Q projection complete");
            }

            executor.matmul_f32(&x_ln1, &k_weight_f32, &mut k_out, seq_len, k_out_dim, d_model)?;

            if debug {
                eprintln!("     ✓ K projection complete");
            }

            executor.matmul_f32(&x_ln1, &v_weight_f32, &mut v_out, seq_len, v_out_dim, d_model)?;

            if debug {
                eprintln!("     ✓ V projection complete");
            }

            if debug && layer_idx == 0 {
                eprintln!("     ✓ Q, K, V projections complete (with autoreleasepool)");
            }

            // Proper attention with RoPE
            // RoPE付き適切なattention
            let head_dim = q_out_dim / self.config.num_heads;
            let num_kv_heads = self.config.num_kv_heads;
            let rope_theta = self.config.rope_theta;

            // Apply RoPE to Q and K using Metal GPU
            executor.apply_rope_f32(&mut q_out, start_position, seq_len, self.config.num_heads, head_dim, rope_theta)?;
            executor.apply_rope_f32(&mut k_out, start_position, seq_len, num_kv_heads, head_dim, rope_theta)?;

            if debug {
                eprintln!("     ✓ RoPE applied to Q and K (Metal GPU)");
            }

            // Update KV cache with new K/V values
            if let Some(ref mut cache) = self.kv_cache {
                // Append new K/V to cache
                let cache_start = cache.cached_tokens;
                let cache_end = cache_start + seq_len;

                if cache_end > cache.max_cache_size {
                    return Err(RusTorchError::tensor_op(format!(
                        "KV cache overflow: {} + {} > {}",
                        cache_start, seq_len, cache.max_cache_size
                    )));
                }

                // Copy new K/V into cache
                for i in 0..seq_len {
                    let src_offset = i * k_out_dim;
                    let dst_offset = (cache_start + i) * k_out_dim;
                    cache.k_cache[layer_idx][dst_offset..dst_offset + k_out_dim]
                        .copy_from_slice(&k_out[src_offset..src_offset + k_out_dim]);
                    cache.v_cache[layer_idx][dst_offset..dst_offset + v_out_dim]
                        .copy_from_slice(&v_out[src_offset..src_offset + v_out_dim]);
                }

                if debug {
                    eprintln!("     ✓ KV cache updated (tokens: {} -> {})", cache_start, cache_end);
                }
            }

            // Get full K/V from cache (including both cached and new tokens)
            let total_seq_len = if let Some(ref cache) = self.kv_cache {
                cache.cached_tokens + seq_len
            } else {
                seq_len
            };

            // Expand K and V to match number of Q heads (GQA)
            let heads_per_kv = self.config.num_heads / num_kv_heads;
            let kv_head_size = v_out_dim / num_kv_heads;

            // Expand full cached K/V for attention
            let mut k_expanded = vec![0.0f32; total_seq_len * self.config.num_heads * head_dim];
            let mut v_expanded = vec![0.0f32; total_seq_len * self.config.num_heads * head_dim];

            if let Some(ref cache) = self.kv_cache {
                // Expand K/V from cache
                for seq in 0..total_seq_len {
                    for kv_h in 0..num_kv_heads {
                        let kv_offset = seq * v_out_dim + kv_h * kv_head_size;
                        for rep in 0..heads_per_kv {
                            let q_head_idx = kv_h * heads_per_kv + rep;
                            let expanded_offset = seq * (self.config.num_heads * head_dim) + q_head_idx * head_dim;
                            for d in 0..kv_head_size {
                                k_expanded[expanded_offset + d] = cache.k_cache[layer_idx][kv_offset + d];
                                v_expanded[expanded_offset + d] = cache.v_cache[layer_idx][kv_offset + d];
                            }
                        }
                    }
                }
            } else {
                // Fallback: expand only current K/V (no cache)
                for seq in 0..seq_len {
                    for kv_h in 0..num_kv_heads {
                        let kv_offset = seq * v_out_dim + kv_h * kv_head_size;
                        for rep in 0..heads_per_kv {
                            let q_head_idx = kv_h * heads_per_kv + rep;
                            let expanded_offset = seq * (self.config.num_heads * head_dim) + q_head_idx * head_dim;
                            for d in 0..kv_head_size {
                                k_expanded[expanded_offset + d] = k_out[kv_offset + d];
                                v_expanded[expanded_offset + d] = v_out[kv_offset + d];
                            }
                        }
                    }
                }
            }

            // Compute attention with softmax on GPU
            let mut attn_out = vec![0.0f32; q_out.len()];
            executor.compute_attention_with_softmax_f32(
                &q_out,
                &k_expanded,
                &v_expanded,
                &mut attn_out,
                seq_len,
                total_seq_len,
                self.config.num_heads,
                head_dim,
            )?;

            if debug {
                eprintln!("     ✓ Attention computed on GPU (q_len={}, kv_len={})", seq_len, total_seq_len);
            }

            // Attention output projection
            // Attention出力射影
            let attn_proj_key = format!("blk.{}.attn_output.weight", layer_idx);
            let attn_proj_weight = self.weights.get(&attn_proj_key)
                .ok_or_else(|| RusTorchError::tensor_op(format!("Attention output weight not found: {}", attn_proj_key)))?;
            let attn_proj_weight_f32: Vec<f32> = attn_proj_weight.data.iter().map(|&v| v as f32).collect();

            let mut attn_proj = vec![0.0f32; seq_len * d_model];
            executor.matmul_f32(&attn_out, &attn_proj_weight_f32, &mut attn_proj, seq_len, d_model, d_model)?;

            // Residual connection
            for i in 0..x_f32.len() {
                x_f32[i] += attn_proj[i];
            }

            if debug {
                eprintln!("     ✓ Attention complete (with RoPE and proper attention calculation)");
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

            // Gate and Up projections
            let mut gate_out = vec![0.0f32; seq_len * intermediate_size];
            let mut up_out = vec![0.0f32; seq_len * intermediate_size];

            executor.matmul_f32(&x_ln2, &gate_weight_f32, &mut gate_out, seq_len, intermediate_size, d_model)?;
            executor.matmul_f32(&x_ln2, &up_weight_f32, &mut up_out, seq_len, intermediate_size, d_model)?;

            // SwiGLU: gate(x) * SiLU(up(x))
            for i in 0..gate_out.len() {
                let silu = up_out[i] / (1.0 + (-up_out[i]).exp()); // SiLU activation
                gate_out[i] *= silu;
            }

            // Down projection
            let mut ffn_out = vec![0.0f32; seq_len * d_model];
            executor.matmul_f32(&gate_out, &down_weight_f32, &mut ffn_out, seq_len, d_model, intermediate_size)?;

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
        let output_key = "output.weight";
        let output_weight = self.weights.get(output_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Output weight not found: {}", output_key)))?;

        let output_weight_f32: Vec<f32> = output_weight.data.iter().map(|&v| v as f32).collect();
        let output_vocab_size = output_weight.data.shape()[1];

        let mut logits_f32 = vec![0.0f32; seq_len * output_vocab_size];
        executor.matmul_f32(&x_final, &output_weight_f32, &mut logits_f32, seq_len, output_vocab_size, d_model)?;

        if debug {
            eprintln!("✓ Output projection complete");
            eprintln!("🎉 Llama forward pass complete!");
        }

        // Update KV cache token count
        if let Some(ref mut cache) = self.kv_cache {
            cache.cached_tokens += seq_len;
            if debug {
                eprintln!("✓ KV cache updated: total cached tokens = {}", cache.cached_tokens);
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

    /// Apply RoPE (Rotary Position Embedding) to Q or K tensors
    /// QまたはKテンソルにRoPE（回転位置埋め込み）を適用
    fn apply_rope(
        x: &mut [f32],
        start_pos: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) {
        let half_head_dim = head_dim / 2;

        for pos in 0..seq_len {
            let absolute_pos = start_pos + pos;

            for head in 0..num_heads {
                let head_offset = pos * (num_heads * head_dim) + head * head_dim;

                for dim_pair in 0..half_head_dim {
                    let dim = dim_pair * 2;

                    // Compute frequency: 1 / (theta ^ (2 * dim / head_dim))
                    let freq = 1.0 / rope_theta.powf((dim as f32) / (head_dim as f32));
                    let angle = (absolute_pos as f32) * freq;

                    let cos_val = angle.cos();
                    let sin_val = angle.sin();

                    // Rotate (x[dim], x[dim+1]) pair
                    let x0 = x[head_offset + dim];
                    let x1 = x[head_offset + dim + 1];

                    x[head_offset + dim] = x0 * cos_val - x1 * sin_val;
                    x[head_offset + dim + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    /// Compute attention scores: softmax(Q @ K^T / sqrt(head_dim))
    /// Attention スコア計算: softmax(Q @ K^T / sqrt(head_dim))
    ///
    /// # Parameters
    /// - q: Query tensor [q_len, num_heads * head_dim]
    /// - k: Key tensor [kv_len, num_heads * head_dim]
    /// - q_len: Query sequence length
    /// - kv_len: Key/Value sequence length (may be different from q_len when using KV cache)
    /// - num_heads: Number of attention heads
    /// - head_dim: Dimension of each head
    fn compute_attention_scores(
        q: &[f32],
        k: &[f32],
        q_len: usize,
        kv_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; q_len * kv_len * num_heads];

        for head in 0..num_heads {
            for i in 0..q_len {
                let q_offset = i * (num_heads * head_dim) + head * head_dim;
                let score_row_offset = head * q_len * kv_len + i * kv_len;

                for j in 0..kv_len {
                    let k_offset = j * (num_heads * head_dim) + head * head_dim;

                    // Compute dot product Q[i] @ K[j]^T
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_offset + d] * k[k_offset + d];
                    }

                    scores[score_row_offset + j] = dot * scale;
                }

                // Apply softmax to the row
                let row_offset = score_row_offset;
                let mut max_score = scores[row_offset];
                for j in 1..kv_len {
                    if scores[row_offset + j] > max_score {
                        max_score = scores[row_offset + j];
                    }
                }

                let mut sum_exp = 0.0f32;
                for j in 0..kv_len {
                    let exp_val = (scores[row_offset + j] - max_score).exp();
                    scores[row_offset + j] = exp_val;
                    sum_exp += exp_val;
                }

                for j in 0..kv_len {
                    scores[row_offset + j] /= sum_exp;
                }
            }
        }

        scores
    }

    /// Apply attention: output = attention_scores @ V
    /// Attentionを適用: output = attention_scores @ V
    ///
    /// # Parameters
    /// - scores: Attention scores [q_len, kv_len] for each head
    /// - v: Value tensor [kv_len, num_heads * head_dim]
    /// - q_len: Query sequence length
    /// - kv_len: Key/Value sequence length
    /// - num_heads: Number of attention heads
    /// - head_dim: Dimension of each head
    fn apply_attention_to_values(
        scores: &[f32],
        v: &[f32],
        q_len: usize,
        kv_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; q_len * num_heads * head_dim];

        for head in 0..num_heads {
            for i in 0..q_len {
                let score_row_offset = head * q_len * kv_len + i * kv_len;
                let out_offset = i * (num_heads * head_dim) + head * head_dim;

                for j in 0..kv_len {
                    let v_offset = j * (num_heads * head_dim) + head * head_dim;
                    let attention_weight = scores[score_row_offset + j];

                    for d in 0..head_dim {
                        output[out_offset + d] += attention_weight * v[v_offset + d];
                    }
                }
            }
        }

        output
    }
}
