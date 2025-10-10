//! GPT model implementation for RusTorch
//! GPTモデル実装

use crate::backends::DeviceType;
use crate::error::{RusTorchError, RusTorchResult};
use crate::formats::gguf::{GGUFLoader, ModelParams};
use crate::formats::mlx::MLXLoader;
use crate::formats::safetensors::SafetensorsLoader;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "metal")]
use crate::gpu::metal_kernels::MetalKernelExecutor;
#[cfg(feature = "metal")]
use std::sync::{Arc, Mutex};


/// GPT model configuration
#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,  // For Grouped Query Attention (GQA)
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
    pub rope_theta: f32,  // RoPE base frequency (default: 10000.0)
}

impl GPTConfig {
    /// Create config from GGUF model parameters
    pub fn from_model_params(params: &ModelParams) -> Self {
        Self {
            vocab_size: params.vocab_size as usize,
            d_model: params.hidden_size as usize,
            num_layers: params.num_layers as usize,
            num_heads: params.num_heads as usize,
            num_kv_heads: params.num_kv_heads as usize,  // GQA support
            d_ff: (params.hidden_size * 4) as usize, // Standard FFN size
            max_seq_len: params.context_length as usize,
            dropout: 0.1,
            rope_theta: 10000.0,  // Standard RoPE base frequency
        }
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }
}

/// KV Cache for a single layer
#[cfg(feature = "metal")]
#[derive(Clone)]
pub struct LayerKVCache {
    /// Cached keys: [cached_seq_len, num_kv_heads, head_dim]
    pub keys: Vec<f32>,
    /// Cached values: [cached_seq_len, num_kv_heads, head_dim]
    pub values: Vec<f32>,
    /// Number of cached tokens
    pub cached_len: usize,
}

#[cfg(feature = "metal")]
impl LayerKVCache {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            cached_len: 0,
        }
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.cached_len = 0;
    }
}

/// GPT model structure
pub struct GPTModel {
    config: GPTConfig,
    weights: HashMap<String, Tensor<f64>>,
    device_type: DeviceType,
    #[cfg(feature = "metal")]
    has_metal: bool,
    #[cfg(feature = "metal")]
    rope_cos: Vec<f32>,  // Precomputed RoPE cosine values
    #[cfg(feature = "metal")]
    rope_sin: Vec<f32>,  // Precomputed RoPE sine values
    #[cfg(feature = "metal")]
    kv_cache: Vec<LayerKVCache>,  // KV cache for each layer
}

impl GPTModel {
    /// Precompute RoPE (Rotary Position Embedding) frequencies
    /// RoPE周波数を事前計算
    #[cfg(feature = "metal")]
    fn precompute_rope_frequencies(config: &GPTConfig) -> (Vec<f32>, Vec<f32>) {
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

    /// Create a new GPT model with given configuration (CPU backend)
    pub fn new(config: GPTConfig) -> RusTorchResult<Self> {
        Self::with_backend(config, DeviceType::Cpu)
    }

    /// Create a new GPT model with specified backend
    pub fn with_backend(config: GPTConfig, device_type: DeviceType) -> RusTorchResult<Self> {
        let actual_device = match device_type {
            DeviceType::Cpu => DeviceType::Cpu,
            #[cfg(feature = "metal")]
            DeviceType::Metal => {
                eprintln!("🚀 Metal backend selected - initializing GPU acceleration");
                DeviceType::Metal
            }
            #[cfg(not(feature = "metal"))]
            DeviceType::Metal => {
                eprintln!("⚠️  Metal feature not enabled, falling back to CPU");
                DeviceType::Cpu
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => {
                eprintln!("⚠️  CUDA backend selected, but tensor operations use CPU");
                eprintln!("    GPU acceleration will be added in future updates");
                DeviceType::Cuda
            }
            _ => {
                eprintln!("⚠️  Unsupported backend: {:?}, using CPU", device_type);
                DeviceType::Cpu
            }
        };

        // Initialize Metal executor if Metal backend is selected
        #[cfg(feature = "metal")]
        let has_metal = if actual_device == DeviceType::Metal {
            match MetalKernelExecutor::get() {
                Ok(_) => {
                    eprintln!("✅ Metal GPU initialized successfully");
                    true
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to initialize Metal: {}", e);
                    eprintln!("   Falling back to CPU");
                    false
                }
            }
        } else {
            false
        };

        #[cfg(feature = "metal")]
        let (rope_cos, rope_sin) = if has_metal {
            Self::precompute_rope_frequencies(&config)
        } else {
            (Vec::new(), Vec::new())
        };

        #[cfg(feature = "metal")]
        let kv_cache = if has_metal {
            (0..config.num_layers)
                .map(|_| LayerKVCache::new())
                .collect()
        } else {
            Vec::new()
        };

        Ok(Self {
            config,
            weights: HashMap::new(),
            device_type: actual_device,
            #[cfg(feature = "metal")]
            has_metal,
            #[cfg(feature = "metal")]
            rope_cos,
            #[cfg(feature = "metal")]
            rope_sin,
            #[cfg(feature = "metal")]
            kv_cache,
        })
    }

    /// Get backend device type
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Clear KV cache for all layers (use between different prompts)
    #[cfg(feature = "metal")]
    pub fn clear_kv_cache(&mut self) {
        for cache in &mut self.kv_cache {
            cache.clear();
        }
    }

    /// Load GPT model from GGUF file (CPU backend)
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> RusTorchResult<Self> {
        Self::from_gguf_with_backend(path, DeviceType::Cpu)
    }

    /// Load GPT model from GGUF file with specified backend
    pub fn from_gguf_with_backend<P: AsRef<Path>>(
        path: P,
        device_type: DeviceType,
    ) -> RusTorchResult<Self> {
        let loader = GGUFLoader::from_file(path)?;

        // Extract model parameters
        let params = loader.get_model_params()?;
        let config = GPTConfig::from_model_params(&params);

        // Create model with backend
        let mut model = Self::with_backend(config, device_type)?;

        eprintln!(
            "📊 Loading GPT model on {:?} backend",
            model.device_type
        );

        // Load weights
        let tensor_names = loader.tensor_names();

        for name in tensor_names.iter() {
            match loader.load_tensor(name) {
                Ok(tensor) => {
                    model.weights.insert(name.to_string(), tensor);
                }
                Err(_e) => {
                    // Continue loading other tensors even if some fail
                }
            }
        }

        Ok(model)
    }

    /// Load GPT model from Safetensors file
    /// SafetensorsファイルからGPTモデルを読み込み
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        config: GPTConfig,
    ) -> RusTorchResult<Self> {
        let loader = SafetensorsLoader::from_file(path)?;

        // Create model with provided config
        let mut model = Self::new(config)?;

        // Load weights
        let tensor_names = loader.tensor_names();

        for name in tensor_names.iter() {
            match loader.load_tensor::<f64>(name) {
                Ok(tensor) => {
                    model.weights.insert(name.to_string(), tensor);
                }
                Err(_e) => {
                    // Continue loading other tensors even if some fail
                }
            }
        }

        Ok(model)
    }

    /// Load GPT model from MLX file
    /// MLXファイルからGPTモデルを読み込み
    pub fn from_mlx<P: AsRef<Path>>(path: P, config: GPTConfig) -> RusTorchResult<Self> {
        let loader = MLXLoader::from_file(path)?;

        // Create model with provided config
        let mut model = Self::new(config)?;

        // Load weights
        let tensor_names = loader.tensor_names();

        for name in tensor_names.iter() {
            match loader.load_tensor::<f64>(name) {
                Ok(tensor) => {
                    model.weights.insert(name.to_string(), tensor);
                }
                Err(_e) => {
                    // Continue loading other tensors even if some fail
                }
            }
        }

        Ok(model)
    }

    /// Get model configuration
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    /// Get a weight tensor by name
    pub fn get_weight(&self, name: &str) -> Option<&Tensor<f64>> {
        self.weights.get(name)
    }

    /// List all weight names
    pub fn weight_names(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }

    /// Create LayerNorm with loaded weights if available
    /// 読み込まれた重みでLayerNormを作成（利用可能な場合）
    fn create_layer_norm_variable(&self, weight_key: &str, d_model: usize) -> crate::autograd::Variable<f64> {
        use crate::autograd::Variable;

        // Create a default weight (all ones) and bias (all zeros)
        let weight_data = if let Some(loaded_weight) = self.weights.get(weight_key) {
            // Use loaded weights if available
            loaded_weight.clone()
        } else {
            // Default initialization: ones
            Tensor::from_vec(vec![1.0; d_model], vec![d_model])
        };

        let bias_data = Tensor::from_vec(vec![0.0; d_model], vec![d_model]);

        // Return weight and bias as Variables
        // For now, just return the weight variable (bias will be handled separately)
        Variable::new(weight_data, true)
    }

    /// Apply manual LayerNorm with loaded weights
    /// 読み込まれた重みで手動LayerNormを適用
    fn apply_layer_norm(&self, input: &crate::autograd::Variable<f64>, weight_key: &str, d_model: usize) -> crate::autograd::Variable<f64> {
        use crate::autograd::Variable;

        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();

        // Get weight and bias
        let (weight, using_loaded_weight) = if let Some(w) = self.weights.get(weight_key) {
            (w.clone(), true)
        } else {
            (Tensor::from_vec(vec![1.0; d_model], vec![d_model]), false)
        };

        #[cfg(debug_assertions)]
        if using_loaded_weight {
            eprintln!("✓ Using loaded GGUF weight: {}", weight_key);
        } else {
            eprintln!("✗ Weight not found, using default: {}", weight_key);
        }

        let bias = Tensor::from_vec(vec![0.0; d_model], vec![d_model]);
        let eps = 1e-5;

        // Manual LayerNorm computation
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let features = input_shape[2];

        let mut output_data = Vec::with_capacity(batch_size * seq_len * features);

        // Process each position
        for b in 0..batch_size {
            for s in 0..seq_len {
                // Extract features for this position
                let mut position_features = Vec::with_capacity(features);
                for f in 0..features {
                    let idx = (b * seq_len * features) + (s * features) + f;
                    if let Some(slice) = input_data.as_array().as_slice() {
                        position_features.push(slice[idx]);
                    }
                }

                // Calculate mean and variance
                let mean: f64 = position_features.iter().sum::<f64>() / features as f64;
                let variance: f64 = position_features.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / features as f64;
                let std = (variance + eps).sqrt();

                // Normalize and apply affine transformation
                for (f, &feature_val) in position_features.iter().enumerate() {
                    let normalized = (feature_val - mean) / std;
                    let gamma = if let Some(slice) = weight.as_array().as_slice() {
                        slice[f]
                    } else {
                        1.0
                    };
                    let beta = if let Some(slice) = bias.as_array().as_slice() {
                        slice[f]
                    } else {
                        0.0
                    };
                    let final_val = gamma * normalized + beta;
                    output_data.push(final_val);
                }
            }
        }

        let output = Tensor::from_vec(output_data, input_shape.to_vec());
        Variable::new(output, input.requires_grad())
    }

    /// GPT forward pass with Transformer implementation
    /// TransformerフォワードパスによるGPT実装
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    ///
    /// # Returns
    /// Logits tensor of shape [batch_size, seq_len, vocab_size]
    ///
    /// # Note
    /// For development/testing, uses only first 2 layers by default.
    /// Use `forward_with_layers(input_ids, None)` for full model inference.
    pub fn forward(&self, input_ids: &[usize]) -> RusTorchResult<Tensor<f64>> {
        // Default: start at position 0 for backward compatibility
        self.forward_with_position(input_ids, 0)
    }

    /// Forward pass with explicit position tracking
    /// 明示的な位置追跡付きフォワードパス
    pub fn forward_with_position(&self, input_ids: &[usize], start_position: usize) -> RusTorchResult<Tensor<f64>> {
        let debug = std::env::var("RUSTORCH_DEBUG").is_ok();

        // Route to Metal or CPU implementation based on backend
        #[cfg(feature = "metal")]
        if self.has_metal && self.device_type == DeviceType::Metal {
            if debug {
                eprintln!("✅ Routing to Metal GPU backend (device_type={:?})",
                    self.device_type);
            }
            return self.forward_metal(input_ids, start_position);
        }

        // CPU fallback
        if debug {
            eprintln!("⚠️  GPT forward pass using CPU (device_type={:?})",
                self.device_type);
        }
        let max_layers = Some(2);
        self.forward_with_layers(input_ids, max_layers)
    }

    /// Metal GPU-accelerated forward pass with position tracking
    /// Metal GPU加速フォワードパス（位置追跡付き）
    #[cfg(feature = "metal")]
    fn forward_metal(&self, input_ids: &[usize], start_position: usize) -> RusTorchResult<Tensor<f64>> {
        use crate::gpu::metal_kernels::MetalKernelExecutor;

        // Debug output controlled by RUSTORCH_DEBUG environment variable
        let debug = std::env::var("RUSTORCH_DEBUG").is_ok();

        // FORCE OUTPUT for debugging
        eprintln!("🚀 GPT forward_metal called (input_len={}, start_pos={}, debug={})",
            input_ids.len(), start_position, debug);

        // Get Metal executor
        let executor_mutex = MetalKernelExecutor::get()?;
        let executor_guard = executor_mutex.lock().unwrap();
        let executor = executor_guard.as_ref()
            .ok_or_else(|| RusTorchError::tensor_op("Metal executor not initialized"))?;

        let batch_size = 1;
        let seq_len = input_ids.len();
        let d_model = self.config.d_model;

        // 1. Token Embedding Lookup (CPU for now - embeddings are quantized)
        let token_emb_key = "token_embd.weight";
        let token_emb_tensor = self.weights.get(token_emb_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Token embedding not found: {}", token_emb_key)))?;

        // Perform embedding lookup - need to handle ndarray shape
        // GGUF shape [2048, 32000] means data is stored as:
        //   32000 rows of 2048 elements each (row-major layout)
        //   Row 0 = token 0's embedding (2048 values)
        //   Row 1 = token 1's embedding (2048 values)
        //   ...
        // So: embedding for token N starts at index N * d_model
        let mut embedding_data = Vec::with_capacity(seq_len * d_model);
        let emb_shape = token_emb_tensor.data.shape();
        let hidden_size = emb_shape[0];  // 2048
        let vocab_size = emb_shape[1];   // 32000

        // Get flat data as slice for efficient access
        let emb_data = token_emb_tensor.data.as_slice()
            .ok_or_else(|| RusTorchError::tensor_op("Failed to get embedding data as slice"))?;

        for (token_idx, &token_id) in input_ids.iter().enumerate() {
            if token_id >= vocab_size {
                return Err(RusTorchError::tensor_op(
                    format!("Token ID {} out of range (vocab_size={})", token_id, vocab_size)
                ));
            }

            // IMPORTANT: GGML/GGUF dimension ordering for quantized tensors
            // GGUF dims [2048, 32000] where ne[0]=2048 is INNERMOST (fastest-changing)
            // Memory layout: [token0_emb (2048), token1_emb (2048), ..., token31999_emb (2048)]
            // So token N's embedding is at indices: N * hidden_size .. (N+1) * hidden_size

            let start = token_id * hidden_size;
            let end = start + hidden_size;

            if end > emb_data.len() {
                return Err(RusTorchError::tensor_op(
                    format!("Embedding index out of range: token_id={}, start={}, end={}, data_len={}",
                        token_id, start, end, emb_data.len())
                ));
            }

            // Extract the embedding for this token
            embedding_data.extend_from_slice(&emb_data[start..end]);

            // Log first 3 tokens' embeddings for llama.cpp comparison
            if token_idx < 3 && debug {
                // Get first 10 elements of the embedding we just extracted
                let current_emb_start = embedding_data.len() - hidden_size;
                let emb_slice = &embedding_data[current_emb_start..current_emb_start + 10.min(hidden_size)];
                let emb_full = &embedding_data[current_emb_start..];
                let mean: f64 = emb_full.iter().map(|&v| v as f64).sum::<f64>() / emb_full.len() as f64;
                let sq_sum: f64 = emb_full.iter().map(|&v| (v as f64).powi(2)).sum();
                let rms = (sq_sum / emb_full.len() as f64).sqrt();
                eprintln!("🔍 [EMBEDDING] Token {} (ID={}): embedding[0..10]: {:?}", token_idx, token_id, emb_slice);
                eprintln!("   📊 Stats: mean={:.9}, rms={:.9}, len={}", mean, rms, emb_full.len());
            }
        }

        // Convert to f32 for Metal operations
        let mut x_f32: Vec<f32> = embedding_data.iter().map(|&v| v as f32).collect();

        // Log initial embedding statistics
        if debug {
            let mean: f32 = x_f32.iter().sum::<f32>() / x_f32.len() as f32;
            let sq_sum: f32 = x_f32.iter().map(|&v| v * v).sum();
            let rms = (sq_sum / x_f32.len() as f32).sqrt();
            eprintln!("🎯 [INPUT] After embedding: mean={:.6}, rms={:.6}", mean, rms);
        }

        // 2. Process through all Transformer layers with Metal
        let num_layers = self.config.num_layers;

        if debug {
            eprintln!("   🔄 Processing {} transformer layers (seq_len={})", num_layers, seq_len);
            eprintln!("   📊 Memory estimate: attn_scores={}KB, embedding={}KB",
                      (seq_len * seq_len * 4) / 1024,
                      (seq_len * d_model * 4) / 1024);
        }

        for layer_idx in 0..num_layers {
            if debug {
                eprintln!("   📍 Layer {}/{}", layer_idx + 1, num_layers);
                // Check Pos 1 and Pos 17 input to this layer
                if seq_len > 17 {
                    let pos1_start = 1 * d_model;
                    let pos1_input = &x_f32[pos1_start..pos1_start + d_model];
                    let pos1_rms = (pos1_input.iter().map(|&v| v * v).sum::<f32>() / d_model as f32).sqrt();

                    let pos17_start = 17 * d_model;
                    let pos17_input = &x_f32[pos17_start..pos17_start + d_model];
                    let pos17_rms = (pos17_input.iter().map(|&v| v * v).sum::<f32>() / d_model as f32).sqrt();
                    eprintln!("   🔍 [LAYER INPUT] Pos 1 RMS: {:.6}, Pos 17 RMS: {:.6}, Ratio: {:.2}",
                              pos1_rms, pos17_rms, pos17_rms / pos1_rms);
                }
            }

        // Layer Norm 1 (Pre-Attention) - Metal
        let ln1_key = format!("blk.{}.attn_norm.weight", layer_idx);
        let ln1_weight = self.weights.get(&ln1_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Layer norm weight not found: {}", ln1_key)))?;

        // Convert gamma weights to f32
        let ln1_gamma_f32: Vec<f32> = ln1_weight.data.iter().map(|&v| v as f32).collect();

        // 🔍 Debug: Dump RMS Norm weight for Layer 0
        if debug && layer_idx == 0 {
            let w_mean: f32 = ln1_gamma_f32.iter().sum::<f32>() / ln1_gamma_f32.len() as f32;
            let w_rms = (ln1_gamma_f32.iter().map(|&v| v * v).sum::<f32>() / ln1_gamma_f32.len() as f32).sqrt();
            let w_min = ln1_gamma_f32.iter().cloned().fold(f32::INFINITY, f32::min);
            let w_max = ln1_gamma_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("\n   🔍 [LAYER 0 RMS NORM WEIGHT] blk.0.attn_norm.weight:");
            eprintln!("      Length: {}", ln1_gamma_f32.len());
            eprintln!("      Stats: mean={:.8}, rms={:.8}, min={:.8}, max={:.8}", w_mean, w_rms, w_min, w_max);
            eprintln!("      First 20 values:");
            for i in 0..20 {
                eprint!("{:.8} ", ln1_gamma_f32[i]);
                if (i + 1) % 5 == 0 { eprintln!(); }
            }
        }

        // Use RMS Norm instead of Layer Norm (TinyLlama uses RMS Norm with eps=1e-5)
        let mut x_ln1 = vec![0.0f32; x_f32.len()];
        Self::rms_norm_f32(&x_f32, &ln1_gamma_f32, &mut x_ln1, seq_len, d_model, 1e-5);

        if debug {
            eprintln!("     ✓ Layer Norm 1 complete");
            // Check RMS Norm output for first 6 layers
            if layer_idx < 6 && seq_len > 17 {
                let pos17_start = 17 * d_model;
                let pos17_input = &x_f32[pos17_start..pos17_start + d_model];
                let pos17_ln1 = &x_ln1[pos17_start..pos17_start + d_model];

                let input_rms = (pos17_input.iter().map(|&v| v * v).sum::<f32>() / d_model as f32).sqrt();
                let pos17_ln1_rms = (pos17_ln1.iter().map(|&v| v * v).sum::<f32>() / d_model as f32).sqrt();

                // Check LN1 weight statistics
                let ln1_weight_mean = ln1_gamma_f32.iter().sum::<f32>() / ln1_gamma_f32.len() as f32;
                let ln1_weight_rms = (ln1_gamma_f32.iter().map(|&v| v * v).sum::<f32>() / ln1_gamma_f32.len() as f32).sqrt();

                eprintln!("   🔍 [LAYER {} LN1] Pos 17: input_rms={:.6}, output_rms={:.6}, weight_rms={:.6}, ratio={:.2}x",
                          layer_idx, input_rms, pos17_ln1_rms, ln1_weight_rms, pos17_ln1_rms / input_rms);
            }
        }

        // Attention Mechanism (Simplified - treating as single-head for now)
        // TODO: Implement full multi-head attention in future phase
        if debug {
            eprintln!("     • Attention (Metal + CPU softmax)");
        }

        // Load Q, K, V, O projection weights
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

        // Convert weights to f32
        let q_weight_f32: Vec<f32> = q_weight.data.iter().map(|&v| v as f32).collect();
        let k_weight_f32: Vec<f32> = k_weight.data.iter().map(|&v| v as f32).collect();
        let v_weight_f32: Vec<f32> = v_weight.data.iter().map(|&v| v as f32).collect();
        let o_weight_f32: Vec<f32> = o_weight.data.iter().map(|&v| v as f32).collect();

        if debug && layer_idx == 0 {
            eprintln!("   📐 [WEIGHT INFO] Layer 0:");
            eprintln!("      Q weight len={}, first 10: {:?}", q_weight_f32.len(), &q_weight_f32[0..10]);
            eprintln!("      Expected Q weight size: {} x {} = {}", d_model, d_model, d_model * d_model);
            eprintln!("      x_ln1 (input) len={}, first 10: {:?}", x_ln1.len(), &x_ln1[0..10.min(x_ln1.len())]);
        }

        // 1. Q, K, V projections (Metal GPU)
        if debug {
            eprintln!("       - Q, K, V projections");
        }

        // For GQA: K/V projections are smaller (num_kv_heads * head_dim)
        let num_kv_heads = self.config.num_kv_heads;
        let num_q_heads = self.config.num_heads;
        let head_dim = d_model / num_q_heads;
        let kv_dim = num_kv_heads * head_dim;

        let mut q_proj = vec![0.0f32; seq_len * d_model];
        let mut k_proj = vec![0.0f32; seq_len * kv_dim];
        let mut v_proj = vec![0.0f32; seq_len * kv_dim];

        if debug && layer_idx == 0 {
            eprintln!("   🔧 [MATMUL PARAMS] Q projection:");
            eprintln!("      Input (x_ln1): shape=[{}, {}], len={}", seq_len, d_model, x_ln1.len());
            eprintln!("      Weight (Q): shape=[{}, {}], len={}", d_model, d_model, q_weight_f32.len());
            eprintln!("      Output (q_proj): shape=[{}, {}], len={}", seq_len, d_model, q_proj.len());
            eprintln!("      Parameters: m={}, n={}, k={}", seq_len, d_model, d_model);
        }

        // Use transposed matmul since GGUF weights are stored as [out_dim, in_dim]
        // We need: output = input @ weight^T
        executor.matmul_transposed_f32(&x_ln1, &q_weight_f32, &mut q_proj, seq_len, d_model, d_model)?;

        if debug && layer_idx == 0 {
            eprintln!("   ✅ [MATMUL OUTPUT] Q projection (with transpose):");
            eprintln!("      q_proj first 10: {:?}", &q_proj[0..10.min(q_proj.len())]);
            eprintln!("      q_proj last 10: {:?}", &q_proj[q_proj.len().saturating_sub(10)..]);
        }

        executor.matmul_transposed_f32(&x_ln1, &k_weight_f32, &mut k_proj, seq_len, kv_dim, d_model)?;
        executor.matmul_transposed_f32(&x_ln1, &v_weight_f32, &mut v_proj, seq_len, kv_dim, d_model)?;

        // Debug: Log Q/K/V projection statistics for first layer
        if debug && layer_idx == 0 {
            let q_mean: f32 = q_proj.iter().sum::<f32>() / q_proj.len() as f32;
            let q_sq_sum: f32 = q_proj.iter().map(|&v| v * v).sum();
            let q_rms = (q_sq_sum / q_proj.len() as f32).sqrt();
            let q_min = q_proj.iter().cloned().fold(f32::INFINITY, f32::min);
            let q_max = q_proj.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let k_mean: f32 = k_proj.iter().sum::<f32>() / k_proj.len() as f32;
            let k_rms = (k_proj.iter().map(|&v| v * v).sum::<f32>() / k_proj.len() as f32).sqrt();
            let k_min = k_proj.iter().cloned().fold(f32::INFINITY, f32::min);
            let k_max = k_proj.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let v_mean: f32 = v_proj.iter().sum::<f32>() / v_proj.len() as f32;
            let v_rms = (v_proj.iter().map(|&v| v * v).sum::<f32>() / v_proj.len() as f32).sqrt();

            eprintln!("   📊 [Q/K/V PROJECTIONS] Layer 0 statistics:");
            eprintln!("      Q_proj: mean={:.6}, rms={:.6}, min={:.6}, max={:.6}", q_mean, q_rms, q_min, q_max);
            eprintln!("      K_proj: mean={:.6}, rms={:.6}, min={:.6}, max={:.6}", k_mean, k_rms, k_min, k_max);
            eprintln!("      V_proj: mean={:.6}, rms={:.6}", v_mean, v_rms);

            // Also log weight statistics
            let q_w_mean: f32 = q_weight_f32.iter().sum::<f32>() / q_weight_f32.len() as f32;
            let q_w_rms = (q_weight_f32.iter().map(|&v| v * v).sum::<f32>() / q_weight_f32.len() as f32).sqrt();
            let q_w_min = q_weight_f32.iter().cloned().fold(f32::INFINITY, f32::min);
            let q_w_max = q_weight_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("      Q_weight: mean={:.6}, rms={:.6}, min={:.6}, max={:.6}", q_w_mean, q_w_rms, q_w_min, q_w_max);
            eprintln!("      Q_weight[0..10]: {:?}", &q_weight_f32[0..10]);
        }

        // 1.5 Apply RoPE to Q and K projections
        if debug {
            eprintln!("       - Apply RoPE (position={})", start_position);
            // Check Pos 17, Head 0 Q values BEFORE RoPE
            if seq_len > 17 {
                let pos17_h0_start = 17 * d_model + 0 * head_dim;
                let pos17_h0_q = &q_proj[pos17_h0_start..pos17_h0_start + head_dim];
                let pos17_q_rms = (pos17_h0_q.iter().map(|&v| v * v).sum::<f32>() / head_dim as f32).sqrt();
                eprintln!("       [BEFORE ROPE] Pos 17, Head 0: Q_rms={:.6}, Q[0..5]={:?}", pos17_q_rms, &pos17_h0_q[0..5]);
            }
        }
        let q_proj = self.apply_rope(&q_proj, seq_len, num_q_heads, head_dim, start_position);
        let k_proj = self.apply_rope(&k_proj, seq_len, num_kv_heads, head_dim, start_position);

        // Debug: Log Q/K after RoPE for first layer
        if debug && layer_idx == 0 {
            let q_rope_mean: f32 = q_proj.iter().sum::<f32>() / q_proj.len() as f32;
            let q_rope_rms = (q_proj.iter().map(|&v| v * v).sum::<f32>() / q_proj.len() as f32).sqrt();
            let k_rope_mean: f32 = k_proj.iter().sum::<f32>() / k_proj.len() as f32;
            let k_rope_rms = (k_proj.iter().map(|&v| v * v).sum::<f32>() / k_proj.len() as f32).sqrt();

            eprintln!("   🔄 [AFTER ROPE] Layer 0:");
            eprintln!("      Q_rope: mean={:.6}, rms={:.6}", q_rope_mean, q_rope_rms);
            eprintln!("      K_rope: mean={:.6}, rms={:.6}", k_rope_mean, k_rope_rms);
            eprintln!("      Q[0..5]: {:?}", &q_proj[0..5]);
            eprintln!("      K[0..5]: {:?}", &k_proj[0..5]);
        }

        // 2. Repeat KV heads to match Q heads for simplified GQA
        // K/V: [seq_len, kv_dim=256] -> [seq_len, d_model=2048]
        if debug {
            eprintln!("       - Repeat KV heads ({} -> {})", kv_dim, d_model);
        }
        let k_expanded = Self::repeat_kv_heads(&k_proj, seq_len, num_kv_heads, num_q_heads, head_dim);
        let v_expanded = Self::repeat_kv_heads(&v_proj, seq_len, num_kv_heads, num_q_heads, head_dim);

        // 3. Multi-head Attention (CPU implementation following hybrid_f32 logic)
        if debug {
            eprintln!("       - Multi-head Attention (CPU-style, num_heads={})", num_q_heads);
            eprintln!("         seq_len={}, d_model={}, head_dim={}", seq_len, d_model, head_dim);
            // Check first few values of Q, K, V
            eprintln!("         Q_proj[0..5]: {:?}", &q_proj[..5.min(q_proj.len())]);
            eprintln!("         K_expanded[0..5]: {:?}", &k_expanded[..5.min(k_expanded.len())]);
            eprintln!("         V_expanded[0..5]: {:?}", &v_expanded[..5.min(v_expanded.len())]);
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_output = vec![0.0f32; seq_len * d_model];

        // For each query position
        eprintln!("   🔄 [UNCONDITIONAL] layer_idx={}, debug={}, seq_len={}", layer_idx, debug, seq_len);
        if debug && layer_idx == 0 {
            eprintln!("   🔄 [LOOP-START] About to loop q_pos from 0 to {} (seq_len={})", seq_len - 1, seq_len);
        }
        for q_pos in 0..seq_len {
            if debug && layer_idx == 0 && q_pos >= seq_len - 2 {
                eprintln!("   🔄 [LOOP] Processing q_pos={} (seq_len={})", q_pos, seq_len);
            }
            // For each query head
            for h in 0..num_q_heads {
                // Get query vector for this head at this position
                let q_start = q_pos * d_model + h * head_dim;
                let q_vec = &q_proj[q_start..q_start + head_dim];

                // Compute attention scores for all key positions
                let mut scores = Vec::with_capacity(q_pos + 1);  // Causal: only attend to current and previous

                for kv_pos in 0..=q_pos {
                    let k_start = kv_pos * d_model + h * head_dim;
                    let k_vec = &k_expanded[k_start..k_start + head_dim];

                    // Dot product: Q · K
                    let score: f32 = q_vec.iter().zip(k_vec.iter())
                        .map(|(&q, &k)| q * k)
                        .sum();

                    scores.push(score * scale);
                }

                // Debug: Log Q_rms for all positions to find anomaly
                if debug && layer_idx == 0 && h == 0 {
                    let q_sq_sum_quick: f32 = q_vec.iter().map(|&v| v * v).sum();
                    let q_rms_quick = (q_sq_sum_quick / head_dim as f32).sqrt();
                    eprintln!("   📊 [Q_RMS] Pos {}: Q_rms={:.6}", q_pos, q_rms_quick);
                }

                let is_last_pos = q_pos == seq_len - 1;
                if debug && layer_idx == 0 && h == 0 && (q_pos == 0 || is_last_pos) {
                    eprintln!("   🎯 [ATTENTION] Layer 0, Head 0, Pos {} {}", q_pos, if is_last_pos { "(LAST)" } else { "" });
                    eprintln!("      q_start index: {}", q_start);
                    eprintln!("      Q[0..5]: {:?}", &q_vec[0..5]);

                    // Log Q and K statistics
                    let q_sq_sum: f32 = q_vec.iter().map(|&v| v * v).sum();
                    let q_rms = (q_sq_sum / head_dim as f32).sqrt();
                    eprintln!("      Q_rms={:.6}, Q_len={}", q_rms, q_vec.len());

                    // Compare with full q_proj first 64 elements (should be same as q_vec if h=0, q_pos=0)
                    let first_64_rms = (q_proj[0..64].iter().map(|&v| v * v).sum::<f32>() / 64.0).sqrt();
                    eprintln!("      q_proj[0..64] rms (should match Q_rms if indexing correct): {:.6}", first_64_rms);

                    // Check other heads to see if they have similar small values
                    for check_h in [0, 1, 15, 31].iter() {
                        let h_start = q_pos * d_model + check_h * head_dim;
                        let h_vec = &q_proj[h_start..h_start + head_dim];
                        let h_rms = (h_vec.iter().map(|&v| v * v).sum::<f32>() / head_dim as f32).sqrt();
                        eprintln!("      Head {} RMS: {:.6}", check_h, h_rms);
                    }

                    // Log first K vector (for kv_pos=0, same head h=0)
                    let kv_pos_0 = 0;
                    let k_start_0 = kv_pos_0 * d_model + 0 * head_dim;  // h=0
                    let k_vec_check = &k_expanded[k_start_0..k_start_0 + head_dim];
                    let k_sq_sum: f32 = k_vec_check.iter().map(|&v| v * v).sum();
                    let k_rms = (k_sq_sum / head_dim as f32).sqrt();
                    eprintln!("      k_start index (for kv_pos=0, h=0): {}", k_start_0);
                    eprintln!("      K[0..5]: {:?}", &k_vec_check[0..5]);
                    eprintln!("      K_rms={:.6}, K_len={}", k_rms, k_vec_check.len());

                    // Calculate expected score magnitude
                    // Expected: Q_rms * K_rms * sqrt(d) WITHOUT scaling
                    let expected_unscaled = q_rms * k_rms * (head_dim as f32).sqrt();
                    // Expected WITH scaling: (Q_rms * K_rms * sqrt(d)) / sqrt(d) = Q_rms * K_rms
                    let expected_scaled = q_rms * k_rms;
                    eprintln!("      Expected unscaled score magnitude: {:.6}", expected_unscaled);
                    eprintln!("      Expected scaled score magnitude: {:.6}", expected_scaled);

                    // Manually calculate dot product for verification
                    let manual_dot: f32 = q_vec.iter().zip(k_vec_check.iter())
                        .map(|(&q, &k)| q * k)
                        .sum();
                    eprintln!("      Manual Q·K dot product (unscaled): {:.9}", manual_dot);
                    eprintln!("      Manual Q·K scaled: {:.9}", manual_dot * scale);

                    eprintln!("      Actual scores (already scaled by 1/sqrt(d)): {:?}", &scores);
                    eprintln!("      Scale factor (1/sqrt(d)): {:.6}", scale);
                }

                // Softmax with temperature scaling for numerical stability
                // Temperature: lower = sharper distribution, higher = softer
                // With value clipping in place, we can use standard temperature
                let temperature = 1.0;

                // Scale scores by temperature before softmax
                let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();

                if debug && layer_idx == 0 && h == 0 && is_last_pos {
                    eprintln!("      Raw scores range: [{:.6}, {:.6}]",
                             scores.iter().cloned().fold(f32::INFINITY, f32::min),
                             scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
                }

                let max_score = scaled_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scaled_scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();

                if debug && layer_idx == 0 && h == 0 && is_last_pos {
                    eprintln!("      After max subtraction: [{:.6}, 0.0]",
                             scaled_scores.iter().map(|&s| s - max_score).fold(f32::INFINITY, f32::min));
                    eprintln!("      Exp scores range: [{:.9}, {:.9}]",
                             exp_scores.iter().cloned().fold(f32::INFINITY, f32::min),
                             exp_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
                    eprintln!("      Sum of exp scores: {:.9}", sum_exp);
                }

                let attn_weights: Vec<f32> = if sum_exp <= 0.0 || !sum_exp.is_finite() {
                    // Fallback: uniform weights
                    if debug && layer_idx == 0 && h == 0 && (q_pos == 0 || is_last_pos) {
                        eprintln!("      ⚠️  Softmax fallback triggered! sum_exp={}", sum_exp);
                    }
                    vec![1.0 / scores.len() as f32; scores.len()]
                } else {
                    exp_scores.iter().map(|&e| e / sum_exp).collect()
                };

                if debug && layer_idx == 0 && h == 0 && (q_pos == 0 || is_last_pos) {
                    eprintln!("      Attention weights (after softmax): {:?}", &attn_weights);

                    // Log first V vector statistics
                    let v_start = 0;
                    let v_vec = &v_expanded[v_start..v_start + head_dim];
                    let v_sq_sum: f32 = v_vec.iter().map(|&v| v * v).sum();
                    let v_rms = (v_sq_sum / head_dim as f32).sqrt();
                    eprintln!("      V[0..5]: {:?}", &v_vec[0..5]);
                    eprintln!("      V_rms={:.6}", v_rms);
                }

                // Weighted sum of values (output for this head at this position)
                for dim in 0..head_dim {
                    let mut weighted_sum = 0.0f32;
                    for (weight_idx, kv_pos) in (0..=q_pos).enumerate() {
                        let v_start = kv_pos * d_model + h * head_dim;
                        weighted_sum += attn_weights[weight_idx] * v_expanded[v_start + dim];
                    }
                    attn_output[q_pos * d_model + h * head_dim + dim] = weighted_sum;
                }
            }
        }

        // 🔍 Layer 0 Attention Output detailed dump
        if debug && layer_idx == 0 {
            eprintln!("\n   🔍 [LAYER 0 ATTENTION OUTPUT] Before output projection:");
            let last_pos = (seq_len - 1) * d_model;
            let attn_out_last = &attn_output[last_pos..last_pos + d_model];
            eprintln!("      Position {}, first 20 values:", seq_len - 1);
            for i in 0..20 {
                eprint!("{:.8} ", attn_out_last[i]);
                if (i + 1) % 5 == 0 { eprintln!(); }
            }
            let rms = (attn_out_last.iter().map(|&v| v*v).sum::<f32>() / d_model as f32).sqrt();
            eprintln!("      RMS: {:.8}", rms);

            // Output projection weight statistics
            let o_w_mean: f32 = o_weight_f32.iter().sum::<f32>() / o_weight_f32.len() as f32;
            let o_w_rms = (o_weight_f32.iter().map(|&v| v * v).sum::<f32>() / o_weight_f32.len() as f32).sqrt();
            let o_w_min = o_weight_f32.iter().cloned().fold(f32::INFINITY, f32::min);
            let o_w_max = o_weight_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("      O_weight (Q6_K): mean={:.8}, rms={:.8}, min={:.8}, max={:.8}", o_w_mean, o_w_rms, o_w_min, o_w_max);
            eprintln!("      O_weight[0..20]: {:?}", &o_weight_f32[0..20]);
        }

        // 7. Output projection (Metal GPU)
        if debug {
            eprintln!("       - Output projection");
        }
        let mut x_post_attn = vec![0.0f32; seq_len * d_model];
        // Use transposed matmul since GGUF weights are [out_dim, in_dim]
        executor.matmul_transposed_f32(&attn_output, &o_weight_f32, &mut x_post_attn, seq_len, d_model, d_model)?;

        // 🔍 Layer 0 After Output Projection
        if debug && layer_idx == 0 {
            eprintln!("\n   🔍 [LAYER 0 AFTER OUTPUT PROJECTION]:");
            let last_pos = (seq_len - 1) * d_model;
            let post_proj = &x_post_attn[last_pos..last_pos + d_model];
            eprintln!("      Position {}, first 20 values:", seq_len - 1);
            for i in 0..20 {
                eprint!("{:.8} ", post_proj[i]);
                if (i + 1) % 5 == 0 { eprintln!(); }
            }
            let rms = (post_proj.iter().map(|&v| v*v).sum::<f32>() / d_model as f32).sqrt();
            eprintln!("      RMS: {:.8}", rms);
        }

        if debug {
            eprintln!("     ✓ Attention complete");
        }

        // Residual connection 1: x = x + attention_output
        if debug {
            eprintln!("     • Residual connection 1 (Metal)");
        }
        let mut x_residual1 = vec![0.0f32; x_f32.len()];
        executor.elementwise_add_f32(&x_f32, &x_post_attn, &mut x_residual1)?;

        // 📊 Track residual accumulation in Layer 0
        if debug && layer_idx == 0 {
            let last_pos = (seq_len - 1) * d_model;
            let x_before = &x_f32[last_pos..last_pos + d_model];
            let attn_delta = &x_post_attn[last_pos..last_pos + d_model];
            let x_after = &x_residual1[last_pos..last_pos + d_model];

            let x_before_rms = (x_before.iter().map(|&v| v*v).sum::<f32>() / d_model as f32).sqrt();
            let attn_delta_rms = (attn_delta.iter().map(|&v| v*v).sum::<f32>() / d_model as f32).sqrt();
            let x_after_rms = (x_after.iter().map(|&v| v*v).sum::<f32>() / d_model as f32).sqrt();

            eprintln!("   📊 [RESIDUAL 1] Layer {}, pos {}:", layer_idx, seq_len - 1);
            eprintln!("      Before (x): RMS={:.6}, first 5: {:?}", x_before_rms, &x_before[0..5]);
            eprintln!("      Delta (attn): RMS={:.6}, first 5: {:?}", attn_delta_rms, &attn_delta[0..5]);
            eprintln!("      After (x+attn): RMS={:.6}, first 5: {:?}", x_after_rms, &x_after[0..5]);
        }

        if debug {
            eprintln!("     ✓ Residual 1 complete");
        }

        // Layer Norm 2 (Pre-FFN) - Metal
        if debug {
            eprintln!("     • Layer Norm 2 (Metal)");
        }
        let ln2_key = format!("blk.{}.ffn_norm.weight", layer_idx);
        let ln2_weight = self.weights.get(&ln2_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Layer norm weight not found: {}", ln2_key)))?;
        let ln2_gamma_f32: Vec<f32> = ln2_weight.data.iter().map(|&v| v as f32).collect();

        // Use RMS Norm instead of Layer Norm
        let mut x_ln2 = vec![0.0f32; x_residual1.len()];
        Self::rms_norm_f32(&x_residual1, &ln2_gamma_f32, &mut x_ln2, seq_len, d_model, 1e-5);
        if debug {
            eprintln!("     ✓ Layer Norm 2 complete");
        }

        // Feed-Forward Network with Metal
        // FFN structure: down_proj(GELU(gate_proj(x)) * up_proj(x))
        if debug {
            eprintln!("     • Feed-Forward Network (Metal)");
        }

        // Load FFN weights
        let gate_key = format!("blk.{}.ffn_gate.weight", layer_idx);
        let up_key = format!("blk.{}.ffn_up.weight", layer_idx);
        let down_key = format!("blk.{}.ffn_down.weight", layer_idx);

        let gate_weight = self.weights.get(&gate_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Gate weight not found: {}", gate_key)))?;
        let up_weight = self.weights.get(&up_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Up weight not found: {}", up_key)))?;
        let down_weight = self.weights.get(&down_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Down weight not found: {}", down_key)))?;

        // Convert weights to f32
        let gate_weight_f32: Vec<f32> = gate_weight.data.iter().map(|&v| v as f32).collect();
        let up_weight_f32: Vec<f32> = up_weight.data.iter().map(|&v| v as f32).collect();
        let down_weight_f32: Vec<f32> = down_weight.data.iter().map(|&v| v as f32).collect();

        // Calculate actual d_ff from gate weight size (gate_weight: [d_ff, d_model])
        let actual_d_ff = gate_weight_f32.len() / d_model;

        if debug {
            eprintln!("       FFN weights: gate={}, up={}, down={}",
                      gate_weight_f32.len(), up_weight_f32.len(), down_weight_f32.len());
            eprintln!("       Calculated d_ff={} (from gate_weight.len={} / d_model={})",
                      actual_d_ff, gate_weight_f32.len(), d_model);
        }

        let d_ff = actual_d_ff;

        // 1. Gate projection: x @ gate_weight^T
        // x_ln2: [seq_len * d_model], gate_weight: [d_ff, d_model] (transposed in GGUF)
        // Result: [seq_len, d_ff]
        if debug {
            eprintln!("   🔶 [FFN GATE] Layer {}:", layer_idx);
            if layer_idx == 0 {
                eprintln!("      Gate weight len={}, first 10: {:?}", gate_weight_f32.len(), &gate_weight_f32[0..10]);
                eprintln!("      Input (x_ln2) len={}, first 10: {:?}", x_ln2.len(), &x_ln2[0..10.min(x_ln2.len())]);
                eprintln!("      Matmul params: m={}, n={}, k={}", seq_len, d_ff, d_model);
            }
        }
        if debug {
            eprintln!("       - Gate projection: [{}, {}] @ [{}, {}]^T", seq_len, d_model, d_ff, d_model);
        }
        let mut gate_out = vec![0.0f32; seq_len * d_ff];
        // Use transposed matmul since GGUF weights are [out_dim, in_dim]
        executor.matmul_transposed_f32(&x_ln2, &gate_weight_f32, &mut gate_out, seq_len, d_ff, d_model)?;

        if debug {
            if layer_idx == 0 {
                eprintln!("      Gate output first 10: {:?}", &gate_out[0..10]);
                eprintln!("      Gate output last 10: {:?}", &gate_out[gate_out.len().saturating_sub(10)..]);
            }
        }

        // 2. Apply GELU to gate output
        if debug {
            eprintln!("       - GELU activation");
        }
        let mut gate_activated = vec![0.0f32; gate_out.len()];
        executor.gelu_f32(&gate_out, &mut gate_activated)?;

        // 3. Up projection: x @ up_weight^T
        if debug {
            eprintln!("       - Up projection: [{}, {}] @ [{}, {}]^T", seq_len, d_model, d_ff, d_model);
        }
        let mut up_out = vec![0.0f32; seq_len * d_ff];
        // Use transposed matmul since GGUF weights are [out_dim, in_dim]
        executor.matmul_transposed_f32(&x_ln2, &up_weight_f32, &mut up_out, seq_len, d_ff, d_model)?;

        // 4. Element-wise multiply: gate_activated * up_out
        if debug {
            eprintln!("       - Element-wise multiply");
        }
        let mut ffn_intermediate = vec![0.0f32; gate_activated.len()];
        executor.elementwise_mul_f32(&gate_activated, &up_out, &mut ffn_intermediate)?;

        // 5. Down projection: ffn_intermediate @ down_weight^T
        // down_weight: [d_model, d_ff] (transposed in GGUF)
        // Result: [seq_len, d_model]
        if debug {
            eprintln!("       - Down projection: [{}, {}] @ [{}, {}]^T", seq_len, d_ff, d_model, d_ff);
        }
        let mut ffn_out = vec![0.0f32; seq_len * d_model];
        // Use transposed matmul since GGUF weights are [out_dim, in_dim]
        executor.matmul_transposed_f32(&ffn_intermediate, &down_weight_f32, &mut ffn_out, seq_len, d_model, d_ff)?;

        // 🔍 Layer 0 FFN Output detailed dump
        if debug && layer_idx == 0 {
            eprintln!("\n   🔍 [LAYER 0 FFN OUTPUT]:");
            let last_pos = (seq_len - 1) * d_model;
            let ffn_last = &ffn_out[last_pos..last_pos + d_model];
            eprintln!("      Position {}, first 20 values:", seq_len - 1);
            for i in 0..20 {
                eprint!("{:.8} ", ffn_last[i]);
                if (i + 1) % 5 == 0 { eprintln!(); }
            }
            let rms = (ffn_last.iter().map(|&v| v*v).sum::<f32>() / d_model as f32).sqrt();
            eprintln!("      RMS: {:.8}", rms);
        }

        if debug {
            eprintln!("     ✓ FFN complete");
        }

        // Residual connection 2: x = x_residual1 + ffn_out
        if debug {
            eprintln!("     • Residual connection 2 (Metal)");
        }
        let mut x_residual2 = vec![0.0f32; x_residual1.len()];
        executor.elementwise_add_f32(&x_residual1, &ffn_out, &mut x_residual2)?;

        // 📊 Track residual 2 accumulation in Layer 0
        if debug && layer_idx == 0 {
            let last_pos = (seq_len - 1) * d_model;
            let x_before = &x_residual1[last_pos..last_pos + d_model];
            let ffn_delta = &ffn_out[last_pos..last_pos + d_model];
            let x_after_no_clip = &x_residual2[last_pos..last_pos + d_model];

            let x_before_rms = (x_before.iter().map(|&v| v*v).sum::<f32>() / d_model as f32).sqrt();
            let ffn_delta_rms = (ffn_delta.iter().map(|&v| v*v).sum::<f32>() / d_model as f32).sqrt();
            let x_after_rms = (x_after_no_clip.iter().map(|&v| v*v).sum::<f32>() / d_model as f32).sqrt();

            eprintln!("   📊 [RESIDUAL 2] Layer {}, pos {} (before clipping):", layer_idx, seq_len - 1);
            eprintln!("      Before (x_res1): RMS={:.6}, first 5: {:?}", x_before_rms, &x_before[0..5]);
            eprintln!("      Delta (ffn): RMS={:.6}, first 5: {:?}", ffn_delta_rms, &ffn_delta[0..5]);
            eprintln!("      After (x_res1+ffn): RMS={:.6}, first 5: {:?}", x_after_rms, &x_after_no_clip[0..5]);
        }

        // Value clipping to prevent numerical instability from 93x amplification
        // Clip to reasonable range to prevent softmax collapse in later layers
        let clip_max = 10.0f32;
        let clip_min = -10.0f32;
        for val in x_residual2.iter_mut() {
            *val = val.clamp(clip_min, clip_max);
        }

        // Track clipping effect in Layer 0
        if debug && layer_idx == 0 {
            let last_pos = (seq_len - 1) * d_model;
            let x_after_clip = &x_residual2[last_pos..last_pos + d_model];
            let x_after_rms = (x_after_clip.iter().map(|&v| v*v).sum::<f32>() / d_model as f32).sqrt();
            eprintln!("      After clipping: RMS={:.6}, first 5: {:?}", x_after_rms, &x_after_clip[0..5]);
        }

        if debug {
            eprintln!("     ✓ Residual 2 complete (with clipping)");

            // Log RMS after clipping for key positions and layers
            // Track every 5 layers + first and last
            if (layer_idx == 0 || layer_idx == 5 || layer_idx == 10 || layer_idx == 15 || layer_idx == 20 || layer_idx == 21) && seq_len > 1 {
                let pos1_start = 1.min(seq_len - 1) * d_model;
                let pos1_clipped = &x_residual2[pos1_start..pos1_start + d_model];
                let pos1_rms = (pos1_clipped.iter().map(|&v| v * v).sum::<f32>() / d_model as f32).sqrt();

                // Check max absolute value to verify clipping is working
                let max_abs = pos1_clipped.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);

                eprintln!("   🔒 [Layer {}] Pos 1: RMS={:.6}, max_abs={:.6}", layer_idx, pos1_rms, max_abs);

                // For last position if available
                if seq_len > 17 {
                    let pos17_start = 17 * d_model;
                    let pos17_clipped = &x_residual2[pos17_start..pos17_start + d_model];
                    let pos17_rms = (pos17_clipped.iter().map(|&v| v * v).sum::<f32>() / d_model as f32).sqrt();
                    let max_abs_17 = pos17_clipped.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
                    eprintln!("   🔒 [Layer {}] Pos 17: RMS={:.6}, max_abs={:.6}", layer_idx, pos17_rms, max_abs_17);
                }
            }
        }

            // 🔬 [LAYER OUTPUT] Debug: dump layer outputs for tracking divergence
            if debug && (layer_idx == 0 || layer_idx == 5 || layer_idx == 10 || layer_idx == 15 || layer_idx == 20 || layer_idx == 21) {
                let last_token_start = (seq_len - 1) * d_model;
                let last_token_out = &x_residual2[last_token_start..last_token_start + d_model];
                let mean: f32 = last_token_out.iter().sum::<f32>() / d_model as f32;
                let sq_sum: f32 = last_token_out.iter().map(|&v| v * v).sum();
                let rms = (sq_sum / d_model as f32).sqrt();
                let max_val = last_token_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min_val = last_token_out.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!("🔬 [LAYER {} OUTPUT] Last token (pos={}): first 10: {:?}", layer_idx, seq_len - 1, &last_token_out[0..10]);
                eprintln!("   📊 Stats: mean={:.9}, rms={:.9}, min={:.6}, max={:.6}", mean, rms, min_val, max_val);
            }

            // Update x_f32 for next layer
            x_f32 = x_residual2;
        }

        if debug {
            eprintln!("   ✅ All {} transformer layers complete", num_layers);
        }

        // 3. Final Layer Normalization
        if debug {
            eprintln!("   • Final Layer Norm (Metal)");
        }
        let output_norm_key = "output_norm.weight";
        let output_norm_weight = self.weights.get(output_norm_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Output norm weight not found: {}", output_norm_key)))?;

        if debug {
            eprintln!("   • Converting output_norm to f32...");
        }
        let output_norm_gamma_f32: Vec<f32> = output_norm_weight.data.iter().map(|&v| v as f32).collect();

        if debug {
            eprintln!("   • Applying final RMS Norm...");
        }
        // Use RMS Norm instead of Layer Norm
        let mut x_final_norm = vec![0.0f32; x_f32.len()];
        Self::rms_norm_f32(&x_f32, &output_norm_gamma_f32, &mut x_final_norm, seq_len, d_model, 1e-5);
        if debug {
            eprintln!("   ✓ Final RMS Norm complete");
            eprintln!("   • Getting LM head weight...");
        }

        // LM Head projection: hidden states -> logits
        let (lm_head_weight, lm_head_key) = if let Some(w) = self.weights.get("output.weight") {
            (w, "output.weight")
        } else if let Some(w) = self.weights.get("lm_head.weight") {
            (w, "lm_head.weight")
        } else if let Some(w) = self.weights.get("token_embd.weight") {
            (w, "token_embd.weight")
        } else {
            return Err(RusTorchError::tensor_op("LM head weight not found".to_string()));
        };

        if debug {
            eprintln!("   ✓ LM head weight retrieved from key: {}", lm_head_key);
        }

        let vocab_size = self.config.vocab_size;

        // Get last token's hidden state
        let last_token_start = (seq_len - 1) * d_model;
        let last_hidden = &x_final_norm[last_token_start..last_token_start + d_model];

        if debug {
            eprintln!("   🎯 [LAST HIDDEN] Token pos={}, first 10: {:?}", seq_len - 1, &last_hidden[0..10]);
            eprintln!("   🎯 [LAST HIDDEN] Last 10: {:?}", &last_hidden[last_hidden.len().saturating_sub(10)..]);
            let last_hidden_rms = (last_hidden.iter().map(|&v| v * v).sum::<f32>() / last_hidden.len() as f32).sqrt();
            eprintln!("   🎯 [LAST HIDDEN] RMS={:.6}", last_hidden_rms);
        }

        // Compute logits: last_hidden @ lm_head^T -> [vocab_size]
        // lm_head shape: [hidden_size, vocab_size] stored in row-major
        let lm_head_data = &lm_head_weight.data;
        let lm_head_shape = lm_head_data.shape();

        if debug {
            eprintln!("🔍 [LM_HEAD] Shape: {:?}, d_model={}, vocab_size={}", lm_head_shape, d_model, vocab_size);
            eprintln!("🔍 [LM_HEAD] Data len: {}, expected: {}", lm_head_data.len(), d_model * vocab_size);

            // Test memory layout hypothesis
            // If ndarray interprets shape [2048, 32000] as row-major:
            //   [[h, v]] → h * 32000 + v
            // If GGUF layout is [token0(2048), token1(2048), ...]:
            //   Token v, element h → v * 2048 + h

            eprintln!("🧪 [LAYOUT TEST] ndarray 2D access [[0, 0]]: value = {:.6}", lm_head_data[[0, 0]]);
            eprintln!("🧪 [LAYOUT TEST] ndarray 2D access [[1, 0]]: value = {:.6}", lm_head_data[[1, 0]]);
            eprintln!("🧪 [LAYOUT TEST] ndarray 2D access [[0, 1]]: value = {:.6}", lm_head_data[[0, 1]]);

            // Sample LM head weights for first few vocab tokens
            eprintln!("🔍 [LM_HEAD] Weight samples for token 0 (first 10): {:?}",
                (0..10).map(|h| lm_head_data[[h, 0]]).collect::<Vec<_>>());
            eprintln!("🔍 [LM_HEAD] Weight samples for token 1 (first 10): {:?}",
                (0..10).map(|h| lm_head_data[[h, 1]]).collect::<Vec<_>>());
        }

        let mut logits = vec![0.0f32; vocab_size];

        if debug {
            eprintln!("   • Computing logits...");
        }

        // Access lm_head_data as 2D array [d_model, vocab_size]
        for v in 0..vocab_size {
            let mut sum = 0.0f64;
            for h in 0..d_model {
                sum += (last_hidden[h] as f64) * lm_head_data[[h, v]];
            }
            logits[v] = sum as f32;
        }

        if debug {
            eprintln!("   ✓ Logits computed");
        }

        // Always log logits stats for verification
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let mean_logit: f32 = logits.iter().sum::<f32>() / logits.len() as f32;

        eprintln!("🔍 [LOGITS] Stats: max={:.4}, min={:.4}, mean={:.4}", max_logit, min_logit, mean_logit);

        // Show top-5 logits
        let mut indexed_logits: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("🔍 [LOGITS] Top-5 tokens:");
        for (i, &(token_id, logit_val)) in indexed_logits.iter().take(5).enumerate() {
            eprintln!("   {}. Token {}: {:.4}", i + 1, token_id, logit_val);
        }

        // Check specific token logits for debugging
        if debug {
            eprintln!("🔍 [LOGITS] Specific tokens:");
            for &token_id in &[29896, 9134, 1, 2, 0] {
                if token_id < logits.len() {
                    eprintln!("   Token {}: {:.4}", token_id, logits[token_id]);
                }
            }

            // Manual verification for token 29896
            let token_id = 29896;
            let mut manual_logit = 0.0f64;
            for h in 0..10 {
                manual_logit += (last_hidden[h] as f64) * lm_head_data[[h, token_id]];
            }
            eprintln!("🧮 [MANUAL] Token {} partial logit (first 10 hidden dims): {:.6}", token_id, manual_logit);

            // Full manual calculation
            let mut full_manual_logit = 0.0f64;
            for h in 0..d_model {
                full_manual_logit += (last_hidden[h] as f64) * lm_head_data[[h, token_id]];
            }
            eprintln!("🧮 [MANUAL] Token {} full logit: {:.6}", token_id, full_manual_logit);
            eprintln!("🧮 [MANUAL] Difference from computed: {:.6}", (full_manual_logit as f32 - logits[token_id]).abs());
        }

        // Convert logits to f64 and create tensor [1, 1, vocab_size]
        let output_data: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
        let output_tensor = Tensor::from_vec(output_data, vec![batch_size, 1, vocab_size]);

        eprintln!("✅ Metal forward pass complete");

        Ok(output_tensor)
    }

    /// Convert Tensor<f64> to Vec<f32> for Metal processing
    /// Tensor<f64>をMetal処理用のVec<f32>に変換
    #[cfg(feature = "metal")]
    fn tensor_to_f32_vec(tensor: &Tensor<f64>) -> Vec<f32> {
        tensor.data.iter().map(|&x| x as f32).collect()
    }

    /// Convert Vec<f32> to Tensor<f64> after Metal processing
    /// Metal処理後のVec<f32>をTensor<f64>に変換
    #[cfg(feature = "metal")]
    fn f32_vec_to_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f64> {
        let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        Tensor::from_vec(data_f64, shape)
    }

    /// Transpose 2D matrix stored as flattened 1D array
    /// 1D配列として格納された2D行列を転置
    #[cfg(feature = "metal")]
    fn transpose_2d_f32(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                output[j * rows + i] = input[i * cols + j];
            }
        }
        output
    }

    /// Row-wise softmax for 2D matrix
    /// 2D行列の行ごとのsoftmax
    #[cfg(feature = "metal")]
    fn softmax_2d_f32(data: &mut [f32], rows: usize, cols: usize) {
        for i in 0..rows {
            let row_start = i * cols;
            let row_end = row_start + cols;
            let row = &mut data[row_start..row_end];

            // Find max for numerical stability
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Compute exp and sum
            let mut sum_exp = 0.0f32;
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
                sum_exp += *val;
            }

            // Normalize
            for val in row.iter_mut() {
                *val /= sum_exp;
            }
        }
    }

    /// Reshape and split tensor for multi-head attention
    /// [seq_len, d_model] -> [seq_len, num_heads, head_dim]
    #[cfg(feature = "metal")]
    fn reshape_for_heads(input: &[f32], seq_len: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
        // Input: [seq_len * (num_heads * head_dim)]
        // Output: [seq_len * num_heads * head_dim] (same data, conceptually reshaped)
        input.to_vec()
    }

    /// Repeat KV heads for Grouped Query Attention
    /// Repeats each KV head (num_q_heads / num_kv_heads) times
    /// RMS Norm (Root Mean Square Layer Normalization) - CPU implementation
    /// RMSノルム（二乗平均平方根正規化）- CPU実装
    #[cfg(feature = "metal")]
    fn rms_norm_f32(
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) {
        let debug = std::env::var("RUSTORCH_DEBUG").is_ok();

        for seq_idx in 0..seq_len {
            let offset = seq_idx * hidden_size;
            let row = &input[offset..offset + hidden_size];

            // Compute RMS (Root Mean Square)
            let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / (hidden_size as f32);
            let rms = (mean_sq + eps).sqrt();

            // 🔥 Debug: Log first RMS Norm computation (Layer 0, Position 0)
            if debug && seq_idx == 0 && seq_len > 1 {
                eprintln!("\n      🔥 [RMS_NORM FIRST CALL] Pos 0:");
                eprintln!("         Input[0..10]: {:?}", &row[0..10]);
                eprintln!("         mean_sq: {:.12}", mean_sq);
                eprintln!("         eps: {:.12}", eps);
                eprintln!("         rms: {:.12}", rms);
                eprintln!("         1/rms: {:.12}", 1.0 / rms);
            }

            // Normalize and scale with weight
            // Use pre-computed scale to match llama.cpp's multiplication order
            let scale = 1.0 / rms;
            for i in 0..hidden_size {
                output[offset + i] = row[i] * scale * weight[i];
            }

            // 🧪 Debug: Verify normalization step for Layer 0, Pos 17
            if debug && seq_idx == 17 && seq_len > 17 {
                // Calculate RMS of normalized vector (before weight multiplication)
                let mut normalized_rms_sq = 0.0f32;
                for &v in row.iter() {
                    let normalized_v = v / rms;
                    normalized_rms_sq += normalized_v * normalized_v;
                }
                let normalized_rms = (normalized_rms_sq / hidden_size as f32).sqrt();

                eprintln!("      🧪 [RMS_NORM DEBUG] Pos 17:");
                eprintln!("         hidden_size parameter: {}", hidden_size);
                eprintln!("         weight.len(): {}", weight.len());
                eprintln!("         row.len(): {}", row.len());
                eprintln!("         Input RMS: {:.9}", row.iter().map(|&v| v*v).sum::<f32>() / (hidden_size as f32));
                eprintln!("         Divisor (rms + eps): {:.9}", rms);
                eprintln!("         Normalized RMS (should be ≈1.0): {:.9}", normalized_rms);
                eprintln!("         Weight[0..5]: {:?}", &weight[0..5]);
                eprintln!("         Output[0..5]: {:?}", &output[offset..offset+5]);
            }

            // Debug: Log Pos 17 RMS Norm computation details
            if debug && seq_idx == 17 && seq_len > 17 {
                let input_rms_actual = row.iter().map(|&v| v * v).sum::<f32>() / (hidden_size as f32);
                let input_rms_actual = input_rms_actual.sqrt();
                let output_rms_actual = output[offset..offset + hidden_size].iter().map(|&v| v * v).sum::<f32>() / (hidden_size as f32);
                let output_rms_actual = output_rms_actual.sqrt();
                eprintln!("      🔬 [RMS_NORM Pos 17] input_rms={:.6}, rms_divisor={:.6}, output_rms={:.6}",
                          input_rms_actual, rms, output_rms_actual);
            }

            // Debug: Log first position RMS Norm stats
            if debug && seq_idx == 0 {
                // Calculate actual input RMS for comparison
                let input_sq_sum: f32 = row.iter().map(|&v| v * v).sum();
                let input_actual_rms = (input_sq_sum / hidden_size as f32).sqrt();

                // 🧮 Precision check: Compare f32 vs f64 computation
                let input_sq_sum_f64: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let mean_sq_f64 = input_sq_sum_f64 / (hidden_size as f64);
                let rms_f64 = (mean_sq_f64 + (eps as f64)).sqrt();
                let rms_diff = ((rms_f64 as f32) - rms).abs();
                eprintln!("   🧮 [PRECISION] RMS: f32={:.9}, f64={:.9}, diff={:.12}", rms, rms_f64, rms_diff);

                let out_row = &output[offset..offset + hidden_size];
                let out_mean: f32 = out_row.iter().sum::<f32>() / hidden_size as f32;
                let out_sq_sum: f32 = out_row.iter().map(|&v| v * v).sum();
                let out_rms = (out_sq_sum / hidden_size as f32).sqrt();
                let weight_mean: f32 = weight.iter().sum::<f32>() / weight.len() as f32;
                let weight_rms: f32 = (weight.iter().map(|&v| v * v).sum::<f32>() / weight.len() as f32).sqrt();

                eprintln!("   🔧 [RMS_NORM] pos={}:", seq_idx);
                eprintln!("      Input actual RMS={:.6}", input_actual_rms);
                eprintln!("      Normalization RMS (sqrt(mean(x²)+eps))={:.6}", rms);
                eprintln!("      Output RMS={:.6}", out_rms);
                eprintln!("      Weight: mean={:.6}, rms={:.6}", weight_mean, weight_rms);
                eprintln!("      Input[0..5]: {:?}", &row[0..5]);
                eprintln!("      Output[0..5]: {:?}", &out_row[0..5]);
            }
        }
    }

    /// Apply RoPE (Rotary Position Embedding) to Q or K projections
    /// RoPE（回転位置埋め込み）をQ/K投影に適用
    #[cfg(feature = "metal")]
    fn apply_rope(
        &self,
        x: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        start_position: usize,
    ) -> Vec<f32> {
        let debug = std::env::var("RUSTORCH_DEBUG").is_ok();
        let total_dim = num_heads * head_dim;
        let mut output = Vec::with_capacity(x.len());

        // Apply rotation for each token in sequence
        for token_idx in 0..seq_len {
            let position = start_position + token_idx;

            // For each head of this token
            for head_idx in 0..num_heads {
                let head_offset = token_idx * total_dim + head_idx * head_dim;
                let head_data = &x[head_offset..head_offset + head_dim];

                for i in 0..(head_dim / 2) {
                    let rope_idx = position * (head_dim / 2) + i;

                    let cos = self.rope_cos[rope_idx];
                    let sin = self.rope_sin[rope_idx];

                    // Debug: Log ALL tokens for first head, first rotation pair
                    if debug && head_idx == 0 && i == 0 {
                        eprintln!("   🌀 [ROPE] token_idx={}, pos={}, head={}, pair={}, rope_idx={}", token_idx, position, head_idx, i, rope_idx);
                    }

                    let x0 = head_data[2 * i];
                    let x1 = head_data[2 * i + 1];

                    // Rotate: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
                    let rotated_0 = x0 * cos - x1 * sin;
                    let rotated_1 = x0 * sin + x1 * cos;

                    if debug && token_idx == 0 && head_idx == 0 && i == 0 {
                        eprintln!("      After: rotated_0={:.6}, rotated_1={:.6}", rotated_0, rotated_1);
                    }

                    output.push(rotated_0);
                    output.push(rotated_1);
                }
            }
        }

        output
    }

    /// [seq_len, num_kv_heads, head_dim] -> [seq_len, num_q_heads, head_dim]
    #[cfg(feature = "metal")]
    fn repeat_kv_heads(
        input: &[f32],
        seq_len: usize,
        num_kv_heads: usize,
        num_q_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let repeat_factor = num_q_heads / num_kv_heads;
        let mut output = Vec::with_capacity(seq_len * num_q_heads * head_dim);

        for s in 0..seq_len {
            for kv_h in 0..num_kv_heads {
                // Repeat this KV head `repeat_factor` times
                for _ in 0..repeat_factor {
                    let src_offset = (s * num_kv_heads + kv_h) * head_dim;
                    for d in 0..head_dim {
                        output.push(input[src_offset + d]);
                    }
                }
            }
        }

        output
    }

    /// Concatenate attention heads
    /// [seq_len, num_heads, head_dim] -> [seq_len, num_heads * head_dim]
    #[cfg(feature = "metal")]
    fn concat_heads(head_outputs: &[Vec<f32>], seq_len: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
        // Concatenate head outputs: [num_heads][seq_len * head_dim] -> [seq_len * (num_heads * head_dim)]
        let mut output = Vec::with_capacity(seq_len * num_heads * head_dim);

        // Each head_output is [seq_len, head_dim] stored as seq_len * head_dim elements
        for s in 0..seq_len {
            for h in 0..num_heads {
                let head_offset = s * head_dim;
                let head_data = &head_outputs[h];

                // Safety check
                if head_offset + head_dim > head_data.len() {
                    eprintln!("⚠️  concat_heads: index out of bounds! s={}, h={}, head_offset={}, head_dim={}, head_data.len()={}",
                              s, h, head_offset, head_dim, head_data.len());
                    // Pad with zeros if out of bounds
                    for _ in 0..head_dim {
                        output.push(0.0);
                    }
                    continue;
                }

                // Copy head_dim values
                output.extend_from_slice(&head_data[head_offset..head_offset + head_dim]);
            }
        }

        output
    }

    /// Test Metal matmul operation with simple matrix
    /// 簡単な行列でMetal matmul演算をテスト
    #[cfg(feature = "metal")]
    fn test_metal_matmul() -> RusTorchResult<()> {
        use crate::gpu::metal_kernels::MetalKernelExecutor;

        eprintln!("🧪 Testing Metal matmul operation...");

        // Get Metal executor
        let executor_mutex = MetalKernelExecutor::get()?;
        let executor_guard = executor_mutex.lock().unwrap();
        let executor = executor_guard.as_ref()
            .ok_or_else(|| RusTorchError::tensor_op("Metal executor not initialized"))?;

        // Test data: 2x3 matrix × 3x2 matrix = 2x2 matrix
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let mut c = vec![0.0f32; 4]; // 2x2

        // Execute Metal matmul: matmul_f32(a, b, c, m, n, k) where A is m×k, B is k×n, C is m×n
        executor.matmul_f32(&a, &b, &mut c, 2, 2, 3)?;

        eprintln!("✅ Metal matmul test passed");
        eprintln!("   Input A (2x3): [{}, {}, {}], [{}, {}, {}]",
            a[0], a[1], a[2], a[3], a[4], a[5]);
        eprintln!("   Input B (3x2): [{}, {}], [{}, {}], [{}, {}]",
            b[0], b[1], b[2], b[3], b[4], b[5]);
        eprintln!("   Result C (2x2): [{}, {}], [{}, {}]",
            c[0], c[1], c[2], c[3]);

        // Verify result (expected: [22, 28], [49, 64])
        let expected = vec![22.0f32, 28.0, 49.0, 64.0];
        let epsilon = 0.001;
        for i in 0..4 {
            if (c[i] - expected[i]).abs() > epsilon {
                return Err(RusTorchError::tensor_op(
                    format!("Metal matmul result mismatch at index {}: got {}, expected {}",
                        i, c[i], expected[i])
                ));
            }
        }

        Ok(())
    }

    /// GPT forward pass with configurable number of layers
    /// レイヤー数を設定可能なGPTフォワードパス
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    /// * `max_layers` - Maximum number of layers to use (None = use all)
    ///
    /// # Returns
    /// Logits tensor of shape [batch_size, seq_len, vocab_size]
    pub fn forward_with_layers(&self, input_ids: &[usize], max_layers: Option<usize>) -> RusTorchResult<Tensor<f64>> {
        use crate::autograd::Variable;
        use crate::nn::{Embedding, SinusoidalPositionalEncoding, MultiheadAttention, Linear, GELU, Module};

        let batch_size = 1;
        let seq_len = input_ids.len();
        let vocab_size = self.config.vocab_size;
        let d_model = self.config.d_model;
        let num_heads = self.config.num_heads;
        let d_ff = self.config.d_ff;

        // 1. Token Embedding Lookup
        // トークン埋め込み変換: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        let token_emb = Embedding::<f64>::new(vocab_size, d_model, None, None, None);

        // Convert input_ids to f64 tensor: [batch_size, seq_len]
        let input_data: Vec<f64> = input_ids.iter().map(|&id| id as f64).collect();
        let input_tensor = Tensor::from_vec(input_data, vec![batch_size, seq_len]);
        let input_var = Variable::new(input_tensor, false);

        // Get token embeddings: [batch_size, seq_len, d_model]
        let mut x = token_emb.forward(&input_var);

        // 2. Add Positional Encoding
        // 位置エンコーディング追加
        let pos_encoding = SinusoidalPositionalEncoding::<f64>::new(
            self.config.max_seq_len,
            d_model
        );
        x = pos_encoding.forward(&x);

        // 3. Apply Transformer Blocks
        // Transformerブロック適用
        let num_layers = max_layers.unwrap_or(self.config.num_layers).min(self.config.num_layers);

        #[cfg(debug_assertions)]
        if let Some(max) = max_layers {
            eprintln!("Using {} layers (out of {})", num_layers, self.config.num_layers);
        }

        for layer_idx in 0..num_layers {
            // Save input for residual connection
            let residual = x.clone();

            // Layer Norm 1 (Pre-Attention) with loaded weights
            let ln1_key = format!("blk.{}.attn_norm.weight", layer_idx);
            x = self.apply_layer_norm(&x, &ln1_key, d_model);

            // Multi-Head Self-Attention
            let attention = MultiheadAttention::<f64>::new(
                d_model,
                num_heads,
                Some(0.0),  // dropout
                Some(true), // bias
                None,       // kdim
                None,       // vdim
                Some(true), // batch_first
            )?;
            let (attn_output, _) = attention.forward(&x, &x, &x, None, Some(false), None, Some(true))?;

            // Residual connection 1
            x = self.add_variables(&residual, &attn_output)?;

            // Save for second residual
            let residual2 = x.clone();

            // Layer Norm 2 (Pre-FFN) with loaded weights
            let ln2_key = format!("blk.{}.ffn_norm.weight", layer_idx);
            x = self.apply_layer_norm(&x, &ln2_key, d_model);

            // Feed-Forward Network
            let fc1 = Linear::<f64>::new(d_model, d_ff);
            let gelu = GELU::<f64>::new();
            let fc2 = Linear::<f64>::new(d_ff, d_model);

            let mut ffn_out = fc1.forward(&x);
            ffn_out = gelu.forward(&ffn_out);
            ffn_out = fc2.forward(&ffn_out);

            // Residual connection 2
            x = self.add_variables(&residual2, &ffn_out)?;
        }

        // Final Layer Norm (Output Norm) with loaded weights
        x = self.apply_layer_norm(&x, "output_norm.weight", d_model);

        // 4. Output Projection to Vocabulary Logits
        // 語彙サイズへの出力射影: [batch_size, seq_len, d_model] -> [batch_size, seq_len, vocab_size]
        let lm_head = Linear::<f64>::new(d_model, vocab_size);
        let logits_var = lm_head.forward(&x);

        // Extract tensor from Variable
        let logits_binding = logits_var.data();
        let logits_data = logits_binding.read().unwrap();
        let logits = logits_data.clone();

        Ok(logits)
    }

    /// Add two Variables element-wise (for residual connections)
    /// 2つのVariableを要素ごとに加算（残差接続用）
    fn add_variables(&self, a: &crate::autograd::Variable<f64>, b: &crate::autograd::Variable<f64>) -> RusTorchResult<crate::autograd::Variable<f64>> {
        let a_binding = a.data();
        let a_data = a_binding.read().unwrap();
        let b_binding = b.data();
        let b_data = b_binding.read().unwrap();

        if a_data.shape() != b_data.shape() {
            return Err(RusTorchError::shape_mismatch(a_data.shape(), b_data.shape()));
        }

        let result_data: Vec<f64> = a_data.as_array().iter()
            .zip(b_data.as_array().iter())
            .map(|(x, y)| x + y)
            .collect();

        let result = Tensor::from_vec(result_data, a_data.shape().to_vec());
        let requires_grad = a.requires_grad() || b.requires_grad();
        Ok(crate::autograd::Variable::new(result, requires_grad))
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
            num_kv_heads: 12,  // Standard MHA (same as num_heads)
            d_ff: 3072,
            max_seq_len: 1024,
            dropout: 0.1,
            rope_theta: 10000.0,
        };

        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.num_layers, 12);
    }

    #[test]
    fn test_gpt_model_creation() {
        let config = GPTConfig {
            vocab_size: 1000,
            d_model: 128,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,  // Standard MHA
            d_ff: 512,
            max_seq_len: 256,
            dropout: 0.0,
            rope_theta: 10000.0,
        };

        let model = GPTModel::new(config).unwrap();
        assert_eq!(model.config().vocab_size, 1000);
        assert_eq!(model.config().num_layers, 2);
    }
}
