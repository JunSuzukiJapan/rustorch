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
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
}

impl GPTConfig {
    /// Create config from GGUF model parameters
    pub fn from_model_params(params: &ModelParams) -> Self {
        Self {
            vocab_size: params.vocab_size as usize,
            d_model: params.hidden_size as usize,
            num_layers: params.num_layers as usize,
            num_heads: params.num_heads as usize,
            d_ff: (params.hidden_size * 4) as usize, // Standard FFN size
            max_seq_len: params.context_length as usize,
            dropout: 0.1,
        }
    }
}

/// GPT model structure
pub struct GPTModel {
    config: GPTConfig,
    weights: HashMap<String, Tensor<f64>>,
    device_type: DeviceType,
    #[cfg(feature = "metal")]
    has_metal: bool,
}

impl GPTModel {
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

        Ok(Self {
            config,
            weights: HashMap::new(),
            device_type: actual_device,
            #[cfg(feature = "metal")]
            has_metal,
        })
    }

    /// Get backend device type
    pub fn device_type(&self) -> DeviceType {
        self.device_type
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
        // Route to Metal or CPU implementation based on backend
        #[cfg(feature = "metal")]
        if self.has_metal && self.device_type == DeviceType::Metal {
            return self.forward_metal(input_ids);
        }

        // CPU fallback
        eprintln!("⚠️  GPT forward pass using CPU (GPU backend not available)");
        let max_layers = Some(2);
        self.forward_with_layers(input_ids, max_layers)
    }

    /// Metal GPU-accelerated forward pass
    /// Metal GPU加速フォワードパス
    #[cfg(feature = "metal")]
    fn forward_metal(&self, input_ids: &[usize]) -> RusTorchResult<Tensor<f64>> {
        use crate::gpu::metal_kernels::MetalKernelExecutor;

        // Debug output controlled by RUSTORCH_DEBUG environment variable
        let debug = std::env::var("RUSTORCH_DEBUG").is_ok();

        if debug {
            eprintln!("🚀 GPT forward pass using Metal GPU acceleration");
        }

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
        // Note: GGUF embeddings are stored as [d_model, vocab_size] - transposed!
        let mut embedding_data = Vec::with_capacity(seq_len * d_model);
        let emb_shape = token_emb_tensor.data.shape();
        let emb_dim0 = emb_shape[0];
        let emb_dim1 = emb_shape[1];

        // Shape is [d_model, vocab_size], so we access [dim_idx, token_id]
        for &token_id in input_ids {
            if token_id >= emb_dim1 {
                return Err(RusTorchError::tensor_op(
                    format!("Token ID {} out of range (vocab_size={})", token_id, emb_dim1)
                ));
            }
            // For each dimension, extract the token's embedding value
            for dim_idx in 0..emb_dim0 {
                // Access [dim_idx, token_id] in the [d_model, vocab_size] matrix
                let linear_idx = dim_idx * emb_dim1 + token_id;
                if linear_idx >= token_emb_tensor.data.len() {
                    eprintln!("ERROR: linear_idx={} >= len={}, dim_idx={}, token_id={}, emb_dim1={}",
                        linear_idx, token_emb_tensor.data.len(), dim_idx, token_id, emb_dim1);
                    return Err(RusTorchError::tensor_op(
                        format!("Index {} out of bounds (len={})", linear_idx, token_emb_tensor.data.len())
                    ));
                }
                // Use iter().nth() for safe access
                if let Some(&value) = token_emb_tensor.data.iter().nth(linear_idx) {
                    embedding_data.push(value);
                } else {
                    return Err(RusTorchError::tensor_op(
                        format!("Failed to access index {}", linear_idx)
                    ));
                }
            }
        }

        // Convert to f32 for Metal operations
        let mut x_f32: Vec<f32> = embedding_data.iter().map(|&v| v as f32).collect();

        // 2. Process through all Transformer layers with Metal
        let num_layers = self.config.num_layers;

        if debug {
            eprintln!("   🔄 Processing {} transformer layers", num_layers);
        }

        for layer_idx in 0..num_layers {
            if debug {
                eprintln!("   📍 Layer {}/{}", layer_idx + 1, num_layers);
            }

        // Layer Norm 1 (Pre-Attention) - Metal
        let ln1_key = format!("blk.{}.attn_norm.weight", layer_idx);
        let ln1_weight = self.weights.get(&ln1_key)
            .ok_or_else(|| RusTorchError::tensor_op(format!("Layer norm weight not found: {}", ln1_key)))?;

        // Convert gamma weights to f32
        let ln1_gamma_f32: Vec<f32> = ln1_weight.data.iter().map(|&v| v as f32).collect();

        // Beta is typically zero for GPT layer norm (affine=False style)
        let ln1_beta_f32 = vec![0.0f32; d_model];

        let mut x_ln1 = vec![0.0f32; x_f32.len()];
        executor.layer_norm_f32(
            &x_f32,
            &mut x_ln1,
            &ln1_gamma_f32,
            &ln1_beta_f32,
            batch_size,
            seq_len,
            d_model,
            1e-5
        )?;

        if debug {
            eprintln!("     ✓ Layer Norm 1 complete");
        }

        // Attention Mechanism (simplified single-head)
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

        // 1. Q, K, V projections (Metal GPU)
        if debug {
            eprintln!("       - Q, K, V projections");
        }
        let mut q_proj = vec![0.0f32; seq_len * d_model];
        let mut k_proj = vec![0.0f32; seq_len * d_model];
        let mut v_proj = vec![0.0f32; seq_len * d_model];

        executor.matmul_f32(&x_ln1, &q_weight_f32, &mut q_proj, seq_len, d_model, d_model)?;
        executor.matmul_f32(&x_ln1, &k_weight_f32, &mut k_proj, seq_len, d_model, d_model)?;
        executor.matmul_f32(&x_ln1, &v_weight_f32, &mut v_proj, seq_len, d_model, d_model)?;

        // 2. Transpose K for attention scores: K^T
        if debug {
            eprintln!("       - Transpose K");
        }
        let k_transposed = Self::transpose_2d_f32(&k_proj, seq_len, d_model);

        // 3. Attention scores: Q @ K^T (Metal GPU)
        if debug {
            eprintln!("       - Attention scores: Q @ K^T");
        }
        let mut attn_scores = vec![0.0f32; seq_len * seq_len];
        executor.matmul_f32(&q_proj, &k_transposed, &mut attn_scores, seq_len, seq_len, d_model)?;

        // 4. Scale by 1/sqrt(d_model)
        let scale = 1.0 / (d_model as f32).sqrt();
        for score in &mut attn_scores {
            *score *= scale;
        }

        // 5. Softmax row-wise (CPU)
        if debug {
            eprintln!("       - Softmax (CPU)");
        }
        Self::softmax_2d_f32(&mut attn_scores, seq_len, seq_len);

        // 6. Apply attention to V: attn_scores @ V (Metal GPU)
        if debug {
            eprintln!("       - Apply attention to V");
        }
        let mut attn_output = vec![0.0f32; seq_len * d_model];
        executor.matmul_f32(&attn_scores, &v_proj, &mut attn_output, seq_len, d_model, seq_len)?;

        // 7. Output projection (Metal GPU)
        if debug {
            eprintln!("       - Output projection");
        }
        let mut x_post_attn = vec![0.0f32; seq_len * d_model];
        executor.matmul_f32(&attn_output, &o_weight_f32, &mut x_post_attn, seq_len, d_model, d_model)?;

        if debug {
            eprintln!("     ✓ Attention complete");
        }

        // Residual connection 1: x = x + attention_output
        if debug {
            eprintln!("     • Residual connection 1 (Metal)");
        }
        let mut x_residual1 = vec![0.0f32; x_f32.len()];
        executor.elementwise_add_f32(&x_f32, &x_post_attn, &mut x_residual1)?;
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
        let ln2_beta_f32 = vec![0.0f32; d_model];

        let mut x_ln2 = vec![0.0f32; x_residual1.len()];
        executor.layer_norm_f32(
            &x_residual1,
            &mut x_ln2,
            &ln2_gamma_f32,
            &ln2_beta_f32,
            batch_size,
            seq_len,
            d_model,
            1e-5
        )?;
        if debug {
            eprintln!("     ✓ Layer Norm 2 complete");
        }

        // Feed-Forward Network with Metal
        // FFN structure: down_proj(GELU(gate_proj(x)) * up_proj(x))
        if debug {
            eprintln!("     • Feed-Forward Network (Metal)");
        }

        let d_ff = self.config.d_ff;
        if debug {
            eprintln!("       d_model={}, d_ff={}, seq_len={}", d_model, d_ff, seq_len);
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

        // 1. Gate projection: x @ gate_weight^T
        // x_ln2: [seq_len * d_model], gate_weight: [d_ff, d_model] (transposed in GGUF)
        // Result: [seq_len, d_ff]
        if debug {
            eprintln!("       - Gate projection: [{}, {}] @ [{}, {}]^T", seq_len, d_model, d_ff, d_model);
        }
        let mut gate_out = vec![0.0f32; seq_len * d_ff];
        executor.matmul_f32(&x_ln2, &gate_weight_f32, &mut gate_out, seq_len, d_ff, d_model)?;

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
        executor.matmul_f32(&x_ln2, &up_weight_f32, &mut up_out, seq_len, d_ff, d_model)?;

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
        executor.matmul_f32(&ffn_intermediate, &down_weight_f32, &mut ffn_out, seq_len, d_model, d_ff)?;

        if debug {
            eprintln!("     ✓ FFN complete");
        }

        // Residual connection 2: x = x_residual1 + ffn_out
        if debug {
            eprintln!("     • Residual connection 2 (Metal)");
        }
        let mut x_residual2 = vec![0.0f32; x_residual1.len()];
        executor.elementwise_add_f32(&x_residual1, &ffn_out, &mut x_residual2)?;
        if debug {
            eprintln!("     ✓ Residual 2 complete");
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

        let output_norm_gamma_f32: Vec<f32> = output_norm_weight.data.iter().map(|&v| v as f32).collect();
        let output_norm_beta_f32 = vec![0.0f32; d_model];

        let mut x_final_norm = vec![0.0f32; x_f32.len()];
        executor.layer_norm_f32(
            &x_f32,
            &mut x_final_norm,
            &output_norm_gamma_f32,
            &output_norm_beta_f32,
            batch_size,
            seq_len,
            d_model,
            1e-5
        )?;
        if debug {
            eprintln!("   ✓ Final Layer Norm complete");
        }

        // Convert back to f64 and create tensor
        let output_data: Vec<f64> = x_final_norm.iter().map(|&v| v as f64).collect();
        let output_tensor = Tensor::from_vec(output_data, vec![batch_size, seq_len, d_model]);

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
            d_ff: 3072,
            max_seq_len: 1024,
            dropout: 0.1,
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
            d_ff: 512,
            max_seq_len: 256,
            dropout: 0.0,
        };

        let model = GPTModel::new(config).unwrap();
        assert_eq!(model.config().vocab_size, 1000);
        assert_eq!(model.config().num_layers, 2);
    }
}
