use anyhow::Result;
use rustorch::models::{ModelArchitecture, detect_architecture};
use std::path::Path;
use tracing;

use crate::cli::Backend as CliBackend;
use crate::model::inference::InferenceEngine;

/// Model architecture types (now using rustorch's ModelArchitecture)
pub type Architecture = ModelArchitecture;

/// Helper trait for architecture detection
pub trait ArchitectureDetection {
    /// Detect architecture from GGUF file metadata (preferred method)
    /// Or fall back to filename-based detection
    fn detect_smart(model_path: &Path) -> Self;

    /// Legacy filename-based detection (fallback only)
    fn detect_from_filename(model_path: &Path) -> Self;
}

impl ArchitectureDetection for Architecture {
    fn detect_smart(model_path: &Path) -> Self {
        // Try GGUF metadata detection first (most accurate)
        if let Some(model_path_str) = model_path.to_str() {
            if let Ok(arch) = detect_architecture(model_path_str) {
                tracing::info!("🔍 Detected architecture from GGUF metadata: {}", arch);
                return arch;
            }
        }

        // Fall back to filename-based detection
        tracing::warn!("⚠️  Could not detect from GGUF metadata, using filename heuristics");
        Self::detect_from_filename(model_path)
    }

    fn detect_from_filename(model_path: &Path) -> Self {
        let model_name_lower = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        tracing::info!("🔍 Model filename: {}", model_name_lower);

        if model_name_lower.contains("mistral") || model_name_lower.contains("mixtral") {
            tracing::info!("🔍 Detected Mistral architecture (filename)");
            ModelArchitecture::Mistral
        } else if model_name_lower.contains("phi") {
            tracing::info!("🔍 Detected Phi architecture (filename)");
            ModelArchitecture::Phi
        } else if model_name_lower.contains("gemma") {
            tracing::info!("🔍 Detected Gemma architecture (filename)");
            ModelArchitecture::Gemma
        } else if model_name_lower.contains("qwen") {
            tracing::info!("🔍 Detected Qwen architecture (filename)");
            ModelArchitecture::Qwen
        } else {
            // Default to LLaMA for llama models or unknown
            tracing::info!("🔍 Detected LLaMA architecture (filename or default)");
            ModelArchitecture::LLaMA
        }
    }
}

/// Backend loader for unified model loading
pub struct BackendLoader;

impl BackendLoader {
    /// Load model with hybrid-f32 backend (Metal GPU)
    #[cfg(feature = "hybrid-f32")]
    pub fn load_hybrid_f32(
        model_path: &Path,
        engine: &mut InferenceEngine,
    ) -> Result<()> {
        use rustorch::hybrid_f32::models::{DeviceType, F32GPTModel, F32LlamaModel, LlamaConfig};
        use rustorch::formats::gguf::GGUFLoader;

        let device_type = DeviceType::Metal;
        tracing::info!("Loading model with hybrid-f32 backend (Metal GPU)");

        let arch = Architecture::detect_smart(model_path);

        match arch {
            ModelArchitecture::LLaMA | ModelArchitecture::Mistral |
            ModelArchitecture::Phi | ModelArchitecture::Gemma | ModelArchitecture::Qwen => {
                tracing::info!("🦙 Loading {}-architecture model with hybrid-f32", arch);
                Self::load_f32_llama(model_path, device_type, engine)?;
            }
        }

        Ok(())
    }

    /// Load model with mac-hybrid backend (Metal/CoreML)
    #[cfg(feature = "mac-hybrid")]
    pub fn load_mac_hybrid(
        model_path: &Path,
        engine: &mut InferenceEngine,
    ) -> Result<()> {
        use rustorch::hybrid_f32::models::{DeviceType as F32DeviceType, F32GPTModel, F32LlamaModel, LlamaConfig};
        use rustorch::formats::gguf::GGUFLoader;

        let device_type = F32DeviceType::Hybrid;
        tracing::info!("🚀 Loading model with mac-hybrid backend (Metal/CoreML)");

        let arch = Architecture::detect_smart(model_path);

        match arch {
            ModelArchitecture::LLaMA | ModelArchitecture::Mistral |
            ModelArchitecture::Phi | ModelArchitecture::Gemma | ModelArchitecture::Qwen => {
                tracing::info!("🦙 Detected {}-architecture model", arch);
                Self::load_f32_llama(model_path, device_type, engine)?;
            }
        }

        Ok(())
    }

    /// Load F32 Llama model (shared by hybrid-f32 and mac-hybrid)
    #[cfg(any(feature = "hybrid-f32", feature = "mac-hybrid"))]
    fn load_f32_llama(
        model_path: &Path,
        device_type: rustorch::hybrid_f32::models::DeviceType,
        engine: &mut InferenceEngine,
    ) -> Result<()> {
        use rustorch::hybrid_f32::models::{F32LlamaModel, LlamaConfig};
        use rustorch::formats::gguf::GGUFLoader;

        // Load GGUF to extract config
        let loader = GGUFLoader::from_file(model_path)?;
        let params = loader.get_model_params()?;

        // Calculate num_kv_heads from K weight shape
        let head_dim = params.hidden_size as usize / params.num_heads as usize;
        let num_kv_heads = loader
            .get_tensor("blk.0.attn_k.weight")
            .and_then(|tensor_info| {
                // K weight shape: [hidden_size, num_kv_heads * head_dim]
                tensor_info.dims.get(1).map(|&kv_dim| kv_dim as usize / head_dim)
            })
            .unwrap_or(params.num_heads as usize); // Fallback to MHA if not found

        let config = LlamaConfig {
            vocab_size: params.vocab_size as usize,
            hidden_size: params.hidden_size as usize,
            num_layers: params.num_layers as usize,
            num_heads: params.num_heads as usize,
            num_kv_heads,
            intermediate_size: (params.hidden_size * 4) as usize, // Standard 4x expansion
            max_seq_len: params.context_length as usize,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        tracing::info!(
            "📋 Llama Config: vocab={}, hidden={}, layers={}, heads={}, kv_heads={}",
            config.vocab_size,
            config.hidden_size,
            config.num_layers,
            config.num_heads,
            config.num_kv_heads
        );

        let llama_model = F32LlamaModel::from_gguf_with_config(model_path, config, device_type)?;
        tracing::info!("✅ F32 Llama model loaded successfully");
        engine.set_f32_llama_model(llama_model);

        Ok(())
    }

    /// Load F32 GPT model (shared by hybrid-f32 and mac-hybrid)
    #[cfg(any(feature = "hybrid-f32", feature = "mac-hybrid"))]
    fn load_f32_gpt(
        model_path: &Path,
        device_type: rustorch::hybrid_f32::models::DeviceType,
        engine: &mut InferenceEngine,
    ) -> Result<()> {
        use rustorch::hybrid_f32::models::F32GPTModel;

        let f32_model = F32GPTModel::from_gguf_with_device(model_path, device_type)?;
        tracing::info!("✅ F32 GPT model loaded successfully");
        engine.set_f32_gpt_model(f32_model);

        Ok(())
    }

    /// Load standard model with specified backend
    pub fn load_standard(
        model_path: &Path,
        engine: &mut InferenceEngine,
    ) -> Result<()> {
        use rustorch::models::GPTModel;
        use rustorch::backends::DeviceType;

        // Determine device type from engine's current backend
        // For now, default to CPU (Metal routing will be added)
        tracing::info!("Loading standard GPT model (f64)");
        let model = GPTModel::from_gguf(model_path)?;
        tracing::info!("✅ Standard GPT model loaded successfully");
        engine.set_gpt_model(model);

        Ok(())
    }

    /// Load model with Metal backend
    #[cfg(feature = "metal")]
    pub fn load_with_metal(
        model_path: &Path,
        engine: &mut InferenceEngine,
    ) -> Result<()> {
        use rustorch::models::{GPTModel, LlamaModel};
        use rustorch::backends::DeviceType;

        // Detect model architecture using smart detection (GGUF metadata + filename fallback)
        let arch = Architecture::detect_smart(model_path);

        tracing::info!("🔍 Detected architecture: {}", arch);

        // All currently supported architectures use the Llama implementation
        match arch {
            ModelArchitecture::LLaMA | ModelArchitecture::Mistral |
            ModelArchitecture::Phi | ModelArchitecture::Gemma | ModelArchitecture::Qwen => {
                tracing::info!("🦙 Loading {} model with Metal GPU backend", arch);
                let model = LlamaModel::from_gguf_with_backend(model_path, DeviceType::Metal)?;
                tracing::info!("✅ {} model loaded with Metal backend", arch);
                engine.set_llama_model(model);
            }
        }

        Ok(())
    }

    /// Main entry point for loading models based on backend
    pub fn load(
        backend: &CliBackend,
        model_path: &Path,
        engine: &mut InferenceEngine,
    ) -> Result<()> {
        match backend {
            #[cfg(feature = "hybrid-f32")]
            CliBackend::HybridF32 => {
                Self::load_hybrid_f32(model_path, engine)
                    .unwrap_or_else(|e| {
                        tracing::warn!("Failed to load hybrid-f32 model: {}", e);
                        tracing::warn!("Falling back to dummy inference");
                    });
                Ok(())
            }

            #[cfg(feature = "mac-hybrid")]
            CliBackend::Hybrid => {
                Self::load_mac_hybrid(model_path, engine)
                    .unwrap_or_else(|e| {
                        tracing::warn!("Failed to load mac-hybrid model: {}", e);
                        tracing::warn!("Falling back to dummy inference");
                    });
                Ok(())
            }

            // Metal backend: Use standard model with Metal acceleration
            #[cfg(feature = "metal")]
            CliBackend::Metal => {
                Self::load_with_metal(model_path, engine)
                    .unwrap_or_else(|e| {
                        tracing::warn!("Failed to load Metal model: {}", e);
                        tracing::warn!("Falling back to CPU");
                        let _ = Self::load_standard(model_path, engine);
                    });
                Ok(())
            }

            _ => {
                Self::load_standard(model_path, engine)
                    .unwrap_or_else(|e| {
                        tracing::warn!("Failed to load standard model: {}", e);
                        tracing::warn!("Falling back to dummy inference");
                    });
                Ok(())
            }
        }
    }
}
