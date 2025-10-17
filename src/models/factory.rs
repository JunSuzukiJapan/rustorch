/// Factory functions for automatic model creation from GGUF files
/// GGUFファイルから自動的にモデルを作成するファクトリー関数

use crate::error::{RusTorchError, RusTorchResult};
use crate::formats::gguf::GGUFLoader;
use crate::models::architecture::ModelArchitecture;
use crate::models::llama::LlamaModel;

/// Create a model from GGUF file with automatic architecture detection
/// アーキテクチャを自動検出してGGUFファイルからモデルを作成
///
/// This function reads the GGUF metadata to determine the model architecture
/// and creates the appropriate model type.
/// この関数はGGUFメタデータを読み取ってモデルアーキテクチャを判定し、
/// 適切なモデルタイプを作成します。
///
/// # Arguments
/// * `gguf_path` - Path to the GGUF model file
///
/// # Returns
/// * `LlamaModel` - Currently only LLaMA models are supported
///
/// # Example
/// ```no_run
/// use rustorch::models::factory::create_model_from_gguf;
///
/// let model = create_model_from_gguf("models/tinyllama.gguf").unwrap();
/// ```
///
/// # Supported Architectures
/// - LLaMA (Meta)
/// - Mistral (mapped to LLaMA for compatibility)
/// - Future: Phi, Gemma, Qwen
pub fn create_model_from_gguf(gguf_path: &str) -> RusTorchResult<LlamaModel> {
    // Load GGUF metadata to determine architecture
    let loader = GGUFLoader::from_file(gguf_path)?;

    // Detect architecture from metadata
    let arch_str = loader.get_architecture()
        .unwrap_or_else(|| "llama".to_string());

    let architecture = ModelArchitecture::from_gguf_string(&arch_str)
        .unwrap_or(ModelArchitecture::LLaMA);

    eprintln!("🔍 Detected architecture: {} (from GGUF metadata: {})", architecture, arch_str);

    // Create appropriate model based on architecture
    match architecture {
        ModelArchitecture::LLaMA | ModelArchitecture::Mistral => {
            // Mistral uses the same architecture as LLaMA with minor differences
            // MistralはLLaMAと同じアーキテクチャを使用（わずかな違いあり）
            LlamaModel::from_gguf(gguf_path)
        }
        _ => {
            // For now, fall back to LLaMA for unsupported architectures
            // 現時点ではサポートされていないアーキテクチャはLLaMAにフォールバック
            eprintln!("⚠️  Architecture {} not yet fully supported, using LLaMA compatibility mode", architecture);
            LlamaModel::from_gguf(gguf_path)
        }
    }
}

/// Detect model architecture from GGUF file without loading the full model
/// フルモデルをロードせずにGGUFファイルからアーキテクチャを検出
///
/// # Arguments
/// * `gguf_path` - Path to the GGUF model file
///
/// # Returns
/// * `ModelArchitecture` - Detected architecture
///
/// # Example
/// ```no_run
/// use rustorch::models::factory::detect_architecture;
///
/// let arch = detect_architecture("models/tinyllama.gguf").unwrap();
/// println!("Detected: {}", arch);
/// ```
pub fn detect_architecture(gguf_path: &str) -> RusTorchResult<ModelArchitecture> {
    let loader = GGUFLoader::from_file(gguf_path)?;

    let arch_str = loader.get_architecture()
        .ok_or_else(|| RusTorchError::ParseError(
            "No architecture metadata found in GGUF file".to_string()
        ))?;

    ModelArchitecture::from_gguf_string(&arch_str)
        .ok_or_else(|| RusTorchError::ParseError(
            format!("Unknown architecture: {}", arch_str)
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_detection_unit() {
        // Unit test for architecture string parsing
        let arch = ModelArchitecture::from_gguf_string("llama");
        assert_eq!(arch, Some(ModelArchitecture::LLaMA));

        let arch = ModelArchitecture::from_gguf_string("mistral");
        assert_eq!(arch, Some(ModelArchitecture::Mistral));
    }
}
