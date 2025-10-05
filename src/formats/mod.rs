//! Model format support for RusTorch
//! RusTorchのモデルフォーマットサポート

pub mod safetensors; // Safetensors format support (always available)

#[cfg(feature = "onnx")]
pub mod onnx;

pub mod pytorch; // PyTorch互換フォーマット
pub mod gguf; // GGUF/GGML quantized format support
