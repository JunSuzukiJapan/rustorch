//! Model format support for RusTorch
//! RusTorchのモデルフォーマットサポート

#[cfg(feature = "safetensors")]
pub mod safetensors;

#[cfg(feature = "onnx")]
pub mod onnx;

pub mod pytorch; // PyTorch互換フォーマット
pub mod gguf; // GGUF/GGML quantized format support
