//! PyTorch to RusTorch conversion module
//! PyTorchからRusTorchへの変換モジュール

/// Simplified PyTorch to RusTorch conversion implementation
/// PyTorchからRusTorchへの簡略変換実装
pub mod pytorch_to_rustorch_simplified;

/// Model architecture parsing and analysis (modular structure)
/// モデルアーキテクチャの解析と分析（モジュラー構造）
pub mod parser;

/// Model architecture parsing and analysis (legacy interface)
/// モデルアーキテクチャの解析と分析（レガシーインターフェース）
pub mod model_parser;

pub use model_parser::*;
pub use pytorch_to_rustorch_simplified::*;