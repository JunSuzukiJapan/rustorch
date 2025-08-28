//! PyTorch model architecture parsing and analysis
//! PyTorchモデルアーキテクチャの解析と分析
//!
//! This module provides backward compatibility by re-exporting the new modular structure.
//! The actual implementation has been moved to the `parser` module for better organization.
//!
//! このモジュールは新しいモジュラー構造を再エクスポートすることで後方互換性を提供します。
//! 実際の実装は、より良い組織化のために `parser` モジュールに移動されました。

// Re-export everything from the new parser module structure
pub use super::parser::*;
