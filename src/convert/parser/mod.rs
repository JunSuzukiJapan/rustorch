//! PyTorch model architecture parsing and analysis
//! PyTorchモデルアーキテクチャの解析と分析

pub mod core;
pub mod errors;
pub mod formats;
pub mod types;
pub mod validation;

#[cfg(test)]
pub mod tests;

// Re-export the main types and functions for backward compatibility
pub use core::ModelParser;
pub use errors::{ExecutionResult, ParsingError, ParsingResult};
pub use formats::{ArchitectureDescription, ConnectionDefinition, LayerDefinition, ModelMetadata};
pub use types::{LayerInfo, LayerType, ModelGraph};
pub use validation::ModelValidator;