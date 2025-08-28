//! Error types for model parsing
//! モデル解析エラータイプ

use std::error::Error;
use std::fmt;

/// Type alias for execution result containing layer order and dependencies
pub type ExecutionResult = (Vec<String>, std::collections::HashMap<String, Vec<String>>);
/// Type alias for parsing result containing layer order and dependencies  
pub type ParsingResult = Result<ExecutionResult, ParsingError>;

/// Model parsing errors
/// モデル解析エラー
#[derive(Debug)]
pub enum ParsingError {
    /// Invalid architecture format
    /// 無効なアーキテクチャ形式
    InvalidArchitecture(String),
    /// Circular dependency detected
    /// 循環依存を検出
    CircularDependency(String),
    /// Missing layer connection
    /// レイヤー接続が見つからない
    MissingConnection(String),
    /// Incompatible layer dimensions
    /// 互換性のないレイヤー次元
    IncompatibleDimensions {
        /// First layer name
        layer1: String,
        /// Second layer name
        layer2: String,
    },
}

impl fmt::Display for ParsingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParsingError::InvalidArchitecture(msg) => write!(f, "Invalid architecture: {}", msg),
            ParsingError::CircularDependency(msg) => write!(f, "Circular dependency: {}", msg),
            ParsingError::MissingConnection(msg) => write!(f, "Missing connection: {}", msg),
            ParsingError::IncompatibleDimensions { layer1, layer2 } => write!(
                f,
                "Incompatible dimensions between {} and {}",
                layer1, layer2
            ),
        }
    }
}

impl Error for ParsingError {}