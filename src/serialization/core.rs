//! Core serialization traits and error types for Phase 9
//! フェーズ9用コアシリアライゼーショントレイトとエラータイプ

use crate::error::RusTorchError;
use crate::tensor::Tensor;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Serialization error types
/// シリアライゼーションエラータイプ
#[derive(Debug, Clone)]
pub enum SerializationError {
    /// File I/O error
    IoError(String),
    /// Format error (invalid file format)
    FormatError(String),
    /// Version incompatibility
    VersionError { 
        expected: String, 
        found: String 
    },
    /// Missing required field
    MissingField(String),
    /// Type mismatch during deserialization
    TypeMismatch { 
        expected: String, 
        found: String 
    },
    /// Corruption detected
    CorruptionError(String),
    /// Unsupported operation
    UnsupportedOperation(String),
}

impl fmt::Display for SerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerializationError::IoError(msg) => write!(f, "I/O error: {}", msg),
            SerializationError::FormatError(msg) => write!(f, "Format error: {}", msg),
            SerializationError::VersionError { expected, found } => {
                write!(f, "Version mismatch: expected {}, found {}", expected, found)
            }
            SerializationError::MissingField(field) => write!(f, "Missing required field: {}", field),
            SerializationError::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {}, found {}", expected, found)
            }
            SerializationError::CorruptionError(msg) => write!(f, "Data corruption: {}", msg),
            SerializationError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
        }
    }
}

impl std::error::Error for SerializationError {}

impl From<std::io::Error> for SerializationError {
    fn from(error: std::io::Error) -> Self {
        SerializationError::IoError(error.to_string())
    }
}

impl From<SerializationError> for RusTorchError {
    fn from(error: SerializationError) -> Self {
        RusTorchError::SerializationError {
            operation: "serialization".to_string(),
            message: error.to_string(),
        }
    }
}

pub type SerializationResult<T> = Result<T, SerializationError>;

/// Core trait for objects that can be saved
/// 保存可能オブジェクトのコアトレイト
pub trait Saveable {
    /// Save object to binary format
    /// オブジェクトをバイナリ形式で保存
    fn save_binary(&self) -> SerializationResult<Vec<u8>>;
    
    /// Get object type identifier
    /// オブジェクトタイプ識別子を取得
    fn type_id(&self) -> &'static str;
    
    /// Get version information
    /// バージョン情報を取得
    fn version(&self) -> String {
        "1.0.0".to_string()
    }
    
    /// Get metadata for object
    /// オブジェクトのメタデータを取得
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Core trait for objects that can be loaded
/// 読み込み可能オブジェクトのコアトレイト
pub trait Loadable: Sized {
    /// Load object from binary format
    /// バイナリ形式からオブジェクトを読み込み
    fn load_binary(data: &[u8]) -> SerializationResult<Self>;
    
    /// Get expected type identifier
    /// 期待されるタイプ識別子を取得
    fn expected_type_id() -> &'static str;
    
    /// Validate version compatibility
    /// バージョン互換性を検証
    fn validate_version(version: &str) -> SerializationResult<()> {
        if version.starts_with("1.") {
            Ok(())
        } else {
            Err(SerializationError::VersionError {
                expected: "1.x".to_string(),
                found: version.to_string(),
            })
        }
    }
}

/// File header for RusTorch serialization format
/// RusTorchシリアライゼーション形式用ファイルヘッダー
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHeader {
    pub magic: [u8; 8],           // "RUSTORCH"
    pub version: String,          // Version string
    pub object_type: String,      // Object type identifier
    pub metadata: HashMap<String, String>, // Additional metadata
    pub checksum: u64,           // Data integrity checksum
}

impl FileHeader {
    /// Create new file header
    /// 新しいファイルヘッダーを作成
    pub fn new(object_type: String, metadata: HashMap<String, String>) -> Self {
        Self {
            magic: *b"RUSTORCH",
            version: "1.0.0".to_string(),
            object_type,
            metadata,
            checksum: 0, // Will be computed during save
        }
    }
    
    /// Validate header magic and version
    /// ヘッダーマジックとバージョンを検証
    pub fn validate(&self) -> SerializationResult<()> {
        if self.magic != *b"RUSTORCH" {
            return Err(SerializationError::FormatError(
                "Invalid file magic".to_string()
            ));
        }
        
        if !self.version.starts_with("1.") {
            return Err(SerializationError::VersionError {
                expected: "1.x".to_string(),
                found: self.version.clone(),
            });
        }
        
        Ok(())
    }
}

/// Tensor serialization metadata
/// テンソルシリアライゼーションメタデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub requires_grad: bool,
    pub data_offset: u64,
    pub data_size: u64,
}

/// Model serialization metadata
/// モデルシリアライゼーションメタデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_type: String,
    pub parameters: HashMap<String, TensorMetadata>,
    pub buffers: HashMap<String, TensorMetadata>,
    pub config: HashMap<String, String>,
    pub training_state: bool,
}

/// Computation graph node for JIT
/// JIT用計算グラフノード
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: usize,
    pub op_type: String,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub attributes: HashMap<String, String>,
}

/// Computation graph for JIT compilation
/// JITコンパイル用計算グラフ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationGraph<T: Float> {
    pub nodes: Vec<GraphNode>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    #[serde(skip)]
    pub constants: HashMap<String, Tensor<T>>,
}

impl<T: Float> ComputationGraph<T> {
    /// Create new computation graph
    /// 新しい計算グラフを作成
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            constants: HashMap::new(),
        }
    }
    
    /// Add node to graph
    /// グラフにノードを追加
    pub fn add_node(&mut self, node: GraphNode) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }
    
    /// Validate graph structure
    /// グラフ構造を検証
    pub fn validate(&self) -> SerializationResult<()> {
        // Check for cycles, validate connections, etc.
        for node in &self.nodes {
            for &input_id in &node.inputs {
                if input_id >= self.nodes.len() {
                    return Err(SerializationError::FormatError(
                        format!("Invalid input node ID: {}", input_id)
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Utilities for checksum computation
/// チェックサム計算ユーティリティ
pub fn compute_checksum(data: &[u8]) -> u64 {
    // Simple CRC64 implementation
    let mut crc: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    for &byte in data {
        crc ^= byte as u64;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xC96C_5795_D787_0F42;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF_FFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_header_creation() {
        let metadata = HashMap::new();
        let header = FileHeader::new("tensor".to_string(), metadata);
        
        assert_eq!(header.magic, *b"RUSTORCH");
        assert_eq!(header.version, "1.0.0");
        assert_eq!(header.object_type, "tensor");
    }

    #[test]
    fn test_file_header_validation() {
        let metadata = HashMap::new();
        let mut header = FileHeader::new("tensor".to_string(), metadata);
        
        // Valid header should pass
        assert!(header.validate().is_ok());
        
        // Invalid magic should fail
        header.magic = *b"INVALID ";
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_serialization_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let ser_error: SerializationError = io_error.into();
        let rust_error: RusTorchError = ser_error.into();
        
        match rust_error {
            RusTorchError::SerializationError { .. } => (),
            _ => panic!("Expected SerializationError"),
        }
    }

    #[test]
    fn test_computation_graph() {
        let mut graph: ComputationGraph<f32> = ComputationGraph::new();
        
        let node = GraphNode {
            id: 0,
            op_type: "add".to_string(),
            inputs: vec![],
            outputs: vec![0],
            attributes: HashMap::new(),
        };
        
        let id = graph.add_node(node);
        assert_eq!(id, 0);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_checksum_computation() {
        let data = b"test data";
        let checksum1 = compute_checksum(data);
        let checksum2 = compute_checksum(data);
        
        // Same data should produce same checksum
        assert_eq!(checksum1, checksum2);
        
        // Different data should produce different checksum
        let different_data = b"different test data";
        let checksum3 = compute_checksum(different_data);
        assert_ne!(checksum1, checksum3);
    }
}