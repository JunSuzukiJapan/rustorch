/// Model format definitions and utilities
/// モデルフォーマット定義とユーティリティ
use std::collections::HashMap;

/// Supported model formats
/// サポートされているモデルフォーマット
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// ONNX (Open Neural Network Exchange)
    Onnx,
    /// PyTorch (.pth, .pt)
    PyTorch,
    /// TensorFlow SavedModel
    TensorFlow,
    /// Keras (.h5)
    Keras,
    /// TensorFlow Lite (.tflite)
    TensorFlowLite,
    /// CoreML (.mlmodel)
    CoreML,
    /// Caffe (.caffemodel)
    Caffe,
    /// MXNet (.params)
    MXNet,
}

/// Custom model format with name
/// 名前付きカスタムモデルフォーマット
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CustomFormat {
    /// Custom format name
    /// カスタムフォーマット名
    pub name: String,
}

/// Extended model format enum
/// 拡張モデルフォーマット列挙型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExtendedModelFormat {
    /// Standard formats
    Standard(ModelFormat),
    /// Custom format
    Custom(String),
}

impl ModelFormat {
    /// Get format from file extension
    /// ファイル拡張子からフォーマットを取得
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "onnx" => Some(ModelFormat::Onnx),
            "pth" | "pt" => Some(ModelFormat::PyTorch),
            "pb" => Some(ModelFormat::TensorFlow),
            "h5" => Some(ModelFormat::Keras),
            "tflite" => Some(ModelFormat::TensorFlowLite),
            "mlmodel" => Some(ModelFormat::CoreML),
            "caffemodel" => Some(ModelFormat::Caffe),
            "params" => Some(ModelFormat::MXNet),
            _ => None,
        }
    }

    /// Get typical file extensions for this format
    /// このフォーマットの一般的なファイル拡張子を取得
    pub fn extensions(&self) -> &[&str] {
        match self {
            ModelFormat::Onnx => &["onnx"],
            ModelFormat::PyTorch => &["pth", "pt"],
            ModelFormat::TensorFlow => &["pb"],
            ModelFormat::Keras => &["h5"],
            ModelFormat::TensorFlowLite => &["tflite"],
            ModelFormat::CoreML => &["mlmodel"],
            ModelFormat::Caffe => &["caffemodel"],
            ModelFormat::MXNet => &["params"],
        }
    }

    /// Check if format supports certain features
    /// フォーマットが特定の機能をサポートしているかチェック
    pub fn supports_feature(&self, feature: FormatFeature) -> bool {
        matches!(
            (self, feature),
            (ModelFormat::Onnx, FormatFeature::GraphStructure)
                | (ModelFormat::Onnx, FormatFeature::Metadata)
                | (ModelFormat::Onnx, FormatFeature::Quantization)
                | (ModelFormat::PyTorch, FormatFeature::DynamicShapes)
                | (ModelFormat::PyTorch, FormatFeature::StateDict)
                | (ModelFormat::TensorFlowLite, FormatFeature::Quantization)
                | (ModelFormat::TensorFlowLite, FormatFeature::MobileOptimized)
        )
    }

    /// Get format description
    /// フォーマット説明を取得
    pub fn description(&self) -> &str {
        match self {
            ModelFormat::Onnx => "Open Neural Network Exchange format",
            ModelFormat::PyTorch => "PyTorch native format",
            ModelFormat::TensorFlow => "TensorFlow SavedModel format",
            ModelFormat::Keras => "Keras HDF5 format",
            ModelFormat::TensorFlowLite => "TensorFlow Lite format",
            ModelFormat::CoreML => "Apple CoreML format",
            ModelFormat::Caffe => "Caffe model format",
            ModelFormat::MXNet => "Apache MXNet format",
        }
    }
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ModelFormat::Onnx => "ONNX",
            ModelFormat::PyTorch => "PyTorch",
            ModelFormat::TensorFlow => "TensorFlow",
            ModelFormat::Keras => "Keras",
            ModelFormat::TensorFlowLite => "TensorFlow Lite",
            ModelFormat::CoreML => "CoreML",
            ModelFormat::Caffe => "Caffe",
            ModelFormat::MXNet => "MXNet",
        };
        write!(f, "{}", name)
    }
}

/// Format-specific features
/// フォーマット固有の機能
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatFeature {
    /// Support for graph structure representation
    GraphStructure,
    /// Support for metadata storage
    Metadata,
    /// Support for quantized models
    Quantization,
    /// Support for dynamic input shapes
    DynamicShapes,
    /// Support for state dictionary format
    StateDict,
    /// Optimized for mobile deployment
    MobileOptimized,
    /// Support for custom operators
    CustomOperators,
    /// Support for training mode
    TrainingMode,
    /// Support for inference mode only
    InferenceOnly,
}

/// Model compression types
/// モデル圧縮タイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression
    None,
    /// 8-bit quantization
    Int8,
    /// 16-bit quantization
    Int16,
    /// Half precision (FP16)
    Float16,
    /// Pruning
    Pruned,
    /// Knowledge distillation
    Distilled,
}

/// Model deployment target
/// モデルデプロイメントターゲット
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentTarget {
    /// Server/Desktop CPU
    ServerCpu,
    /// Server/Desktop GPU
    ServerGpu,
    /// Mobile CPU
    MobileCpu,
    /// Mobile GPU
    MobileGpu,
    /// Edge devices
    Edge,
    /// Web browsers (WebAssembly)
    WebAssembly,
    /// Embedded systems
    Embedded,
}

/// Format compatibility matrix
/// フォーマット互換性マトリックス
pub struct FormatCompatibility {
    /// Compatibility scores between format pairs
    /// フォーマットペア間の互換性スコア
    compatibility_map: HashMap<(ModelFormat, ModelFormat), f32>,
}

impl FormatCompatibility {
    /// Create new compatibility matrix
    /// 新しい互換性マトリックスを作成
    pub fn new() -> Self {
        let mut compatibility_map = HashMap::new();

        // Perfect compatibility (same format)
        for format in [
            ModelFormat::Onnx,
            ModelFormat::PyTorch,
            ModelFormat::TensorFlow,
            ModelFormat::Keras,
        ] {
            compatibility_map.insert((format, format), 1.0);
        }

        // High compatibility
        compatibility_map.insert((ModelFormat::PyTorch, ModelFormat::Onnx), 0.9);
        compatibility_map.insert((ModelFormat::Onnx, ModelFormat::PyTorch), 0.8);
        compatibility_map.insert((ModelFormat::TensorFlow, ModelFormat::Onnx), 0.8);
        compatibility_map.insert((ModelFormat::Keras, ModelFormat::TensorFlow), 0.95);

        // Medium compatibility
        compatibility_map.insert((ModelFormat::PyTorch, ModelFormat::TensorFlow), 0.6);
        compatibility_map.insert((ModelFormat::Onnx, ModelFormat::TensorFlowLite), 0.7);

        // Low compatibility
        compatibility_map.insert((ModelFormat::Caffe, ModelFormat::Onnx), 0.5);
        compatibility_map.insert((ModelFormat::MXNet, ModelFormat::Onnx), 0.5);

        Self { compatibility_map }
    }

    /// Get compatibility score between two formats
    /// 2つのフォーマット間の互換性スコアを取得
    pub fn get_compatibility(&self, from: ModelFormat, to: ModelFormat) -> f32 {
        self.compatibility_map
            .get(&(from, to))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get recommended conversion path
    /// 推奨変換パスを取得
    pub fn get_conversion_path(&self, from: ModelFormat, to: ModelFormat) -> Vec<ModelFormat> {
        if from == to {
            return vec![from];
        }

        // Direct conversion
        if self.get_compatibility(from, to) > 0.5 {
            return vec![from, to];
        }

        // Via ONNX (most common intermediate format)
        if from != ModelFormat::Onnx && to != ModelFormat::Onnx {
            if self.get_compatibility(from, ModelFormat::Onnx) > 0.5
                && self.get_compatibility(ModelFormat::Onnx, to) > 0.5
            {
                return vec![from, ModelFormat::Onnx, to];
            }
        }

        // No good path found
        vec![from, to]
    }
}

impl Default for FormatCompatibility {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization profiles for different deployment scenarios
/// 異なるデプロイメントシナリオ用の最適化プロファイル
#[derive(Debug, Clone)]
pub struct OptimizationProfile {
    /// Profile name
    /// プロファイル名
    pub name: String,
    /// Deployment target
    /// デプロイメントターゲット
    pub target: DeploymentTarget,
    /// Compression type
    /// 圧縮タイプ
    pub compression: CompressionType,
    /// Maximum model size in bytes
    /// モデルの最大サイズ（バイト）
    pub max_model_size: Option<usize>,
    /// Target inference latency in milliseconds
    /// 目標推論レイテンシ（ミリ秒）
    pub target_latency_ms: Option<f32>,
    /// Memory limit in megabytes
    /// メモリ制限（MB）
    pub memory_limit_mb: Option<usize>,
    /// Minimum accuracy to preserve (0.0-1.0)
    /// 保持する最小精度（0.0-1.0）
    pub preserve_accuracy: f32,
}

impl OptimizationProfile {
    /// Create profile for server deployment
    /// サーバーデプロイメント用プロファイルを作成
    pub fn server() -> Self {
        Self {
            name: "Server".to_string(),
            target: DeploymentTarget::ServerGpu,
            compression: CompressionType::None,
            max_model_size: None,
            target_latency_ms: Some(100.0),
            memory_limit_mb: None,
            preserve_accuracy: 0.99,
        }
    }

    /// Create profile for mobile deployment
    /// モバイルデプロイメント用プロファイルを作成
    pub fn mobile() -> Self {
        Self {
            name: "Mobile".to_string(),
            target: DeploymentTarget::MobileCpu,
            compression: CompressionType::Int8,
            max_model_size: Some(50 * 1024 * 1024), // 50MB
            target_latency_ms: Some(50.0),
            memory_limit_mb: Some(100),
            preserve_accuracy: 0.95,
        }
    }

    /// Create profile for edge deployment
    /// エッジデプロイメント用プロファイルを作成
    pub fn edge() -> Self {
        Self {
            name: "Edge".to_string(),
            target: DeploymentTarget::Edge,
            compression: CompressionType::Int8,
            max_model_size: Some(10 * 1024 * 1024), // 10MB
            target_latency_ms: Some(20.0),
            memory_limit_mb: Some(50),
            preserve_accuracy: 0.90,
        }
    }

    /// Create profile for web deployment
    /// Webデプロイメント用プロファイルを作成
    pub fn web() -> Self {
        Self {
            name: "Web".to_string(),
            target: DeploymentTarget::WebAssembly,
            compression: CompressionType::Float16,
            max_model_size: Some(5 * 1024 * 1024), // 5MB
            target_latency_ms: Some(100.0),
            memory_limit_mb: Some(100),
            preserve_accuracy: 0.95,
        }
    }
}

/// Format validation utilities
/// フォーマット検証ユーティリティ
pub struct FormatValidator;

impl FormatValidator {
    /// Validate model format
    /// モデルフォーマットを検証
    pub fn validate_format(data: &[u8], expected_format: ModelFormat) -> bool {
        match expected_format {
            ModelFormat::Onnx => Self::validate_onnx(data),
            ModelFormat::PyTorch => Self::validate_pytorch(data),
            ModelFormat::TensorFlow => Self::validate_tensorflow(data),
            _ => false, // Other formats not implemented
        }
    }

    /// Validate ONNX format
    /// ONNX形式を検証
    fn validate_onnx(data: &[u8]) -> bool {
        // Check for protobuf magic bytes and ONNX-specific patterns
        data.len() > 8 && data.starts_with(&[0x08]) // Protobuf field 1, varint
    }

    /// Validate PyTorch format
    /// PyTorch形式を検証
    fn validate_pytorch(data: &[u8]) -> bool {
        // Check for pickle protocol markers
        data.len() > 2 && (data[0] == 0x80 || data.starts_with(b"PK")) // Pickle or ZIP
    }

    /// Validate TensorFlow format
    /// TensorFlow形式を検証
    fn validate_tensorflow(data: &[u8]) -> bool {
        // Check for TensorFlow SavedModel signatures
        data.len() > 10
            && (
                data.starts_with(b"\x08\x01") || // SavedModel signature
            data.starts_with(b"TensorFlow")
                // GraphDef signature
            )
    }

    /// Get format confidence score
    /// フォーマット信頼度スコアを取得
    pub fn get_format_confidence(data: &[u8]) -> HashMap<ModelFormat, f32> {
        let mut scores = HashMap::new();

        // ONNX confidence
        if data.len() > 8 && data.starts_with(&[0x08]) {
            scores.insert(ModelFormat::Onnx, 0.8);
        }

        // PyTorch confidence
        if data.len() > 2 && data[0] == 0x80 {
            scores.insert(ModelFormat::PyTorch, 0.9);
        }

        // TensorFlow confidence
        if data.len() > 10 && data.starts_with(b"\x08\x01") {
            scores.insert(ModelFormat::TensorFlow, 0.7);
        }

        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_from_extension() {
        assert_eq!(ModelFormat::from_extension("onnx"), Some(ModelFormat::Onnx));
        assert_eq!(
            ModelFormat::from_extension("pth"),
            Some(ModelFormat::PyTorch)
        );
        assert_eq!(
            ModelFormat::from_extension("pt"),
            Some(ModelFormat::PyTorch)
        );
        assert_eq!(ModelFormat::from_extension("h5"), Some(ModelFormat::Keras));
        assert_eq!(ModelFormat::from_extension("unknown"), None);
    }

    #[test]
    fn test_format_features() {
        assert!(ModelFormat::Onnx.supports_feature(FormatFeature::GraphStructure));
        assert!(ModelFormat::PyTorch.supports_feature(FormatFeature::DynamicShapes));
        assert!(!ModelFormat::Onnx.supports_feature(FormatFeature::StateDict));
    }

    #[test]
    fn test_compatibility_matrix() {
        let compat = FormatCompatibility::new();

        // Perfect self-compatibility
        assert_eq!(
            compat.get_compatibility(ModelFormat::Onnx, ModelFormat::Onnx),
            1.0
        );

        // High PyTorch -> ONNX compatibility
        assert!(compat.get_compatibility(ModelFormat::PyTorch, ModelFormat::Onnx) > 0.8);

        // Low unknown compatibility
        assert_eq!(
            compat.get_compatibility(ModelFormat::Onnx, ModelFormat::Caffe),
            0.0
        );
    }

    #[test]
    fn test_conversion_path() {
        let compat = FormatCompatibility::new();

        // Direct path
        let path = compat.get_conversion_path(ModelFormat::PyTorch, ModelFormat::Onnx);
        assert_eq!(path, vec![ModelFormat::PyTorch, ModelFormat::Onnx]);

        // Via ONNX path
        let path = compat.get_conversion_path(ModelFormat::PyTorch, ModelFormat::TensorFlowLite);
        assert!(path.contains(&ModelFormat::Onnx));
    }

    #[test]
    fn test_optimization_profiles() {
        let mobile = OptimizationProfile::mobile();
        assert_eq!(mobile.target, DeploymentTarget::MobileCpu);
        assert_eq!(mobile.compression, CompressionType::Int8);
        assert!(mobile.max_model_size.is_some());

        let server = OptimizationProfile::server();
        assert_eq!(server.target, DeploymentTarget::ServerGpu);
        assert_eq!(server.compression, CompressionType::None);
        assert!(server.max_model_size.is_none());
    }

    #[test]
    fn test_format_validation() {
        // Mock ONNX data (needs >8 bytes)
        let onnx_data = vec![0x08, 0x01, 0x12, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(FormatValidator::validate_format(
            &onnx_data,
            ModelFormat::Onnx
        ));

        // Mock PyTorch data (needs >2 bytes)
        let pytorch_data = vec![0x80, 0x02, 0x00];
        assert!(FormatValidator::validate_format(
            &pytorch_data,
            ModelFormat::PyTorch
        ));

        // Invalid data
        let invalid_data = vec![0x00, 0x00];
        assert!(!FormatValidator::validate_format(
            &invalid_data,
            ModelFormat::Onnx
        ));
    }
}
