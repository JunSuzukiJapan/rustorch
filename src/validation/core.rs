//! Core Data Validation Engine
//! コアデータ検証エンジン

use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Core validation engine for tensor data
/// テンソルデータのコア検証エンジン
#[derive(Debug)]
pub struct ValidationEngine {
    /// Validation configuration
    /// 検証設定
    config: ValidationConfig,
    /// Registered validation rules
    /// 登録された検証ルール
    rules: Vec<Box<dyn ValidationRule>>,
    /// Validation statistics
    /// 検証統計
    stats: ValidationStatistics,
}

/// Validation configuration
/// 検証設定
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable strict validation mode
    /// 厳密検証モードを有効化
    pub strict_mode: bool,
    /// Maximum allowed NaN percentage
    /// 許可される最大NaN率
    pub max_nan_percentage: f64,
    /// Maximum allowed infinite values percentage
    /// 許可される最大無限値率
    pub max_inf_percentage: f64,
    /// Minimum allowed finite values percentage
    /// 必要な最小有限値率
    pub min_finite_percentage: f64,
    /// Performance validation budget (microseconds)
    /// パフォーマンス検証予算（マイクロ秒）
    pub performance_budget_us: u64,
    /// Enable schema validation
    /// スキーマ検証を有効化
    pub enable_schema_validation: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            max_nan_percentage: 0.01, // 1% NaN allowed
            max_inf_percentage: 0.001, // 0.1% Inf allowed
            min_finite_percentage: 0.95, // 95% finite values required
            performance_budget_us: 500, // 500 microseconds
            enable_schema_validation: true,
        }
    }
}

/// Validation result with detailed information
/// 詳細情報付き検証結果
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    /// 検証が合格したか
    pub is_valid: bool,
    /// Validation level applied
    /// 適用された検証レベル
    pub level: ValidationLevel,
    /// Detected issues
    /// 検出された問題
    pub issues: Vec<ValidationIssue>,
    /// Validation metrics
    /// 検証メトリクス
    pub metrics: ValidationMetrics,
    /// Time taken for validation
    /// 検証にかかった時間
    pub validation_time: Duration,
}

/// Validation levels
/// 検証レベル
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationLevel {
    /// Basic validation (shape, type)
    /// 基本検証（形状、型）
    Basic,
    /// Standard validation (includes NaN/Inf checks)
    /// 標準検証（NaN/無限値チェックを含む）
    Standard,
    /// Comprehensive validation (includes statistical checks)
    /// 包括的検証（統計チェックを含む）
    Comprehensive,
    /// Strict validation (zero tolerance for issues)
    /// 厳密検証（問題に対する寛容度ゼロ）
    Strict,
}

/// Validation issue with severity and context
/// 重要度とコンテキスト付き検証問題
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Issue type
    /// 問題タイプ
    pub issue_type: IssueType,
    /// Issue severity
    /// 問題重要度
    pub severity: IssueSeverity,
    /// Human-readable message
    /// 人間が読める形式のメッセージ
    pub message: String,
    /// Additional context
    /// 追加コンテキスト
    pub context: HashMap<String, String>,
}

/// Types of validation issues
/// 検証問題のタイプ
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssueType {
    /// Invalid shape or dimensions
    /// 無効な形状または次元
    InvalidShape,
    /// Presence of NaN values
    /// NaN値の存在
    NaNValues,
    /// Presence of infinite values
    /// 無限値の存在
    InfiniteValues,
    /// Values outside expected range
    /// 期待範囲外の値
    OutOfRange,
    /// Type mismatch
    /// 型不一致
    TypeMismatch,
    /// Schema violation
    /// スキーマ違反
    SchemaViolation,
    /// Performance issue
    /// パフォーマンス問題
    PerformanceIssue,
    /// Memory constraint violation
    /// メモリ制約違反
    MemoryConstraint,
    /// Custom validation failure
    /// カスタム検証失敗
    CustomValidation,
}

/// Severity levels for issues
/// 問題の重要度レベル
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Low severity - informational
    /// 低重要度 - 情報提供
    Low,
    /// Medium severity - warning
    /// 中重要度 - 警告
    Medium,
    /// High severity - error
    /// 高重要度 - エラー
    High,
    /// Critical severity - system failure
    /// 最重要 - システム障害
    Critical,
}

/// Validation metrics collected during validation
/// 検証中に収集される検証メトリクス
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Total elements validated
    /// 検証された総要素数
    pub total_elements: usize,
    /// Number of NaN values found
    /// 発見されたNaN値数
    pub nan_count: usize,
    /// Number of infinite values found
    /// 発見された無限値数
    pub inf_count: usize,
    /// Number of finite values
    /// 有限値数
    pub finite_count: usize,
    /// Value range statistics
    /// 値範囲統計
    pub value_range: Option<(f64, f64)>, // (min, max)
    /// Memory usage during validation
    /// 検証中のメモリ使用量
    pub memory_usage_bytes: usize,
    /// Performance metrics
    /// パフォーマンスメトリクス
    pub performance_metrics: PerformanceMetrics,
}

/// Performance metrics for validation
/// 検証のパフォーマンスメトリクス
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Elements processed per second
    /// 秒間処理要素数
    pub elements_per_second: f64,
    /// Memory throughput (MB/s)
    /// メモリスループット（MB/s）
    pub memory_throughput_mb_per_sec: f64,
    /// Cache hit rate for repeated validations
    /// 繰り返し検証のキャッシュヒット率
    pub cache_hit_rate: f64,
}

/// Data schema definition for validation
/// 検証用データスキーマ定義
#[derive(Debug, Clone)]
pub struct DataSchema {
    /// Expected tensor shape
    /// 期待されるテンソル形状
    pub expected_shape: Option<Vec<usize>>,
    /// Expected data type
    /// 期待されるデータ型
    pub expected_dtype: String,
    /// Value constraints
    /// 値制約
    pub value_constraints: ValueConstraints,
    /// Custom validation rules
    /// カスタム検証ルール
    pub custom_rules: Vec<String>,
}

/// Value constraints for data validation
/// データ検証の値制約
#[derive(Debug, Clone)]
pub struct ValueConstraints {
    /// Minimum allowed value
    /// 許可される最小値
    pub min_value: Option<f64>,
    /// Maximum allowed value
    /// 許可される最大値
    pub max_value: Option<f64>,
    /// Allow NaN values
    /// NaN値を許可
    pub allow_nan: bool,
    /// Allow infinite values
    /// 無限値を許可
    pub allow_infinite: bool,
    /// Required statistical properties
    /// 必要な統計特性
    pub statistical_constraints: Option<StatisticalConstraints>,
}

/// Statistical constraints for advanced validation
/// 高度検証のための統計制約
#[derive(Debug, Clone)]
pub struct StatisticalConstraints {
    /// Expected mean range
    /// 期待される平均範囲
    pub mean_range: Option<(f64, f64)>,
    /// Expected standard deviation range
    /// 期待される標準偏差範囲
    pub std_dev_range: Option<(f64, f64)>,
    /// Expected distribution type
    /// 期待される分布型
    pub distribution_type: Option<DistributionType>,
}

/// Distribution types for statistical validation
/// 統計検証の分布型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionType {
    /// Normal (Gaussian) distribution
    /// 正規（ガウス）分布
    Normal,
    /// Uniform distribution
    /// 均等分布
    Uniform,
    /// Exponential distribution
    /// 指数分布
    Exponential,
    /// Custom distribution
    /// カスタム分布
    Custom(String),
}

/// Trait for custom validation rules
/// カスタム検証ルールのトレイト
pub trait ValidationRule: fmt::Debug + Send + Sync {
    /// Rule name for identification
    /// 識別用ルール名
    fn name(&self) -> &str;
    
    /// Apply validation rule to f32 tensor
    /// f32テンソルに検証ルールを適用
    fn validate_f32(&self, tensor: &crate::tensor::Tensor<f32>) -> RusTorchResult<Vec<ValidationIssue>>;
    
    /// Apply validation rule to f64 tensor
    /// f64テンソルに検証ルールを適用
    fn validate_f64(&self, tensor: &crate::tensor::Tensor<f64>) -> RusTorchResult<Vec<ValidationIssue>>;
}

/// Schema validation implementation
/// スキーマ検証実装
#[derive(Debug)]
pub struct SchemaValidation {
    /// Schema to validate against
    /// 検証対象スキーマ
    schema: DataSchema,
}

impl SchemaValidation {
    /// Create new schema validation
    /// 新しいスキーマ検証を作成
    pub fn new(schema: DataSchema) -> Self {
        Self { schema }
    }
}

impl ValidationRule for SchemaValidation {
    fn name(&self) -> &str {
        "schema_validation"
    }
    
    fn validate_f32(&self, tensor: &crate::tensor::Tensor<f32>) -> RusTorchResult<Vec<ValidationIssue>> {
        self.validate_tensor_generic(tensor)
    }
    
    fn validate_f64(&self, tensor: &crate::tensor::Tensor<f64>) -> RusTorchResult<Vec<ValidationIssue>> {
        self.validate_tensor_generic(tensor)
    }
}

impl SchemaValidation {
    /// Generic validation implementation for any float type
    /// 任意の浮動小数点型の汎用検証実装
    fn validate_tensor_generic<T>(&self, tensor: &crate::tensor::Tensor<T>) -> RusTorchResult<Vec<ValidationIssue>>
    where
        T: num_traits::Float + std::fmt::Debug + Clone + Send + Sync + 'static,
    {
        let mut issues = Vec::new();
        
        // Validate shape
        if let Some(ref expected_shape) = self.schema.expected_shape {
            if &tensor.shape() != expected_shape {
                issues.push(ValidationIssue {
                    issue_type: IssueType::InvalidShape,
                    severity: IssueSeverity::High,
                    message: format!(
                        "Shape mismatch: expected {:?}, got {:?}",
                        expected_shape,
                        tensor.shape()
                    ),
                    context: {
                        let mut ctx = HashMap::new();
                        ctx.insert("expected".to_string(), format!("{:?}", expected_shape));
                        ctx.insert("actual".to_string(), format!("{:?}", tensor.shape()));
                        ctx
                    },
                });
            }
        }
        
        // Validate value constraints
        if !self.schema.value_constraints.allow_nan {
            // Check for NaN values (placeholder implementation)
            let nan_count = 0; // Would implement actual NaN counting
            if nan_count > 0 {
                issues.push(ValidationIssue {
                    issue_type: IssueType::NaNValues,
                    severity: IssueSeverity::Medium,
                    message: format!("Found {} NaN values (not allowed by schema)", nan_count),
                    context: {
                        let mut ctx = HashMap::new();
                        ctx.insert("nan_count".to_string(), nan_count.to_string());
                        ctx
                    },
                });
            }
        }
        
        Ok(issues)
    }
}

/// Validation statistics
/// 検証統計
#[derive(Debug, Default)]
pub struct ValidationStatistics {
    /// Total validations performed
    /// 実行された総検証数
    pub total_validations: usize,
    /// Successful validations
    /// 成功した検証数
    pub successful_validations: usize,
    /// Failed validations
    /// 失敗した検証数
    pub failed_validations: usize,
    /// Average validation time
    /// 平均検証時間
    pub average_validation_time: Duration,
    /// Total validation time
    /// 総検証時間
    pub total_validation_time: Duration,
}

impl ValidationEngine {
    /// Create new validation engine
    /// 新しい検証エンジンを作成
    pub fn new(config: ValidationConfig) -> RusTorchResult<Self> {
        Ok(Self {
            config,
            rules: Vec::new(),
            stats: ValidationStatistics::default(),
        })
    }
    
    /// Add validation rule
    /// 検証ルールを追加
    pub fn add_rule(&mut self, rule: Box<dyn ValidationRule>) {
        self.rules.push(rule);
    }
    
    /// Validate tensor with comprehensive checks
    /// 包括的チェックでテンソルを検証
    pub fn validate_tensor<T>(&mut self, tensor: &crate::tensor::Tensor<T>) -> RusTorchResult<ValidationResult>
    where
        T: num_traits::Float + std::fmt::Debug + Clone + Send + Sync + 'static,
    {
        let start_time = Instant::now();
        let mut issues = Vec::new();
        
        // Determine validation level
        let level = if self.config.strict_mode {
            ValidationLevel::Strict
        } else {
            ValidationLevel::Standard
        };
        
        // Basic validation
        let shape = tensor.shape();
        if shape.is_empty() {
            issues.push(ValidationIssue {
                issue_type: IssueType::InvalidShape,
                severity: IssueSeverity::High,
                message: "Empty tensor shape detected".to_string(),
                context: HashMap::new(),
            });
        }
        
        // Calculate basic metrics
        let total_elements = shape.iter().product();
        let metrics = ValidationMetrics {
            total_elements,
            nan_count: 0, // Placeholder - would implement actual counting
            inf_count: 0, // Placeholder - would implement actual counting
            finite_count: total_elements, // Placeholder
            value_range: Some((0.0, 1.0)), // Placeholder
            memory_usage_bytes: total_elements * std::mem::size_of::<T>(),
            performance_metrics: PerformanceMetrics {
                elements_per_second: total_elements as f64 / start_time.elapsed().as_secs_f64().max(1e-9),
                memory_throughput_mb_per_sec: 0.0, // Placeholder
                cache_hit_rate: 0.0, // Placeholder
            },
        };
        
        // Apply custom rules - dispatch based on type
        use std::any::{Any, TypeId};
        let tensor_any = tensor as &dyn Any;
        
        for rule in &self.rules {
            let rule_result = if TypeId::of::<T>() == TypeId::of::<f32>() {
                if let Some(f32_tensor) = tensor_any.downcast_ref::<crate::tensor::Tensor<f32>>() {
                    rule.validate_f32(f32_tensor)
                } else {
                    continue;
                }
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                if let Some(f64_tensor) = tensor_any.downcast_ref::<crate::tensor::Tensor<f64>>() {
                    rule.validate_f64(f64_tensor)
                } else {
                    continue;
                }
            } else {
                // Skip unsupported types
                continue;
            };
            
            match rule_result {
                Ok(mut rule_issues) => issues.append(&mut rule_issues),
                Err(e) => {
                    issues.push(ValidationIssue {
                        issue_type: IssueType::CustomValidation,
                        severity: IssueSeverity::High,
                        message: format!("Custom validation rule '{}' failed: {}", rule.name(), e),
                        context: HashMap::new(),
                    });
                }
            }
        }
        
        let validation_time = start_time.elapsed();
        
        // Check performance budget
        if validation_time.as_micros() as u64 > self.config.performance_budget_us {
            issues.push(ValidationIssue {
                issue_type: IssueType::PerformanceIssue,
                severity: IssueSeverity::Medium,
                message: format!(
                    "Validation exceeded performance budget: {}μs > {}μs",
                    validation_time.as_micros(),
                    self.config.performance_budget_us
                ),
                context: {
                    let mut ctx = HashMap::new();
                    ctx.insert("actual_time_us".to_string(), validation_time.as_micros().to_string());
                    ctx.insert("budget_us".to_string(), self.config.performance_budget_us.to_string());
                    ctx
                },
            });
        }
        
        // Determine if validation passed
        let is_valid = match level {
            ValidationLevel::Strict => issues.is_empty(),
            _ => !issues.iter().any(|issue| issue.severity >= IssueSeverity::High),
        };
        
        // Update statistics
        self.stats.total_validations += 1;
        if is_valid {
            self.stats.successful_validations += 1;
        } else {
            self.stats.failed_validations += 1;
        }
        self.stats.total_validation_time += validation_time;
        self.stats.average_validation_time = 
            self.stats.total_validation_time / self.stats.total_validations as u32;
        
        Ok(ValidationResult {
            is_valid,
            level,
            issues,
            metrics,
            validation_time,
        })
    }
    
    /// Get validation statistics
    /// 検証統計を取得
    pub fn get_statistics(&self) -> &ValidationStatistics {
        &self.stats
    }
    
    /// Reset validation statistics
    /// 検証統計をリセット
    pub fn reset_statistics(&mut self) {
        self.stats = ValidationStatistics::default();
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.is_valid { "✅ VALID" } else { "❌ INVALID" };
        write!(f,
            "🔍 Validation Result\n\
             ===================\n\
             Status: {}\n\
             Level: {:?}\n\
             Issues: {}\n\
             Elements: {}\n\
             Time: {:.3}ms",
            status,
            self.level,
            self.issues.len(),
            self.metrics.total_elements,
            self.validation_time.as_secs_f64() * 1000.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!(!config.strict_mode);
        assert_eq!(config.max_nan_percentage, 0.01);
        assert_eq!(config.performance_budget_us, 500);
    }

    #[test]
    fn test_validation_engine_creation() {
        let config = ValidationConfig::default();
        let result = ValidationEngine::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_issue_creation() {
        let issue = ValidationIssue {
            issue_type: IssueType::NaNValues,
            severity: IssueSeverity::Medium,
            message: "Test issue".to_string(),
            context: HashMap::new(),
        };
        
        assert_eq!(issue.issue_type, IssueType::NaNValues);
        assert_eq!(issue.severity, IssueSeverity::Medium);
    }
}