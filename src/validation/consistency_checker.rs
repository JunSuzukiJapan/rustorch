//! Data Consistency Checking System
//! データ整合性チェックシステム

use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;
use std::fmt;

/// Consistency checker for data integrity validation
/// データ整合性検証のための整合性チェッカー
#[derive(Debug)]
pub struct ConsistencyChecker {
    /// Consistency rules
    /// 整合性ルール
    rules: Vec<Box<dyn ConsistencyRule>>,
    /// Violation statistics
    /// 違反統計
    stats: ConsistencyStatistics,
}

/// Consistency rule trait
/// 整合性ルールトレイト
pub trait ConsistencyRule: fmt::Debug + Send + Sync {
    /// Rule name
    /// ルール名
    fn name(&self) -> &str;
    
    /// Check consistency for f32 tensor
    /// f32テンソルの整合性をチェック
    fn check_f32(&self, tensor: &crate::tensor::Tensor<f32>) -> RusTorchResult<Vec<ConsistencyViolation>>;
    
    /// Check consistency for f64 tensor
    /// f64テンソルの整合性をチェック
    fn check_f64(&self, tensor: &crate::tensor::Tensor<f64>) -> RusTorchResult<Vec<ConsistencyViolation>>;
}

/// Consistency check result
/// 整合性チェック結果
#[derive(Debug, Clone)]
pub struct ConsistencyResult {
    /// Whether data is consistent
    /// データが一貫しているか
    pub is_consistent: bool,
    /// Violations found
    /// 発見された違反
    pub violations: Vec<ConsistencyViolation>,
    /// Overall consistency score
    /// 総合整合性スコア
    pub consistency_score: f64,
}

/// Consistency violation details
/// 整合性違反詳細
#[derive(Debug, Clone)]
pub struct ConsistencyViolation {
    /// Rule that was violated
    /// 違反されたルール
    pub rule_name: String,
    /// Severity of violation
    /// 違反の重要度
    pub severity: ViolationSeverity,
    /// Description of violation
    /// 違反の説明
    pub description: String,
    /// Location in data
    /// データ内の位置
    pub location: Option<DataLocation>,
}

/// Violation severity levels
/// 違反重要度レベル
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ViolationSeverity {
    /// Minor violation
    /// 軽微な違反
    Minor,
    /// Moderate violation
    /// 中程度の違反
    Moderate,
    /// Major violation
    /// 重大な違反
    Major,
    /// Critical violation
    /// 重要な違反
    Critical,
}

/// Data location specification
/// データ位置仕様
#[derive(Debug, Clone)]
pub struct DataLocation {
    /// Indices in tensor
    /// テンソル内のインデックス
    pub indices: Vec<usize>,
    /// Range specification
    /// 範囲仕様
    pub range: Option<(usize, usize)>,
}

/// Data consistency validation
/// データ整合性検証
pub struct DataConsistency;

impl DataConsistency {
    /// Shape consistency check
    /// 形状整合性チェック
    pub fn check_shape_consistency<T>(tensor: &crate::tensor::Tensor<T>) -> bool
    where
        T: num_traits::Float + std::fmt::Debug + Clone + Send + Sync + 'static,
    {
        let shape = tensor.shape();
        !shape.is_empty() && shape.iter().all(|&dim| dim > 0)
    }
    
    /// Value range consistency check
    /// 値範囲整合性チェック
    pub fn check_value_range_consistency<T>(_tensor: &crate::tensor::Tensor<T>, _min: T, _max: T) -> bool
    where
        T: num_traits::Float + std::fmt::Debug + Clone + Send + Sync + 'static,
    {
        // Placeholder - would check if all values are within range
        true
    }
}

/// Referential integrity checker
/// 参照整合性チェッカー
pub struct ReferentialIntegrity;

impl ReferentialIntegrity {
    /// Check referential integrity between tensors
    /// テンソル間の参照整合性をチェック
    pub fn check_referential_integrity<T>(_primary: &crate::tensor::Tensor<T>, _foreign: &crate::tensor::Tensor<T>) -> bool
    where
        T: num_traits::Float + std::fmt::Debug + Clone + Send + Sync + 'static,
    {
        // Placeholder - would check referential integrity
        true
    }
}

/// Basic shape consistency rule
/// 基本形状整合性ルール
#[derive(Debug)]
pub struct ShapeConsistencyRule;

impl ConsistencyRule for ShapeConsistencyRule {
    fn name(&self) -> &str {
        "shape_consistency"
    }
    
    fn check_f32(&self, tensor: &crate::tensor::Tensor<f32>) -> RusTorchResult<Vec<ConsistencyViolation>> {
        self.check_tensor_generic(tensor)
    }
    
    fn check_f64(&self, tensor: &crate::tensor::Tensor<f64>) -> RusTorchResult<Vec<ConsistencyViolation>> {
        self.check_tensor_generic(tensor)
    }
}

impl ShapeConsistencyRule {
    /// Generic consistency check implementation for any float type
    /// 任意の浮動小数点型の汎用整合性チェック実装
    fn check_tensor_generic<T>(&self, tensor: &crate::tensor::Tensor<T>) -> RusTorchResult<Vec<ConsistencyViolation>>
    where
        T: num_traits::Float + std::fmt::Debug + Clone + Send + Sync + 'static,
    {
        let mut violations = Vec::new();
        
        if !DataConsistency::check_shape_consistency(tensor) {
            violations.push(ConsistencyViolation {
                rule_name: self.name().to_string(),
                severity: ViolationSeverity::Major,
                description: "Invalid tensor shape detected".to_string(),
                location: None,
            });
        }
        
        Ok(violations)
    }
}

/// Consistency statistics
/// 整合性統計
#[derive(Debug, Default)]
pub struct ConsistencyStatistics {
    /// Total consistency checks
    /// 総整合性チェック数
    pub total_checks: usize,
    /// Total violations found
    /// 発見された総違反数
    pub total_violations: usize,
    /// Violations by severity
    /// 重要度別違反
    pub violations_by_severity: HashMap<ViolationSeverity, usize>,
}

impl ConsistencyChecker {
    /// Create new consistency checker
    /// 新しい整合性チェッカーを作成
    pub fn new() -> Self {
        let mut checker = Self {
            rules: Vec::new(),
            stats: ConsistencyStatistics::default(),
        };
        
        // Add default rules
        checker.add_rule(Box::new(ShapeConsistencyRule));
        
        checker
    }
    
    /// Add consistency rule
    /// 整合性ルールを追加
    pub fn add_rule(&mut self, rule: Box<dyn ConsistencyRule>) {
        self.rules.push(rule);
    }
    
    /// Check consistency of tensor data
    /// テンソルデータの整合性をチェック
    pub fn check_consistency<T>(&mut self, tensor: &crate::tensor::Tensor<T>) -> RusTorchResult<ConsistencyResult>
    where
        T: num_traits::Float + std::fmt::Debug + Clone + Send + Sync + 'static,
    {
        let mut all_violations = Vec::new();
        
        // Apply all rules - dispatch based on type
        use std::any::{Any, TypeId};
        let tensor_any = tensor as &dyn Any;
        
        for rule in &self.rules {
            let rule_result = if TypeId::of::<T>() == TypeId::of::<f32>() {
                if let Some(f32_tensor) = tensor_any.downcast_ref::<crate::tensor::Tensor<f32>>() {
                    rule.check_f32(f32_tensor)
                } else {
                    continue;
                }
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                if let Some(f64_tensor) = tensor_any.downcast_ref::<crate::tensor::Tensor<f64>>() {
                    rule.check_f64(f64_tensor)
                } else {
                    continue;
                }
            } else {
                // Skip unsupported types
                continue;
            };
            
            match rule_result {
                Ok(mut violations) => all_violations.append(&mut violations),
                Err(e) => {
                    all_violations.push(ConsistencyViolation {
                        rule_name: rule.name().to_string(),
                        severity: ViolationSeverity::Critical,
                        description: format!("Rule execution failed: {}", e),
                        location: None,
                    });
                }
            }
        }
        
        // Update statistics
        self.stats.total_checks += 1;
        self.stats.total_violations += all_violations.len();
        
        for violation in &all_violations {
            *self.stats.violations_by_severity.entry(violation.severity.clone()).or_insert(0) += 1;
        }
        
        let is_consistent = all_violations.is_empty();
        let consistency_score = if is_consistent { 1.0 } else {
            1.0 - (all_violations.len() as f64 / 10.0).min(1.0)
        };
        
        Ok(ConsistencyResult {
            is_consistent,
            violations: all_violations,
            consistency_score,
        })
    }
    
    /// Get violation count
    /// 違反数を取得
    pub fn get_violation_count(&self) -> usize {
        self.stats.total_violations
    }
}