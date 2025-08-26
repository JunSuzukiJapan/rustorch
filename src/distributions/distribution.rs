/// Base Distribution types and error handling
/// 基本分布型とエラー処理
///
/// This module defines the core Distribution trait and error types
/// used throughout the distributions system.
use std::fmt;

/// Distribution error types
/// 分布エラー型
#[derive(Debug, Clone)]
pub enum DistributionError {
    /// Invalid parameter value
    /// 無効なパラメータ値
    InvalidParameter(String),

    /// Shape mismatch between tensors
    /// テンソル間の形状不一致
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape received
        got: Vec<usize>,
    },

    /// Sampling error
    /// サンプリングエラー
    SamplingError(String),

    /// Numerical computation error
    /// 数値計算エラー
    NumericalError(String),

    /// Unsupported operation
    /// サポートされていない操作
    UnsupportedOperation(String),

    /// Tensor operation error
    /// テンソル操作エラー
    TensorError(String),
}

impl fmt::Display for DistributionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributionError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            DistributionError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            DistributionError::SamplingError(msg) => write!(f, "Sampling error: {}", msg),
            DistributionError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            DistributionError::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {}", msg)
            }
            DistributionError::TensorError(msg) => write!(f, "Tensor error: {}", msg),
        }
    }
}

impl std::error::Error for DistributionError {}

/// Base Distribution struct for common functionality
/// 共通機能のための基本分布構造体
#[derive(Debug, Clone)]
pub struct Distribution {
    /// Batch shape of the distribution
    /// 分布のバッチ形状
    pub batch_shape: Vec<usize>,

    /// Event shape of the distribution
    /// 分布のイベント形状
    pub event_shape: Vec<usize>,

    /// Whether to validate parameters
    /// パラメータを検証するかどうか
    pub validate_args: bool,
}

impl Distribution {
    /// Create a new Distribution
    /// 新しい分布を作成
    pub fn new(batch_shape: Vec<usize>, event_shape: Vec<usize>, validate_args: bool) -> Self {
        Self {
            batch_shape,
            event_shape,
            validate_args,
        }
    }

    /// Get the shape of a single sample
    /// 単一サンプルの形状を取得
    pub fn sample_shape(&self) -> Vec<usize> {
        [self.batch_shape.clone(), self.event_shape.clone()].concat()
    }

    /// Expand batch dimensions for sampling
    /// サンプリング用にバッチ次元を展開
    pub fn expand_shape(&self, sample_shape: Option<&[usize]>) -> Vec<usize> {
        match sample_shape {
            Some(shape) => [shape, &self.batch_shape, &self.event_shape].concat(),
            None => self.sample_shape(),
        }
    }

    /// Validate tensor shape matches expected shape
    /// テンソル形状が期待される形状と一致するかを検証
    pub fn validate_sample_shape(
        &self,
        shape: &[usize],
        expected: &[usize],
    ) -> Result<(), DistributionError> {
        if shape != expected {
            Err(DistributionError::ShapeMismatch {
                expected: expected.to_vec(),
                got: shape.to_vec(),
            })
        } else {
            Ok(())
        }
    }

    /// Broadcast two shapes together
    /// 2つの形状を一緒にブロードキャスト
    pub fn broadcast_shapes(
        shape1: &[usize],
        shape2: &[usize],
    ) -> Result<Vec<usize>, DistributionError> {
        let max_len = shape1.len().max(shape2.len());
        let mut result = vec![1; max_len];

        // Pad shapes with 1s on the left
        let padded1 = Self::pad_shape_left(shape1, max_len);
        let padded2 = Self::pad_shape_left(shape2, max_len);

        for i in 0..max_len {
            let dim1 = padded1[i];
            let dim2 = padded2[i];

            if dim1 == 1 {
                result[i] = dim2;
            } else if dim2 == 1 || dim1 == dim2 {
                result[i] = dim1;
            } else {
                return Err(DistributionError::ShapeMismatch {
                    expected: shape1.to_vec(),
                    got: shape2.to_vec(),
                });
            }
        }

        Ok(result)
    }

    /// Pad shape with 1s on the left
    /// 左側に1でパディングして形状を調整
    fn pad_shape_left(shape: &[usize], target_len: usize) -> Vec<usize> {
        if shape.len() >= target_len {
            shape.to_vec()
        } else {
            let padding = target_len - shape.len();
            [vec![1; padding], shape.to_vec()].concat()
        }
    }
}

/// Distribution registry for factory pattern
/// ファクトリパターン用の分布レジストリ
pub struct DistributionRegistry;

impl DistributionRegistry {
    /// Get distribution name for a given type
    /// 指定された型の分布名を取得
    pub fn get_distribution_name(dist_type: &str) -> Result<String, DistributionError> {
        match dist_type.to_lowercase().as_str() {
            "normal" | "gaussian" => Ok("Normal".to_string()),
            "bernoulli" | "binomial" => Ok("Bernoulli".to_string()),
            "categorical" | "multinomial" => Ok("Categorical".to_string()),
            "gamma" => Ok("Gamma".to_string()),
            "uniform" => Ok("Uniform".to_string()),
            "beta" => Ok("Beta".to_string()),
            "exponential" => Ok("Exponential".to_string()),
            _ => Err(DistributionError::UnsupportedOperation(format!(
                "Unknown distribution type: {}",
                dist_type
            ))),
        }
    }

    /// List all available distribution types
    /// 利用可能なすべての分布型をリスト
    pub fn available_distributions() -> Vec<&'static str> {
        vec![
            "Normal",
            "Bernoulli",
            "Categorical",
            "Gamma",
            "Uniform",
            "Beta",
            "Exponential",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_creation() {
        let dist = Distribution::new(vec![2, 3], vec![4], true);
        assert_eq!(dist.batch_shape, vec![2, 3]);
        assert_eq!(dist.event_shape, vec![4]);
        assert_eq!(dist.sample_shape(), vec![2, 3, 4]);
    }

    #[test]
    fn test_shape_broadcasting() {
        let shape1 = vec![2, 1, 3];
        let shape2 = vec![1, 4, 1];

        let result = Distribution::broadcast_shapes(&shape1, &shape2).unwrap();
        assert_eq!(result, vec![2, 4, 3]);

        // Test incompatible shapes
        let shape3 = vec![2, 3];
        let shape4 = vec![2, 4];
        assert!(Distribution::broadcast_shapes(&shape3, &shape4).is_err());
    }

    #[test]
    fn test_expand_shape() {
        let dist = Distribution::new(vec![2], vec![3], true);

        // Without sample shape
        assert_eq!(dist.expand_shape(None), vec![2, 3]);

        // With sample shape
        assert_eq!(dist.expand_shape(Some(&[5, 4])), vec![5, 4, 2, 3]);
    }

    #[test]
    fn test_distribution_registry() {
        assert_eq!(
            DistributionRegistry::get_distribution_name("normal").unwrap(),
            "Normal"
        );
        assert_eq!(
            DistributionRegistry::get_distribution_name("gaussian").unwrap(),
            "Normal"
        );

        assert!(DistributionRegistry::get_distribution_name("unknown").is_err());

        let available = DistributionRegistry::available_distributions();
        assert!(available.contains(&"Normal"));
        assert!(available.contains(&"Gamma"));
    }
}
