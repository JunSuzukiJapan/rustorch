//! Sparse tensor utilities and conversions
//! スパーステンソルユーティリティと変換

use crate::error::{RusTorchError, RusTorchResult};
use super::{SparseTensor, SparseFormat};
use ndarray::{ArrayD, Array1, Array2};
use num_traits::{Float, Zero, One};
use std::collections::{HashMap, HashSet};

/// Sparse tensor analysis and statistics
/// スパーステンソル解析と統計
pub struct SparseAnalyzer<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Copy + PartialOrd> SparseAnalyzer<T> {
    /// Analyze sparsity patterns in tensor
    /// テンソルのスパースパターンを解析
    pub fn analyze_pattern(tensor: &SparseTensor<T>) -> SparsePatternAnalysis<T> {
        let mut analysis = SparsePatternAnalysis::new();
        
        // Basic statistics
        analysis.total_elements = tensor.dense_size();
        analysis.non_zero_elements = tensor.nnz;
        analysis.sparsity_ratio = tensor.sparsity();
        analysis.format = tensor.format;
        
        // Value distribution statistics
        if !tensor.values.is_empty() {
            let values_slice = tensor.values.as_slice().unwrap();
            analysis.min_value = values_slice.iter().fold(T::infinity(), |a, &b| if a < b { a } else { b });
            analysis.max_value = values_slice.iter().fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });
            analysis.mean_abs_value = values_slice.iter().map(|&x| x.abs()).sum::<T>() / T::from(tensor.nnz).unwrap();
        }

        // Pattern regularity analysis
        analysis.pattern_regularity = Self::compute_pattern_regularity(tensor);
        
        // Memory efficiency
        let dense_memory = tensor.dense_size() * std::mem::size_of::<T>();
        let sparse_memory = tensor.memory_usage();
        analysis.memory_efficiency = 1.0 - (sparse_memory as f64 / dense_memory as f64);

        analysis
    }

    /// Compute pattern regularity score (0.0 = random, 1.0 = highly structured)
    /// パターン規則性スコアを計算（0.0 = ランダム, 1.0 = 高度に構造化）
    fn compute_pattern_regularity(tensor: &SparseTensor<T>) -> f64 {
        if tensor.format != SparseFormat::COO || tensor.shape.len() != 2 {
            return 0.0; // Can only analyze 2D COO tensors for now
        }

        let row_indices = &tensor.indices[0];
        let col_indices = &tensor.indices[1];
        
        // Analyze row distribution uniformity
        let mut row_counts = HashMap::new();
        for &row in row_indices.iter() {
            *row_counts.entry(row).or_insert(0) += 1;
        }
        
        // Calculate coefficient of variation for row distribution
        let row_count_values: Vec<_> = row_counts.values().collect();
        if row_count_values.is_empty() {
            return 0.0;
        }
        
        let mean = row_count_values.iter().map(|&&x| x as f64).sum::<f64>() / row_count_values.len() as f64;
        let variance = row_count_values.iter()
            .map(|&&x| (x as f64 - mean).powi(2))
            .sum::<f64>() / row_count_values.len() as f64;
        
        let cv = variance.sqrt() / mean;
        
        // Lower coefficient of variation indicates more regular pattern
        (1.0 / (1.0 + cv)).clamp(0.0, 1.0)
    }

    /// Suggest optimal sparse format for given access pattern
    /// アクセスパターンに最適なスパース形式を提案
    pub fn suggest_optimal_format(
        tensor: &SparseTensor<T>,
        access_pattern: AccessPattern,
    ) -> SparseFormat {
        match access_pattern {
            AccessPattern::RowMajor | AccessPattern::MatrixVector => SparseFormat::CSR,
            AccessPattern::ColumnMajor => SparseFormat::CSC,
            AccessPattern::Random | AccessPattern::Unknown => {
                // Choose based on sparsity level
                if tensor.sparsity() > 0.95 {
                    SparseFormat::COO // Very sparse - COO is more memory efficient
                } else {
                    SparseFormat::CSR // Moderately sparse - CSR for better access patterns
                }
            }
        }
    }
}

/// Sparse tensor access pattern classification
/// スパーステンソルアクセスパターン分類
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    /// Row-major sequential access
    /// 行メジャー順次アクセス
    RowMajor,
    /// Column-major sequential access
    /// 列メジャー順次アクセス
    ColumnMajor,
    /// Matrix-vector multiplication pattern
    /// 行列ベクトル乗算パターン
    MatrixVector,
    /// Random access pattern
    /// ランダムアクセスパターン
    Random,
    /// Unknown or mixed pattern
    /// 不明または混合パターン
    Unknown,
}

/// Results of sparse pattern analysis
/// スパースパターン解析結果
#[derive(Debug, Clone)]
pub struct SparsePatternAnalysis<T: Float> {
    /// Total number of elements in dense representation
    /// 密表現での総要素数
    pub total_elements: usize,
    /// Number of non-zero elements
    /// 非ゼロ要素数
    pub non_zero_elements: usize,
    /// Sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    /// スパース率（0.0 = 密, 1.0 = 完全スパース）
    pub sparsity_ratio: f64,
    /// Current storage format
    /// 現在の格納形式
    pub format: SparseFormat,
    /// Minimum non-zero value
    /// 最小非ゼロ値
    pub min_value: T,
    /// Maximum non-zero value
    /// 最大非ゼロ値
    pub max_value: T,
    /// Mean absolute value of non-zero elements
    /// 非ゼロ要素の平均絶対値
    pub mean_abs_value: T,
    /// Pattern regularity score (0.0 = random, 1.0 = structured)
    /// パターン規則性スコア（0.0 = ランダム, 1.0 = 構造化）
    pub pattern_regularity: f64,
    /// Memory efficiency compared to dense storage
    /// 密格納と比較したメモリ効率
    pub memory_efficiency: f64,
}

impl<T: Float> SparsePatternAnalysis<T> {
    fn new() -> Self {
        Self {
            total_elements: 0,
            non_zero_elements: 0,
            sparsity_ratio: 0.0,
            format: SparseFormat::COO,
            min_value: T::zero(),
            max_value: T::zero(),
            mean_abs_value: T::zero(),
            pattern_regularity: 0.0,
            memory_efficiency: 0.0,
        }
    }

    /// Generate comprehensive analysis report
    /// 包括的解析レポートを生成
    pub fn report(&self) -> String {
        format!(
            "Sparse Tensor Analysis Report:\n\
            ================================\n\
            Format: {:?}\n\
            Total elements: {}\n\
            Non-zero elements: {}\n\
            Sparsity: {:.2}%\n\
            Pattern regularity: {:.2}\n\
            Memory efficiency: {:.2}%\n\
            Value range: [{:.6}, {:.6}]\n\
            Mean |value|: {:.6}",
            self.format,
            self.total_elements,
            self.non_zero_elements,
            self.sparsity_ratio * 100.0,
            self.pattern_regularity,
            self.memory_efficiency * 100.0,
            self.min_value,
            self.max_value,
            self.mean_abs_value
        )
    }

    /// Recommend optimizations based on analysis
    /// 解析に基づく最適化を推奨
    pub fn optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if self.sparsity_ratio > 0.95 {
            recommendations.push("Very high sparsity - consider COO format for memory efficiency".to_string());
        } else if self.sparsity_ratio < 0.5 {
            recommendations.push("Low sparsity - consider dense representation".to_string());
        }
        
        if self.pattern_regularity > 0.8 {
            recommendations.push("High pattern regularity - structured pruning may be beneficial".to_string());
        }
        
        if self.memory_efficiency < 0.3 {
            recommendations.push("Low memory efficiency - sparse format may not be optimal".to_string());
        }
        
        recommendations
    }
}

/// Sparse tensor validation utilities
/// スパーステンソル検証ユーティリティ
pub struct SparseValidator;

impl SparseValidator {
    /// Validate sparse tensor integrity
    /// スパーステンソル整合性を検証
    pub fn validate<T: Float + Copy + PartialOrd>(tensor: &SparseTensor<T>) -> RusTorchResult<()> {
        // Check basic consistency
        if tensor.values.len() != tensor.nnz {
            return Err(RusTorchError::InvalidParameters {
                operation: "sparse_validation".to_string(),
                message: "Values length doesn't match nnz count".to_string(),
            });
        }

        // Check indices validity
        for (dim, indices) in tensor.indices.iter().enumerate() {
            if indices.len() != tensor.nnz {
                return Err(RusTorchError::InvalidParameters {
                    operation: "sparse_validation".to_string(),
                    message: format!("Indices dimension {} length mismatch", dim),
                });
            }
            
            if dim < tensor.shape.len() {
                let max_allowed = tensor.shape[dim];
                for &idx in indices.iter() {
                    if idx >= max_allowed {
                        return Err(RusTorchError::InvalidParameters {
                            operation: "sparse_validation".to_string(),
                            message: format!("Index {} exceeds dimension {} size {}", idx, dim, max_allowed),
                        });
                    }
                }
            }
        }

        // Format-specific validation
        match tensor.format {
            SparseFormat::CSR => Self::validate_csr(tensor)?,
            SparseFormat::COO => Self::validate_coo(tensor)?,
            SparseFormat::CSC => {
                return Err(RusTorchError::NotImplemented {
                    feature: "CSC format validation".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Validate CSR format specific constraints
    /// CSR形式特有制約を検証
    fn validate_csr<T: Float>(tensor: &SparseTensor<T>) -> RusTorchResult<()> {
        if tensor.shape.len() != 2 {
            return Err(RusTorchError::InvalidParameters {
                operation: "csr_validation".to_string(),
                message: "CSR format requires 2D tensors".to_string(),
            });
        }

        if tensor.indices.len() != 2 {
            return Err(RusTorchError::InvalidParameters {
                operation: "csr_validation".to_string(),
                message: "CSR format requires exactly 2 index arrays".to_string(),
            });
        }

        let row_ptr = &tensor.indices[0];
        let col_indices = &tensor.indices[1];

        // Validate row pointer
        if row_ptr.len() != tensor.shape[0] + 1 {
            return Err(RusTorchError::InvalidParameters {
                operation: "csr_validation".to_string(),
                message: "Row pointer length must be rows + 1".to_string(),
            });
        }

        // Check row pointer is non-decreasing
        for i in 1..row_ptr.len() {
            if row_ptr[i] < row_ptr[i - 1] {
                return Err(RusTorchError::InvalidParameters {
                    operation: "csr_validation".to_string(),
                    message: "Row pointer must be non-decreasing".to_string(),
                });
            }
        }

        // Check last row pointer equals nnz
        if row_ptr[row_ptr.len() - 1] != tensor.nnz {
            return Err(RusTorchError::InvalidParameters {
                operation: "csr_validation".to_string(),
                message: "Last row pointer must equal nnz".to_string(),
            });
        }

        Ok(())
    }

    /// Validate COO format specific constraints
    /// COO形式特有制約を検証
    fn validate_coo<T: Float>(tensor: &SparseTensor<T>) -> RusTorchResult<()> {
        if tensor.indices.len() != tensor.shape.len() {
            return Err(RusTorchError::InvalidParameters {
                operation: "coo_validation".to_string(),
                message: "COO format requires one index array per dimension".to_string(),
            });
        }

        // Check for duplicate indices (optional - could be expensive)
        if tensor.shape.len() == 2 {
            let mut coordinate_set = HashSet::new();
            
            for i in 0..tensor.nnz {
                let coord = (tensor.indices[0][i], tensor.indices[1][i]);
                if coordinate_set.contains(&coord) {
                    return Err(RusTorchError::InvalidParameters {
                        operation: "coo_validation".to_string(),
                        message: "Duplicate coordinates found in COO tensor".to_string(),
                    });
                }
                coordinate_set.insert(coord);
            }
        }

        Ok(())
    }
}

/// Sparse tensor format conversion utilities
/// スパーステンソル形式変換ユーティリティ
pub struct SparseConverter;

impl SparseConverter {
    /// Convert between sparse formats with validation
    /// 検証付きスパース形式間変換
    pub fn convert<T: Float + Zero + One + Copy>(
        tensor: &SparseTensor<T>,
        target_format: SparseFormat,
    ) -> RusTorchResult<SparseTensor<T>> {
        // Validate input tensor first
        SparseValidator::validate(tensor)?;

        let result = match (tensor.format, target_format) {
            (SparseFormat::COO, SparseFormat::CSR) => tensor.to_csr()?,
            (SparseFormat::CSR, SparseFormat::COO) => tensor.to_coo()?,
            (format, target) if format == target => tensor.clone(),
            _ => {
                return Err(RusTorchError::NotImplemented {
                    feature: format!("Conversion from {:?} to {:?}", tensor.format, target_format),
                });
            }
        };

        // Validate result
        SparseValidator::validate(&result)?;
        Ok(result)
    }

    /// Batch convert multiple tensors efficiently
    /// 複数テンソルの効率的バッチ変換
    pub fn batch_convert<T: Float + Zero + One + Copy>(
        tensors: &[SparseTensor<T>],
        target_format: SparseFormat,
    ) -> RusTorchResult<Vec<SparseTensor<T>>> {
        let mut results = Vec::with_capacity(tensors.len());
        
        for tensor in tensors {
            let converted = Self::convert(tensor, target_format)?;
            results.push(converted);
        }
        
        Ok(results)
    }
}

/// Sparse tensor I/O operations
/// スパーステンソルI/O演算
pub struct SparseIO;

impl SparseIO {
    /// Save sparse tensor in efficient binary format
    /// 効率的バイナリ形式でスパーステンソルを保存
    pub fn save_binary<T: Float + serde::Serialize>(
        tensor: &SparseTensor<T>,
        path: &std::path::Path,
    ) -> RusTorchResult<()> {
        use std::io::Write;
        
        let mut file = std::fs::File::create(path)?;
        
        // Write header
        file.write_all(b"RUSTORCH_SPARSE_V1")?;
        
        // Write format
        let format_byte = match tensor.format {
            SparseFormat::COO => 0u8,
            SparseFormat::CSR => 1u8,
            SparseFormat::CSC => 2u8,
        };
        file.write_all(&[format_byte])?;
        
        // Serialize tensor data
        let serialized = bincode::serialize(tensor)
            .map_err(|e| RusTorchError::Serialization {
                message: format!("Failed to serialize sparse tensor: {}", e),
            })?;
        
        file.write_all(&serialized)?;
        Ok(())
    }

    /// Load sparse tensor from binary format
    /// バイナリ形式からスパーステンソルを読み込み
    pub fn load_binary<T: Float + serde::de::DeserializeOwned>(
        path: &std::path::Path,
    ) -> RusTorchResult<SparseTensor<T>> {
        use std::io::Read;
        
        let mut file = std::fs::File::open(path)?;
        
        // Check header
        let mut header = [0u8; 18];
        file.read_exact(&mut header)?;
        if &header != b"RUSTORCH_SPARSE_V1" {
            return Err(RusTorchError::Serialization {
                message: "Invalid sparse tensor file header".to_string(),
            });
        }
        
        // Read format
        let mut format_byte = [0u8; 1];
        file.read_exact(&mut format_byte)?;
        
        // Read tensor data
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        
        let tensor: SparseTensor<T> = bincode::deserialize(&data)
            .map_err(|e| RusTorchError::Serialization {
                message: format!("Failed to deserialize sparse tensor: {}", e),
            })?;
        
        // Validate loaded tensor
        SparseValidator::validate(&tensor)?;
        
        Ok(tensor)
    }
}

/// Performance benchmarking for sparse operations
/// スパース演算のパフォーマンスベンチマーク
pub struct SparseBenchmark<T: Float> {
    /// Benchmark results storage
    /// ベンチマーク結果格納
    pub results: HashMap<String, BenchmarkResult>,
    _phantom: std::marker::PhantomData<T>,
}

/// Individual benchmark result
/// 個別ベンチマーク結果
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation name
    /// 演算名
    pub operation: String,
    /// Execution time in nanoseconds
    /// 実行時間（ナノ秒）
    pub time_ns: u64,
    /// Memory usage in bytes
    /// メモリ使用量（バイト）
    pub memory_bytes: usize,
    /// Throughput (operations per second)
    /// スループット（1秒あたりの演算数）
    pub throughput_ops: f64,
}

impl<T: Float + Copy + Zero + One + std::ops::AddAssign + PartialOrd + 'static> SparseBenchmark<T> {
    /// Create new benchmark suite
    /// 新しいベンチマークスイートを作成
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Benchmark sparse matrix-vector multiplication
    /// スパース行列ベクトル乗算のベンチマーク
    pub fn benchmark_spmv(&mut self, tensor: &SparseTensor<T>, vector: &Array1<T>, iterations: usize) -> RusTorchResult<()> {
        let start_time = std::time::Instant::now();
        
        for _ in 0..iterations {
            let _ = tensor.spmv(vector)?;
        }
        
        let elapsed = start_time.elapsed();
        let time_per_op = elapsed.as_nanos() / iterations as u128;
        
        let result = BenchmarkResult {
            operation: "spmv".to_string(),
            time_ns: time_per_op as u64,
            memory_bytes: tensor.memory_usage(),
            throughput_ops: 1_000_000_000.0 / time_per_op as f64,
        };
        
        self.results.insert("spmv".to_string(), result);
        Ok(())
    }

    /// Compare sparse vs dense operation performance
    /// スパースvs密演算パフォーマンスを比較
    pub fn compare_with_dense(
        &mut self,
        sparse_tensor: &SparseTensor<T>,
        dense_equivalent: &Array2<T>,
        vector: &Array1<T>,
    ) -> RusTorchResult<f64> {
        // Benchmark sparse operation
        self.benchmark_spmv(sparse_tensor, vector, 100)?;
        let sparse_time = self.results["spmv"].time_ns;
        
        // Benchmark dense operation
        let start_time = std::time::Instant::now();
        for _ in 0..100 {
            let _ = dense_equivalent.dot(vector);
        }
        let dense_time = start_time.elapsed().as_nanos() / 100;
        
        // Return speedup ratio (> 1.0 means sparse is faster)
        Ok(dense_time as f64 / sparse_time as f64)
    }

    /// Generate comprehensive benchmark report
    /// 包括的ベンチマークレポートを生成
    pub fn report(&self) -> String {
        let mut report = String::from("Sparse Operations Benchmark Report:\n");
        report.push_str("=====================================\n");
        
        for (op, result) in &self.results {
            report.push_str(&format!(
                "{}: {:.2}μs, {:.1}MB/s throughput\n",
                op,
                result.time_ns as f64 / 1000.0,
                result.throughput_ops / 1_000_000.0
            ));
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_analyzer() {
        let indices = vec![
            Array1::from_vec(vec![0, 1, 2]),
            Array1::from_vec(vec![0, 1, 2]),
        ];
        let values = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let shape = vec![4, 4];
        
        let sparse_tensor = SparseTensor::from_coo(indices, values, shape).unwrap();
        let analysis = SparseAnalyzer::analyze_pattern(&sparse_tensor);
        
        assert_eq!(analysis.total_elements, 16);
        assert_eq!(analysis.non_zero_elements, 3);
        assert!(analysis.sparsity_ratio > 0.8);
    }

    #[test]
    fn test_sparse_validator() {
        let indices = vec![
            Array1::from_vec(vec![0, 1]),
            Array1::from_vec(vec![0, 1]),
        ];
        let values = Array1::from_vec(vec![1.0f32, 2.0]);
        let shape = vec![2, 2];
        
        let sparse_tensor = SparseTensor::from_coo(indices, values, shape).unwrap();
        assert!(SparseValidator::validate(&sparse_tensor).is_ok());
        
        // Test invalid tensor
        let invalid_indices = vec![
            Array1::from_vec(vec![0, 5]), // Invalid index 5 for 2x2 tensor
            Array1::from_vec(vec![0, 1]),
        ];
        let invalid_tensor = SparseTensor::from_coo(invalid_indices, values, vec![2, 2]).unwrap();
        assert!(SparseValidator::validate(&invalid_tensor).is_err());
    }

    #[test]
    fn test_sparse_converter() {
        let indices = vec![
            Array1::from_vec(vec![0, 1, 2]),
            Array1::from_vec(vec![1, 2, 0]),
        ];
        let values = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let shape = vec![3, 3];
        
        let coo_tensor = SparseTensor::from_coo(indices, values, shape).unwrap();
        let csr_tensor = SparseConverter::convert(&coo_tensor, SparseFormat::CSR).unwrap();
        
        assert_eq!(csr_tensor.format, SparseFormat::CSR);
        assert_eq!(csr_tensor.nnz, coo_tensor.nnz);
        
        // Convert back and verify
        let coo_again = SparseConverter::convert(&csr_tensor, SparseFormat::COO).unwrap();
        assert_eq!(coo_again.format, SparseFormat::COO);
    }

    #[test]
    fn test_sparse_benchmark() {
        let mut benchmark = SparseBenchmark::new();
        
        let sparse_tensor = SparseTensor::from_coo(
            vec![Array1::from_vec(vec![0, 1]), Array1::from_vec(vec![0, 1])],
            Array1::from_vec(vec![1.0f32, 2.0]),
            vec![2, 2],
        ).unwrap().to_csr().unwrap();
        
        let vector = Array1::from_vec(vec![1.0, 2.0]);
        
        benchmark.benchmark_spmv(&sparse_tensor, &vector, 10).unwrap();
        assert!(benchmark.results.contains_key("spmv"));
        
        let report = benchmark.report();
        assert!(report.contains("spmv"));
    }
}