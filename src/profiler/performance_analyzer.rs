//! Performance Analysis & Optimization Engine
//! パフォーマンス分析・最適化エンジン

use crate::error::{RusTorchError, RusTorchResult};
use crate::profiler::benchmark_suite::{BenchmarkResult, BenchmarkStatistics};
use crate::profiler::core::{PerformanceStatistics, SessionSnapshot};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance trend direction
/// パフォーマンス傾向方向
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Performance is improving
    /// パフォーマンス向上中
    Improving,
    /// Performance is degrading
    /// パフォーマンス劣化中
    Degrading,
    /// Performance is stable
    /// パフォーマンス安定
    Stable,
    /// Not enough data
    /// データ不足
    Unknown,
}

/// Performance trend analysis
/// パフォーマンス傾向分析
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Operation name
    /// 操作名
    pub operation_name: String,
    /// Trend direction
    /// 傾向方向
    pub direction: TrendDirection,
    /// Rate of change (% per measurement)
    /// 変化率（測定ごとの%）
    pub change_rate_percent: f64,
    /// Confidence level (0.0 to 1.0)
    /// 信頼度（0.0から1.0）
    pub confidence: f64,
    /// Historical measurements
    /// 過去の測定値
    pub measurements: Vec<TrendMeasurement>,
    /// Projected performance
    /// 予測パフォーマンス
    pub projection: Option<PerformanceProjection>,
}

/// Individual trend measurement
/// 個別傾向測定
#[derive(Debug, Clone)]
pub struct TrendMeasurement {
    /// Measurement timestamp
    /// 測定タイムスタンプ
    pub timestamp: Instant,
    /// Performance value (e.g., execution time in ms)
    /// パフォーマンス値（例：実行時間のms）
    pub value: f64,
    /// Measurement context
    /// 測定コンテキスト
    pub context: MeasurementContext,
}

/// Context for performance measurements
/// パフォーマンス測定のコンテキスト
#[derive(Debug, Clone)]
pub struct MeasurementContext {
    /// System load at measurement time
    /// 測定時システム負荷
    pub system_load: Option<f64>,
    /// Memory pressure
    /// メモリプレッシャー
    pub memory_pressure: Option<f64>,
    /// GPU utilization
    /// GPU使用率
    pub gpu_utilization: Option<f64>,
    /// Temperature factors
    /// 温度要因
    pub temperature_celsius: Option<f32>,
    /// Additional metadata
    /// 追加メタデータ
    pub metadata: HashMap<String, String>,
}

/// Performance projection
/// パフォーマンス予測
#[derive(Debug, Clone)]
pub struct PerformanceProjection {
    /// Predicted value at next measurement
    /// 次の測定での予測値
    pub next_value: f64,
    /// Prediction confidence (0.0 to 1.0)
    /// 予測信頼度（0.0から1.0）
    pub confidence: f64,
    /// Time horizon for prediction
    /// 予測の時間軸
    pub time_horizon: Duration,
    /// Potential performance range
    /// 潜在的パフォーマンス範囲
    pub value_range: (f64, f64),
}

/// Optimization recommendation
/// 最適化推奨事項
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    /// 推奨事項ID
    pub id: String,
    /// Target operation or system component
    /// 対象操作またはシステムコンポーネント
    pub target: String,
    /// Recommendation type
    /// 推奨事項タイプ
    pub recommendation_type: RecommendationType,
    /// Priority level
    /// 優先度レベル
    pub priority: RecommendationPriority,
    /// Description of the issue
    /// 問題の説明
    pub description: String,
    /// Recommended action
    /// 推奨アクション
    pub action: String,
    /// Expected improvement
    /// 期待される改善
    pub expected_improvement: ExpectedImprovement,
    /// Implementation complexity
    /// 実装複雑度
    pub complexity: ImplementationComplexity,
    /// Supporting evidence
    /// 裏付け証拠
    pub evidence: Vec<String>,
    /// Recommendation timestamp
    /// 推奨事項タイムスタンプ
    pub timestamp: Instant,
}

/// Type of optimization recommendation
/// 最適化推奨事項のタイプ
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendationType {
    /// Algorithm optimization
    /// アルゴリズム最適化
    Algorithm,
    /// Memory optimization
    /// メモリ最適化
    Memory,
    /// GPU optimization
    /// GPU最適化
    Gpu,
    /// Caching strategy
    /// キャッシュ戦略
    Caching,
    /// Parallelization
    /// 並列化
    Parallelization,
    /// Data structure optimization
    /// データ構造最適化
    DataStructure,
    /// System configuration
    /// システム設定
    SystemConfig,
    /// Hardware upgrade
    /// ハードウェアアップグレード
    Hardware,
}

/// Recommendation priority
/// 推奨事項優先度
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    /// Critical performance issue
    /// 重要なパフォーマンス問題
    Critical,
    /// High impact optimization
    /// 高影響最適化
    High,
    /// Medium impact optimization
    /// 中影響最適化
    Medium,
    /// Low impact optimization
    /// 低影響最適化
    Low,
    /// Optional improvement
    /// オプション改善
    Optional,
}

/// Expected improvement from recommendation
/// 推奨事項による期待される改善
#[derive(Debug, Clone)]
pub struct ExpectedImprovement {
    /// Performance improvement percentage
    /// パフォーマンス改善率
    pub performance_gain_percent: f64,
    /// Memory reduction percentage
    /// メモリ削減率
    pub memory_reduction_percent: Option<f64>,
    /// Energy efficiency improvement
    /// エネルギー効率改善
    pub energy_efficiency_gain_percent: Option<f64>,
    /// Confidence in estimate (0.0 to 1.0)
    /// 見積もりの信頼度（0.0から1.0）
    pub confidence: f64,
}

/// Implementation complexity
/// 実装複雑度
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplementationComplexity {
    /// Simple configuration change
    /// シンプルな設定変更
    Trivial,
    /// Minor code changes
    /// 軽微なコード変更
    Simple,
    /// Moderate refactoring required
    /// 中程度のリファクタリング必要
    Moderate,
    /// Significant architectural changes
    /// 重要なアーキテクチャ変更
    Complex,
    /// Major system redesign
    /// 主要システム再設計
    Expert,
}

/// Trend analysis configuration
/// 傾向分析設定
#[derive(Debug, Clone)]
pub struct TrendAnalysisConfig {
    /// Minimum measurements for trend analysis
    /// 傾向分析の最小測定数
    pub min_measurements: usize,
    /// Lookback window for trend analysis
    /// 傾向分析の振り返り期間
    pub lookback_window: Duration,
    /// Significance threshold for trend detection
    /// 傾向検出の有意性閾値
    pub significance_threshold: f64,
    /// Minimum confidence for recommendations
    /// 推奨事項の最小信頼度
    pub min_recommendation_confidence: f64,
}

impl Default for TrendAnalysisConfig {
    fn default() -> Self {
        Self {
            min_measurements: 5,
            lookback_window: Duration::from_secs(3600), // 1 hour
            significance_threshold: 0.05,               // 5% change
            min_recommendation_confidence: 0.7,
        }
    }
}

/// Performance analyzer engine
/// パフォーマンス分析エンジン
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Analysis configuration
    /// 分析設定
    config: TrendAnalysisConfig,
    /// Historical performance data
    /// 過去のパフォーマンスデータ
    performance_history: HashMap<String, Vec<TrendMeasurement>>,
    /// Cached trend analysis results
    /// キャッシュされた傾向分析結果
    trend_cache: HashMap<String, PerformanceTrend>,
    /// Generated recommendations
    /// 生成された推奨事項
    recommendations: Vec<OptimizationRecommendation>,
    /// Baseline performance metrics
    /// ベースラインパフォーマンスメトリクス
    baselines: HashMap<String, f64>,
    /// Analysis timestamp
    /// 分析タイムスタンプ
    last_analysis: Option<Instant>,
}

impl PerformanceAnalyzer {
    /// Create new performance analyzer
    /// 新しいパフォーマンス分析器を作成
    pub fn new() -> Self {
        Self {
            config: TrendAnalysisConfig::default(),
            performance_history: HashMap::new(),
            trend_cache: HashMap::new(),
            recommendations: Vec::new(),
            baselines: HashMap::new(),
            last_analysis: None,
        }
    }

    /// Create analyzer with custom configuration
    /// カスタム設定で分析器を作成
    pub fn with_config(config: TrendAnalysisConfig) -> Self {
        Self {
            config,
            performance_history: HashMap::new(),
            trend_cache: HashMap::new(),
            recommendations: Vec::new(),
            baselines: HashMap::new(),
            last_analysis: None,
        }
    }

    /// Add performance measurement
    /// パフォーマンス測定を追加
    pub fn add_measurement(
        &mut self,
        operation_name: String,
        value: f64,
        context: MeasurementContext,
    ) {
        let measurement = TrendMeasurement {
            timestamp: Instant::now(),
            value,
            context,
        };

        self.performance_history
            .entry(operation_name.clone())
            .or_default()
            .push(measurement);

        // Invalidate cache for this operation
        self.trend_cache.remove(&operation_name);

        // Maintain history size (keep last 1000 measurements)
        if let Some(history) = self.performance_history.get_mut(&operation_name) {
            if history.len() > 1000 {
                history.drain(0..500);
            }
        }
    }

    /// Analyze session performance
    /// セッションパフォーマンスを分析
    pub fn analyze_session(&mut self, session: &SessionSnapshot) -> RusTorchResult<TrendAnalysis> {
        println!("🔍 Analyzing session: {}", session.session_name);

        let mut trends = HashMap::new();

        for operation in &session.operations {
            let stats = operation.get_statistics();

            // Add measurement from session
            let context = MeasurementContext {
                system_load: None,
                memory_pressure: None,
                gpu_utilization: None,
                temperature_celsius: None,
                metadata: HashMap::new(),
            };

            self.add_measurement(operation.name.clone(), stats.avg_time_ms, context);

            // Analyze trend for this operation
            if let Ok(trend) = self.analyze_operation_trend(&operation.name) {
                trends.insert(operation.name.clone(), trend);
            }
        }

        // Generate recommendations based on analysis
        self.generate_recommendations(&trends)?;

        self.last_analysis = Some(Instant::now());

        Ok(TrendAnalysis {
            session_id: session.session_id.clone(),
            session_name: session.session_name.clone(),
            analysis_timestamp: Instant::now(),
            operation_trends: trends.clone(),
            overall_performance_score: self.calculate_overall_score(&session.operations)?,
            recommendations: self.recommendations.clone(),
            summary: self.generate_analysis_summary(&trends),
        })
    }

    /// Analyze benchmark results
    /// ベンチマーク結果を分析
    pub fn analyze_benchmark_results(
        &mut self,
        results: &HashMap<String, BenchmarkResult>,
    ) -> RusTorchResult<BenchmarkAnalysis> {
        println!(
            "📊 Analyzing benchmark results ({} benchmarks)",
            results.len()
        );

        let mut analysis_results = HashMap::new();
        let mut performance_insights = Vec::new();

        for (name, result) in results {
            if result.error.is_none() {
                // Add benchmark data to history
                let context = MeasurementContext {
                    system_load: result
                        .system_metrics
                        .as_ref()
                        .map(|s| s.cpu_utilization_percent),
                    memory_pressure: result
                        .memory_metrics
                        .as_ref()
                        .map(|m| m.fragmentation_score),
                    gpu_utilization: result
                        .gpu_metrics
                        .as_ref()
                        .map(|g| g.gpu_utilization_percent),
                    temperature_celsius: result
                        .gpu_metrics
                        .as_ref()
                        .and_then(|g| g.gpu_temperature_celsius),
                    metadata: HashMap::new(),
                };

                self.add_measurement(name.clone(), result.statistics.mean_ms, context);

                // Analyze benchmark performance
                let analysis = self.analyze_benchmark_performance(result)?;
                analysis_results.insert(name.clone(), analysis);

                // Generate performance insights
                if let Some(insights) = self.generate_benchmark_insights(result) {
                    performance_insights.extend(insights);
                }
            }
        }

        Ok(BenchmarkAnalysis {
            analysis_timestamp: Instant::now(),
            benchmark_results: analysis_results,
            performance_insights,
            comparative_analysis: self.generate_comparative_analysis(results)?,
            optimization_opportunities: self.identify_optimization_opportunities(results)?,
        })
    }

    /// Set baseline for operation
    /// 操作のベースラインを設定
    pub fn set_baseline(&mut self, operation_name: String, baseline_value: f64) {
        self.baselines.insert(operation_name, baseline_value);
    }

    /// Get performance trend for operation
    /// 操作のパフォーマンス傾向を取得
    pub fn get_trend(&mut self, operation_name: &str) -> RusTorchResult<PerformanceTrend> {
        if let Some(cached_trend) = self.trend_cache.get(operation_name) {
            return Ok(cached_trend.clone());
        }

        let trend = self.analyze_operation_trend(operation_name)?;
        self.trend_cache
            .insert(operation_name.to_string(), trend.clone());
        Ok(trend)
    }

    /// Get all current recommendations
    /// 現在の全推奨事項を取得
    pub fn get_recommendations(&self) -> &[OptimizationRecommendation] {
        &self.recommendations
    }

    /// Get recommendations by priority
    /// 優先度別推奨事項を取得
    pub fn get_recommendations_by_priority(
        &self,
        priority: RecommendationPriority,
    ) -> Vec<&OptimizationRecommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.priority == priority)
            .collect()
    }

    /// Clear analysis data
    /// 分析データをクリア
    pub fn clear(&mut self) {
        self.performance_history.clear();
        self.trend_cache.clear();
        self.recommendations.clear();
        self.baselines.clear();
        self.last_analysis = None;
    }

    // Private analysis methods

    fn analyze_operation_trend(&self, operation_name: &str) -> RusTorchResult<PerformanceTrend> {
        let measurements = self
            .performance_history
            .get(operation_name)
            .ok_or_else(|| RusTorchError::Profiling {
                message: format!("No measurements for operation: {}", operation_name),
            })?;

        if measurements.len() < self.config.min_measurements {
            return Ok(PerformanceTrend {
                operation_name: operation_name.to_string(),
                direction: TrendDirection::Unknown,
                change_rate_percent: 0.0,
                confidence: 0.0,
                measurements: measurements.clone(),
                projection: None,
            });
        }

        // Calculate trend using linear regression
        let (slope, confidence) = self.calculate_linear_trend(measurements)?;

        let direction = if slope.abs() < self.config.significance_threshold {
            TrendDirection::Stable
        } else if slope < 0.0 {
            TrendDirection::Improving // Negative slope = decreasing time = better performance
        } else {
            TrendDirection::Degrading
        };

        let change_rate_percent = slope * 100.0;

        // Generate projection if trend is significant
        let projection = if confidence > 0.5 {
            Some(self.generate_projection(measurements, slope, confidence)?)
        } else {
            None
        };

        Ok(PerformanceTrend {
            operation_name: operation_name.to_string(),
            direction,
            change_rate_percent,
            confidence,
            measurements: measurements.clone(),
            projection,
        })
    }

    fn calculate_linear_trend(
        &self,
        measurements: &[TrendMeasurement],
    ) -> RusTorchResult<(f64, f64)> {
        if measurements.len() < 2 {
            return Ok((0.0, 0.0));
        }

        // Convert timestamps to seconds from first measurement
        let first_time = measurements[0].timestamp;
        let x_values: Vec<f64> = measurements
            .iter()
            .map(|m| m.timestamp.duration_since(first_time).as_secs_f64())
            .collect();

        let y_values: Vec<f64> = measurements.iter().map(|m| m.value).collect();

        // Calculate linear regression
        let n = measurements.len() as f64;
        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = y_values.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(&y_values).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = x_values.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        // Calculate R-squared for confidence
        let mean_y = sum_y / n;
        let ss_tot: f64 = y_values.iter().map(|y| (y - mean_y).powi(2)).sum();
        let intercept = (sum_y - slope * sum_x) / n;
        let ss_res: f64 = x_values
            .iter()
            .zip(&y_values)
            .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
            .sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };
        let confidence = r_squared.max(0.0).min(1.0);

        Ok((slope, confidence))
    }

    fn generate_projection(
        &self,
        measurements: &[TrendMeasurement],
        slope: f64,
        confidence: f64,
    ) -> RusTorchResult<PerformanceProjection> {
        let last_measurement = &measurements[measurements.len() - 1];
        let time_horizon = Duration::from_secs(3600); // 1 hour projection

        let next_value = last_measurement.value + slope * time_horizon.as_secs_f64();

        // Calculate uncertainty range based on confidence
        let uncertainty_factor = (1.0 - confidence) * 0.2; // Max 20% uncertainty
        let range_width = next_value * uncertainty_factor;

        Ok(PerformanceProjection {
            next_value,
            confidence,
            time_horizon,
            value_range: (next_value - range_width, next_value + range_width),
        })
    }

    fn generate_recommendations(
        &mut self,
        trends: &HashMap<String, PerformanceTrend>,
    ) -> RusTorchResult<()> {
        self.recommendations.clear();

        for (operation_name, trend) in trends {
            match trend.direction {
                TrendDirection::Degrading
                    if trend.confidence > self.config.min_recommendation_confidence =>
                {
                    let recommendation =
                        self.create_degradation_recommendation(operation_name, trend)?;
                    self.recommendations.push(recommendation);
                }
                TrendDirection::Stable => {
                    // Look for optimization opportunities even in stable performance
                    if let Some(recommendation) =
                        self.create_optimization_recommendation(operation_name, trend)?
                    {
                        self.recommendations.push(recommendation);
                    }
                }
                _ => {}
            }
        }

        // Sort recommendations by priority
        self.recommendations
            .sort_by(|a, b| a.priority.cmp(&b.priority));

        Ok(())
    }

    fn create_degradation_recommendation(
        &self,
        operation_name: &str,
        trend: &PerformanceTrend,
    ) -> RusTorchResult<OptimizationRecommendation> {
        let priority = if trend.change_rate_percent > 20.0 {
            RecommendationPriority::Critical
        } else if trend.change_rate_percent > 10.0 {
            RecommendationPriority::High
        } else {
            RecommendationPriority::Medium
        };

        Ok(OptimizationRecommendation {
            id: format!("degradation_{}", operation_name),
            target: operation_name.to_string(),
            recommendation_type: RecommendationType::Algorithm,
            priority,
            description: format!(
                "Performance degradation detected: {:.1}% slower over recent measurements",
                trend.change_rate_percent
            ),
            action: "Investigate recent changes and profile operation for bottlenecks".to_string(),
            expected_improvement: ExpectedImprovement {
                performance_gain_percent: trend.change_rate_percent,
                memory_reduction_percent: None,
                energy_efficiency_gain_percent: None,
                confidence: trend.confidence,
            },
            complexity: ImplementationComplexity::Moderate,
            evidence: vec![
                format!("Trend confidence: {:.1}%", trend.confidence * 100.0),
                format!("Change rate: {:.1}%", trend.change_rate_percent),
            ],
            timestamp: Instant::now(),
        })
    }

    fn create_optimization_recommendation(
        &self,
        _operation_name: &str,
        _trend: &PerformanceTrend,
    ) -> RusTorchResult<Option<OptimizationRecommendation>> {
        // Placeholder for optimization opportunity detection
        Ok(None)
    }

    fn analyze_benchmark_performance(
        &self,
        _result: &BenchmarkResult,
    ) -> RusTorchResult<BenchmarkPerformanceAnalysis> {
        // Placeholder implementation
        Ok(BenchmarkPerformanceAnalysis {
            stability_score: 0.9,
            efficiency_score: 0.8,
            comparison_to_baseline: None,
            bottleneck_indicators: Vec::new(),
        })
    }

    fn generate_benchmark_insights(
        &self,
        _result: &BenchmarkResult,
    ) -> Option<Vec<PerformanceInsight>> {
        // Placeholder implementation
        None
    }

    fn generate_comparative_analysis(
        &self,
        _results: &HashMap<String, BenchmarkResult>,
    ) -> RusTorchResult<ComparativeAnalysis> {
        // Placeholder implementation
        Ok(ComparativeAnalysis {
            performance_ranking: Vec::new(),
            relative_performance: HashMap::new(),
            category_analysis: HashMap::new(),
        })
    }

    fn identify_optimization_opportunities(
        &self,
        _results: &HashMap<String, BenchmarkResult>,
    ) -> RusTorchResult<Vec<OptimizationOpportunity>> {
        // Placeholder implementation
        Ok(Vec::new())
    }

    fn calculate_overall_score(
        &self,
        _operations: &[crate::profiler::core::OperationMetrics],
    ) -> RusTorchResult<f64> {
        // Simplified overall score calculation
        Ok(0.85) // Placeholder
    }

    fn generate_analysis_summary(
        &self,
        trends: &HashMap<String, PerformanceTrend>,
    ) -> AnalysisSummary {
        let total_operations = trends.len();
        let improving_count = trends
            .values()
            .filter(|t| t.direction == TrendDirection::Improving)
            .count();
        let degrading_count = trends
            .values()
            .filter(|t| t.direction == TrendDirection::Degrading)
            .count();
        let stable_count = trends
            .values()
            .filter(|t| t.direction == TrendDirection::Stable)
            .count();

        AnalysisSummary {
            total_operations,
            improving_count,
            degrading_count,
            stable_count,
            critical_recommendations: self
                .get_recommendations_by_priority(RecommendationPriority::Critical)
                .len(),
            high_recommendations: self
                .get_recommendations_by_priority(RecommendationPriority::High)
                .len(),
        }
    }
}

/// Overall trend analysis results
/// 全体傾向分析結果
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Source session ID
    /// ソースセッションID
    pub session_id: String,
    /// Source session name
    /// ソースセッション名
    pub session_name: String,
    /// Analysis timestamp
    /// 分析タイムスタンプ
    pub analysis_timestamp: Instant,
    /// Trends for individual operations
    /// 個別操作の傾向
    pub operation_trends: HashMap<String, PerformanceTrend>,
    /// Overall performance score (0.0 to 1.0)
    /// 全体パフォーマンススコア（0.0から1.0）
    pub overall_performance_score: f64,
    /// Generated recommendations
    /// 生成された推奨事項
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Analysis summary
    /// 分析サマリー
    pub summary: AnalysisSummary,
}

/// Analysis summary
/// 分析サマリー
#[derive(Debug, Clone)]
pub struct AnalysisSummary {
    /// Total operations analyzed
    /// 分析した総操作数
    pub total_operations: usize,
    /// Operations with improving performance
    /// パフォーマンス向上中の操作数
    pub improving_count: usize,
    /// Operations with degrading performance
    /// パフォーマンス劣化中の操作数
    pub degrading_count: usize,
    /// Operations with stable performance
    /// パフォーマンス安定の操作数
    pub stable_count: usize,
    /// Number of critical recommendations
    /// 重要推奨事項数
    pub critical_recommendations: usize,
    /// Number of high priority recommendations
    /// 高優先度推奨事項数
    pub high_recommendations: usize,
}

/// Benchmark analysis results
/// ベンチマーク分析結果
#[derive(Debug, Clone)]
pub struct BenchmarkAnalysis {
    /// Analysis timestamp
    /// 分析タイムスタンプ
    pub analysis_timestamp: Instant,
    /// Results for individual benchmarks
    /// 個別ベンチマークの結果
    pub benchmark_results: HashMap<String, BenchmarkPerformanceAnalysis>,
    /// Performance insights
    /// パフォーマンス洞察
    pub performance_insights: Vec<PerformanceInsight>,
    /// Comparative analysis
    /// 比較分析
    pub comparative_analysis: ComparativeAnalysis,
    /// Optimization opportunities
    /// 最適化機会
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

// Supporting data structures (placeholders for full implementation)

/// Benchmark performance analysis results
/// ベンチマークパフォーマンス分析結果
#[derive(Debug, Clone)]
pub struct BenchmarkPerformanceAnalysis {
    /// Stability score (0.0 - 1.0)
    /// 安定性スコア（0.0 - 1.0）
    pub stability_score: f64,
    /// Efficiency score (0.0 - 1.0)
    /// 効率性スコア（0.0 - 1.0）
    pub efficiency_score: f64,
    /// Comparison to baseline performance
    /// ベースラインパフォーマンスとの比較
    pub comparison_to_baseline: Option<f64>,
    /// Identified bottleneck indicators
    /// 特定されたボトルネック指標
    pub bottleneck_indicators: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PerformanceInsight {
    pub category: String,
    pub description: String,
    pub impact_level: RecommendationPriority,
}

#[derive(Debug, Clone)]
pub struct ComparativeAnalysis {
    pub performance_ranking: Vec<String>,
    pub relative_performance: HashMap<String, f64>,
    pub category_analysis: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    pub operation: String,
    pub opportunity_type: RecommendationType,
    pub potential_gain: f64,
    pub implementation_effort: ImplementationComplexity,
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = PerformanceAnalyzer::new();
        assert_eq!(analyzer.recommendations.len(), 0);
        assert_eq!(analyzer.performance_history.len(), 0);
    }

    #[test]
    fn test_measurement_addition() {
        let mut analyzer = PerformanceAnalyzer::new();

        let context = MeasurementContext {
            system_load: Some(0.5),
            memory_pressure: None,
            gpu_utilization: None,
            temperature_celsius: None,
            metadata: HashMap::new(),
        };

        analyzer.add_measurement("test_op".to_string(), 100.0, context);

        assert_eq!(analyzer.performance_history.len(), 1);
        assert!(analyzer.performance_history.contains_key("test_op"));
        assert_eq!(analyzer.performance_history["test_op"].len(), 1);
    }

    #[test]
    fn test_trend_analysis_insufficient_data() {
        let analyzer = PerformanceAnalyzer::new();
        let result = analyzer.analyze_operation_trend("nonexistent_op");
        assert!(result.is_err());
    }

    #[test]
    fn test_baseline_setting() {
        let mut analyzer = PerformanceAnalyzer::new();
        analyzer.set_baseline("test_op".to_string(), 50.0);

        assert!(analyzer.baselines.contains_key("test_op"));
        assert_eq!(analyzer.baselines["test_op"], 50.0);
    }

    #[test]
    fn test_linear_trend_calculation() {
        let analyzer = PerformanceAnalyzer::new();
        let start_time = Instant::now();

        let measurements = vec![
            TrendMeasurement {
                timestamp: start_time,
                value: 100.0,
                context: MeasurementContext {
                    system_load: None,
                    memory_pressure: None,
                    gpu_utilization: None,
                    temperature_celsius: None,
                    metadata: HashMap::new(),
                },
            },
            TrendMeasurement {
                timestamp: start_time + Duration::from_secs(60),
                value: 110.0,
                context: MeasurementContext {
                    system_load: None,
                    memory_pressure: None,
                    gpu_utilization: None,
                    temperature_celsius: None,
                    metadata: HashMap::new(),
                },
            },
        ];

        let result = analyzer.calculate_linear_trend(&measurements);
        assert!(result.is_ok());
        let (slope, confidence) = result.unwrap();
        assert!(slope > 0.0); // Increasing trend
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}
