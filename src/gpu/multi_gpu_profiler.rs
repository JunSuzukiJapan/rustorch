//! Multi-GPU Performance Profiler
//! マルチGPUパフォーマンスプロファイラー
//!
//! Advanced profiling system specifically designed for multi-GPU distributed operations.
//! Provides detailed performance metrics, bottleneck identification, and optimization recommendations.

use crate::error::{RusTorchError, RusTorchResult};
use crate::gpu::distributed_training::{DistributedTrainer, TrainingMetrics};
use crate::gpu::multi_gpu::{MultiGpuContext, ParallelismStrategy};
use crate::gpu::sync_primitives::{MultiGpuBarrier, StreamManager};
use crate::profiler::ProfilerConfig;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Multi-GPU performance profiler
#[derive(Debug)]
pub struct MultiGpuProfiler {
    /// Performance metrics per GPU
    gpu_metrics: Arc<RwLock<HashMap<usize, GpuMetrics>>>,
    /// Communication metrics
    comm_metrics: Arc<RwLock<CommunicationMetrics>>,
    /// Training session metrics
    training_metrics: Arc<RwLock<TrainingSessionMetrics>>,
    /// Profiling start time
    start_time: Instant,
    /// Profiling enabled flag
    enabled: bool,
    /// Performance history
    history: Vec<PerformanceSnapshot>,
}

/// GPU-specific performance metrics
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// GPU device ID
    pub device_id: usize,
    /// Compute utilization percentage
    pub compute_utilization: f32,
    /// Memory utilization percentage
    pub memory_utilization: f32,
    /// Temperature in Celsius
    pub temperature: f32,
    /// Power consumption in watts
    pub power_consumption: f32,
    /// Memory bandwidth utilization
    pub memory_bandwidth: f32,
    /// Kernel execution times
    pub kernel_times: HashMap<String, Duration>,
    /// Memory transfer times
    pub transfer_times: Vec<Duration>,
    /// Stream synchronization times
    pub sync_times: Vec<Duration>,
}

/// Communication performance metrics
#[derive(Debug, Clone)]
pub struct CommunicationMetrics {
    /// All-reduce operation times
    pub allreduce_times: HashMap<String, Vec<Duration>>,
    /// Broadcast operation times
    pub broadcast_times: HashMap<String, Vec<Duration>>,
    /// P2P transfer times
    pub p2p_times: HashMap<(usize, usize), Vec<Duration>>,
    /// Barrier synchronization times
    pub barrier_times: Vec<Duration>,
    /// Communication overhead percentage
    pub overhead_percentage: f32,
    /// Network bandwidth utilization
    pub bandwidth_utilization: f32,
}

/// Training session performance metrics
#[derive(Debug, Clone)]
pub struct TrainingSessionMetrics {
    /// Total training steps
    pub total_steps: usize,
    /// Average step time
    pub avg_step_time: Duration,
    /// Forward pass times
    pub forward_times: Vec<Duration>,
    /// Backward pass times
    pub backward_times: Vec<Duration>,
    /// Parameter update times
    pub update_times: Vec<Duration>,
    /// Gradient synchronization times
    pub gradient_sync_times: Vec<Duration>,
    /// Throughput (samples per second)
    pub throughput: f32,
    /// GPU efficiency score
    pub efficiency_score: f32,
}

/// Performance snapshot for trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// GPU metrics snapshot
    pub gpu_metrics: HashMap<usize, GpuMetrics>,
    /// Communication metrics snapshot
    pub comm_metrics: CommunicationMetrics,
    /// Training metrics snapshot
    pub training_metrics: TrainingSessionMetrics,
}

/// Performance analysis report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Overall performance score (0-100)
    pub overall_score: f32,
    /// GPU efficiency scores
    pub gpu_efficiency: HashMap<usize, f32>,
    /// Communication efficiency
    pub communication_efficiency: f32,
    /// Bottleneck identification
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Performance trends
    pub trends: PerformanceTrends,
    /// Session duration
    pub session_duration: Duration,
    /// Total training steps
    pub total_training_steps: usize,
    /// Average step time
    pub avg_step_time: Duration,
    /// Communication overhead ratio
    pub communication_overhead_ratio: f32,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity (0-100)
    pub severity: f32,
    /// Description
    pub description: String,
    /// Affected components
    pub affected_components: Vec<String>,
    /// Impact on overall performance
    pub performance_impact: f32,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckType {
    /// GPU compute bound
    ComputeBound,
    /// Memory bandwidth bound
    MemoryBound,
    /// Communication bound
    CommunicationBound,
    /// Synchronization bound
    SynchronizationBound,
    /// Load imbalance
    LoadImbalance,
    /// Memory fragmentation
    MemoryFragmentation,
}

/// Optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level (1-10)
    pub priority: u8,
    /// Expected performance gain percentage
    pub expected_gain: f32,
    /// Implementation complexity (1-10)
    pub complexity: u8,
    /// Description
    pub description: String,
    /// Specific actions to take
    pub actions: Vec<String>,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationType {
    /// Adjust parallelism strategy
    ParallelismStrategy,
    /// Optimize memory usage
    MemoryOptimization,
    /// Improve communication patterns
    CommunicationOptimization,
    /// Load balancing adjustment
    LoadBalancing,
    /// Batch size optimization
    BatchSizeOptimization,
    /// Hardware configuration
    HardwareConfiguration,
}

/// Performance trends analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Throughput trend (samples/sec over time)
    pub throughput_trend: Vec<(Instant, f32)>,
    /// GPU utilization trend
    pub gpu_utilization_trend: HashMap<usize, Vec<(Instant, f32)>>,
    /// Communication overhead trend
    pub communication_overhead_trend: Vec<(Instant, f32)>,
    /// Memory efficiency trend
    pub memory_efficiency_trend: Vec<(Instant, f32)>,
}

impl MultiGpuProfiler {
    /// Create new multi-GPU profiler
    pub fn new(gpu_ids: Vec<usize>, config: ProfilerConfig) -> RusTorchResult<Self> {
        // Initialize GPU metrics for each GPU
        let mut gpu_metrics_map = HashMap::new();
        for gpu_id in gpu_ids {
            gpu_metrics_map.insert(gpu_id, GpuMetrics::new(gpu_id));
        }

        Ok(Self {
            gpu_metrics: Arc::new(RwLock::new(gpu_metrics_map)),
            comm_metrics: Arc::new(RwLock::new(CommunicationMetrics::new())),
            training_metrics: Arc::new(RwLock::new(TrainingSessionMetrics::new())),
            start_time: Instant::now(),
            enabled: config.enable_gpu_profiling,
            history: Vec::new(),
        })
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
        self.start_time = Instant::now();
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Start profiling a multi-GPU operation
    pub fn start_operation(
        &mut self,
        operation_name: &str,
        gpu_ids: &[usize],
    ) -> RusTorchResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // Initialize GPU metrics for all participating GPUs
        let mut gpu_metrics = self.gpu_metrics.write().unwrap();
        for &gpu_id in gpu_ids {
            gpu_metrics.insert(gpu_id, GpuMetrics::new(gpu_id));
        }

        // Record operation start
        println!(
            "🔍 Profiling started: {} (GPUs: {:?})",
            operation_name, gpu_ids
        );
        Ok(())
    }

    /// Record GPU kernel execution
    pub fn record_kernel_execution(
        &self,
        gpu_id: usize,
        kernel_name: &str,
        execution_time: Duration,
    ) -> RusTorchResult<()> {
        if !self.enabled {
            return Ok(());
        }

        if let Ok(mut gpu_metrics) = self.gpu_metrics.write() {
            if let Some(metrics) = gpu_metrics.get_mut(&gpu_id) {
                metrics
                    .kernel_times
                    .insert(kernel_name.to_string(), execution_time);
            }
        }

        Ok(())
    }

    /// Record communication operation
    pub fn record_communication(
        &self,
        operation_type: &str,
        duration: Duration,
        data_size: usize,
    ) -> RusTorchResult<()> {
        if !self.enabled {
            return Ok(());
        }

        if let Ok(mut comm_metrics) = self.comm_metrics.write() {
            match operation_type {
                "allreduce" => {
                    comm_metrics
                        .allreduce_times
                        .entry("default".to_string())
                        .or_insert_with(Vec::new)
                        .push(duration);
                }
                "broadcast" => {
                    comm_metrics
                        .broadcast_times
                        .entry("default".to_string())
                        .or_insert_with(Vec::new)
                        .push(duration);
                }
                "barrier" => {
                    comm_metrics.barrier_times.push(duration);
                }
                _ => {}
            }

            // Calculate bandwidth
            let bandwidth_gbps = (data_size as f64 * 8.0) / (duration.as_secs_f64() * 1e9);
            comm_metrics.bandwidth_utilization = bandwidth_gbps as f32;
        }

        Ok(())
    }

    /// Record training step metrics
    pub fn record_training_step(
        &self,
        forward_time: Duration,
        backward_time: Duration,
        update_time: Duration,
        sync_time: Duration,
        throughput: f32,
    ) -> RusTorchResult<()> {
        if !self.enabled {
            return Ok(());
        }

        if let Ok(mut training_metrics) = self.training_metrics.write() {
            training_metrics.total_steps += 1;
            training_metrics.forward_times.push(forward_time);
            training_metrics.backward_times.push(backward_time);
            training_metrics.update_times.push(update_time);
            training_metrics.gradient_sync_times.push(sync_time);
            training_metrics.throughput = throughput;

            // Calculate average step time
            let total_step_time = forward_time + backward_time + update_time + sync_time;
            training_metrics.avg_step_time = (training_metrics.avg_step_time
                * (training_metrics.total_steps - 1) as u32
                + total_step_time)
                / training_metrics.total_steps as u32;
        }

        Ok(())
    }

    /// Analyze performance and generate report
    pub fn analyze_performance(&self) -> PerformanceReport {
        if !self.enabled {
            return PerformanceReport::default();
        }

        // Take performance snapshot
        self.take_snapshot();

        let gpu_metrics = self.gpu_metrics.read().unwrap();
        let comm_metrics = self.comm_metrics.read().unwrap();
        let training_metrics = self.training_metrics.read().unwrap();

        // Calculate overall performance score
        let overall_score =
            self.calculate_overall_score(&gpu_metrics, &comm_metrics, &training_metrics);

        // Calculate GPU efficiency scores
        let mut gpu_efficiency = HashMap::new();
        for (&gpu_id, metrics) in gpu_metrics.iter() {
            let efficiency = self.calculate_gpu_efficiency(metrics);
            gpu_efficiency.insert(gpu_id, efficiency);
        }

        // Calculate communication efficiency
        let communication_efficiency = self.calculate_communication_efficiency(&comm_metrics);

        // Identify bottlenecks
        let bottlenecks = self.identify_bottlenecks(&gpu_metrics, &comm_metrics, &training_metrics);

        // Generate optimization recommendations
        let recommendations = self.generate_recommendations(&bottlenecks, &gpu_metrics);

        // Analyze trends
        let trends = self.analyze_trends();

        PerformanceReport {
            overall_score,
            gpu_efficiency,
            communication_efficiency,
            bottlenecks,
            recommendations,
            trends,
            session_duration: self.start_time.elapsed(),
            total_training_steps: training_metrics.total_steps,
            avg_step_time: if training_metrics.total_steps > 0 {
                training_metrics.avg_step_time
            } else {
                Duration::ZERO
            },
            communication_overhead_ratio: comm_metrics.overhead_percentage / 100.0,
        }
    }

    /// Take performance snapshot
    fn take_snapshot(&self) {
        // Note: Snapshot functionality disabled for const-correct implementation
        // Would require Arc<Mutex<Vec<PerformanceSnapshot>>> for history
    }

    /// Calculate overall performance score
    fn calculate_overall_score(
        &self,
        gpu_metrics: &HashMap<usize, GpuMetrics>,
        comm_metrics: &CommunicationMetrics,
        training_metrics: &TrainingSessionMetrics,
    ) -> f32 {
        let mut scores = Vec::new();

        // GPU utilization score (weight: 40%)
        let avg_gpu_utilization: f32 = gpu_metrics
            .values()
            .map(|m| m.compute_utilization)
            .sum::<f32>()
            / gpu_metrics.len() as f32;
        scores.push(avg_gpu_utilization * 0.4);

        // Communication efficiency score (weight: 30%)
        let comm_efficiency = 100.0 - comm_metrics.overhead_percentage;
        scores.push(comm_efficiency * 0.3);

        // Training efficiency score (weight: 30%)
        scores.push(training_metrics.efficiency_score * 0.3);

        scores.iter().sum()
    }

    /// Calculate GPU efficiency
    fn calculate_gpu_efficiency(&self, metrics: &GpuMetrics) -> f32 {
        let compute_score = metrics.compute_utilization;
        let memory_score = 100.0 - metrics.memory_utilization; // Lower memory usage is better
        let thermal_score = if metrics.temperature < 80.0 {
            100.0
        } else {
            100.0 - (metrics.temperature - 80.0) * 2.0
        };

        (compute_score + memory_score + thermal_score) / 3.0
    }

    /// Calculate communication efficiency
    fn calculate_communication_efficiency(&self, metrics: &CommunicationMetrics) -> f32 {
        let overhead_score = 100.0 - metrics.overhead_percentage;
        let bandwidth_score = metrics.bandwidth_utilization;

        (overhead_score + bandwidth_score) / 2.0
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(
        &self,
        gpu_metrics: &HashMap<usize, GpuMetrics>,
        comm_metrics: &CommunicationMetrics,
        _training_metrics: &TrainingSessionMetrics,
    ) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        // Check for compute bottlenecks
        for (gpu_id, metrics) in gpu_metrics {
            if metrics.compute_utilization > 95.0 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::ComputeBound,
                    severity: metrics.compute_utilization,
                    description: format!(
                        "GPU {} is compute bound ({}% utilization)",
                        gpu_id, metrics.compute_utilization
                    ),
                    affected_components: vec![format!("GPU {}", gpu_id)],
                    performance_impact: (metrics.compute_utilization - 80.0) / 20.0 * 100.0,
                });
            }

            if metrics.memory_utilization > 90.0 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::MemoryBound,
                    severity: metrics.memory_utilization,
                    description: format!(
                        "GPU {} is memory bound ({}% utilization)",
                        gpu_id, metrics.memory_utilization
                    ),
                    affected_components: vec![format!("GPU {} memory", gpu_id)],
                    performance_impact: (metrics.memory_utilization - 80.0) / 20.0 * 100.0,
                });
            }
        }

        // Check for communication bottlenecks
        if comm_metrics.overhead_percentage > 30.0 {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::CommunicationBound,
                severity: comm_metrics.overhead_percentage,
                description: format!(
                    "High communication overhead ({}%)",
                    comm_metrics.overhead_percentage
                ),
                affected_components: vec!["Multi-GPU communication".to_string()],
                performance_impact: comm_metrics.overhead_percentage,
            });
        }

        bottlenecks
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        bottlenecks: &[PerformanceBottleneck],
        gpu_metrics: &HashMap<usize, GpuMetrics>,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        for bottleneck in bottlenecks {
            match bottleneck.bottleneck_type {
                BottleneckType::ComputeBound => {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: RecommendationType::ParallelismStrategy,
                        priority: 8,
                        expected_gain: 15.0,
                        complexity: 6,
                        description: "Consider switching to model parallelism to better distribute compute load".to_string(),
                        actions: vec![
                            "Evaluate model parallelism vs data parallelism".to_string(),
                            "Implement pipeline parallelism for large models".to_string(),
                            "Optimize kernel fusion to reduce compute overhead".to_string(),
                        ],
                    });
                }
                BottleneckType::MemoryBound => {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: RecommendationType::MemoryOptimization,
                        priority: 9,
                        expected_gain: 25.0,
                        complexity: 4,
                        description: "Optimize memory usage to reduce pressure on GPU memory"
                            .to_string(),
                        actions: vec![
                            "Implement gradient checkpointing".to_string(),
                            "Use mixed precision training".to_string(),
                            "Optimize batch sizes per GPU".to_string(),
                        ],
                    });
                }
                BottleneckType::CommunicationBound => {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: RecommendationType::CommunicationOptimization,
                        priority: 7,
                        expected_gain: 20.0,
                        complexity: 5,
                        description: "Optimize communication patterns to reduce overhead"
                            .to_string(),
                        actions: vec![
                            "Implement gradient compression".to_string(),
                            "Use NCCL for optimized all-reduce".to_string(),
                            "Overlap computation with communication".to_string(),
                        ],
                    });
                }
                _ => {}
            }
        }

        // Load balancing recommendations
        let utilizations: Vec<f32> = gpu_metrics
            .values()
            .map(|m| m.compute_utilization)
            .collect();

        if let (Some(&min_util), Some(&max_util)) = (
            utilizations.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
            utilizations.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
        ) {
            if max_util - min_util > 20.0 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::LoadBalancing,
                    priority: 8,
                    expected_gain: (max_util - min_util) / 2.0,
                    complexity: 7,
                    description: format!(
                        "Load imbalance detected: {:.1}% - {:.1}% utilization range",
                        min_util, max_util
                    ),
                    actions: vec![
                        "Implement dynamic load balancing".to_string(),
                        "Adjust work distribution across GPUs".to_string(),
                        "Consider different parallelism strategies".to_string(),
                    ],
                });
            }
        }

        recommendations
    }

    /// Analyze performance trends
    fn analyze_trends(&self) -> PerformanceTrends {
        let throughput_trend = Vec::new();
        let gpu_utilization_trend = HashMap::new();
        let communication_overhead_trend = Vec::new();
        let memory_efficiency_trend = Vec::new();

        // Note: Trend analysis disabled since history snapshots are not available
        // in const-correct implementation. Return empty trends.

        PerformanceTrends {
            throughput_trend,
            gpu_utilization_trend,
            communication_overhead_trend,
            memory_efficiency_trend,
        }
    }

    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> PerformanceReport {
        self.analyze_performance()
    }

    /// Export profiling data for external analysis
    pub fn export_data(&self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();

        // Export GPU metrics
        let gpu_data: HashMap<String, serde_json::Value> = self
            .gpu_metrics
            .read()
            .unwrap()
            .iter()
            .map(|(&gpu_id, metrics)| {
                (
                    format!("gpu_{}", gpu_id),
                    serde_json::json!({
                        "compute_utilization": metrics.compute_utilization,
                        "memory_utilization": metrics.memory_utilization,
                        "temperature": metrics.temperature,
                        "power_consumption": metrics.power_consumption,
                    }),
                )
            })
            .collect();
        data.insert(
            "gpu_metrics".to_string(),
            serde_json::Value::Object(gpu_data.into_iter().collect()),
        );

        // Export communication metrics
        if let Ok(comm_metrics) = self.comm_metrics.read() {
            data.insert(
                "communication_metrics".to_string(),
                serde_json::json!({
                    "overhead_percentage": comm_metrics.overhead_percentage,
                    "bandwidth_utilization": comm_metrics.bandwidth_utilization,
                    "barrier_count": comm_metrics.barrier_times.len(),
                }),
            );
        }

        // Export training metrics
        if let Ok(training_metrics) = self.training_metrics.read() {
            data.insert(
                "training_metrics".to_string(),
                serde_json::json!({
                    "total_steps": training_metrics.total_steps,
                    "avg_step_time_ms": training_metrics.avg_step_time.as_millis(),
                    "throughput": training_metrics.throughput,
                    "efficiency_score": training_metrics.efficiency_score,
                }),
            );
        }

        data
    }
}

impl Default for MultiGpuProfiler {
    fn default() -> Self {
        Self::new(vec![0], ProfilerConfig::default())
            .unwrap_or_else(|_| panic!("Failed to create default profiler"))
    }
}

// Implementation of helper structs

impl GpuMetrics {
    pub fn new(device_id: usize) -> Self {
        Self {
            device_id,
            compute_utilization: 0.0,
            memory_utilization: 0.0,
            temperature: 25.0,
            power_consumption: 0.0,
            memory_bandwidth: 0.0,
            kernel_times: HashMap::new(),
            transfer_times: Vec::new(),
            sync_times: Vec::new(),
        }
    }
}

impl CommunicationMetrics {
    pub fn new() -> Self {
        Self {
            allreduce_times: HashMap::new(),
            broadcast_times: HashMap::new(),
            p2p_times: HashMap::new(),
            barrier_times: Vec::new(),
            overhead_percentage: 0.0,
            bandwidth_utilization: 0.0,
        }
    }
}

impl TrainingSessionMetrics {
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            avg_step_time: Duration::ZERO,
            forward_times: Vec::new(),
            backward_times: Vec::new(),
            update_times: Vec::new(),
            gradient_sync_times: Vec::new(),
            throughput: 0.0,
            efficiency_score: 0.0,
        }
    }
}

/// Convenience function to profile multi-GPU operations
pub fn profile_multi_gpu_operation<F, R>(
    operation_name: &str,
    gpu_ids: &[usize],
    operation: F,
) -> RusTorchResult<(R, PerformanceReport)>
where
    F: FnOnce() -> RusTorchResult<R>,
{
    let mut profiler = MultiGpuProfiler::new(gpu_ids.to_vec(), ProfilerConfig::default())?;
    profiler.enable();
    profiler.start_operation(operation_name, gpu_ids)?;

    let start_time = Instant::now();
    let result = operation()?;
    let duration = start_time.elapsed();

    // Record the operation
    profiler.record_communication("operation", duration, 0)?;

    let report = profiler.generate_report();
    Ok((result, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_gpu_profiler_creation() -> RusTorchResult<()> {
        let profiler = MultiGpuProfiler::new(vec![0], ProfilerConfig::default())?;
        assert!(!profiler.enabled);
        assert_eq!(profiler.history.len(), 0);
        Ok(())
    }

    #[test]
    fn test_profiler_enable_disable() -> RusTorchResult<()> {
        let mut profiler = MultiGpuProfiler::new(vec![0], ProfilerConfig::default())?;

        profiler.enable();
        assert!(profiler.enabled);

        profiler.disable();
        assert!(!profiler.enabled);
        Ok(())
    }

    #[test]
    fn test_kernel_execution_recording() -> RusTorchResult<()> {
        let mut profiler = MultiGpuProfiler::new(vec![0], ProfilerConfig::default())?;
        profiler.enable();

        let result = profiler.record_kernel_execution(0, "test_kernel", Duration::from_millis(10));
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_communication_recording() -> RusTorchResult<()> {
        let mut profiler = MultiGpuProfiler::new(vec![0, 1], ProfilerConfig::default())?;
        profiler.enable();

        let result = profiler.record_communication("allreduce", Duration::from_millis(5), 1024);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_training_step_recording() -> RusTorchResult<()> {
        let mut profiler = MultiGpuProfiler::new(vec![0, 1], ProfilerConfig::default())?;
        profiler.enable();

        let result = profiler.record_training_step(
            Duration::from_millis(10),
            Duration::from_millis(15),
            Duration::from_millis(5),
            Duration::from_millis(3),
            100.0,
        );
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_profile_multi_gpu_operation() -> RusTorchResult<()> {
        let result = profile_multi_gpu_operation("test_operation", &[0], || {
            std::thread::sleep(Duration::from_millis(10));
            Ok(42)
        });

        assert!(result.is_ok());
        let (value, _report) = result.unwrap();
        assert_eq!(value, 42);
        Ok(())
    }
}

impl Default for PerformanceReport {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            gpu_efficiency: HashMap::new(),
            communication_efficiency: 0.0,
            bottlenecks: Vec::new(),
            recommendations: Vec::new(),
            trends: PerformanceTrends::default(),
            session_duration: Duration::ZERO,
            total_training_steps: 0,
            avg_step_time: Duration::ZERO,
            communication_overhead_ratio: 0.0,
        }
    }
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            throughput_trend: Vec::new(),
            gpu_utilization_trend: HashMap::new(),
            communication_overhead_trend: Vec::new(),
            memory_efficiency_trend: Vec::new(),
        }
    }
}
