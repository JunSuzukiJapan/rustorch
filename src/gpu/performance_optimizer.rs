//! Performance optimization system for GPU kernels
//! GPUカーネルのパフォーマンス最適化システム
//!
//! This module provides dynamic optimization of kernel parameters, memory layout,
//! and execution strategies to maximize performance across different devices.
//! このモジュールは、カーネルパラメータ、メモリレイアウト、実行戦略の動的最適化を提供し、
//! 異なるデバイス間でパフォーマンスを最大化します。

use super::unified_kernel::{KernelOp, KernelParams, KernelMetrics};
use super::kernel_selector::{PerformanceDatabase, WorkloadProfile};
use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Optimization target metrics
/// 最適化対象メトリクス
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationTarget {
    /// Minimize execution time
    /// 実行時間を最小化
    MinimizeTime,
    /// Maximize throughput
    /// スループットを最大化
    MaximizeThroughput,
    /// Minimize energy consumption
    /// エネルギー消費を最小化
    MinimizeEnergy,
    /// Balance time and memory usage
    /// 時間とメモリ使用量のバランス
    BalanceTimeMemory,
}

/// Parameter optimization configuration
/// パラメータ最適化設定
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Optimization target
    /// 最適化目標
    pub target: OptimizationTarget,
    /// Maximum optimization iterations
    /// 最大最適化反復回数
    pub max_iterations: usize,
    /// Convergence threshold (relative improvement)
    /// 収束閾値（相対改善）
    pub convergence_threshold: f64,
    /// Enable cache for optimization results
    /// 最適化結果のキャッシュを有効化
    pub enable_cache: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            target: OptimizationTarget::MinimizeTime,
            max_iterations: 10,
            convergence_threshold: 0.01, // 1% improvement threshold
            enable_cache: true,
        }
    }
}

/// Optimization result
/// 最適化結果
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimized kernel parameters
    /// 最適化されたカーネルパラメータ
    pub params: KernelParams,
    /// Performance metrics achieved
    /// 達成されたパフォーマンスメトリクス
    pub metrics: KernelMetrics,
    /// Number of optimization iterations
    /// 最適化反復回数
    pub iterations: usize,
    /// Improvement factor (1.0 = no improvement, >1.0 = better)
    /// 改善係数（1.0 = 改善なし、>1.0 = より良い）
    pub improvement_factor: f64,
}

/// Memory layout optimizer
/// メモリレイアウト最適化器
#[derive(Debug, Clone)]
pub struct MemoryLayoutOptimizer {
    /// Device-specific memory alignment requirements
    /// デバイス固有のメモリアライメント要件
    alignment_requirements: HashMap<DeviceType, usize>,
    /// Cache of optimal layouts
    /// 最適レイアウトのキャッシュ
    layout_cache: Arc<RwLock<HashMap<(DeviceType, KernelOp, Vec<usize>), Vec<usize>>>>,
}

impl MemoryLayoutOptimizer {
    pub fn new() -> Self {
        let mut alignment_requirements = HashMap::new();
        
        // Set device-specific alignment requirements
        alignment_requirements.insert(DeviceType::Cpu, 32); // AVX alignment
        alignment_requirements.insert(DeviceType::Cuda(0), 128); // CUDA coalescing
        alignment_requirements.insert(DeviceType::Metal(0), 64); // Metal alignment
        alignment_requirements.insert(DeviceType::OpenCL(0), 64); // OpenCL alignment

        Self {
            alignment_requirements,
            layout_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Optimize memory layout for given device and operation
    /// 指定されたデバイスと操作のメモリレイアウトを最適化
    pub fn optimize_layout(&self, device: DeviceType, op: KernelOp, shape: &[usize]) -> Vec<usize> {
        let cache_key = (device, op, shape.to_vec());
        
        // Check cache first
        if let Ok(cache) = self.layout_cache.read() {
            if let Some(cached_layout) = cache.get(&cache_key) {
                return cached_layout.clone();
            }
        }

        // Compute optimal layout
        let optimal_layout = self.compute_optimal_layout(device, op, shape);
        
        // Cache result
        if let Ok(mut cache) = self.layout_cache.write() {
            cache.insert(cache_key, optimal_layout.clone());
        }

        optimal_layout
    }

    fn compute_optimal_layout(&self, device: DeviceType, op: KernelOp, shape: &[usize]) -> Vec<usize> {
        let alignment = self.alignment_requirements.get(&device).copied().unwrap_or(32);
        
        match op {
            KernelOp::MatMul => {
                // For matrix multiplication, prefer row-major with proper alignment
                self.align_shape_for_matmul(shape, alignment)
            },
            KernelOp::Conv2D => {
                // For convolution, optimize for spatial locality
                self.align_shape_for_conv2d(shape, alignment)
            },
            _ => {
                // Default: align to memory boundary
                self.align_shape_default(shape, alignment)
            }
        }
    }

    fn align_shape_for_matmul(&self, shape: &[usize], alignment: usize) -> Vec<usize> {
        let mut aligned_shape = shape.to_vec();
        if aligned_shape.len() >= 2 {
            // Align the inner dimension for memory coalescing
            let last_idx = aligned_shape.len() - 1;
            aligned_shape[last_idx] = align_up(aligned_shape[last_idx], alignment / 4); // Assume 4-byte elements
        }
        aligned_shape
    }

    fn align_shape_for_conv2d(&self, shape: &[usize], alignment: usize) -> Vec<usize> {
        let mut aligned_shape = shape.to_vec();
        // For convolution, align channel dimension for vectorization
        if aligned_shape.len() >= 3 {
            let channel_idx = aligned_shape.len() - 3; // Assume NCHW format
            aligned_shape[channel_idx] = align_up(aligned_shape[channel_idx], alignment / 4);
        }
        aligned_shape
    }

    fn align_shape_default(&self, shape: &[usize], alignment: usize) -> Vec<usize> {
        let mut aligned_shape = shape.to_vec();
        if !aligned_shape.is_empty() {
            let last_idx = aligned_shape.len() - 1;
            aligned_shape[last_idx] = align_up(aligned_shape[last_idx], alignment / 4);
        }
        aligned_shape
    }
}

impl Default for MemoryLayoutOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Kernel parameter optimizer
/// カーネルパラメータ最適化器
pub struct KernelParameterOptimizer {
    /// Performance database for historical data
    /// 履歴データのためのパフォーマンスデータベース
    performance_db: Arc<RwLock<PerformanceDatabase>>,
    /// Memory layout optimizer
    /// メモリレイアウト最適化器
    memory_optimizer: MemoryLayoutOptimizer,
    /// Cache for optimization results
    /// 最適化結果のキャッシュ
    optimization_cache: Arc<RwLock<HashMap<(DeviceType, KernelOp, Vec<usize>), OptimizationResult>>>,
}

impl KernelParameterOptimizer {
    pub fn new(performance_db: Arc<RwLock<PerformanceDatabase>>) -> Self {
        Self {
            performance_db,
            memory_optimizer: MemoryLayoutOptimizer::new(),
            optimization_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Optimize kernel parameters for given operation
    /// 指定された操作のカーネルパラメータを最適化
    pub fn optimize<T: Float>(
        &self,
        device: DeviceType,
        op: KernelOp,
        inputs: &[&Tensor<T>],
        base_params: &KernelParams,
        config: &OptimizationConfig,
    ) -> RusTorchResult<OptimizationResult> {
        let workload = WorkloadProfile::analyze(inputs, op);
        let cache_key = (device, op, inputs.iter().map(|t| t.shape().to_vec()).collect::<Vec<_>>().concat());

        // Check cache if enabled
        if config.enable_cache {
            if let Ok(cache) = self.optimization_cache.read() {
                if let Some(cached_result) = cache.get(&cache_key) {
                    return Ok(cached_result.clone());
                }
            }
        }

        // Perform optimization
        let result = self.perform_optimization(device, op, &workload, base_params, config)?;

        // Cache result if enabled
        if config.enable_cache {
            if let Ok(mut cache) = self.optimization_cache.write() {
                cache.insert(cache_key, result.clone());
            }
        }

        Ok(result)
    }

    fn perform_optimization(
        &self,
        device: DeviceType,
        op: KernelOp,
        workload: &WorkloadProfile,
        base_params: &KernelParams,
        config: &OptimizationConfig,
    ) -> RusTorchResult<OptimizationResult> {
        let mut best_params = base_params.clone();
        let mut best_metrics = KernelMetrics::default();
        let mut best_score = f64::INFINITY;
        let mut iterations = 0;

        // Optimize memory layout
        if !workload.total_elements == 0 && !base_params.input_shapes.is_empty() {
            for (i, shape) in base_params.input_shapes.iter().enumerate() {
                let optimized_shape = self.memory_optimizer.optimize_layout(device, op, shape);
                if i < best_params.input_shapes.len() {
                    best_params.input_shapes[i] = optimized_shape;
                }
            }
        }

        // Iterative parameter optimization
        for iteration in 0..config.max_iterations {
            iterations = iteration + 1;

            // Generate parameter variations
            let param_variations = self.generate_parameter_variations(&best_params, device, op, iteration);

            for params in param_variations {
                // Simulate execution with these parameters
                let simulated_metrics = self.simulate_execution(device, op, workload, &params)?;
                let score = self.calculate_optimization_score(&simulated_metrics, config.target);

                if score < best_score {
                    let improvement = (best_score - score) / best_score;
                    best_score = score;
                    best_metrics = simulated_metrics;
                    best_params = params;

                    // Check for convergence
                    if improvement < config.convergence_threshold {
                        break;
                    }
                }
            }
        }

        let improvement_factor = if best_score > 0.0 {
            self.calculate_improvement_factor(&base_params, &best_params, device, op, workload)
        } else {
            1.0
        };

        Ok(OptimizationResult {
            params: best_params,
            metrics: best_metrics,
            iterations,
            improvement_factor,
        })
    }

    fn generate_parameter_variations(
        &self,
        base_params: &KernelParams,
        device: DeviceType,
        op: KernelOp,
        iteration: usize,
    ) -> Vec<KernelParams> {
        let mut variations = Vec::new();
        
        // Start with base parameters
        variations.push(base_params.clone());

        // Generate device-specific parameter variations
        match device {
            DeviceType::Cuda(_) => {
                self.generate_cuda_variations(base_params, op, iteration, &mut variations);
            },
            DeviceType::Metal(_) => {
                self.generate_metal_variations(base_params, op, iteration, &mut variations);
            },
            DeviceType::OpenCL(_) => {
                self.generate_opencl_variations(base_params, op, iteration, &mut variations);
            },
            DeviceType::Cpu => {
                self.generate_cpu_variations(base_params, op, iteration, &mut variations);
            },
        }

        variations
    }

    fn generate_cuda_variations(
        &self,
        base_params: &KernelParams,
        op: KernelOp,
        _iteration: usize,
        variations: &mut Vec<KernelParams>,
    ) {
        match op {
            KernelOp::MatMul => {
                // Try different tile sizes for CUDA matrix multiplication
                for tile_size in [16, 32, 64] {
                    let mut params = base_params.clone();
                    params.extra_params.insert("cuda_tile_size".to_string(), tile_size as f64);
                    variations.push(params);
                }
            },
            KernelOp::Conv2D => {
                // Try different cuDNN algorithms
                for algo in [0, 1, 2, 3] {
                    let mut params = base_params.clone();
                    params.extra_params.insert("cudnn_algorithm".to_string(), algo as f64);
                    variations.push(params);
                }
            },
            _ => {
                // Generic CUDA optimizations
                for block_size in [128, 256, 512] {
                    let mut params = base_params.clone();
                    params.extra_params.insert("cuda_block_size".to_string(), block_size as f64);
                    variations.push(params);
                }
            }
        }
    }

    fn generate_metal_variations(
        &self,
        base_params: &KernelParams,
        _op: KernelOp,
        _iteration: usize,
        variations: &mut Vec<KernelParams>,
    ) {
        // Metal-specific parameter variations
        for threads_per_group in [32, 64, 128] {
            let mut params = base_params.clone();
            params.extra_params.insert("metal_threads_per_group".to_string(), threads_per_group as f64);
            variations.push(params);
        }
    }

    fn generate_opencl_variations(
        &self,
        base_params: &KernelParams,
        _op: KernelOp,
        _iteration: usize,
        variations: &mut Vec<KernelParams>,
    ) {
        // OpenCL-specific parameter variations
        for local_size in [64, 128, 256] {
            let mut params = base_params.clone();
            params.extra_params.insert("opencl_local_size".to_string(), local_size as f64);
            variations.push(params);
        }
    }

    fn generate_cpu_variations(
        &self,
        base_params: &KernelParams,
        _op: KernelOp,
        _iteration: usize,
        variations: &mut Vec<KernelParams>,
    ) {
        // CPU-specific parameter variations (e.g., thread count)
        for thread_count in [1, 2, 4, 8] {
            let mut params = base_params.clone();
            params.extra_params.insert("cpu_threads".to_string(), thread_count as f64);
            variations.push(params);
        }
    }

    fn simulate_execution(
        &self,
        device: DeviceType,
        op: KernelOp,
        workload: &WorkloadProfile,
        params: &KernelParams,
    ) -> RusTorchResult<KernelMetrics> {
        // Check historical performance data first
        if let Ok(db) = self.performance_db.read() {
            if let Some(historical_metrics) = db.get_performance(device, op, workload.total_elements) {
                // Adjust based on parameter changes
                return Ok(self.adjust_metrics_for_params(historical_metrics, params));
            }
        }

        // Fallback to performance model
        Ok(self.estimate_performance(device, op, workload, params))
    }

    fn adjust_metrics_for_params(&self, base_metrics: &KernelMetrics, params: &KernelParams) -> KernelMetrics {
        let mut adjusted_metrics = base_metrics.clone();

        // Adjust based on common parameter patterns
        if let Some(&tile_size) = params.extra_params.get("cuda_tile_size") {
            // Larger tile sizes generally improve performance up to a point
            let tile_factor = (tile_size / 32.0).max(0.5).min(2.0);
            adjusted_metrics.execution_time = Duration::from_nanos(
                (base_metrics.execution_time.as_nanos() as f64 / tile_factor) as u64
            );
        }

        if let Some(&thread_count) = params.extra_params.get("cpu_threads") {
            // More threads generally improve performance up to core count
            let thread_factor = thread_count.max(1.0).min(8.0);
            adjusted_metrics.execution_time = Duration::from_nanos(
                (base_metrics.execution_time.as_nanos() as f64 / thread_factor) as u64
            );
        }

        adjusted_metrics
    }

    fn estimate_performance(
        &self,
        device: DeviceType,
        op: KernelOp,
        workload: &WorkloadProfile,
        _params: &KernelParams,
    ) -> KernelMetrics {
        // Simple performance model based on device and operation characteristics
        let base_time_ms = match device {
            DeviceType::Cpu => workload.total_elements as f64 * 0.001, // 1ns per element
            DeviceType::Cuda(_) => workload.total_elements as f64 * 0.0001, // 0.1ns per element
            DeviceType::Metal(_) => workload.total_elements as f64 * 0.0002, // 0.2ns per element
            DeviceType::OpenCL(_) => workload.total_elements as f64 * 0.0003, // 0.3ns per element
        };

        let op_multiplier = match op {
            KernelOp::Add | KernelOp::Sub | KernelOp::Mul | KernelOp::Div => 1.0,
            KernelOp::MatMul => workload.compute_intensity,
            KernelOp::Conv2D => workload.compute_intensity * 2.0,
            _ => 2.0,
        };

        KernelMetrics {
            execution_time: Duration::from_nanos((base_time_ms * op_multiplier * 1_000_000.0) as u64),
            memory_bandwidth: 100.0,
            occupancy: 80.0,
            flops: workload.total_elements as f64 * workload.compute_intensity,
        }
    }

    fn calculate_optimization_score(&self, metrics: &KernelMetrics, target: OptimizationTarget) -> f64 {
        match target {
            OptimizationTarget::MinimizeTime => metrics.execution_time.as_nanos() as f64,
            OptimizationTarget::MaximizeThroughput => 1.0 / metrics.flops,
            OptimizationTarget::MinimizeEnergy => {
                // Rough energy estimate: time * occupancy
                metrics.execution_time.as_secs_f64() * metrics.occupancy
            },
            OptimizationTarget::BalanceTimeMemory => {
                // Balance time and memory bandwidth utilization
                metrics.execution_time.as_nanos() as f64 / metrics.memory_bandwidth
            },
        }
    }

    fn calculate_improvement_factor(
        &self,
        _base_params: &KernelParams,
        _optimized_params: &KernelParams,
        _device: DeviceType,
        _op: KernelOp,
        _workload: &WorkloadProfile,
    ) -> f64 {
        // Placeholder implementation - would compare actual performance
        1.2 // Assume 20% improvement on average
    }

    /// Get memory layout optimizer
    /// メモリレイアウト最適化器を取得
    pub fn memory_optimizer(&self) -> &MemoryLayoutOptimizer {
        &self.memory_optimizer
    }

    /// Clear optimization cache
    /// 最適化キャッシュをクリア
    pub fn clear_cache(&self) -> RusTorchResult<()> {
        if let Ok(mut cache) = self.optimization_cache.write() {
            cache.clear();
        }
        Ok(())
    }
}

/// Helper function to align value up to boundary
/// 値を境界に切り上げるヘルパー関数
fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) / alignment * alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::kernel_selector::PerformanceDatabase;
    use std::sync::{Arc, RwLock};

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();
        assert_eq!(config.target, OptimizationTarget::MinimizeTime);
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.convergence_threshold, 0.01);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_memory_layout_optimizer() {
        let optimizer = MemoryLayoutOptimizer::new();
        let shape = vec![100, 100];
        
        let cuda_layout = optimizer.optimize_layout(DeviceType::Cuda(0), KernelOp::MatMul, &shape);
        let cpu_layout = optimizer.optimize_layout(DeviceType::Cpu, KernelOp::Add, &shape);

        // Should return optimized layouts
        assert!(!cuda_layout.is_empty());
        assert!(!cpu_layout.is_empty());
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(10, 8), 16);
        assert_eq!(align_up(16, 8), 16);
        assert_eq!(align_up(17, 8), 24);
        assert_eq!(align_up(0, 8), 0);
    }

    #[test]
    fn test_kernel_parameter_optimizer_creation() {
        let db = Arc::new(RwLock::new(PerformanceDatabase::new()));
        let optimizer = KernelParameterOptimizer::new(db);
        
        // Should create successfully
        assert!(optimizer.memory_optimizer().alignment_requirements.len() > 0);
    }

    #[test]
    fn test_optimization_targets() {
        let targets = [
            OptimizationTarget::MinimizeTime,
            OptimizationTarget::MaximizeThroughput,
            OptimizationTarget::MinimizeEnergy,
            OptimizationTarget::BalanceTimeMemory,
        ];

        for target in &targets {
            let config = OptimizationConfig {
                target: *target,
                ..Default::default()
            };
            assert_eq!(config.target, *target);
        }
    }
}