//! Automatic kernel selection and optimization system
//! 自動カーネル選択と最適化システム
//!
//! This module provides intelligent kernel selection based on device capabilities,
//! workload characteristics, and performance profiling.
//! このモジュールは、デバイス機能、ワークロード特性、パフォーマンスプロファイリング
//! に基づいた、インテリジェントなカーネル選択を提供します。

use super::unified_kernel::{UnifiedKernelExecutor, KernelOp, KernelParams, KernelMetrics};
use super::{DeviceType, GpuError, GpuResult};
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

/// Performance database for kernel selection
/// カーネル選択のためのパフォーマンスデータベース
#[derive(Debug, Clone)]
pub struct PerformanceDatabase {
    /// Performance records: (device, operation, input_size) -> metrics
    /// パフォーマンスレコード: (デバイス, 操作, 入力サイズ) -> メトリクス
    records: HashMap<(DeviceType, KernelOp, usize), KernelMetrics>,
}

impl PerformanceDatabase {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Record kernel performance
    /// カーネルパフォーマンスを記録
    pub fn record(&mut self, device: DeviceType, op: KernelOp, input_size: usize, metrics: KernelMetrics) {
        let key = (device, op, input_size);
        self.records.insert(key, metrics);
    }

    /// Get best performance for operation
    /// 操作の最高パフォーマンスを取得
    pub fn get_best_performance(&self, op: KernelOp, input_size: usize) -> Option<(DeviceType, &KernelMetrics)> {
        self.records
            .iter()
            .filter(|((_, stored_op, stored_size), _)| *stored_op == op && *stored_size == input_size)
            .min_by_key(|(_, metrics)| metrics.execution_time)
            .map(|((device, _, _), metrics)| (*device, metrics))
    }

    /// Get performance for specific device and operation
    /// 特定デバイス・操作のパフォーマンスを取得
    pub fn get_performance(&self, device: DeviceType, op: KernelOp, input_size: usize) -> Option<&KernelMetrics> {
        let key = (device, op, input_size);
        self.records.get(&key)
    }
}

impl Default for PerformanceDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Workload characteristics for kernel selection
/// カーネル選択のためのワークロード特性
#[derive(Debug, Clone)]
pub struct WorkloadProfile {
    /// Total number of elements to process
    /// 処理する要素の総数
    pub total_elements: usize,
    /// Memory requirement in bytes
    /// バイト単位のメモリ要求量
    pub memory_requirement: usize,
    /// Computational intensity (FLOPS per byte)
    /// 計算強度（バイトあたりのFLOPS）
    pub compute_intensity: f64,
    /// Parallelization potential (0.0 to 1.0)
    /// 並列化ポテンシャル（0.0から1.0）
    pub parallelization: f64,
}

impl WorkloadProfile {
    /// Analyze workload from tensor inputs
    /// テンソル入力からワークロードを分析
    pub fn analyze<T: Float>(inputs: &[&Tensor<T>], op: KernelOp) -> Self {
        let total_elements = inputs.iter()
            .map(|t| t.size())
            .sum();

        let memory_requirement = total_elements * std::mem::size_of::<T>();
        
        let compute_intensity = Self::estimate_compute_intensity(op, inputs);
        let parallelization = Self::estimate_parallelization(op, inputs);

        Self {
            total_elements,
            memory_requirement,
            compute_intensity,
            parallelization,
        }
    }

    fn estimate_compute_intensity<T: Float>(op: KernelOp, inputs: &[&Tensor<T>]) -> f64 {
        match op {
            KernelOp::Add | KernelOp::Sub | KernelOp::Mul | KernelOp::Div => 1.0, // Simple element-wise ops
            KernelOp::MatMul => {
                if inputs.len() >= 2 {
                    let m = inputs[0].shape().get(0).unwrap_or(&1);
                    let n = inputs[1].shape().get(1).unwrap_or(&1);
                    let k = inputs[0].shape().get(1).unwrap_or(&1);
                    (*m * n * k) as f64 / (inputs[0].size() + inputs[1].size()) as f64
                } else {
                    1.0
                }
            },
            KernelOp::Conv2D => 10.0, // High compute intensity
            KernelOp::BatchNorm => 3.0, // Medium compute intensity
            _ => 2.0, // Default estimate
        }
    }

    fn estimate_parallelization<T: Float>(op: KernelOp, inputs: &[&Tensor<T>]) -> f64 {
        match op {
            KernelOp::Add | KernelOp::Sub | KernelOp::Mul | KernelOp::Div => 1.0, // Perfectly parallelizable
            KernelOp::MatMul => 0.9, // Highly parallelizable
            KernelOp::Conv2D => 0.95, // Very parallelizable
            KernelOp::ReduceSum | KernelOp::ReduceMean => 0.7, // Limited by reduction
            KernelOp::BatchNorm => 0.8, // Good parallelization
            _ => 0.8, // Default estimate
        }
    }
}

/// Kernel selection strategy
/// カーネル選択戦略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Always prefer fastest device
    /// 常に最速デバイスを優先
    FastestDevice,
    /// Prefer energy-efficient device
    /// エネルギー効率の良いデバイスを優先
    EnergyEfficient,
    /// Balance performance and memory usage
    /// パフォーマンスとメモリ使用量のバランス
    Balanced,
    /// Prefer specific device type
    /// 特定デバイスタイプを優先
    PreferDevice(DeviceType),
}

/// Automatic kernel selector
/// 自動カーネル選択器
pub struct KernelSelector {
    /// Available executors
    /// 利用可能な実行者
    executors: Vec<Box<dyn UnifiedKernelExecutor>>,
    /// Performance database
    /// パフォーマンスデータベース
    performance_db: Arc<RwLock<PerformanceDatabase>>,
    /// Selection strategy
    /// 選択戦略
    strategy: SelectionStrategy,
    /// Device benchmarking results
    /// デバイスベンチマーク結果
    benchmark_cache: Arc<Mutex<HashMap<DeviceType, f64>>>,
}

impl KernelSelector {
    /// Create new kernel selector
    /// 新しいカーネル選択器を作成
    pub fn new(strategy: SelectionStrategy) -> Self {
        Self {
            executors: Vec::new(),
            performance_db: Arc::new(RwLock::new(PerformanceDatabase::new())),
            strategy,
            benchmark_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add executor to selector
    /// 選択器に実行者を追加
    pub fn add_executor(&mut self, executor: Box<dyn UnifiedKernelExecutor>) {
        self.executors.push(executor);
    }

    /// Select best executor for operation
    /// 操作に最適な実行者を選択
    pub fn select_executor<T: Float>(
        &self,
        op: KernelOp,
        inputs: &[&Tensor<T>],
        params: &KernelParams
    ) -> GpuResult<&dyn UnifiedKernelExecutor> {
        if self.executors.is_empty() {
            return Err(GpuError::DeviceNotFound(0));
        }

        let workload = WorkloadProfile::analyze(inputs, op);
        
        match self.strategy {
            SelectionStrategy::FastestDevice => self.select_fastest(op, &workload),
            SelectionStrategy::EnergyEfficient => self.select_energy_efficient(op, &workload),
            SelectionStrategy::Balanced => self.select_balanced(op, &workload),
            SelectionStrategy::PreferDevice(device_type) => self.select_preferred(device_type, op),
        }
    }

    /// Execute operation with automatic executor selection
    /// 自動実行者選択で操作を実行
    pub fn execute<T: Float + 'static + Send + Sync>(
        &self,
        op: KernelOp,
        inputs: &[&Tensor<T>],
        params: &KernelParams
    ) -> GpuResult<Tensor<T>> {
        let executor = self.select_executor(op, inputs, params)?;
        let result = executor.execute(op, inputs, params)?;
        
        // Record performance for future selections
        let metrics = executor.get_metrics();
        let workload = WorkloadProfile::analyze(inputs, op);
        
        if let Ok(mut db) = self.performance_db.write() {
            db.record(executor.device_type(), op, workload.total_elements, metrics);
        }

        Ok(result)
    }

    fn select_fastest<T: Float>(
        &self,
        op: KernelOp,
        workload: &WorkloadProfile
    ) -> GpuResult<&dyn UnifiedKernelExecutor> {
        // Check performance database first
        if let Ok(db) = self.performance_db.read() {
            if let Some((best_device, _)) = db.get_best_performance(op, workload.total_elements) {
                if let Some(executor) = self.find_executor(best_device) {
                    return Ok(executor);
                }
            }
        }

        // Fallback to heuristic selection
        self.select_by_heuristics(op, workload)
    }

    fn select_energy_efficient<T: Float>(
        &self,
        op: KernelOp,
        workload: &WorkloadProfile
    ) -> GpuResult<&dyn UnifiedKernelExecutor> {
        // For small workloads, prefer CPU to save energy
        if workload.total_elements < 1000 {
            if let Some(executor) = self.find_executor(DeviceType::Cpu) {
                if executor.supports_operation(op) {
                    return Ok(executor);
                }
            }
        }

        // For larger workloads, prefer Metal (energy efficient GPU)
        if let Some(executor) = self.find_executor(DeviceType::Metal(0)) {
            if executor.supports_operation(op) {
                return Ok(executor);
            }
        }

        // Fallback to fastest available
        self.select_fastest::<T>(op, workload)
    }

    fn select_balanced<T: Float>(
        &self,
        op: KernelOp,
        workload: &WorkloadProfile
    ) -> GpuResult<&dyn UnifiedKernelExecutor> {
        // Balance compute intensity with memory requirements
        let score_threshold = workload.compute_intensity * workload.parallelization;
        
        if score_threshold > 5.0 {
            // High compute, prefer GPU
            for device in &[DeviceType::Cuda(0), DeviceType::Metal(0), DeviceType::OpenCL(0)] {
                if let Some(executor) = self.find_executor(*device) {
                    if executor.supports_operation(op) {
                        return Ok(executor);
                    }
                }
            }
        }

        // Medium/low compute, CPU might be sufficient
        if let Some(executor) = self.find_executor(DeviceType::Cpu) {
            if executor.supports_operation(op) {
                return Ok(executor);
            }
        }

        // Fallback
        self.select_any_available(op)
    }

    fn select_preferred(
        &self,
        preferred_device: DeviceType,
        op: KernelOp
    ) -> GpuResult<&dyn UnifiedKernelExecutor> {
        if let Some(executor) = self.find_executor(preferred_device) {
            if executor.supports_operation(op) {
                return Ok(executor);
            }
        }

        // Fallback to any available
        self.select_any_available(op)
    }

    fn select_by_heuristics<T: Float>(
        &self,
        op: KernelOp,
        workload: &WorkloadProfile
    ) -> GpuResult<&dyn UnifiedKernelExecutor> {
        // Simple heuristics based on operation type and workload size
        match op {
            KernelOp::MatMul if workload.total_elements > 10000 => {
                // Large matrix multiplication - prefer CUDA
                if let Some(executor) = self.find_executor(DeviceType::Cuda(0)) {
                    if executor.supports_operation(op) {
                        return Ok(executor);
                    }
                }
            },
            KernelOp::Conv2D => {
                // Convolution - prefer CUDA for best performance
                if let Some(executor) = self.find_executor(DeviceType::Cuda(0)) {
                    if executor.supports_operation(op) {
                        return Ok(executor);
                    }
                }
            },
            KernelOp::Add | KernelOp::Mul if workload.total_elements < 1000 => {
                // Small element-wise ops - CPU might be faster due to overhead
                if let Some(executor) = self.find_executor(DeviceType::Cpu) {
                    if executor.supports_operation(op) {
                        return Ok(executor);
                    }
                }
            },
            _ => {}
        }

        // Default fallback
        self.select_any_available(op)
    }

    fn select_any_available(&self, op: KernelOp) -> GpuResult<&dyn UnifiedKernelExecutor> {
        for executor in &self.executors {
            if executor.supports_operation(op) {
                return Ok(executor.as_ref());
            }
        }

        Err(GpuError::UnsupportedOperation(format!("No executor supports operation {:?}", op)))
    }

    fn find_executor(&self, device_type: DeviceType) -> Option<&dyn UnifiedKernelExecutor> {
        self.executors
            .iter()
            .find(|e| e.device_type() == device_type)
            .map(|e| e.as_ref())
    }

    /// Benchmark all available devices
    /// 全ての利用可能デバイスをベンチマーク
    pub fn benchmark_devices(&self) -> GpuResult<()> {
        let benchmark_op = KernelOp::MatMul;
        let size = 100;
        
        // Create test tensors
        let test_data: Vec<f32> = (0..size*size).map(|i| i as f32).collect();
        let a = Tensor::from_vec(test_data.clone(), vec![size, size]);
        let b = Tensor::from_vec(test_data, vec![size, size]);
        let params = KernelParams::default();

        let mut benchmark_results = HashMap::new();

        for executor in &self.executors {
            if executor.supports_operation(benchmark_op) {
                // Run benchmark multiple times and take average
                let mut total_time = Duration::ZERO;
                let iterations = 5;

                for _ in 0..iterations {
                    let start = std::time::Instant::now();
                    let _result = executor.execute(benchmark_op, &[&a, &b], &params);
                    total_time += start.elapsed();
                }

                let avg_time = total_time / iterations as u32;
                let performance_score = 1000.0 / avg_time.as_millis() as f64; // Higher is better
                
                benchmark_results.insert(executor.device_type(), performance_score);
            }
        }

        // Store results in cache
        if let Ok(mut cache) = self.benchmark_cache.lock() {
            *cache = benchmark_results;
        }

        Ok(())
    }

    /// Get device benchmark score
    /// デバイスベンチマークスコアを取得
    pub fn get_benchmark_score(&self, device: DeviceType) -> Option<f64> {
        self.benchmark_cache
            .lock()
            .ok()
            .and_then(|cache| cache.get(&device).copied())
    }

    /// Set selection strategy
    /// 選択戦略を設定
    pub fn set_strategy(&mut self, strategy: SelectionStrategy) {
        self.strategy = strategy;
    }

    /// Get current strategy
    /// 現在の戦略を取得
    pub fn strategy(&self) -> SelectionStrategy {
        self.strategy
    }

    /// Get performance database
    /// パフォーマンスデータベースを取得
    pub fn performance_database(&self) -> Arc<RwLock<PerformanceDatabase>> {
        Arc::clone(&self.performance_db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::unified_kernel::CpuFallbackExecutor;

    #[test]
    fn test_performance_database() {
        let mut db = PerformanceDatabase::new();
        let metrics = KernelMetrics {
            execution_time: Duration::from_millis(10),
            memory_bandwidth: 100.0,
            occupancy: 80.0,
            flops: 1000.0,
        };

        db.record(DeviceType::Cpu, KernelOp::Add, 1000, metrics.clone());
        
        let retrieved = db.get_performance(DeviceType::Cpu, KernelOp::Add, 1000);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().execution_time, Duration::from_millis(10));
    }

    #[test]
    fn test_workload_analysis() {
        let a = Tensor::from_vec(vec![1.0f32; 100], vec![10, 10]);
        let b = Tensor::from_vec(vec![1.0f32; 100], vec![10, 10]);
        
        let workload = WorkloadProfile::analyze(&[&a, &b], KernelOp::Add);
        assert_eq!(workload.total_elements, 200);
        assert_eq!(workload.memory_requirement, 200 * 4); // 4 bytes per f32
        assert_eq!(workload.compute_intensity, 1.0); // Simple add operation
        assert_eq!(workload.parallelization, 1.0); // Perfect parallelization
    }

    #[test]
    fn test_kernel_selector_creation() {
        let mut selector = KernelSelector::new(SelectionStrategy::FastestDevice);
        let executor = Box::new(CpuFallbackExecutor::new());
        
        selector.add_executor(executor);
        assert_eq!(selector.executors.len(), 1);
        assert_eq!(selector.strategy(), SelectionStrategy::FastestDevice);
    }

    #[test]
    fn test_selection_strategies() {
        let strategies = [
            SelectionStrategy::FastestDevice,
            SelectionStrategy::EnergyEfficient,
            SelectionStrategy::Balanced,
            SelectionStrategy::PreferDevice(DeviceType::Cpu),
        ];

        for strategy in &strategies {
            let selector = KernelSelector::new(*strategy);
            assert_eq!(selector.strategy(), *strategy);
        }
    }

    #[test]
    fn test_cpu_execution_through_selector() {
        let mut selector = KernelSelector::new(SelectionStrategy::FastestDevice);
        selector.add_executor(Box::new(CpuFallbackExecutor::new()));

        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        let params = KernelParams::default();

        let result = selector.execute(KernelOp::Add, &[&a, &b], &params).unwrap();
        let expected = vec![5.0f32, 7.0, 9.0];
        assert_eq!(result.as_slice().unwrap(), &expected);
    }
}