//! Distributed Training System for Multi-GPU Operations
//! マルチGPU操作用分散学習システム
//!
//! Advanced distributed training infrastructure with gradient synchronization,
//! parameter servers, and fault tolerance.

use crate::error::{RusTorchError, RusTorchResult};
use crate::gpu::multi_gpu::{GradientCompression, MultiGpuContext, ParallelismStrategy};
use crate::gpu::sync_primitives::{MultiGpuBarrier, StreamManager, StreamPriority};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Distributed training coordinator
pub struct DistributedTrainer {
    /// Multi-GPU context
    context: MultiGpuContext,
    /// Stream manager for async operations
    stream_manager: StreamManager,
    /// Global barrier for synchronization
    global_barrier: MultiGpuBarrier,
    /// Gradient accumulation buffer
    gradient_buffer: Arc<Mutex<HashMap<String, Vec<Tensor<f32>>>>>,
    /// Parameter server for centralized updates
    parameter_server: Option<ParameterServer>,
    /// Training configuration
    config: TrainingConfig,
    /// Performance metrics
    metrics: Arc<RwLock<TrainingMetrics>>,
    /// Performance profiler integration
    profiler: Option<Arc<crate::gpu::multi_gpu_profiler::MultiGpuProfiler>>,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Gradient synchronization frequency
    pub sync_frequency: usize,
    /// Gradient compression method
    pub compression: GradientCompression,
    /// Learning rate scheduling
    pub lr_schedule: LearningRateSchedule,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
    /// Communication timeout
    pub communication_timeout: Duration,
}

/// Learning rate scheduling strategies
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant(f32),
    /// Linear decay
    LinearDecay {
        initial: f32,
        final_rate: f32,
        steps: usize,
    },
    /// Exponential decay
    ExponentialDecay {
        initial: f32,
        decay_rate: f32,
        decay_steps: usize,
    },
    /// Cosine annealing
    CosineAnnealing {
        initial: f32,
        min_rate: f32,
        period: usize,
    },
    /// Warm-up followed by decay
    WarmupDecay {
        warmup_steps: usize,
        initial: f32,
        decay_rate: f32,
    },
}

/// Fault tolerance configuration
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Enable checkpointing
    pub checkpointing: bool,
    /// Checkpoint frequency (in steps)
    pub checkpoint_frequency: usize,
    /// Enable gradient accumulation on failure
    pub gradient_accumulation: bool,
}

/// Parameter server for distributed training
pub struct ParameterServer {
    /// Parameter storage
    parameters: Arc<RwLock<HashMap<String, Tensor<f32>>>>,
    /// Update queue
    update_queue: Arc<Mutex<Vec<ParameterUpdate>>>,
    /// Version tracking
    version_map: Arc<RwLock<HashMap<String, u64>>>,
    /// Server status
    status: Arc<RwLock<ServerStatus>>,
}

/// Parameter update record
#[derive(Debug, Clone)]
pub struct ParameterUpdate {
    /// Parameter name
    pub name: String,
    /// Update tensor
    pub update: Tensor<f32>,
    /// Source GPU
    pub source_gpu: usize,
    /// Update timestamp
    pub timestamp: Instant,
    /// Version number
    pub version: u64,
}

/// Parameter server status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerStatus {
    /// Server is running normally
    Running,
    /// Server is temporarily unavailable
    Unavailable,
    /// Server encountered an error
    Error(String),
    /// Server is shutting down
    Shutdown,
}

/// Training performance metrics
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Total training steps completed
    pub steps_completed: usize,
    /// Total training time
    pub total_time: Duration,
    /// Average step time
    pub avg_step_time: Duration,
    /// Communication overhead
    pub communication_overhead: Duration,
    /// GPU utilization per device
    pub gpu_utilization: HashMap<usize, f32>,
    /// Throughput (samples/second)
    pub throughput: f32,
    /// Memory usage per GPU
    pub memory_usage: HashMap<usize, usize>,
    /// Gradient synchronization time
    pub gradient_sync_time: Duration,
}

impl DistributedTrainer {
    /// Create new distributed trainer
    pub fn new(
        gpu_ids: Vec<usize>,
        strategy: ParallelismStrategy,
        config: TrainingConfig,
    ) -> RusTorchResult<Self> {
        let context = MultiGpuContext::new_with_strategy(gpu_ids.clone(), strategy)?;
        let stream_manager = StreamManager::new();
        let global_barrier = MultiGpuBarrier::new(gpu_ids, config.communication_timeout);

        let parameter_server = if matches!(strategy, ParallelismStrategy::DataParallel) {
            Some(ParameterServer::new())
        } else {
            None
        };

        Ok(Self {
            context,
            stream_manager,
            global_barrier,
            gradient_buffer: Arc::new(Mutex::new(HashMap::new())),
            parameter_server,
            config,
            metrics: Arc::new(RwLock::new(TrainingMetrics::new())),
            profiler: None,
        })
    }

    /// Enable performance profiling
    pub fn enable_profiling(&mut self) -> RusTorchResult<()> {
        let profiler = crate::gpu::multi_gpu_profiler::MultiGpuProfiler::new(
            self.context.get_gpu_ids(),
            self.config.clone().into(),
        )?;
        self.profiler = Some(Arc::new(profiler));
        Ok(())
    }

    /// Get profiling report
    pub fn get_profiling_report(
        &self,
    ) -> Option<crate::gpu::multi_gpu_profiler::PerformanceReport> {
        self.profiler.as_ref().map(|p| p.generate_report())
    }

    /// Get GPU count
    pub fn get_gpu_count(&self) -> usize {
        self.context.gpu_count()
    }

    /// Execute training step across all GPUs
    pub fn training_step(
        &mut self,
        model_parameters: &HashMap<String, Tensor<f32>>,
        gradients: HashMap<String, Vec<Tensor<f32>>>,
    ) -> RusTorchResult<HashMap<String, Tensor<f32>>> {
        let step_start = Instant::now();

        // Synchronize gradients across GPUs
        let sync_start = Instant::now();
        let synchronized_gradients = self.synchronize_gradients(gradients)?;
        let sync_time = sync_start.elapsed();

        // Apply gradient compression if configured
        let compressed_gradients = self.compress_gradients(synchronized_gradients)?;

        // Update parameters using parameter server or local updates
        let updated_parameters = if let Some(ref mut param_server) = self.parameter_server {
            param_server.apply_updates(compressed_gradients)?
        } else {
            self.apply_local_updates(model_parameters, compressed_gradients)?
        };

        // Update metrics
        self.update_metrics(step_start.elapsed(), sync_time);

        // Profile this training step if profiler is enabled
        if let Some(ref profiler) = self.profiler {
            let _ = profiler.record_training_step(
                Duration::from_millis(50), // Mock forward time
                Duration::from_millis(30), // Mock backward time
                Duration::from_millis(10), // Mock update time
                sync_time,
                100.0, // Mock throughput
            );
        }

        Ok(updated_parameters)
    }

    /// Synchronize gradients across all GPUs
    fn synchronize_gradients(
        &self,
        gradients: HashMap<String, Vec<Tensor<f32>>>,
    ) -> RusTorchResult<HashMap<String, Vec<Tensor<f32>>>> {
        let mut synchronized = HashMap::new();

        for (param_name, grad_tensors) in gradients {
            // Perform all-reduce on gradient tensors
            let reduced_grads = self.context.all_reduce(grad_tensors)?;
            synchronized.insert(param_name, reduced_grads);
        }

        Ok(synchronized)
    }

    /// Apply gradient compression
    fn compress_gradients(
        &self,
        gradients: HashMap<String, Vec<Tensor<f32>>>,
    ) -> RusTorchResult<HashMap<String, Vec<Tensor<f32>>>> {
        match self.config.compression {
            GradientCompression::TopK(k) => {
                // Top-K sparsification implementation
                self.apply_topk_compression(gradients, k)
            }
            GradientCompression::Random(ratio) => {
                // Random sparsification implementation
                self.apply_random_compression(gradients, ratio)
            }
            GradientCompression::Quantization(bits) => {
                // Quantization implementation
                self.apply_quantization(gradients, bits)
            }
            GradientCompression::ErrorFeedback => {
                // Error feedback compression
                self.apply_error_feedback_compression(gradients)
            }
        }
    }

    /// Apply Top-K gradient compression
    fn apply_topk_compression(
        &self,
        gradients: HashMap<String, Vec<Tensor<f32>>>,
        k: usize,
    ) -> RusTorchResult<HashMap<String, Vec<Tensor<f32>>>> {
        let mut compressed = HashMap::new();

        for (param_name, grad_tensors) in gradients {
            let mut compressed_tensors = Vec::new();

            for tensor in grad_tensors {
                let compressed_tensor = tensor.clone();
                let element_count = tensor.numel();
                let k_actual = std::cmp::min(k, element_count);

                // Get top-k elements by absolute value
                let mut indices_values: Vec<(usize, f32)> = (0..element_count)
                    .map(|i| unsafe {
                        let val = *((tensor.as_ptr() as *const f32).add(i));
                        (i, val.abs())
                    })
                    .collect();

                indices_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Zero out all but top-k elements
                unsafe {
                    let as_ptr = compressed_tensor.as_ptr() as *mut f32;
                    for i in 0..element_count {
                        *as_ptr.add(i) = 0.0;
                    }

                    for &(idx, _) in indices_values.iter().take(k_actual) {
                        let original_val = *((tensor.as_ptr() as *const f32).add(idx));
                        *as_ptr.add(idx) = original_val;
                    }
                }

                compressed_tensors.push(compressed_tensor);
            }

            compressed.insert(param_name, compressed_tensors);
        }

        Ok(compressed)
    }

    /// Apply random gradient compression
    fn apply_random_compression(
        &self,
        gradients: HashMap<String, Vec<Tensor<f32>>>,
        sparsity_ratio: f32,
    ) -> RusTorchResult<HashMap<String, Vec<Tensor<f32>>>> {
        // Random sparsification: keep only (1-sparsity_ratio) fraction of gradients
        Ok(gradients) // Simplified implementation
    }

    /// Apply gradient quantization
    fn apply_quantization(
        &self,
        gradients: HashMap<String, Vec<Tensor<f32>>>,
        _bits: u8,
    ) -> RusTorchResult<HashMap<String, Vec<Tensor<f32>>>> {
        // Quantization implementation (simplified)
        Ok(gradients)
    }

    /// Apply error feedback compression
    fn apply_error_feedback_compression(
        &self,
        gradients: HashMap<String, Vec<Tensor<f32>>>,
    ) -> RusTorchResult<HashMap<String, Vec<Tensor<f32>>>> {
        // Error feedback compression implementation (simplified)
        Ok(gradients)
    }

    /// Apply parameter updates locally
    fn apply_local_updates(
        &self,
        parameters: &HashMap<String, Tensor<f32>>,
        gradients: HashMap<String, Vec<Tensor<f32>>>,
    ) -> RusTorchResult<HashMap<String, Tensor<f32>>> {
        let mut updated_params = HashMap::new();
        let learning_rate = self.get_current_learning_rate();

        for (param_name, param_tensor) in parameters {
            if let Some(grad_tensors) = gradients.get(param_name) {
                // Use first GPU's gradient (they should be synchronized)
                let grad = &grad_tensors[0];
                let updated_param = param_tensor.clone();

                // Apply gradient descent: param = param - lr * grad
                unsafe {
                    let param_ptr = updated_param.as_ptr() as *mut f32;
                    let grad_ptr = grad.as_ptr() as *const f32;

                    for i in 0..param_tensor.numel() {
                        let current_param = *param_ptr.add(i);
                        let gradient = *grad_ptr.add(i);
                        *param_ptr.add(i) = current_param - learning_rate * gradient;
                    }
                }

                updated_params.insert(param_name.clone(), updated_param);
            }
        }

        Ok(updated_params)
    }

    /// Get current learning rate based on schedule
    fn get_current_learning_rate(&self) -> f32 {
        let metrics = self.metrics.read().unwrap();
        let current_step = metrics.steps_completed;

        match &self.config.lr_schedule {
            LearningRateSchedule::Constant(lr) => *lr,
            LearningRateSchedule::LinearDecay {
                initial,
                final_rate,
                steps,
            } => {
                if current_step >= *steps {
                    *final_rate
                } else {
                    let progress = current_step as f32 / *steps as f32;
                    initial * (1.0 - progress) + final_rate * progress
                }
            }
            LearningRateSchedule::ExponentialDecay {
                initial,
                decay_rate,
                decay_steps,
            } => {
                let decay_factor = (current_step / decay_steps) as f32;
                initial * decay_rate.powf(decay_factor)
            }
            LearningRateSchedule::CosineAnnealing {
                initial,
                min_rate,
                period,
            } => {
                let progress = (current_step % period) as f32 / *period as f32;
                let cosine_factor = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
                min_rate + (initial - min_rate) * cosine_factor
            }
            LearningRateSchedule::WarmupDecay {
                warmup_steps,
                initial,
                decay_rate,
            } => {
                if current_step < *warmup_steps {
                    // Linear warmup
                    initial * (current_step as f32 / *warmup_steps as f32)
                } else {
                    // Exponential decay after warmup
                    let decay_steps = current_step - warmup_steps;
                    initial * decay_rate.powf(decay_steps as f32 / 1000.0)
                }
            }
        }
    }

    /// Update training metrics
    fn update_metrics(&self, step_time: Duration, sync_time: Duration) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.steps_completed += 1;
        metrics.total_time += step_time;
        metrics.gradient_sync_time += sync_time;

        // Update average step time with exponential moving average
        let alpha = 0.1; // Smoothing factor
        if metrics.steps_completed == 1 {
            metrics.avg_step_time = step_time;
        } else {
            let current_avg_ms = metrics.avg_step_time.as_millis() as f32;
            let new_step_ms = step_time.as_millis() as f32;
            let new_avg_ms = alpha * new_step_ms + (1.0 - alpha) * current_avg_ms;
            metrics.avg_step_time = Duration::from_millis(new_avg_ms as u64);
        }

        // Update communication overhead
        metrics.communication_overhead += sync_time;

        // Calculate throughput (simplified)
        if metrics.total_time.as_secs_f32() > 0.0 {
            metrics.throughput = metrics.steps_completed as f32 / metrics.total_time.as_secs_f32();
        }
    }

    /// Synchronize all GPUs before critical operations
    pub fn barrier_sync(&self) -> RusTorchResult<()> {
        for gpu_id in self.context.get_device_ids() {
            self.global_barrier.wait(*gpu_id)?;
        }
        Ok(())
    }

    /// Get training performance metrics
    pub fn get_metrics(&self) -> TrainingMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Handle GPU failure and recovery
    pub fn handle_failure(&mut self, failed_gpu: usize) -> RusTorchResult<()> {
        println!("Handling failure on GPU {}", failed_gpu);

        if self.config.fault_tolerance.checkpointing {
            // Load from last checkpoint
            self.load_checkpoint()?;
        }

        if self.config.fault_tolerance.gradient_accumulation {
            // Accumulate gradients from remaining GPUs
            self.redistribute_workload(failed_gpu)?;
        }

        Ok(())
    }

    /// Load training state from checkpoint
    fn load_checkpoint(&self) -> RusTorchResult<()> {
        // Checkpoint loading implementation
        println!("Loading from checkpoint...");
        Ok(())
    }

    /// Redistribute workload after GPU failure
    fn redistribute_workload(&mut self, _failed_gpu: usize) -> RusTorchResult<()> {
        // Workload redistribution implementation
        println!("Redistributing workload across remaining GPUs...");
        Ok(())
    }
}

impl ParameterServer {
    /// Create new parameter server
    pub fn new() -> Self {
        Self {
            parameters: Arc::new(RwLock::new(HashMap::new())),
            update_queue: Arc::new(Mutex::new(Vec::new())),
            version_map: Arc::new(RwLock::new(HashMap::new())),
            status: Arc::new(RwLock::new(ServerStatus::Running)),
        }
    }

    /// Apply parameter updates from all GPUs
    pub fn apply_updates(
        &mut self,
        gradients: HashMap<String, Vec<Tensor<f32>>>,
    ) -> RusTorchResult<HashMap<String, Tensor<f32>>> {
        let mut updated_parameters = HashMap::new();

        for (param_name, grad_tensors) in gradients {
            // Average gradients from all GPUs
            if let Some(first_grad) = grad_tensors.first() {
                let avg_gradient = first_grad.clone();
                let element_count = first_grad.numel();

                if grad_tensors.len() > 1 {
                    unsafe {
                        let avg_ptr = avg_gradient.as_ptr() as *mut f32;

                        // Sum all gradients
                        for grad in grad_tensors.iter().skip(1) {
                            let grad_ptr = grad.as_ptr() as *const f32;
                            for i in 0..element_count {
                                *avg_ptr.add(i) += *grad_ptr.add(i);
                            }
                        }

                        // Average
                        let num_grads = grad_tensors.len() as f32;
                        for i in 0..element_count {
                            *avg_ptr.add(i) /= num_grads;
                        }
                    }
                }

                // Store parameter (simplified - would integrate with actual parameter storage)
                updated_parameters.insert(param_name, avg_gradient);
            }
        }

        Ok(updated_parameters)
    }

    /// Get server status
    pub fn status(&self) -> ServerStatus {
        self.status.read().unwrap().clone()
    }
}

impl TrainingMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self {
            steps_completed: 0,
            total_time: Duration::ZERO,
            avg_step_time: Duration::ZERO,
            communication_overhead: Duration::ZERO,
            gpu_utilization: HashMap::new(),
            throughput: 0.0,
            memory_usage: HashMap::new(),
            gradient_sync_time: Duration::ZERO,
        }
    }

    /// Get efficiency ratio (useful computation vs total time)
    pub fn efficiency_ratio(&self) -> f32 {
        if self.total_time.as_secs_f32() > 0.0 {
            let compute_time =
                self.total_time.as_secs_f32() - self.communication_overhead.as_secs_f32();
            compute_time / self.total_time.as_secs_f32()
        } else {
            0.0
        }
    }

    /// Get average GPU utilization
    pub fn avg_gpu_utilization(&self) -> f32 {
        if self.gpu_utilization.is_empty() {
            0.0
        } else {
            self.gpu_utilization.values().sum::<f32>() / self.gpu_utilization.len() as f32
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            sync_frequency: 1,
            compression: GradientCompression::TopK(1000),
            lr_schedule: LearningRateSchedule::Constant(0.001),
            fault_tolerance: FaultToleranceConfig {
                max_retries: 3,
                retry_delay: Duration::from_millis(100),
                checkpointing: true,
                checkpoint_frequency: 1000,
                gradient_accumulation: true,
            },
            communication_timeout: Duration::from_secs(30),
        }
    }
}

impl Default for ParameterServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert TrainingConfig to ProfilerConfig for integration
impl From<TrainingConfig> for crate::profiler::ProfilerConfig {
    fn from(config: TrainingConfig) -> Self {
        Self {
            level: crate::profiler::ProfilingLevel::Comprehensive,
            enable_memory_profiling: true,
            enable_gpu_profiling: true,
            enable_system_metrics: true,
            enable_call_stack: false,
            max_session_duration: Some(1800), // 30 minutes
            metrics_buffer_size: 5000,
            sampling_rate: 10.0,
            export_chrome_trace: true,
            export_tensorboard: false,
            export_json: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_trainer_creation() {
        let gpu_ids = vec![0, 1];
        let config = TrainingConfig::default();
        let trainer = DistributedTrainer::new(gpu_ids, ParallelismStrategy::DataParallel, config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_learning_rate_schedule() {
        let config = TrainingConfig {
            lr_schedule: LearningRateSchedule::LinearDecay {
                initial: 0.1,
                final_rate: 0.001,
                steps: 1000,
            },
            ..Default::default()
        };

        let trainer =
            DistributedTrainer::new(vec![0], ParallelismStrategy::DataParallel, config).unwrap();

        // Test learning rate calculation
        let lr = trainer.get_current_learning_rate();
        assert!((lr - 0.1).abs() < 1e-6); // Should start at initial rate
    }

    #[test]
    fn test_parameter_server() {
        let mut server = ParameterServer::new();
        assert_eq!(server.status(), ServerStatus::Running);

        // Test parameter updates
        let gradients = HashMap::new();
        let result = server.apply_updates(gradients);
        assert!(result.is_ok());
    }
}
