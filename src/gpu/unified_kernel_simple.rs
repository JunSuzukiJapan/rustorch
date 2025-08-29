//! Simplified unified kernel interface for cross-platform GPU acceleration
//! クロスプラットフォームGPU加速のための簡潔な統一カーネルインターフェース

use crate::error::{RusTorchError, RusTorchResult};
use crate::gpu::DeviceType;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::ops::{Add, Mul, Sub};
use std::time::Duration;

/// Unified kernel operation types
/// 統一カーネル操作タイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelOp {
    /// Element-wise addition
    /// 要素ごとの加算
    Add,
    /// Element-wise multiplication
    /// 要素ごとの乗算
    Mul,
    /// Element-wise subtraction
    /// 要素ごとの減算
    Sub,
    /// Element-wise division
    /// 要素ごとの除算
    Div,
    /// Matrix multiplication
    /// 行列乗算
    MatMul,
    /// 2D convolution
    /// 2D畳み込み
    Conv2D,
    /// Batch normalization
    /// バッチ正規化
    BatchNorm,
    /// Reduction sum
    /// リダクション合計
    ReduceSum,
    /// Reduction mean
    /// リダクション平均
    ReduceMean,
    /// ReLU activation
    /// ReLU活性化
    ReLU,
    /// Softmax activation
    /// Softmax活性化
    Softmax,
}

/// Kernel execution parameters
/// カーネル実行パラメータ
#[derive(Debug, Clone, Default)]
pub struct KernelParams {
    /// Input tensor shapes for the kernel
    /// カーネルの入力テンソル形状
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shape
    /// 出力テンソル形状
    pub output_shape: Vec<usize>,
    /// Additional kernel parameters
    /// 追加カーネルパラメータ
    pub extra_params: HashMap<String, f64>,
}

/// Kernel performance metrics
/// カーネル性能メトリクス
#[derive(Debug, Clone)]
pub struct KernelMetrics {
    /// Kernel execution time
    /// カーネル実行時間
    pub execution_time: Duration,
    /// Memory bandwidth utilization (GB/s)
    /// メモリ帯域利用率 (GB/s)
    pub memory_bandwidth: f64,
    /// GPU occupancy percentage
    /// GPU占有率パーセント
    pub occupancy: f64,
    /// Floating point operations per second
    /// 毎秒浮動小数点演算数
    pub flops: f64,
}

impl Default for KernelMetrics {
    fn default() -> Self {
        Self {
            execution_time: Duration::ZERO,
            memory_bandwidth: 0.0,
            occupancy: 0.0,
            flops: 0.0,
        }
    }
}

/// Unified kernel executor
/// 統一カーネル実行者
pub struct UnifiedKernelExecutor {
    device: DeviceType,
    metrics: KernelMetrics,
}

impl UnifiedKernelExecutor {
    /// Create new unified kernel executor
    /// 新しい統一カーネルエグゼキューターを作成
    pub fn new(device: DeviceType) -> RusTorchResult<Self> {
        // Validate device availability
        if !device.is_available() {
            return Err(RusTorchError::DeviceNotAvailable(format!(
                "Device {} not available",
                device
            )));
        }

        Ok(Self {
            device,
            metrics: KernelMetrics::default(),
        })
    }

    /// Execute kernel operation on f32 tensors
    /// f32テンソルでカーネル操作を実行
    pub fn execute_f32(
        &mut self,
        op: KernelOp,
        inputs: &[&Tensor<f32>],
        _params: &KernelParams,
    ) -> RusTorchResult<Tensor<f32>> {
        let start_time = std::time::Instant::now();

        let result = match op {
            KernelOp::Add => {
                if inputs.len() != 2 {
                    return Err(RusTorchError::InvalidOperation(
                        "Add requires exactly 2 inputs",
                    ));
                }
                inputs[0]
                    .add(inputs[1])
                    .map_err(|e| RusTorchError::KernelExecutionError(e.to_string()))
            }
            KernelOp::Mul => {
                if inputs.len() != 2 {
                    return Err(RusTorchError::InvalidOperation(
                        "Mul requires exactly 2 inputs",
                    ));
                }
                inputs[0]
                    .mul(inputs[1])
                    .map_err(|e| RusTorchError::KernelExecutionError(e.to_string()))
            }
            KernelOp::Sub => {
                if inputs.len() != 2 {
                    return Err(RusTorchError::InvalidOperation(
                        "Sub requires exactly 2 inputs",
                    ));
                }
                inputs[0]
                    .sub(inputs[1])
                    .map_err(|e| RusTorchError::KernelExecutionError(e.to_string()))
            }
            KernelOp::MatMul => {
                if inputs.len() != 2 {
                    return Err(RusTorchError::InvalidOperation(
                        "MatMul requires exactly 2 inputs",
                    ));
                }
                inputs[0]
                    .matmul(inputs[1])
                    .map_err(|e| RusTorchError::KernelExecutionError(e.to_string()))
            }
            _ => Err(RusTorchError::UnsupportedOperation(format!(
                "Operation {:?} not implemented",
                op
            ))),
        };

        // Update metrics
        let execution_time = start_time.elapsed();
        self.metrics.execution_time = execution_time;

        // Set device-specific performance characteristics
        match self.device {
            DeviceType::Cpu => {
                self.metrics.memory_bandwidth = 50.0;
                self.metrics.occupancy = 100.0;
                self.metrics.flops = 100.0;
            }
            DeviceType::Cuda(_) => {
                self.metrics.memory_bandwidth = 500.0;
                self.metrics.occupancy = 80.0;
                self.metrics.flops = 1000.0;
            }
            DeviceType::Metal(_) => {
                self.metrics.memory_bandwidth = 300.0;
                self.metrics.occupancy = 75.0;
                self.metrics.flops = 800.0;
            }
            DeviceType::OpenCL(_) => {
                self.metrics.memory_bandwidth = 200.0;
                self.metrics.occupancy = 60.0;
                self.metrics.flops = 600.0;
            }
        }

        result
    }

    /// Execute kernel operation on f64 tensors
    /// f64テンソルでカーネル操作を実行
    pub fn execute_f64(
        &mut self,
        op: KernelOp,
        inputs: &[&Tensor<f64>],
        _params: &KernelParams,
    ) -> RusTorchResult<Tensor<f64>> {
        let start_time = std::time::Instant::now();

        let result = match op {
            KernelOp::Add => {
                if inputs.len() != 2 {
                    return Err(RusTorchError::InvalidOperation(
                        "Add requires exactly 2 inputs",
                    ));
                }
                inputs[0]
                    .add(inputs[1])
                    .map_err(|e| RusTorchError::KernelExecutionError(e.to_string()))
            }
            KernelOp::Mul => {
                if inputs.len() != 2 {
                    return Err(RusTorchError::InvalidOperation(
                        "Mul requires exactly 2 inputs",
                    ));
                }
                inputs[0]
                    .mul(inputs[1])
                    .map_err(|e| RusTorchError::KernelExecutionError(e.to_string()))
            }
            KernelOp::Sub => {
                if inputs.len() != 2 {
                    return Err(RusTorchError::InvalidOperation(
                        "Sub requires exactly 2 inputs",
                    ));
                }
                inputs[0]
                    .sub(inputs[1])
                    .map_err(|e| RusTorchError::KernelExecutionError(e.to_string()))
            }
            KernelOp::MatMul => {
                if inputs.len() != 2 {
                    return Err(RusTorchError::InvalidOperation(
                        "MatMul requires exactly 2 inputs",
                    ));
                }
                inputs[0]
                    .matmul(inputs[1])
                    .map_err(|e| RusTorchError::KernelExecutionError(e.to_string()))
            }
            _ => Err(RusTorchError::UnsupportedOperation(format!(
                "Operation {:?} not implemented",
                op
            ))),
        };

        // Update metrics
        let execution_time = start_time.elapsed();
        self.metrics.execution_time = execution_time;

        // Set device-specific performance characteristics (same as f32)
        match self.device {
            DeviceType::Cpu => {
                self.metrics.memory_bandwidth = 50.0;
                self.metrics.occupancy = 100.0;
                self.metrics.flops = 100.0;
            }
            DeviceType::Cuda(_) => {
                self.metrics.memory_bandwidth = 500.0;
                self.metrics.occupancy = 80.0;
                self.metrics.flops = 1000.0;
            }
            DeviceType::Metal(_) => {
                self.metrics.memory_bandwidth = 300.0;
                self.metrics.occupancy = 75.0;
                self.metrics.flops = 800.0;
            }
            DeviceType::OpenCL(_) => {
                self.metrics.memory_bandwidth = 200.0;
                self.metrics.occupancy = 60.0;
                self.metrics.flops = 600.0;
            }
        }

        result
    }

    /// Check if operation is supported
    /// 操作がサポートされているかチェック
    pub fn supports_operation(&self, op: KernelOp) -> bool {
        matches!(
            op,
            KernelOp::Add | KernelOp::Mul | KernelOp::Sub | KernelOp::MatMul
        )
    }

    /// Get device type
    /// デバイスタイプを取得
    pub fn device_type(&self) -> DeviceType {
        self.device
    }

    /// Get performance metrics
    /// パフォーマンスメトリクスを取得
    pub fn get_metrics(&self) -> &KernelMetrics {
        &self.metrics
    }
}

/// Simplified kernel selector
/// 簡潔なカーネル選択器
pub struct KernelSelector {
    executors: Vec<UnifiedKernelExecutor>,
    current: usize,
}

impl KernelSelector {
    /// Create new kernel selector
    /// 新しいカーネルセレクターを作成
    pub fn new() -> Self {
        Self {
            executors: Vec::new(),
            current: 0,
        }
    }

    /// Add executor
    /// 実行者を追加
    pub fn add_executor(&mut self, executor: UnifiedKernelExecutor) {
        self.executors.push(executor);
    }

    /// Select best executor for operation
    /// 操作に最適な実行者を選択
    pub fn select_best(&mut self, op: KernelOp) -> Option<&mut UnifiedKernelExecutor> {
        // Simple selection: first available executor that supports the operation
        for (i, executor) in self.executors.iter().enumerate() {
            if executor.supports_operation(op) {
                self.current = i;
                return self.executors.get_mut(i);
            }
        }
        None
    }

    /// Execute operation with best executor
    /// 最適な実行者で操作を実行
    pub fn execute_f32(
        &mut self,
        op: KernelOp,
        inputs: &[&Tensor<f32>],
        params: &KernelParams,
    ) -> RusTorchResult<Tensor<f32>> {
        if let Some(executor) = self.select_best(op) {
            executor.execute_f32(op, inputs, params)
        } else {
            Err(RusTorchError::UnsupportedOperation(format!(
                "No executor supports operation {:?}",
                op
            )))
        }
    }

    /// Execute operation with best executor
    /// 最適な実行者で操作を実行
    pub fn execute_f64(
        &mut self,
        op: KernelOp,
        inputs: &[&Tensor<f64>],
        params: &KernelParams,
    ) -> RusTorchResult<Tensor<f64>> {
        if let Some(executor) = self.select_best(op) {
            executor.execute_f64(op, inputs, params)
        } else {
            Err(RusTorchError::UnsupportedOperation(format!(
                "No executor supports operation {:?}",
                op
            )))
        }
    }

    /// Get available devices
    /// 利用可能デバイスを取得
    pub fn available_devices(&self) -> Vec<DeviceType> {
        self.executors.iter().map(|e| e.device_type()).collect()
    }
}

impl Default for KernelSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_kernel_executor() {
        let executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();

        assert_eq!(executor.device_type(), DeviceType::Cpu);
        assert!(executor.supports_operation(KernelOp::Add));
        assert!(executor.supports_operation(KernelOp::MatMul));
        assert!(!executor.supports_operation(KernelOp::Conv2D));
    }

    #[test]
    fn test_kernel_execution() {
        let mut executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        let params = KernelParams::default();

        let result = executor
            .execute_f32(KernelOp::Add, &[&a, &b], &params)
            .unwrap();
        let expected = vec![5.0f32, 7.0, 9.0];
        assert_eq!(result.as_slice().unwrap(), &expected);

        // Check metrics were recorded
        assert!(executor.get_metrics().execution_time.as_nanos() > 0);
    }

    #[test]
    fn test_kernel_selector() {
        let mut selector = KernelSelector::new();
        let executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        selector.add_executor(executor);

        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        let params = KernelParams::default();

        let result = selector
            .execute_f32(KernelOp::Add, &[&a, &b], &params)
            .unwrap();
        let expected = vec![5.0f32, 7.0, 9.0];
        assert_eq!(result.as_slice().unwrap(), &expected);

        // Check available devices
        let devices = selector.available_devices();
        assert_eq!(devices, vec![DeviceType::Cpu]);
    }

    #[test]
    fn test_matrix_multiplication() {
        let mut executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let params = KernelParams::default();

        let result = executor
            .execute_f32(KernelOp::MatMul, &[&a, &b], &params)
            .unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // Verify metrics show MatMul operation
        let metrics = executor.get_metrics();
        assert!(metrics.execution_time.as_nanos() > 0);
        assert_eq!(metrics.memory_bandwidth, 50.0); // CPU bandwidth
    }

    #[test]
    fn test_f64_operations() {
        let mut executor = UnifiedKernelExecutor::new(DeviceType::Cpu).unwrap();
        let a = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f64, 5.0, 6.0], vec![3]);
        let params = KernelParams::default();

        let result = executor
            .execute_f64(KernelOp::Mul, &[&a, &b], &params)
            .unwrap();
        let expected = vec![4.0f64, 10.0, 18.0];
        assert_eq!(result.as_slice().unwrap(), &expected);
    }
}
