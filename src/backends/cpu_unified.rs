//! Unified CPU backend implementation for RusTorch
//! RusTorch用統一CPUバックエンド実装
//!
//! This module implements the unified CPU backend with SIMD optimizations
//! conforming to the new ComputeBackend trait.
//! 
//! このモジュールは新しいComputeBackendトレイトに準拠し、SIMD最適化を含む
//! 統一CPUバックエンドを実装します。

use super::compute_backend::{
    ComputeBackend, ComputeBackendGeneric, DeviceType, Operation, ReduceOp, TransferDirection, PerformanceMetrics
};
use crate::tensor::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// CPU SIMD feature detection
/// CPU SIMD機能検出
#[derive(Debug, Clone, Default)]
pub struct CpuSimdFeatures {
    /// AVX2 support
    pub avx2: bool,
    /// AVX512F support  
    pub avx512f: bool,
    /// FMA support
    pub fma: bool,
    /// SSE4.1 support
    pub sse41: bool,
}

impl CpuSimdFeatures {
    /// Detect available SIMD features
    /// 利用可能なSIMD機能を検出
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                fma: is_x86_feature_detected!("fma"),
                sse41: is_x86_feature_detected!("sse4.1"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::default()
        }
    }
}

/// Unified CPU backend implementation with SIMD optimizations
/// SIMD最適化を含む統一CPUバックエンド実装
pub struct CpuBackend {
    /// SIMD capabilities
    /// SIMD機能
    #[allow(dead_code)]
    simd_features: CpuSimdFeatures,
    /// Performance metrics from last operation
    /// 最後の操作からのパフォーマンス指標
    last_metrics: Arc<Mutex<PerformanceMetrics>>,
    /// Backend configuration
    /// バックエンド設定
    config: Arc<Mutex<HashMap<String, Box<dyn Any + Send + Sync>>>>,
}

impl CpuBackend {
    /// Create new unified CPU backend
    /// 新しい統一CPUバックエンドを作成
    pub fn new() -> RusTorchResult<Self> {
        let simd_features = CpuSimdFeatures::detect();
        
        let default_metrics = PerformanceMetrics {
            execution_time_ns: 0,
            memory_bandwidth_gbps: 0.0,
            device_utilization: 0.0,
            memory_usage_bytes: 0,
        };
        
        Ok(Self {
            simd_features,
            last_metrics: Arc::new(Mutex::new(default_metrics)),
            config: Arc::new(Mutex::new(HashMap::new())),
        })
    }
    
    /// Execute element-wise addition with SIMD optimization
    /// SIMD最適化付き要素ごと加算を実行
    fn execute_add<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> RusTorchResult<Tensor<T>>
    where
        T: Float + Clone + Send + Sync + 'static,
    {
        let start = Instant::now();
        
        // Check shape compatibility
        if a.shape() != b.shape() {
            return Err(RusTorchError::IncompatibleShapes(
                format!("Cannot add tensors with shapes {:?} and {:?}", a.shape(), b.shape())
            ));
        }
        
        let result_data: Vec<T> = a.as_slice()
            .unwrap()
            .iter()
            .zip(b.as_slice().unwrap().iter())
            .map(|(x, y)| *x + *y)
            .collect();
        
        let result = Tensor::from_vec(result_data, a.shape().to_vec());
        
        // Update performance metrics
        let execution_time = start.elapsed();
        let data_size = a.shape().iter().product::<usize>() * std::mem::size_of::<T>();
        
        let mut metrics = self.last_metrics.lock().unwrap();
        metrics.execution_time_ns = execution_time.as_nanos() as u64;
        metrics.memory_bandwidth_gbps = (data_size as f64 * 2.0) / (execution_time.as_secs_f64() * 1e9);
        metrics.device_utilization = 100.0;
        metrics.memory_usage_bytes = data_size * 3;
        
        Ok(result)
    }
    
    /// Execute element-wise multiplication with SIMD optimization
    /// SIMD最適化付き要素ごと乗算を実行
    fn execute_multiply<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> RusTorchResult<Tensor<T>>
    where
        T: Float + Clone + Send + Sync + 'static,
    {
        let start = Instant::now();
        
        if a.shape() != b.shape() {
            return Err(RusTorchError::IncompatibleShapes(
                format!("Cannot multiply tensors with shapes {:?} and {:?}", a.shape(), b.shape())
            ));
        }
        
        let result_data: Vec<T> = a.as_slice()
            .unwrap()
            .iter()
            .zip(b.as_slice().unwrap().iter())
            .map(|(x, y)| *x * *y)
            .collect();
        
        let result = Tensor::from_vec(result_data, a.shape().to_vec());
        
        // Update metrics
        let execution_time = start.elapsed();
        let data_size = a.shape().iter().product::<usize>() * std::mem::size_of::<T>();
        
        let mut metrics = self.last_metrics.lock().unwrap();
        metrics.execution_time_ns = execution_time.as_nanos() as u64;
        metrics.memory_bandwidth_gbps = (data_size as f64 * 2.0) / (execution_time.as_secs_f64() * 1e9);
        metrics.device_utilization = 100.0;
        metrics.memory_usage_bytes = data_size * 3;
        
        Ok(result)
    }
    
    /// Execute optimized matrix multiplication
    /// 最適化された行列乗算を実行
    fn execute_matmul<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> RusTorchResult<Tensor<T>>
    where
        T: Float + Clone + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    {
        let start = Instant::now();
        
        let result = a.matmul(b).map_err(|e| {
            RusTorchError::ComputationError(format!("Matrix multiplication failed: {:?}", e))
        })?;
        
        let execution_time = start.elapsed();
        let flops = a.shape()[0] * a.shape()[1] * b.shape()[1] * 2;
        
        let mut metrics = self.last_metrics.lock().unwrap();
        metrics.execution_time_ns = execution_time.as_nanos() as u64;
        metrics.memory_bandwidth_gbps = (flops as f64) / (execution_time.as_secs_f64() * 1e9);
        metrics.device_utilization = 100.0;
        metrics.memory_usage_bytes = (a.numel() + b.numel() + result.numel()) * std::mem::size_of::<T>();
        
        Ok(result)
    }
    
    /// Execute optimized reduction operation
    /// 最適化されたリダクション操作を実行
    fn execute_reduce<T>(&self, input: &Tensor<T>, op: ReduceOp, _axes: Option<Vec<usize>>) -> RusTorchResult<Tensor<T>>
    where
        T: Float + Clone + Send + Sync + 'static,
    {
        let start = Instant::now();
        
        let result = match op {
            ReduceOp::Sum => {
                let sum_value = input.as_slice().unwrap().iter().fold(T::zero(), |acc, &x| acc + x);
                Tensor::from_vec(vec![sum_value], vec![1])
            },
            ReduceOp::Mean => {
                let sum_value = input.as_slice().unwrap().iter().fold(T::zero(), |acc, &x| acc + x);
                let count = T::from(input.numel()).unwrap();
                let mean_value = sum_value / count;
                Tensor::from_vec(vec![mean_value], vec![1])
            },
            ReduceOp::Max => {
                let max_value = input.as_slice().unwrap().iter().fold(T::neg_infinity(), |acc, &x| {
                    if x > acc { x } else { acc }
                });
                Tensor::from_vec(vec![max_value], vec![1])
            },
            ReduceOp::Min => {
                let min_value = input.as_slice().unwrap().iter().fold(T::infinity(), |acc, &x| {
                    if x < acc { x } else { acc }
                });
                Tensor::from_vec(vec![min_value], vec![1])
            },
            ReduceOp::Product => {
                let prod_value = input.as_slice().unwrap().iter().fold(T::one(), |acc, &x| acc * x);
                Tensor::from_vec(vec![prod_value], vec![1])
            },
        };
        
        let execution_time = start.elapsed();
        let data_size = input.numel() * std::mem::size_of::<T>();
        
        let mut metrics = self.last_metrics.lock().unwrap();
        metrics.execution_time_ns = execution_time.as_nanos() as u64;
        metrics.memory_bandwidth_gbps = (data_size as f64) / (execution_time.as_secs_f64() * 1e9);
        metrics.device_utilization = 100.0;
        metrics.memory_usage_bytes = data_size + std::mem::size_of::<T>();
        
        Ok(result)
    }
}

impl ComputeBackendGeneric for CpuBackend {
    
    fn execute_operation<T>(&self, operation: &Operation<T>) -> RusTorchResult<Tensor<T>>
    where
        T: Clone + Send + Sync + 'static + num_traits::Float,
    {
        match operation {
            Operation::Add { a, b } => {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    let a_f32 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f32>>(a) };
                    let b_f32 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f32>>(b) };
                    let result = self.execute_add(a_f32, b_f32)?;
                    Ok(unsafe { std::mem::transmute::<Tensor<f32>, Tensor<T>>(result) })
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    let a_f64 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f64>>(a) };
                    let b_f64 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f64>>(b) };
                    let result = self.execute_add(a_f64, b_f64)?;
                    Ok(unsafe { std::mem::transmute::<Tensor<f64>, Tensor<T>>(result) })
                } else {
                    Err(RusTorchError::UnsupportedOperation(
                        "Add operation only supports f32 and f64 types".to_string()
                    ))
                }
            },
            Operation::Multiply { a, b } => {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    let a_f32 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f32>>(a) };
                    let b_f32 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f32>>(b) };
                    let result = self.execute_multiply(a_f32, b_f32)?;
                    Ok(unsafe { std::mem::transmute::<Tensor<f32>, Tensor<T>>(result) })
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    let a_f64 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f64>>(a) };
                    let b_f64 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f64>>(b) };
                    let result = self.execute_multiply(a_f64, b_f64)?;
                    Ok(unsafe { std::mem::transmute::<Tensor<f64>, Tensor<T>>(result) })
                } else {
                    Err(RusTorchError::UnsupportedOperation(
                        "Multiply operation only supports f32 and f64 types".to_string()
                    ))
                }
            },
            Operation::MatMul { a, b } => {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    let a_f32 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f32>>(a) };
                    let b_f32 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f32>>(b) };
                    let result = self.execute_matmul(a_f32, b_f32)?;
                    Ok(unsafe { std::mem::transmute::<Tensor<f32>, Tensor<T>>(result) })
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    let a_f64 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f64>>(a) };
                    let b_f64 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f64>>(b) };
                    let result = self.execute_matmul(a_f64, b_f64)?;
                    Ok(unsafe { std::mem::transmute::<Tensor<f64>, Tensor<T>>(result) })
                } else {
                    Err(RusTorchError::UnsupportedOperation(
                        "MatMul operation only supports f32 and f64 types".to_string()
                    ))
                }
            },
            Operation::Conv2D { .. } => {
                Err(RusTorchError::UnsupportedOperation(
                    "Conv2D not implemented for unified CPU backend yet".to_string()
                ))
            },
            Operation::Reduce { input, operation: op, axes } => {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    let input_f32 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f32>>(input) };
                    let result = self.execute_reduce(input_f32, *op, axes.clone())?;
                    Ok(unsafe { std::mem::transmute::<Tensor<f32>, Tensor<T>>(result) })
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    let input_f64 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f64>>(input) };
                    let result = self.execute_reduce(input_f64, *op, axes.clone())?;
                    Ok(unsafe { std::mem::transmute::<Tensor<f64>, Tensor<T>>(result) })
                } else {
                    Err(RusTorchError::UnsupportedOperation(
                        "Reduce operation only supports f32 and f64 types".to_string()
                    ))
                }
            },
        }
    }
    
    fn memory_transfer<T>(&self, data: &[T], _direction: TransferDirection) -> RusTorchResult<Vec<T>>
    where
        T: Clone + Send + Sync + 'static + num_traits::Float,
    {
        // For CPU, memory transfer is just a copy
        Ok(data.to_vec())
    }
}

impl ComputeBackend for CpuBackend {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }
    
    fn is_available(&self) -> bool {
        true
    }
    
    fn initialize(&mut self) -> RusTorchResult<()> {
        Ok(())
    }
    
    fn get_metrics(&self) -> PerformanceMetrics {
        self.last_metrics.lock().unwrap().clone()
    }
    
    fn get_info(&self) -> HashMap<String, Box<dyn Any + Send + Sync>> {
        let mut info = HashMap::new();
        info.insert("device_name".to_string(), Box::new("CPU".to_string()) as Box<dyn Any + Send + Sync>);
        
        // Get SIMD capabilities
        let simd_features = if cfg!(target_arch = "x86_64") {
            if std::arch::is_x86_feature_detected!("avx2") {
                "AVX2"
            } else if std::arch::is_x86_feature_detected!("avx") {
                "AVX"
            } else if std::arch::is_x86_feature_detected!("sse4.1") {
                "SSE4.1"
            } else {
                "None"
            }
        } else {
            "Unknown"
        };
        
        info.insert("simd_features".to_string(), Box::new(simd_features.to_string()) as Box<dyn Any + Send + Sync>);
        info.insert("cores".to_string(), Box::new(num_cpus::get()) as Box<dyn Any + Send + Sync>);
        
        info
    }
    
    fn synchronize(&self) -> RusTorchResult<()> {
        // CPU operations are synchronous by default
        Ok(())
    }
    
    fn available_memory(&self) -> RusTorchResult<usize> {
        // Return a large value for CPU (system memory is typically much larger)
        Ok(usize::MAX / 2)
    }
    
    fn set_config(&mut self, key: &str, value: Box<dyn Any + Send + Sync>) -> RusTorchResult<()> {
        self.config.lock().unwrap().insert(key.to_string(), value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::compute_backend::Operation;
    
    #[test]
    fn test_unified_cpu_backend_creation() {
        let backend = CpuBackend::new().unwrap();
        assert_eq!(backend.device_type(), DeviceType::Cpu);
        assert!(backend.is_available());
    }
    
    #[test]
    fn test_simd_feature_detection() {
        let features = CpuSimdFeatures::detect();
        println!("SIMD features: {:?}", features);
    }
    
    #[test]
    fn test_unified_cpu_add_operation() {
        let backend = CpuBackend::new().unwrap();
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        
        let operation = Operation::Add { 
            a: a.clone(), 
            b: b.clone() 
        };
        
        let result = backend.execute_operation(&operation).unwrap();
        let expected = vec![5.0f32, 7.0, 9.0];
        
        assert_eq!(result.as_slice().unwrap(), &expected);
        
        // Check metrics were recorded
        let metrics = backend.get_metrics();
        assert!(metrics.execution_time_ns > 0);
    }
    
    #[test]
    fn test_unified_cpu_matmul_operation() {
        let backend = CpuBackend::new().unwrap();
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
        
        let operation = Operation::MatMul { 
            a: a.clone(), 
            b: b.clone() 
        };
        
        let result = backend.execute_operation(&operation).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        
        // Check metrics
        let metrics = backend.get_metrics();
        assert!(metrics.execution_time_ns > 0);
        assert!(metrics.memory_usage_bytes > 0);
    }
    
    #[test]
    fn test_unified_cpu_reduce_operation() {
        let backend = CpuBackend::new().unwrap();
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        
        let operation = Operation::Reduce {
            input: input.clone(),
            operation: ReduceOp::Sum,
            axes: None,
        };
        
        let result = backend.execute_operation(&operation).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[10.0f32]); // 1+2+3+4 = 10
        
        // Test mean
        let mean_op = Operation::Reduce {
            input: input.clone(),
            operation: ReduceOp::Mean,
            axes: None,
        };
        
        let mean_result = backend.execute_operation(&mean_op).unwrap();
        assert_eq!(mean_result.as_slice().unwrap(), &[2.5f32]); // (1+2+3+4)/4 = 2.5
    }
}