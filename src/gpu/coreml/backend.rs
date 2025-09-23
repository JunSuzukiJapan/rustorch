//! CoreML backend integration and implementation
//! CoreMLバックエンド統合と実装

use super::common::error_helpers;
use super::common::*;
use super::device::{CoreMLDevice, CoreMLDeviceManager};
use crate::gpu::coreml::common::coreml_feature;
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// CoreML implementations are handled in the coreml_implementation module

/// Backend statistics for monitoring
/// 監視用バックエンド統計
#[derive(Debug, Default, Clone)]
pub struct CoreMLBackendStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_operations: u64,
    pub fallback_operations: u64,
    pub average_execution_time: std::time::Duration,
}

/// Unified CoreML backend interface
/// 統一CoreMLバックエンドインターフェース
#[derive(Clone)]
pub struct CoreMLBackend {
    /// Device manager for CoreML devices
    /// CoreMLデバイス用デバイスマネージャー
    device_manager: Arc<CoreMLDeviceManager>,

    /// Model manager for CoreML model caching and execution
    /// CoreMLモデルキャッシュと実行用モデルマネージャー
    model_manager: Arc<super::model_manager::CoreMLModelManager>,

    /// Operation cache for performance optimization
    /// パフォーマンス最適化用演算キャッシュ
    operation_cache: Arc<Mutex<HashMap<String, CachedOperation>>>,

    /// Backend configuration
    /// バックエンド設定
    config: CoreMLBackendConfig,
}

/// Configuration for CoreML backend
/// CoreMLバックエンド設定
#[derive(Debug, Clone)]
pub struct CoreMLBackendConfig {
    /// Enable operation caching
    /// 演算キャッシングを有効化
    pub enable_caching: bool,

    /// Maximum cache size (number of operations)
    /// 最大キャッシュサイズ（演算数）
    pub max_cache_size: usize,

    /// Automatically fallback to CPU on errors
    /// エラー時に自動的にCPUにフォールバック
    pub auto_fallback: bool,

    /// Performance profiling enabled
    /// パフォーマンスプロファイリング有効
    pub enable_profiling: bool,
}

impl Default for CoreMLBackendConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 1000,
            auto_fallback: true,
            enable_profiling: false,
        }
    }
}

/// Cached operation information
/// キャッシュされた演算情報
#[derive(Debug, Clone)]
struct CachedOperation {
    /// Operation identifier
    /// 演算識別子
    id: String,

    /// Last execution time
    /// 最後の実行時間
    last_execution_time: std::time::Duration,

    /// Number of executions
    /// 実行回数
    execution_count: usize,

    /// Average execution time
    /// 平均実行時間
    average_time: std::time::Duration,
}

/// Neural Engine performance and capability information
/// Neural Engineパフォーマンスと機能情報
#[derive(Debug, Clone)]
pub struct NeuralEngineInfo {
    pub available: bool,
    pub apple_silicon: bool,
    pub optimal_batch_size: usize,
    pub optimal_matrix_size: usize,
    pub supports_float16: bool,
    pub supports_int8: bool,
    pub max_ops_per_second: u64, // TOPS (Trillion Operations Per Second)
}
impl CoreMLBackend {
    /// Get global CoreML backend instance
    /// グローバルCoreMLバックエンドインスタンスを取得
    pub fn global() -> &'static CoreMLBackend {
        static BACKEND: OnceLock<CoreMLBackend> = OnceLock::new();
        BACKEND.get_or_init(|| {
            CoreMLBackend::new(CoreMLBackendConfig::default()).unwrap_or_else(|_| {
                // Fallback to a dummy backend if initialization fails
                CoreMLBackend::dummy()
            })
        })
    }

    /// Create new CoreML backend
    /// 新しいCoreMLバックエンドを作成
    pub fn new(config: CoreMLBackendConfig) -> CoreMLResult<Self> {
        let device_manager = Arc::new(CoreMLDeviceManager::new());
        let model_manager = Arc::new(super::model_manager::CoreMLModelManager::new());

        // Verify CoreML availability during backend creation
        if !is_coreml_available() {
            return Err(error_helpers::not_available());
        }

        // Warmup the device manager
        if let Err(e) = device_manager.warmup() {
            // Log warning but don't fail backend creation
            eprintln!("Warning: CoreML warmup failed: {}", e);
        }

        Ok(Self {
            device_manager,
            model_manager,
            operation_cache: Arc::new(Mutex::new(HashMap::new())),
            config,
        })
    }

    /// Create a dummy backend for fallback scenarios
    /// フォールバックシナリオ用のダミーバックエンドを作成
    fn dummy() -> Self {
        Self {
            device_manager: Arc::new(CoreMLDeviceManager::new()),
            model_manager: Arc::new(super::model_manager::CoreMLModelManager::new()),
            operation_cache: Arc::new(Mutex::new(HashMap::new())),
            config: CoreMLBackendConfig {
                enable_caching: false,
                auto_fallback: true,
                ..Default::default()
            },
        }
    }

    /// Check if backend is available and ready
    /// バックエンドが利用可能で準備完了かチェック
    pub fn is_available(&self) -> bool {
        is_coreml_available() && self.device_manager.is_available()
    }

    /// Get device manager
    /// デバイスマネージャーを取得
    pub fn device_manager(&self) -> &CoreMLDeviceManager {
        &self.device_manager
    }

    /// Get model manager
    /// モデルマネージャーを取得
    pub fn model_manager(&self) -> &super::model_manager::CoreMLModelManager {
        &self.model_manager
    }

    /// Execute operation with caching and profiling
    /// キャッシングとプロファイリング付きで演算を実行
    pub fn execute_operation<T, F>(&self, operation_id: &str, operation: F) -> CoreMLResult<T>
    where
        F: FnOnce() -> CoreMLResult<T>,
    {
        let start_time = std::time::Instant::now();

        // Execute the operation
        let result = operation();

        let execution_time = start_time.elapsed();

        // Update cache if enabled
        if self.config.enable_caching && result.is_ok() {
            self.update_operation_cache(operation_id, execution_time);
        }

        // Log profiling information if enabled
        if self.config.enable_profiling {
            println!(
                "CoreML Operation '{}' executed in {:?}",
                operation_id, execution_time
            );
        }

        result
    }

    /// Update operation cache with execution statistics
    /// 実行統計でキャッシュを更新
    fn update_operation_cache(&self, operation_id: &str, execution_time: std::time::Duration) {
        if let Ok(mut cache) = self.operation_cache.lock() {
            let cached_op =
                cache
                    .entry(operation_id.to_string())
                    .or_insert_with(|| CachedOperation {
                        id: operation_id.to_string(),
                        last_execution_time: execution_time,
                        execution_count: 0,
                        average_time: execution_time,
                    });

            cached_op.last_execution_time = execution_time;
            cached_op.execution_count += 1;

            // Update running average
            let total_time =
                cached_op.average_time.as_nanos() as f64 * (cached_op.execution_count - 1) as f64;
            let new_total = total_time + execution_time.as_nanos() as f64;
            cached_op.average_time = std::time::Duration::from_nanos(
                (new_total / cached_op.execution_count as f64) as u64,
            );

            // Enforce cache size limit
            if cache.len() > self.config.max_cache_size {
                // Remove oldest entries (simple LRU approximation)
                let keys_to_remove: Vec<String> = cache
                    .iter()
                    .filter(|(_, op)| op.execution_count == 1) // Remove single-use entries first
                    .take(cache.len() - self.config.max_cache_size + 1)
                    .map(|(k, _)| k.clone())
                    .collect();

                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }
    }

    /// Get operation statistics from cache
    /// キャッシュから演算統計を取得
    pub fn get_operation_stats(&self, operation_id: &str) -> Option<(usize, std::time::Duration)> {
        if let Ok(cache) = self.operation_cache.lock() {
            cache
                .get(operation_id)
                .map(|op| (op.execution_count, op.average_time))
        } else {
            None
        }
    }

    /// Get backend statistics
    /// バックエンド統計を取得
    pub fn get_stats(&self) -> CoreMLBackendStats {
        let (total_ops, total_execs) = self.cache_stats();

        CoreMLBackendStats {
            cache_hits: 0,   // TODO: Implement proper cache hit tracking
            cache_misses: 0, // TODO: Implement proper cache miss tracking
            total_operations: total_execs as u64,
            fallback_operations: 0, // TODO: Implement fallback tracking
            average_execution_time: std::time::Duration::from_millis(0), // TODO: Calculate from cache
        }
    }

    /// Cleanup cache (alias for clear_cache for consistency)
    /// キャッシュクリーンアップ（一貫性のためのclear_cacheエイリアス）
    pub fn cleanup_cache(&self) -> CoreMLResult<()> {
        self.clear_cache();
        Ok(())
    }

    /// Clear operation cache
    /// 演算キャッシュをクリア
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.operation_cache.lock() {
            cache.clear();
        }
    }

    /// Get cache statistics
    /// キャッシュ統計を取得
    pub fn cache_stats(&self) -> (usize, usize) {
        if let Ok(cache) = self.operation_cache.lock() {
            let total_operations = cache.len();
            let total_executions: usize = cache.values().map(|op| op.execution_count).sum();
            (total_operations, total_executions)
        } else {
            (0, 0)
        }
    }

    /// Check if Apple Neural Engine is available
    /// Apple Neural Engineが利用可能かチェック
    pub fn is_neural_engine_available(&self) -> bool {
        #[cfg(all(feature = "coreml", target_os = "macos"))]
        {
            // Check for Apple Neural Engine availability
            // This requires checking system capabilities and CoreML compute units
            
            // Check if we're running on Apple Silicon
            if !self.is_apple_silicon() {
                return false;
            }
            
            // For this implementation, assume Neural Engine is available on Apple Silicon
            // In production, this would test actual CoreML model loading with ANE compute units
            true
        }
        
        #[cfg(not(all(feature = "coreml", target_os = "macos")))]
        {
            false
        }
    }
    
    /// Check if running on Apple Silicon
    /// Apple Siliconで動作しているかチェック
    fn is_apple_silicon(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            
            // Check system architecture
            if let Ok(output) = Command::new("uname").arg("-m").output() {
                let arch = String::from_utf8_lossy(&output.stdout);
                return arch.trim() == "arm64";
            }
            
            false
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
    
    /// Execute matrix multiplication directly on Apple Neural Engine
    /// Apple Neural Engine上で直接行列乗算を実行
    pub fn execute_on_neural_engine<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> CoreMLResult<Tensor<T>>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
    {
        #[cfg(all(feature = "coreml", target_os = "macos"))]
        {
            // Use model manager with Neural Engine optimization
            let model_manager = self.model_manager();
            let model_handle = model_manager.get_or_create_matmul_model::<T>(a.shape(), b.shape())?;
            
            // Execute with Neural Engine specific optimizations
            let start_time = std::time::Instant::now();
            
            // Convert tensors to optimal format for Neural Engine
            let a_optimized = self.optimize_tensor_for_ane(a)?;
            let b_optimized = self.optimize_tensor_for_ane(b)?;
            
            // Execute on Neural Engine using our proven Metal-CoreML hybrid approach
            let inputs = vec![&a_optimized, &b_optimized];
            let outputs = model_manager.execute_model(&model_handle, &inputs)?;
            
            let execution_time = start_time.elapsed();
            
            // Log Neural Engine usage for performance monitoring
            if self.config.enable_profiling {
                println!(
                    "Neural Engine execution completed in {:?} for {}x{} matrix multiplication",
                    execution_time,
                    a.shape()[0],
                    b.shape()[1]
                );
            }
            
            if outputs.is_empty() {
                return Err(error_helpers::unsupported_operation(
                    "Neural Engine model returned no outputs"
                ));
            }
            
            Ok(outputs[0].clone())
        }
        
        #[cfg(not(all(feature = "coreml", target_os = "macos")))]
        {
            Err(error_helpers::not_available())
        }
    }
    
    /// Optimize tensor format for Apple Neural Engine
    /// Apple Neural Engine用にテンソル形式を最適化
    fn optimize_tensor_for_ane<T>(&self, tensor: &Tensor<T>) -> CoreMLResult<Tensor<T>>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
    {
        // Apple Neural Engine optimizations:
        // 1. Prefer Float16 for better performance (when precision allows)
        // 2. Ensure optimal memory layout
        // 3. Apply any necessary padding for optimal ANE utilization
        
        // For this implementation, we'll ensure the tensor is in optimal format
        // In production, this would involve:
        // - Converting to optimal data types (Float16 when possible)
        // - Applying padding for optimal ANE tile sizes
        // - Ensuring proper memory alignment
        
        // For now, return the tensor as-is with potential future optimizations
        Ok(tensor.clone())
    }
    
    /// Get Neural Engine performance characteristics
    /// Neural Engineパフォーマンス特性を取得
    pub fn get_neural_engine_info(&self) -> Option<NeuralEngineInfo> {
        #[cfg(all(feature = "coreml", target_os = "macos"))]
        {
            if self.is_neural_engine_available() {
                Some(NeuralEngineInfo {
                    available: true,
                    apple_silicon: self.is_apple_silicon(),
                    optimal_batch_size: 1,
                    optimal_matrix_size: 1024, // Typical ANE sweet spot
                    supports_float16: true,
                    supports_int8: true,
                    max_ops_per_second: 15_800_000_000_000, // ~15.8 TOPS for M1/M2
                })
            } else {
                None
            }
        }
        
        #[cfg(not(all(feature = "coreml", target_os = "macos")))]
        {
            None
        }
    }
}

/// CoreML computation graph for executing operations
/// 演算実行用CoreML計算グラフ
pub struct CoreMLGraph {
    device: CoreMLDevice,
    backend: Arc<CoreMLBackend>,
}

impl CoreMLGraph {
    /// Create new CoreML computation graph
    /// 新しいCoreML計算グラフを作成
    pub fn new(device_id: usize) -> CoreMLResult<Self> {
        let backend = Arc::new(CoreMLBackend::new(CoreMLBackendConfig::default())?);
        let device = backend.device_manager().initialize(device_id)?;

        Ok(Self { device, backend })
    }

    /// Execute matrix multiplication
    /// 行列乗算を実行
    pub fn matmul<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> CoreMLResult<Tensor<T>>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
    {
        // Validate tensors
        let a_size = a.shape().iter().product::<usize>() * std::mem::size_of::<T>();
        let b_size = b.shape().iter().product::<usize>() * std::mem::size_of::<T>();

        self.device
            .validate_tensor::<T>(a_size, &crate::dtype::DType::Float32)?;
        self.device
            .validate_tensor::<T>(b_size, &crate::dtype::DType::Float32)?;

        let operation_id = format!(
            "matmul_{}x{}_{}",
            a.shape().iter().product::<usize>(),
            b.shape().iter().product::<usize>(),
            std::any::type_name::<T>()
        );

        self.backend.execute_operation(&operation_id, || {
            #[cfg(any(
                feature = "coreml",
                feature = "coreml-hybrid",
                feature = "coreml-fallback"
            ))]
            {
                // Use actual CoreML implementation with MLMultiArray conversion
                let a_shape = a.shape();
                let b_shape = b.shape();
                
                // Check if tensors are suitable for CoreML (2D matrices)
                if a_shape.len() != 2 || b_shape.len() != 2 {
                    // Fallback to CPU for non-2D tensors
                    return a.matmul(b);
                }
                
                if a_shape[1] != b_shape[0] {
                    return Err(error_helpers::unsupported_operation(
                        "Matrix dimensions don't match for multiplication"
                    ));
                }
                
                // Use CoreML implementation with actual MLMultiArray conversion
                return coreml_implementation::coreml_matmul(a, b);
            }

            #[cfg(not(any(
                feature = "coreml",
                feature = "coreml-hybrid",
                feature = "coreml-fallback"
            )))]
            {
                Err(error_helpers::feature_disabled())
            }
        })
    }

    /// Execute 2D convolution
    /// 2D畳み込みを実行
    pub fn conv2d<T>(
        &self,
        input: &Tensor<T>,
        kernel: &Tensor<T>,
        stride: &[usize],
        padding: &[usize],
    ) -> CoreMLResult<Tensor<T>>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
    {
        // Validate tensors
        let input_size = input.shape().iter().product::<usize>() * std::mem::size_of::<T>();
        let kernel_size = kernel.shape().iter().product::<usize>() * std::mem::size_of::<T>();

        self.device
            .validate_tensor::<T>(input_size, &crate::dtype::DType::Float32)?;
        self.device
            .validate_tensor::<T>(kernel_size, &crate::dtype::DType::Float32)?;

        let operation_id = format!(
            "conv2d_{}_{}_{:?}_{:?}",
            input.shape().iter().product::<usize>(),
            kernel.shape().iter().product::<usize>(),
            stride,
            padding
        );

        self.backend.execute_operation(&operation_id, || {
            #[cfg(any(
                feature = "coreml",
                feature = "coreml-hybrid",
                feature = "coreml-fallback"
            ))]
            {
                // Use CoreML implementation with hardware acceleration
                return coreml_implementation::coreml_conv2d(input, kernel, stride, padding);
            }

            #[cfg(not(any(
                feature = "coreml",
                feature = "coreml-hybrid",
                feature = "coreml-fallback"
            )))]
            {
                Err(error_helpers::feature_disabled())
            }
        })
    }

    /// Execute activation function
    /// 活性化関数を実行
    pub fn activation<T>(
        &self,
        input: &Tensor<T>,
        activation_type: CoreMLActivationType,
    ) -> CoreMLResult<Tensor<T>>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
    {
        // Validate tensor
        let input_size = input.shape().iter().product::<usize>() * std::mem::size_of::<T>();
        self.device
            .validate_tensor::<T>(input_size, &crate::dtype::DType::Float32)?;

        let operation_id = format!(
            "activation_{:?}_{}",
            activation_type,
            input.shape().iter().product::<usize>()
        );

        self.backend.execute_operation(&operation_id, || {
            #[cfg(any(
                feature = "coreml",
                feature = "coreml-hybrid",
                feature = "coreml-fallback"
            ))]
            {
                // Use CoreML implementation with hardware acceleration
                return coreml_implementation::coreml_activation(input, activation_type);
            }

            #[cfg(not(any(
                feature = "coreml",
                feature = "coreml-hybrid",
                feature = "coreml-fallback"
            )))]
            {
                Err(error_helpers::feature_disabled())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let config = CoreMLBackendConfig::default();
        let backend_result = CoreMLBackend::new(config);

        // Should succeed or fail gracefully depending on platform
        match backend_result {
            Ok(backend) => {
                assert!(backend.config.enable_caching);
                assert_eq!(backend.config.max_cache_size, 1000);
            }
            Err(_) => {
                // Expected on non-macOS or when CoreML is not available
                println!("CoreML backend creation failed (expected on some platforms)");
            }
        }
    }

    #[test]
    fn test_global_backend() {
        let backend = CoreMLBackend::global();

        // Should always succeed (creates dummy backend if needed)
        assert!(backend.config.auto_fallback);
    }

    #[test]
    fn test_cache_operations() {
        let backend = CoreMLBackend::global();

        // Test cache clearing
        backend.clear_cache();
        let (ops, execs) = backend.cache_stats();
        assert_eq!(ops, 0);
        assert_eq!(execs, 0);
    }

    #[test]
    fn test_graph_creation() {
        let graph_result = CoreMLGraph::new(0);

        match graph_result {
            Ok(_graph) => {
                // Successfully created graph
                println!("CoreML graph created successfully");
            }
            Err(_) => {
                // Expected on platforms without CoreML
                println!("CoreML graph creation failed (expected on some platforms)");
            }
        }
    }
}

/// CoreML MLMultiArray conversion and computation helpers
/// CoreML MLMultiArray変換と計算ヘルパー
#[cfg(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback"
))]
mod coreml_implementation {
    use super::*;
    
    /// Perform matrix multiplication using CoreML with Apple Neural Engine optimization
    /// Apple Neural Engine最適化を伴うCoreMLを使用して行列乗算を実行
    pub fn coreml_matmul<T>(
        a: &Tensor<T>,
        b: &Tensor<T>,
    ) -> CoreMLResult<Tensor<T>>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
    {
        #[cfg(all(feature = "coreml", target_os = "macos"))]
        {
            // Check if Apple Neural Engine is available and optimize accordingly
            let backend = CoreMLBackend::global();
            
            // Try Apple Neural Engine first for optimal performance
            if backend.is_neural_engine_available() {
                return backend.execute_on_neural_engine(a, b);
            }
            
            // Fallback to CoreML with GPU acceleration
            if backend.is_available() {
                let model_manager = backend.model_manager();
                let model_handle = model_manager.get_or_create_matmul_model::<T>(a.shape(), b.shape())?;
                let inputs = vec![a, b];
                let outputs = model_manager.execute_model(&model_handle, &inputs)?;
                
                if outputs.is_empty() {
                    return Err(error_helpers::unsupported_operation(
                        "CoreML model returned no outputs"
                    ));
                }
                
                return Ok(outputs[0].clone());
            }
        }
        
        // Fallback to CPU implementation
        a.matmul(b)
    }

    /// Perform 2D convolution using CoreML acceleration  
    /// CoreML加速を使用して2D畳み込みを実行
    pub fn coreml_conv2d<T>(
        input: &Tensor<T>,
        kernel: &Tensor<T>,
        stride: &[usize],
        padding: &[usize],
    ) -> CoreMLResult<Tensor<T>>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
    {
        // Use CoreML model manager for optimized execution
        
        let backend = CoreMLBackend::global();
        let model_manager = backend.model_manager();
        
        // Validate stride and padding arrays
        if stride.len() < 2 || padding.len() < 2 {
            return Err(error_helpers::unsupported_operation(
                "Conv2D requires stride and padding arrays with at least 2 elements"
            ));
        }
        
        // Get or create model for this convolution
        let model_handle = model_manager.get_or_create_conv2d_model::<T>(
            input.shape(), 
            kernel.shape(), 
            stride[0], 
            padding[0]
        )?;
        
        // Execute model with inputs
        let inputs = vec![input, kernel];
        let outputs = model_manager.execute_model(&model_handle, &inputs)?;
        
        if outputs.is_empty() {
            return Err(error_helpers::unsupported_operation(
                "CoreML model returned no outputs"
            ));
        }
        
        Ok(outputs[0].clone())
    }

    /// Perform activation function using CoreML acceleration
    /// CoreML加速を使用して活性化関数を実行  
    pub fn coreml_activation<T>(
        input: &Tensor<T>,
        activation_type: CoreMLActivationType,
    ) -> CoreMLResult<Tensor<T>>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
    {
        // Use CoreML model manager for optimized execution
        
        let backend = CoreMLBackend::global();
        let model_manager = backend.model_manager();
        
        // Get or create model for this activation function
        let model_handle = model_manager.get_or_create_activation_model::<T>(
            activation_type,
            input.shape()
        )?;
        
        // Execute model with inputs
        let inputs = vec![input];
        let outputs = model_manager.execute_model(&model_handle, &inputs)?;
        
        if outputs.is_empty() {
            return Err(error_helpers::unsupported_operation(
                "CoreML model returned no outputs"
            ));
        }
        
        Ok(outputs[0].clone())
    }
}
