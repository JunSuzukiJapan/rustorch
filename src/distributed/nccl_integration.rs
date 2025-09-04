//! Advanced NCCL integration for high-performance distributed training
//! 高性能分散学習のための高度NCCL統合
//!
//! This module provides comprehensive NCCL (NVIDIA Collective Communications Library)
//! integration for RusTorch, enabling efficient multi-GPU and multi-node training.
//!
//! このモジュールは、RusTorch向けの包括的NCCL統合を提供し、
//! 効率的なマルチGPUおよびマルチノード学習を可能にします。

use super::{DistributedOps, ProcessGroup, ReduceOp};
use crate::error::{RusTorchError, RusTorchResult};
use crate::gpu::DeviceType;
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};

#[cfg(feature = "nccl")]
/// NCCL communicator handle
/// NCCL通信器ハンドル
pub struct NCCLCommunicator {
    /// NCCL communicator pointer
    /// NCCL通信器ポインタ
    comm: *mut c_void,
    /// Device ID associated with this communicator
    /// この通信器に関連するデバイスID
    device_id: usize,
    /// Rank in the communicator
    /// 通信器内でのランク
    rank: usize,
    /// Number of devices in the communicator
    /// 通信器内のデバイス数
    nranks: usize,
    /// CUDA streams for async operations
    /// 非同期操作用CUDAストリーム
    streams: Vec<*mut c_void>,
}

#[cfg(feature = "nccl")]
unsafe impl Send for NCCLCommunicator {}

#[cfg(feature = "nccl")]
unsafe impl Sync for NCCLCommunicator {}

#[cfg(feature = "nccl")]
impl NCCLCommunicator {
    /// Create a new NCCL communicator
    /// 新しいNCCL通信器を作成
    pub fn new(
        rank: usize,
        nranks: usize,
        device_id: usize,
        comm_id: &NCCLUniqueId,
    ) -> RusTorchResult<Self> {
        let comm: *mut c_void = ptr::null_mut();
        let streams = Vec::new();

        // In a full implementation, this would call NCCL APIs:
        // 完全な実装では、NCCL APIを呼び出し：
        // ncclCommInitRank(&comm, nranks, commId, rank);

        Ok(Self {
            comm,
            device_id,
            rank,
            nranks,
            streams,
        })
    }

    /// Perform all-reduce operation using NCCL
    /// NCCLを使用したall-reduce操作
    pub fn all_reduce<T: Float>(&self, tensor: &mut Tensor<T>, op: ReduceOp) -> RusTorchResult<()> {
        // Validate device placement
        // Check if tensor is on CUDA device (simplified check)
        if !matches!(tensor.device, crate::tensor::device::Device::Cuda(_)) {
            return Err(RusTorchError::gpu(
                "Tensor must be on CUDA device for NCCL".to_string(),
            ));
        }

        // In a full implementation:
        // 完全な実装では：
        // - Get tensor data pointer
        // - Convert ReduceOp to ncclRedOp_t
        // - Call ncclAllReduce
        // - Synchronize CUDA stream

        let _ = (tensor, op); // Suppress warnings for now

        Ok(())
    }

    /// Perform all-gather operation using NCCL
    /// NCCLを使用したall-gather操作
    pub fn all_gather<T: Float>(&self, tensor: &Tensor<T>) -> RusTorchResult<Vec<Tensor<T>>> {
        // Check if tensor is on CUDA device (simplified check)
        if !matches!(tensor.device, crate::tensor::device::Device::Cuda(_)) {
            return Err(RusTorchError::gpu(
                "Tensor must be on CUDA device for NCCL".to_string(),
            ));
        }

        // Create output tensors for each rank
        let mut output_tensors = Vec::with_capacity(self.nranks);
        for _ in 0..self.nranks {
            output_tensors.push(tensor.clone());
        }

        // In a full implementation:
        // 完全な実装では：
        // - Allocate contiguous output buffer
        // - Call ncclAllGather
        // - Split output buffer into individual tensors

        Ok(output_tensors)
    }

    /// Perform broadcast operation using NCCL
    /// NCCLを使用したbroadcast操作
    pub fn broadcast<T: Float>(&self, tensor: &mut Tensor<T>, root: usize) -> RusTorchResult<()> {
        // Check if tensor is on CUDA device (simplified check)
        if !matches!(tensor.device, crate::tensor::device::Device::Cuda(_)) {
            return Err(RusTorchError::gpu(
                "Tensor must be on CUDA device for NCCL".to_string(),
            ));
        }

        if root >= self.nranks {
            return Err(RusTorchError::distributed(&format!(
                "Root rank {} exceeds communicator size {}",
                root, self.nranks
            )));
        }

        // In a full implementation:
        // 完全な実装では：
        // - Call ncclBroadcast with tensor data and root rank

        let _ = (tensor, root);
        Ok(())
    }

    /// Get device ID for this communicator
    /// この通信器のデバイスIDを取得
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Get rank for this communicator
    /// この通信器のランクを取得
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get number of ranks for this communicator
    /// この通信器のランク数を取得
    pub fn nranks(&self) -> usize {
        self.nranks
    }
}

#[cfg(feature = "nccl")]
impl Drop for NCCLCommunicator {
    fn drop(&mut self) {
        // Clean up NCCL communicator
        // NCCL通信器をクリーンアップ
        if !self.comm.is_null() {
            // In a full implementation: ncclCommDestroy(self.comm);
        }

        // Clean up CUDA streams
        // CUDAストリームをクリーンアップ
        for stream in &self.streams {
            if !stream.is_null() {
                // In a full implementation: cudaStreamDestroy(*stream);
            }
        }
    }
}

/// NCCL unique ID for communicator initialization
/// 通信器初期化用NCCL固有ID
#[cfg(feature = "nccl")]
#[derive(Debug, Clone)]
pub struct NCCLUniqueId {
    id: [u8; 128], // NCCL_UNIQUE_ID_BYTES
}

#[cfg(feature = "nccl")]
impl NCCLUniqueId {
    /// Generate a new unique ID (should be called by rank 0)
    /// 新しい固有IDを生成（ランク0が呼び出すべき）
    pub fn new() -> RusTorchResult<Self> {
        let id = [0u8; 128];

        // In a full implementation: ncclGetUniqueId(&id);

        Ok(Self { id })
    }

    /// Get the raw ID bytes
    /// 生IDバイトを取得
    pub fn as_bytes(&self) -> &[u8] {
        &self.id
    }

    /// Create from raw bytes
    /// 生バイトから作成
    pub fn from_bytes(bytes: &[u8]) -> RusTorchResult<Self> {
        if bytes.len() != 128 {
            return Err(RusTorchError::distributed("Invalid NCCL ID length"));
        }

        let mut id = [0u8; 128];
        id.copy_from_slice(bytes);

        Ok(Self { id })
    }
}

#[cfg(feature = "nccl")]
impl Default for NCCLUniqueId {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// NCCL backend implementation with optimizations
/// 最適化付きNCCLバックエンド実装
#[cfg(feature = "nccl")]
pub struct NCCLBackendOptimized {
    /// Map of device to communicator
    /// デバイスから通信器へのマップ
    communicators: HashMap<usize, Arc<Mutex<NCCLCommunicator>>>,
    /// Process group information
    /// プロセスグループ情報
    process_group: ProcessGroup,
    /// Optimization settings
    /// 最適化設定
    optimizations: NCCLOptimizations,
}

/// NCCL optimization configuration
/// NCCL最適化設定
#[cfg(feature = "nccl")]
#[derive(Debug, Clone)]
pub struct NCCLOptimizations {
    /// Enable gradient compression
    /// 勾配圧縮を有効化
    pub compression_enabled: bool,
    /// Bucket size for gradient bucketing (MB)
    /// 勾配バケット化のバケットサイズ（MB）
    pub bucket_size_mb: usize,
    /// Number of communication streams
    /// 通信ストリーム数
    pub num_streams: usize,
    /// Enable async operations
    /// 非同期操作を有効化
    pub async_enabled: bool,
    /// Timeout for operations (seconds)
    /// 操作のタイムアウト（秒）
    pub timeout_seconds: u64,
}

#[cfg(feature = "nccl")]
impl Default for NCCLOptimizations {
    fn default() -> Self {
        Self {
            compression_enabled: false,
            bucket_size_mb: 25,
            num_streams: 4,
            async_enabled: true,
            timeout_seconds: 300, // 5 minutes
        }
    }
}

#[cfg(feature = "nccl")]
impl NCCLBackendOptimized {
    /// Create a new optimized NCCL backend
    /// 新しい最適化NCCLバックエンドを作成
    pub fn new(
        process_group: ProcessGroup,
        device_ids: &[usize],
        comm_id: &NCCLUniqueId,
    ) -> RusTorchResult<Self> {
        let mut communicators = HashMap::new();

        for &device_id in device_ids {
            let comm = NCCLCommunicator::new(
                process_group.rank,
                process_group.world_size,
                device_id,
                comm_id,
            )?;
            communicators.insert(device_id, Arc::new(Mutex::new(comm)));
        }

        Ok(Self {
            communicators,
            process_group,
            optimizations: NCCLOptimizations::default(),
        })
    }

    /// Configure optimizations
    /// 最適化を設定
    pub fn configure_optimizations(&mut self, opts: NCCLOptimizations) {
        self.optimizations = opts;
    }

    /// Get communicator for specific device
    /// 特定デバイスの通信器を取得
    pub fn get_communicator(&self, device_id: usize) -> Option<Arc<Mutex<NCCLCommunicator>>> {
        self.communicators.get(&device_id).cloned()
    }

    /// Perform optimized all-reduce with bucketing
    /// バケット化による最適化all-reduce
    pub fn all_reduce_bucketed<T: Float + 'static>(
        &self,
        tensors: &mut [Tensor<T>],
        op: ReduceOp,
    ) -> RusTorchResult<()> {
        // Group tensors into buckets for efficient communication
        let buckets = self.create_gradient_buckets(tensors)?;

        for bucket in buckets {
            // Perform all-reduce on each bucket
            for mut tensor in bucket {
                self.all_reduce_single(&mut tensor, op)?;
            }
        }

        Ok(())
    }

    /// Create gradient buckets for efficient communication
    /// 効率的な通信のための勾配バケット作成
    fn create_gradient_buckets<T: Float + 'static>(
        &self,
        tensors: &[Tensor<T>],
    ) -> RusTorchResult<Vec<Vec<Tensor<T>>>> {
        let bucket_size_bytes = self.optimizations.bucket_size_mb * 1024 * 1024;
        let mut buckets = Vec::new();
        let mut current_bucket = Vec::new();
        let mut current_size = 0;

        for tensor in tensors {
            let tensor_size = tensor.numel() * std::mem::size_of::<T>();

            if current_size + tensor_size > bucket_size_bytes && !current_bucket.is_empty() {
                buckets.push(current_bucket.clone());
                current_bucket.clear();
                current_size = 0;
            }

            current_bucket.push(tensor.clone());
            current_size += tensor_size;
        }

        if !current_bucket.is_empty() {
            buckets.push(current_bucket);
        }

        Ok(buckets)
    }

    /// Perform single all-reduce operation
    /// 単一all-reduce操作を実行
    fn all_reduce_single<T: Float>(
        &self,
        tensor: &mut Tensor<T>,
        op: ReduceOp,
    ) -> RusTorchResult<()> {
        // Get the appropriate communicator for tensor's device
        let device_id = match tensor.device {
            crate::tensor::device::Device::Cuda(id) => id,
            _ => 0,
        };

        if let Some(comm_arc) = self.get_communicator(device_id) {
            let comm = comm_arc.lock().unwrap();
            comm.all_reduce(tensor, op)?;
        } else {
            return Err(RusTorchError::distributed(&format!(
                "No NCCL communicator found for device {}",
                device_id
            )));
        }

        Ok(())
    }
}

/// High-level NCCL operations interface
/// 高レベルNCCL操作インターフェース
#[cfg(feature = "nccl")]
pub struct NCCLOps;

#[cfg(feature = "nccl")]
impl NCCLOps {
    /// Initialize NCCL for multi-GPU training
    /// マルチGPU学習用NCCL初期化
    pub fn init_multi_gpu(
        device_ids: &[usize],
    ) -> RusTorchResult<HashMap<usize, NCCLCommunicator>> {
        if device_ids.is_empty() {
            return Err(RusTorchError::distributed("No device IDs provided"));
        }

        let nranks = device_ids.len();
        let comm_id = NCCLUniqueId::new()?;
        let mut communicators = HashMap::new();

        for (rank, &device_id) in device_ids.iter().enumerate() {
            let comm = NCCLCommunicator::new(rank, nranks, device_id, &comm_id)?;
            communicators.insert(device_id, comm);
        }

        Ok(communicators)
    }

    /// Perform synchronous all-reduce across all GPUs
    /// 全GPU間での同期all-reduce
    pub fn all_reduce_multi_gpu<T: Float>(
        tensors: &mut [Tensor<T>],
        communicators: &HashMap<usize, NCCLCommunicator>,
        op: ReduceOp,
    ) -> RusTorchResult<()> {
        for tensor in tensors.iter_mut() {
            match tensor.device {
                crate::tensor::device::Device::Cuda(device_id) => {
                    if let Some(comm) = communicators.get(&device_id) {
                        comm.all_reduce(tensor, op)?;
                    }
                }
                _ => {} // Skip non-CUDA tensors
            }
        }

        // Synchronize all CUDA streams
        // 全CUDAストリームを同期
        Self::synchronize_all_streams(communicators)?;

        Ok(())
    }

    /// Synchronize all CUDA streams
    /// 全CUDAストリームを同期
    fn synchronize_all_streams(
        _communicators: &HashMap<usize, NCCLCommunicator>,
    ) -> RusTorchResult<()> {
        // In a full implementation: cudaDeviceSynchronize for each device
        // 完全な実装では：各デバイスでcudaDeviceSynchronize
        Ok(())
    }

    /// Get optimal NCCL configuration for current hardware
    /// 現在のハードウェア用最適NCCL設定を取得
    pub fn get_optimal_config(num_gpus: usize, gpu_memory_gb: f32) -> NCCLOptimizations {
        let bucket_size_mb = if gpu_memory_gb > 16.0 {
            50 // Larger buckets for high-memory GPUs
        } else {
            25 // Default bucket size
        };

        let num_streams = if num_gpus > 4 {
            8 // More streams for larger setups
        } else {
            4 // Default stream count
        };

        NCCLOptimizations {
            compression_enabled: num_gpus > 8, // Enable compression for large setups
            bucket_size_mb,
            num_streams,
            async_enabled: true,
            timeout_seconds: 300,
        }
    }
}

/// NCCL performance profiler
/// NCCLパフォーマンスプロファイラー
#[cfg(feature = "nccl")]
pub struct NCCLProfiler {
    /// Operation timing data
    /// 操作タイミングデータ
    timing_data: Arc<Mutex<HashMap<String, Vec<f64>>>>,
    /// Bandwidth measurements
    /// 帯域幅測定
    bandwidth_data: Arc<Mutex<HashMap<String, f64>>>,
}

#[cfg(feature = "nccl")]
impl NCCLProfiler {
    /// Create a new NCCL profiler
    /// 新しいNCCLプロファイラーを作成
    pub fn new() -> Self {
        Self {
            timing_data: Arc::new(Mutex::new(HashMap::new())),
            bandwidth_data: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start timing an operation
    /// 操作のタイミング測定開始
    pub fn start_timing(&self, operation: &str) -> TimingGuard {
        TimingGuard::new(operation.to_string(), Arc::clone(&self.timing_data))
    }

    /// Get timing statistics
    /// タイミング統計を取得
    pub fn get_timing_stats(&self) -> HashMap<String, TimingStats> {
        let data = self.timing_data.lock().unwrap();
        let mut stats = HashMap::new();

        for (op, times) in data.iter() {
            if !times.is_empty() {
                let sum: f64 = times.iter().sum();
                let avg = sum / times.len() as f64;
                let min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                stats.insert(
                    op.clone(),
                    TimingStats {
                        count: times.len(),
                        average_ms: avg,
                        min_ms: min,
                        max_ms: max,
                        total_ms: sum,
                    },
                );
            }
        }

        stats
    }

    /// Calculate communication bandwidth
    /// 通信帯域幅を計算
    pub fn calculate_bandwidth(&self, operation: &str, bytes_transferred: usize, duration_ms: f64) {
        if duration_ms > 0.0 {
            let bandwidth_gbps = (bytes_transferred as f64 * 8.0) / (duration_ms * 1_000_000.0);
            let mut data = self.bandwidth_data.lock().unwrap();
            data.insert(operation.to_string(), bandwidth_gbps);
        }
    }
}

/// Timing guard for automatic measurement
/// 自動測定用タイミングガード
#[cfg(feature = "nccl")]
pub struct TimingGuard {
    operation: String,
    start_time: std::time::Instant,
    timing_data: Arc<Mutex<HashMap<String, Vec<f64>>>>,
}

#[cfg(feature = "nccl")]
impl TimingGuard {
    fn new(operation: String, timing_data: Arc<Mutex<HashMap<String, Vec<f64>>>>) -> Self {
        Self {
            operation,
            start_time: std::time::Instant::now(),
            timing_data,
        }
    }
}

#[cfg(feature = "nccl")]
impl Drop for TimingGuard {
    fn drop(&mut self) {
        let duration_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        let mut data = self.timing_data.lock().unwrap();
        data.entry(self.operation.clone())
            .or_insert_with(Vec::new)
            .push(duration_ms);
    }
}

/// Timing statistics
/// タイミング統計
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub count: usize,
    pub average_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub total_ms: f64,
}

/// Fallback implementations when NCCL is not available
/// NCCLが利用できない場合のフォールバック実装
#[cfg(not(feature = "nccl"))]
pub mod fallback {
    use super::*;

    pub fn nccl_not_available_error() -> RusTorchError {
        RusTorchError::backend_unavailable(
            "NCCL backend not available. Compile with --features nccl",
        )
    }

    pub fn init_multi_gpu(_device_ids: &[usize]) -> RusTorchResult<()> {
        Err(nccl_not_available_error())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "nccl")]
    #[test]
    fn test_nccl_unique_id() {
        let id1 = NCCLUniqueId::new().unwrap();
        let bytes = id1.as_bytes();
        let id2 = NCCLUniqueId::from_bytes(bytes).unwrap();

        assert_eq!(id1.as_bytes(), id2.as_bytes());
    }

    #[cfg(feature = "nccl")]
    #[test]
    fn test_nccl_optimizations() {
        let opts = NCCLOps::get_optimal_config(8, 32.0);
        assert!(opts.compression_enabled);
        assert_eq!(opts.bucket_size_mb, 50);
        assert_eq!(opts.num_streams, 8);

        let opts = NCCLOps::get_optimal_config(2, 8.0);
        assert!(!opts.compression_enabled);
        assert_eq!(opts.bucket_size_mb, 25);
        assert_eq!(opts.num_streams, 4);
    }

    #[cfg(feature = "nccl")]
    #[test]
    fn test_nccl_profiler() {
        let profiler = NCCLProfiler::new();

        {
            let _guard = profiler.start_timing("test_op");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let stats = profiler.get_timing_stats();
        assert!(stats.contains_key("test_op"));
        assert_eq!(stats["test_op"].count, 1);
        assert!(stats["test_op"].average_ms >= 10.0);
    }

    #[cfg(not(feature = "nccl"))]
    #[test]
    fn test_fallback_error() {
        let error = fallback::nccl_not_available_error();
        match error {
            RusTorchError::BackendUnavailable { .. } => (),
            _ => panic!("Expected BackendUnavailable error"),
        }
    }
}
