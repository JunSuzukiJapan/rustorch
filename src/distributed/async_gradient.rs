//! Asynchronous gradient synchronization system for distributed training
//! 分散学習用非同期勾配同期システム
//!
//! This module implements advanced asynchronous gradient synchronization mechanisms
//! for efficient distributed training with overlap of computation and communication.
//!
//! このモジュールは、計算と通信のオーバーラップによる効率的な分散学習のための
//! 高度な非同期勾配同期メカニズムを実装します。
//!
//! ## Module Structure / モジュール構造
//! - `AsyncGradientSynchronizer`: Main coordination layer / メイン調整層
//! - `GradientBucketManager`: Bucketing and batching / バケット化とバッチ処理
//! - `compression`: Gradient compression utilities / 勾配圧縮ユーティリティ

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use crate::autograd::Variable;
use std::sync::{Arc, Mutex, mpsc};
use std::collections::{HashMap, VecDeque};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use super::{ReduceOp, api, DistributedScalar};

/// Asynchronous gradient synchronization coordinator
/// 非同期勾配同期コーディネーター
pub struct AsyncGradientSynchronizer<T: DistributedScalar> {
    /// Background worker thread
    /// バックグラウンドワーカースレッド
    worker_handle: Option<JoinHandle<()>>,
    /// Channel for sending gradient sync requests
    /// 勾配同期リクエスト送信チャンネル
    request_sender: mpsc::Sender<GradientSyncRequest<T>>,
    /// Channel for receiving completion notifications
    /// 完了通知受信チャンネル
    completion_receiver: mpsc::Receiver<GradientSyncCompletion>,
    /// Configuration for async operations
    /// 非同期操作設定
    config: AsyncConfig,
    /// Gradient bucket manager
    /// 勾配バケットマネージャー
    bucket_manager: Arc<Mutex<GradientBucketManager<T>>>,
}

/// Configuration for asynchronous gradient synchronization
/// 非同期勾配同期設定
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// Maximum number of concurrent operations
    /// 最大同時操作数
    pub max_concurrent_ops: usize,
    /// Timeout for gradient synchronization
    /// 勾配同期のタイムアウト
    pub sync_timeout: Duration,
    /// Enable gradient compression
    /// 勾配圧縮を有効化
    pub enable_compression: bool,
    /// Compression threshold (bytes)
    /// 圧縮閾値（バイト）
    pub compression_threshold: usize,
    /// Enable gradient bucketing
    /// 勾配バケット化を有効化
    pub enable_bucketing: bool,
    /// Bucket size (MB)
    /// バケットサイズ（MB）
    pub bucket_size_mb: usize,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            max_concurrent_ops: 4,
            sync_timeout: Duration::from_secs(30),
            enable_compression: false,
            compression_threshold: 1024 * 1024, // 1MB
            enable_bucketing: true,
            bucket_size_mb: 25,
        }
    }
}

/// Gradient synchronization request
/// 勾配同期リクエスト
#[derive(Debug)]
pub struct GradientSyncRequest<T: DistributedScalar> {
    /// Request ID for tracking
    /// 追跡用リクエストID
    pub id: u64,
    /// Parameter name
    /// パラメータ名
    pub param_name: String,
    /// Gradient tensor to synchronize
    /// 同期する勾配テンソル
    pub gradient: Tensor<T>,
    /// Reduction operation
    /// リダクション操作
    pub reduce_op: ReduceOp,
    /// Priority level
    /// 優先度レベル
    pub priority: Priority,
}

/// Priority levels for gradient synchronization
/// 勾配同期の優先度レベル
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Gradient synchronization completion notification
/// 勾配同期完了通知
#[derive(Debug)]
pub struct GradientSyncCompletion {
    /// Request ID that completed
    /// 完了したリクエストID
    pub request_id: u64,
    /// Success status
    /// 成功ステータス
    pub success: bool,
    /// Error message if failed
    /// 失敗時のエラーメッセージ
    pub error: Option<String>,
    /// Synchronization duration
    /// 同期時間
    pub duration: Duration,
}

/// Gradient bucket manager for efficient communication
/// 効率的な通信のための勾配バケットマネージャー
pub struct GradientBucketManager<T: DistributedScalar> {
    /// Active buckets
    /// アクティブバケット
    buckets: HashMap<usize, GradientBucket<T>>,
    /// Bucket assignment for parameters
    /// パラメータのバケット割り当て
    param_to_bucket: HashMap<String, usize>,
    /// Next bucket ID
    /// 次のバケットID
    next_bucket_id: usize,
    /// Configuration
    /// 設定
    config: AsyncConfig,
}

/// Individual gradient bucket
/// 個別勾配バケット
#[derive(Debug, Clone)]
pub struct GradientBucket<T: DistributedScalar> {
    /// Bucket ID
    /// バケットID
    pub id: usize,
    /// Parameters in this bucket
    /// このバケット内のパラメータ
    pub parameters: Vec<String>,
    /// Combined gradient tensors
    /// 結合勾配テンソル
    pub gradients: Vec<Tensor<T>>,
    /// Total size in bytes
    /// 総サイズ（バイト）
    pub total_size: usize,
    /// Ready for synchronization
    /// 同期準備完了
    pub ready: bool,
    /// Last update timestamp
    /// 最終更新タイムスタンプ
    pub last_update: Instant,
}

impl<T: DistributedScalar> AsyncGradientSynchronizer<T> {
    /// Create a new asynchronous gradient synchronizer
    /// 新しい非同期勾配同期器を作成
    pub fn new(config: AsyncConfig) -> RusTorchResult<Self> {
        let (request_sender, request_receiver) = mpsc::channel();
        let (completion_sender, completion_receiver) = mpsc::channel();

        let bucket_manager = Arc::new(Mutex::new(GradientBucketManager::new(config.clone())));
        let bucket_manager_worker = Arc::clone(&bucket_manager);

        // Spawn background worker thread
        let worker_handle = thread::spawn(move || {
            Self::worker_loop(request_receiver, completion_sender, bucket_manager_worker);
        });

        Ok(Self {
            worker_handle: Some(worker_handle),
            request_sender,
            completion_receiver,
            config,
            bucket_manager,
        })
    }

    /// Submit gradient for asynchronous synchronization
    /// 非同期同期用勾配を投入
    pub fn submit_gradient(
        &self,
        param_name: String,
        gradient: Tensor<T>,
        priority: Priority,
    ) -> RusTorchResult<u64> {
        static mut REQUEST_ID: u64 = 0;
        
        let request_id = unsafe {
            REQUEST_ID += 1;
            REQUEST_ID
        };

        let request = GradientSyncRequest {
            id: request_id,
            param_name,
            gradient,
            reduce_op: ReduceOp::Average,
            priority,
        };

        self.request_sender
            .send(request)
            .map_err(|_| RusTorchError::distributed("Failed to submit gradient sync request"))?;

        Ok(request_id)
    }

    /// Check for completed synchronizations
    /// 完了した同期をチェック
    pub fn check_completions(&self) -> Vec<GradientSyncCompletion> {
        let mut completions = Vec::new();
        
        while let Ok(completion) = self.completion_receiver.try_recv() {
            completions.push(completion);
        }

        completions
    }

    /// Wait for specific synchronization to complete
    /// 特定の同期完了を待機
    pub fn wait_for_completion(&self, request_id: u64, timeout: Duration) -> RusTorchResult<()> {
        let start = Instant::now();
        
        while start.elapsed() < timeout {
            if let Ok(completion) = self.completion_receiver.recv_timeout(Duration::from_millis(100)) {
                if completion.request_id == request_id {
                    if completion.success {
                        return Ok(());
                    } else {
                        return Err(RusTorchError::distributed(&format!(
                            "Gradient sync failed: {}",
                            completion.error.unwrap_or_else(|| "Unknown error".to_string())
                        )));
                    }
                }
            }
        }

        Err(RusTorchError::distributed("Gradient sync timeout"))
    }

    /// Synchronize all pending gradients
    /// 保留中の全勾配を同期
    pub fn sync_all(&self, timeout: Duration) -> RusTorchResult<()> {
        let ready_buckets = {
            let bucket_manager = self.bucket_manager.lock().unwrap();
            bucket_manager.get_ready_buckets()
        };

        let start = Instant::now();
        let mut pending_ids = Vec::new();

        // Submit all ready buckets for synchronization
        for bucket in ready_buckets {
            for (i, grad) in bucket.gradients.iter().enumerate() {
                if i < bucket.parameters.len() {
                    let id = self.submit_gradient(
                        bucket.parameters[i].clone(),
                        grad.clone(),
                        Priority::High,
                    )?;
                    pending_ids.push(id);
                }
            }
        }

        // Wait for all to complete
        while !pending_ids.is_empty() && start.elapsed() < timeout {
            let completions = self.check_completions();
            for completion in completions {
                if let Some(pos) = pending_ids.iter().position(|&id| id == completion.request_id) {
                    pending_ids.remove(pos);
                    if !completion.success {
                        return Err(RusTorchError::distributed(&format!(
                            "Gradient sync failed: {}",
                            completion.error.unwrap_or_else(|| "Unknown error".to_string())
                        )));
                    }
                }
            }
            thread::sleep(Duration::from_millis(10));
        }

        if !pending_ids.is_empty() {
            return Err(RusTorchError::distributed("Some gradient syncs did not complete"));
        }

        Ok(())
    }

    /// Background worker loop
    /// バックグラウンドワーカーループ
    fn worker_loop(
        receiver: mpsc::Receiver<GradientSyncRequest<T>>,
        completion_sender: mpsc::Sender<GradientSyncCompletion>,
        bucket_manager: Arc<Mutex<GradientBucketManager<T>>>,
    ) {
        let mut pending_requests: VecDeque<GradientSyncRequest<T>> = VecDeque::new();

        loop {
            // Receive new requests
            while let Ok(request) = receiver.try_recv() {
                pending_requests.push_back(request);
            }

            // Sort by priority
            pending_requests.make_contiguous().sort_by_key(|req| req.priority);

            // Process requests
            while let Some(request) = pending_requests.pop_front() {
                let start_time = Instant::now();
                
                let result = Self::process_gradient_sync(&request, &bucket_manager);
                let duration = start_time.elapsed();

                let completion = GradientSyncCompletion {
                    request_id: request.id,
                    success: result.is_ok(),
                    error: result.err().map(|e| e.to_string()),
                    duration,
                };

                if completion_sender.send(completion).is_err() {
                    // Receiver dropped, exit worker
                    break;
                }
            }

            // Small delay to prevent busy waiting
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Process individual gradient synchronization
    /// 個別勾配同期を処理
    fn process_gradient_sync(
        request: &GradientSyncRequest<T>,
        _bucket_manager: &Arc<Mutex<GradientBucketManager<T>>>,
    ) -> RusTorchResult<()> {
        // Perform the actual gradient synchronization
        let mut grad_copy = request.gradient.clone();
        api::all_reduce(&mut grad_copy, request.reduce_op, None, false)?;

        // Update the original parameter (this would need proper parameter tracking)
        // 元のパラメータを更新（適切なパラメータ追跡が必要）
        
        Ok(())
    }
}

impl<T: DistributedScalar> GradientBucketManager<T> {
    /// Create new gradient bucket manager
    /// 新しい勾配バケットマネージャーを作成
    pub fn new(config: AsyncConfig) -> Self {
        Self {
            buckets: HashMap::new(),
            param_to_bucket: HashMap::new(),
            next_bucket_id: 0,
            config,
        }
    }

    /// Add gradient to appropriate bucket
    /// 適切なバケットに勾配を追加
    pub fn add_gradient(&mut self, param_name: String, gradient: Tensor<T>) -> RusTorchResult<()> {
        let bucket_id = if let Some(&existing_id) = self.param_to_bucket.get(&param_name) {
            existing_id
        } else {
            // Find or create appropriate bucket
            self.find_or_create_bucket(&param_name, &gradient)?
        };

        let gradient_size = self.estimate_tensor_size(&gradient);
        if let Some(bucket) = self.buckets.get_mut(&bucket_id) {
            bucket.parameters.push(param_name.clone());
            bucket.gradients.push(gradient);
            bucket.total_size += gradient_size;
            bucket.last_update = Instant::now();
            
            // Mark as ready if bucket is full enough
            if bucket.total_size >= self.config.bucket_size_mb * 1024 * 1024 {
                bucket.ready = true;
            }
        }

        self.param_to_bucket.insert(param_name, bucket_id);
        Ok(())
    }

    /// Find or create appropriate bucket for gradient
    /// 勾配用の適切なバケットを検索または作成
    fn find_or_create_bucket(&mut self, param_name: &str, gradient: &Tensor<T>) -> RusTorchResult<usize> {
        let gradient_size = self.estimate_tensor_size(gradient);
        let bucket_size_limit = self.config.bucket_size_mb * 1024 * 1024;

        // Try to find existing bucket with space
        for (id, bucket) in &self.buckets {
            if !bucket.ready && bucket.total_size + gradient_size <= bucket_size_limit {
                return Ok(*id);
            }
        }

        // Create new bucket
        let bucket_id = self.next_bucket_id;
        self.next_bucket_id += 1;

        let bucket = GradientBucket {
            id: bucket_id,
            parameters: Vec::new(),
            gradients: Vec::new(),
            total_size: 0,
            ready: false,
            last_update: Instant::now(),
        };

        self.buckets.insert(bucket_id, bucket);
        Ok(bucket_id)
    }

    /// Get buckets ready for synchronization
    /// 同期準備完了のバケットを取得
    pub fn get_ready_buckets(&self) -> Vec<GradientBucket<T>> {
        self.buckets
            .values()
            .filter(|bucket| bucket.ready)
            .cloned()
            .collect()
    }

    /// Mark bucket as synchronized and clear
    /// バケットを同期済みとしてマークし、クリア
    pub fn mark_bucket_synced(&mut self, bucket_id: usize) -> RusTorchResult<()> {
        if let Some(bucket) = self.buckets.get_mut(&bucket_id) {
            bucket.ready = false;
            bucket.gradients.clear();
            bucket.parameters.clear();
            bucket.total_size = 0;
        }
        Ok(())
    }

    /// Estimate tensor size in bytes
    /// テンソルサイズをバイト単位で推定
    fn estimate_tensor_size(&self, tensor: &Tensor<T>) -> usize {
        tensor.numel() * std::mem::size_of::<T>()
    }
}

impl<T: DistributedScalar> Drop for AsyncGradientSynchronizer<T> {
    fn drop(&mut self) {
        // Signal worker thread to stop and wait for completion
        // ワーカースレッドに停止を信号し、完了を待機
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Gradient compression utilities
/// 勾配圧縮ユーティリティ
pub mod compression {
    use super::{*, DistributedScalar};

    /// Gradient compression algorithms
    /// 勾配圧縮アルゴリズム
    #[derive(Debug, Clone, Copy)]
    pub enum CompressionAlgorithm {
        /// No compression
        /// 圧縮なし
        None,
        /// Top-K sparsification
        /// Top-Kスパース化
        TopK { k: usize },
        /// Quantization-based compression
        /// 量子化ベース圧縮
        Quantization { bits: u8 },
        /// Error feedback compression
        /// エラーフィードバック圧縮
        ErrorFeedback,
    }

    /// Compress gradient tensor
    /// 勾配テンソルを圧縮
    pub fn compress_gradient<T: DistributedScalar>(
        gradient: &Tensor<T>,
        algorithm: CompressionAlgorithm,
    ) -> RusTorchResult<CompressedGradient<T>> {
        match algorithm {
            CompressionAlgorithm::None => {
                Ok(CompressedGradient {
                    data: gradient.clone(),
                    metadata: CompressionMetadata::None,
                    original_shape: gradient.shape().to_vec(),
                })
            }
            CompressionAlgorithm::TopK { k } => {
                compress_top_k(gradient, k)
            }
            _ => Err(RusTorchError::distributed("Compression algorithm not implemented")),
        }
    }

    /// Decompress gradient tensor
    /// 勾配テンソルを展開
    pub fn decompress_gradient<T: DistributedScalar>(
        compressed: &CompressedGradient<T>,
    ) -> RusTorchResult<Tensor<T>> {
        match &compressed.metadata {
            CompressionMetadata::None => Ok(compressed.data.clone()),
            CompressionMetadata::TopK { .. } => {
                decompress_top_k(compressed)
            }
            _ => Err(RusTorchError::distributed("Decompression not implemented")),
        }
    }

    /// Compressed gradient representation
    /// 圧縮勾配表現
    #[derive(Debug, Clone)]
    pub struct CompressedGradient<T: DistributedScalar> {
        /// Compressed data
        /// 圧縮データ
        pub data: Tensor<T>,
        /// Compression metadata
        /// 圧縮メタデータ
        pub metadata: CompressionMetadata,
        /// Original tensor shape
        /// 元のテンソル形状
        pub original_shape: Vec<usize>,
    }

    /// Compression metadata
    /// 圧縮メタデータ
    #[derive(Debug, Clone)]
    pub enum CompressionMetadata {
        None,
        TopK { k: usize, indices: Vec<usize> },
        Quantization { scale: f32, zero_point: i8 },
    }

    /// Top-K compression implementation
    /// Top-K圧縮実装
    fn compress_top_k<T: DistributedScalar>(gradient: &Tensor<T>, k: usize) -> RusTorchResult<CompressedGradient<T>> {
        let total_elements = gradient.numel();
        if k > total_elements {
            return Err(RusTorchError::tensor_op("K larger than tensor size"));
        }

        // Simplified implementation - in production would find actual top-k elements
        // 簡略化実装 - プロダクションでは実際のtop-k要素を検索
        let indices = (0..k).collect();
        let compressed_data = gradient.clone(); // Placeholder

        Ok(CompressedGradient {
            data: compressed_data,
            metadata: CompressionMetadata::TopK { k, indices },
            original_shape: gradient.shape().to_vec(),
        })
    }

    /// Top-K decompression implementation
    /// Top-K展開実装
    fn decompress_top_k<T: DistributedScalar>(
        compressed: &CompressedGradient<T>,
    ) -> RusTorchResult<Tensor<T>> {
        // Simplified implementation - in production would reconstruct sparse tensor
        // 簡略化実装 - プロダクションではスパーステンソルを再構築
        Ok(compressed.data.clone())
    }
}

/// Asynchronous gradient synchronization context
/// 非同期勾配同期コンテキスト
pub struct AsyncGradContext<T: DistributedScalar> {
    /// Synchronizer instance
    /// 同期器インスタンス
    synchronizer: Arc<Mutex<AsyncGradientSynchronizer<T>>>,
    /// Pending operations
    /// 保留中の操作
    pending_ops: Arc<Mutex<HashMap<u64, String>>>,
}

impl<T: DistributedScalar> AsyncGradContext<T> {
    /// Create new async gradient context
    /// 新しい非同期勾配コンテキストを作成
    pub fn new(config: AsyncConfig) -> RusTorchResult<Self> {
        let synchronizer = AsyncGradientSynchronizer::new(config)?;
        
        Ok(Self {
            synchronizer: Arc::new(Mutex::new(synchronizer)),
            pending_ops: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Submit parameter gradient for async sync
    /// パラメータ勾配を非同期同期に投入
    pub fn sync_parameter_async(&self, parameter: &Variable<T>) -> RusTorchResult<()> {
        let grad_lock = parameter.grad();
        let grad_guard = grad_lock.read().unwrap();
        if let Some(ref gradient) = *grad_guard {
            let param_name = format!("param_{}", parameter.id());
            let gradient_clone = gradient.clone();
            drop(grad_guard);
            
            let synchronizer = self.synchronizer.lock().unwrap();
            let request_id = synchronizer.submit_gradient(
                param_name.clone(),
                gradient_clone,
                Priority::Normal,
            )?;
            
            let mut pending = self.pending_ops.lock().unwrap();
            pending.insert(request_id, param_name);
        }

        Ok(())
    }

    /// Wait for all pending operations to complete
    /// 保留中の全操作の完了を待機
    pub fn synchronize(&self) -> RusTorchResult<()> {
        let timeout = Duration::from_secs(30);
        let synchronizer = self.synchronizer.lock().unwrap();
        synchronizer.sync_all(timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_async_config_default() {
        let config = AsyncConfig::default();
        assert_eq!(config.max_concurrent_ops, 4);
        assert_eq!(config.bucket_size_mb, 25);
        assert!(config.enable_bucketing);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_bucket_manager_creation() {
        let config = AsyncConfig::default();
        let manager = GradientBucketManager::<f32>::new(config);
        assert_eq!(manager.buckets.len(), 0);
        assert_eq!(manager.next_bucket_id, 0);
    }

    #[test]
    fn test_compression_none() {
        let tensor: Tensor<f32> = Tensor::ones(&[2, 2]);
        let compressed = compression::compress_gradient(
            &tensor,
            compression::CompressionAlgorithm::None,
        ).unwrap();
        
        let decompressed = compression::decompress_gradient(&compressed).unwrap();
        assert_eq!(tensor.shape(), decompressed.shape());
    }
}