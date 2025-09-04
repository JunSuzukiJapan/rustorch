//! DistributedDataParallel (DDP) implementation for RusTorch
//! RusTorch用DistributedDataParallel（DDP）実装
//!
//! This module provides PyTorch-compatible DistributedDataParallel functionality
//! for efficient distributed training across multiple devices.
//!
//! このモジュールは、複数デバイス間での効率的な分散学習のための
//! PyTorch互換DistributedDataParallel機能を提供します。

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use crate::nn::Module;
use crate::autograd::Variable;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use super::{ProcessGroup, ReduceOp, DistributedOps, api, DistributedScalar, DistributedDataParallelTrait};

/// DistributedDataParallel wrapper for PyTorch compatibility
/// PyTorch互換性のためのDistributedDataParallelラッパー
#[derive(Debug)]
pub struct DistributedDataParallel<T: DistributedScalar, M: Module<T>> {
    /// The wrapped module
    /// ラップされたモジュール
    module: Arc<Mutex<M>>,
    /// Process group for communication
    /// 通信用プロセスグループ
    process_group: Option<ProcessGroup>,
    /// Device IDs for this process
    /// このプロセスのデバイスID
    device_ids: Vec<usize>,
    /// Output device ID
    /// 出力デバイスID
    output_device: Option<usize>,
    /// Bucket size for gradient communication
    /// 勾配通信用バケットサイズ
    bucket_cap_mb: usize,
    /// Find unused parameters
    /// 未使用パラメータを検出
    find_unused_parameters: bool,
    /// Gradient as bucket view
    /// バケットビューとしての勾配
    gradient_as_bucket_view: bool,
    /// Static graph optimization
    /// 静的グラフ最適化
    static_graph: bool,
    /// Gradient accumulation state
    /// 勾配累積状態
    gradient_state: Arc<Mutex<GradientState<T>>>,
}

/// Gradient accumulation and synchronization state
/// 勾配累積と同期状態
#[derive(Debug)]
struct GradientState<T: DistributedScalar> {
    /// Accumulated gradients per parameter
    /// パラメータ毎の累積勾配
    accumulated_grads: HashMap<String, Tensor<T>>,
    /// Whether gradients are ready for synchronization
    /// 勾配が同期準備完了かどうか
    ready_for_sync: bool,
    /// Bucket management for efficient communication
    /// 効率的な通信のためのバケット管理
    buckets: Vec<GradientBucket<T>>,
}

/// Gradient bucket for efficient communication
/// 効率的な通信のための勾配バケット
#[derive(Debug, Clone)]
struct GradientBucket<T: DistributedScalar> {
    /// Parameters in this bucket
    /// このバケット内のパラメータ
    parameters: Vec<String>,
    /// Combined gradient tensor
    /// 結合勾配テンソル
    gradient: Option<Tensor<T>>,
    /// Bucket size in bytes
    /// バケットサイズ（バイト）
    size_bytes: usize,
}

impl<T: DistributedScalar, M: Module<T> + Send + Sync + 'static> 
    DistributedDataParallel<T, M> 
{
    /// Create a new DistributedDataParallel wrapper
    /// 新しいDistributedDataParallelラッパーを作成
    pub fn new(
        module: M,
        device_ids: Option<Vec<usize>>,
        output_device: Option<usize>,
        dim: Option<usize>,
        broadcast_buffers: bool,
        process_group: Option<ProcessGroup>,
        bucket_cap_mb: Option<usize>,
        find_unused_parameters: Option<bool>,
        check_reduction: Option<bool>,
        gradient_as_bucket_view: Option<bool>,
        static_graph: Option<bool>,
    ) -> RusTorchResult<Self> {
        if !api::is_initialized() {
            return Err(RusTorchError::distributed(
                "Distributed process group not initialized. Call distributed::init_process_group() first."
            ));
        }

        let device_ids = device_ids.unwrap_or_else(|| vec![0]);
        let bucket_cap_mb = bucket_cap_mb.unwrap_or(25); // Default 25MB
        let find_unused_parameters = find_unused_parameters.unwrap_or(false);
        let gradient_as_bucket_view = gradient_as_bucket_view.unwrap_or(false);
        let static_graph = static_graph.unwrap_or(false);

        // Initialize gradient state
        let gradient_state = Arc::new(Mutex::new(GradientState {
            accumulated_grads: HashMap::new(),
            ready_for_sync: false,
            buckets: Vec::new(),
        }));

        let _ = (dim, broadcast_buffers, check_reduction); // TODO: Implement these features

        Ok(Self {
            module: Arc::new(Mutex::new(module)),
            process_group,
            device_ids,
            output_device,
            bucket_cap_mb,
            find_unused_parameters,
            gradient_as_bucket_view,
            static_graph,
            gradient_state,
        })
    }

    /// Perform forward pass with distributed synchronization
    /// 分散同期付きフォワードパス
    pub fn forward(&self, input: &Variable<T>) -> RusTorchResult<Variable<T>> {
        let module = self.module.lock().unwrap();
        let output = module.forward(input);

        // Register backward hook for gradient synchronization
        // 勾配同期のためのバックワードフック登録
        self.register_grad_hooks()?;

        Ok(output)
    }

    /// Register gradient synchronization hooks
    /// 勾配同期フック登録
    fn register_grad_hooks(&self) -> RusTorchResult<()> {
        // In a full implementation, this would register hooks on all parameters
        // 完全な実装では、全パラメータにフックを登録
        Ok(())
    }

    /// Synchronize gradients across all processes
    /// 全プロセス間での勾配同期
    pub fn sync_gradients(&self) -> RusTorchResult<()> {
        let module = self.module.lock().unwrap();
        
        // Get all parameters with gradients
        let parameters = module.parameters();
        
        for param in parameters {
            let grad_lock = param.grad();
            let mut grad_guard = grad_lock.write().unwrap();
            if let Some(ref mut grad) = *grad_guard {
                // Perform all-reduce on gradient
                api::all_reduce(grad, ReduceOp::Average, self.process_group.as_ref(), false)?;
            }
        }

        Ok(())
    }

    /// Get the wrapped module
    /// ラップされたモジュールを取得
    pub fn module(&self) -> Arc<Mutex<M>> {
        Arc::clone(&self.module)
    }

    /// Get device IDs
    /// デバイスIDを取得
    pub fn device_ids(&self) -> &[usize] {
        &self.device_ids
    }

    /// Check if this is a DDP module
    /// DDPモジュールかどうかをチェック
    pub fn is_ddp_module() -> bool {
        true
    }
}

impl<T: DistributedScalar, M: Module<T> + Send + Sync + 'static> Module<T> 
    for DistributedDataParallel<T, M> 
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input).unwrap_or_else(|_| Variable::new(Tensor::zeros(&[1]), false))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let module = self.module.lock().unwrap();
        module.parameters()
    }

    fn eval(&mut self) {
        // Module evaluation mode - no specific action needed for DDP wrapper
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Implement the shared DDP trait
impl<T: DistributedScalar, M: Module<T> + Send + Sync + 'static> DistributedDataParallelTrait<T> 
    for DistributedDataParallel<T, M>
{
    fn device_ids(&self) -> &[usize] {
        &self.device_ids
    }
    
    fn distributed_forward(&self, input: &Variable<T>) -> RusTorchResult<Variable<T>> {
        self.forward(input)
    }
    
    fn sync_gradients(&self) -> RusTorchResult<()> {
        self.sync_gradients()
    }
}

impl<T: DistributedScalar> GradientState<T> {
    /// Create gradient buckets for efficient communication
    /// 効率的な通信のための勾配バケット作成
    fn create_buckets(&mut self, bucket_size_mb: usize) -> RusTorchResult<()> {
        let bucket_size_bytes = bucket_size_mb * 1024 * 1024;
        
        // Group parameters into buckets based on size
        // サイズに基づいてパラメータをバケットにグループ化
        let mut current_bucket = GradientBucket {
            parameters: Vec::new(),
            gradient: None,
            size_bytes: 0,
        };

        for param_name in self.accumulated_grads.keys() {
            if let Some(grad) = self.accumulated_grads.get(param_name) {
                let grad_size = grad.numel() * std::mem::size_of::<T>();
                
                if current_bucket.size_bytes + grad_size > bucket_size_bytes && !current_bucket.parameters.is_empty() {
                    self.buckets.push(current_bucket.clone());
                    current_bucket = GradientBucket {
                        parameters: Vec::new(),
                        gradient: None,
                        size_bytes: 0,
                    };
                }

                current_bucket.parameters.push(param_name.clone());
                current_bucket.size_bytes += grad_size;
            }
        }

        if !current_bucket.parameters.is_empty() {
            self.buckets.push(current_bucket);
        }

        Ok(())
    }
}

/// Convenience function to wrap a module in DDP
/// モジュールをDDPでラップするための便利関数
pub fn wrap_module<T: DistributedScalar, M: Module<T> + Send + Sync + 'static>(
    module: M,
    device_ids: Option<Vec<usize>>,
) -> RusTorchResult<DistributedDataParallel<T, M>> {
    DistributedDataParallel::new(
        module,
        device_ids,
        None, // output_device
        None, // dim
        true, // broadcast_buffers
        None, // process_group
        None, // bucket_cap_mb
        None, // find_unused_parameters
        None, // check_reduction
        None, // gradient_as_bucket_view
        None, // static_graph
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear};

    #[test]
    fn test_ddp_creation() {
        // This test would require distributed initialization
        // このテストは分散初期化が必要
        let linear: Linear<f32> = Linear::new(10, 5);
        let device_ids = vec![0];
        
        // Note: This would fail without proper distributed initialization
        // 注意：適切な分散初期化なしでは失敗する
        let ddp_result = DistributedDataParallel::new(
            linear,
            Some(device_ids),
            None, None, true, None, None, None, None, None, None,
        );
        
        // Test should fail because distributed not initialized
        assert!(ddp_result.is_err());
    }

    #[test]
    fn test_is_ddp_module() {
        assert!(DistributedDataParallel::<f32, Linear<f32>>::is_ddp_module());
    }
}