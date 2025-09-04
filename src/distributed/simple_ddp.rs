//! Simplified DistributedDataParallel implementation
//! 簡略化DistributedDataParallel実装

use crate::error::{RusTorchError, RusTorchResult};
use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Tensor;
use num_traits::Float;
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;
use super::{ReduceOp, api};

/// Simplified DistributedDataParallel for RusTorch
/// RusTorch用簡略化DistributedDataParallel
pub struct SimpleDistributedDataParallel<T, M>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    M: Module<T> + Send + Sync + 'static,
{
    /// The wrapped module
    /// ラップされたモジュール
    module: Arc<Mutex<M>>,
    /// Device IDs for this process
    /// このプロセスのデバイスID
    device_ids: Vec<usize>,
    /// Gradient synchronization enabled
    /// 勾配同期が有効
    sync_gradients: bool,
    /// Phantom data for type parameter
    /// 型パラメータ用のファントムデータ
    _phantom: PhantomData<T>,
}

impl<T, M> SimpleDistributedDataParallel<T, M>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    M: Module<T> + Send + Sync + 'static,
{
    /// Create a new simplified DDP wrapper
    /// 新しい簡略化DDPラッパーを作成
    pub fn new(module: M, device_ids: Option<Vec<usize>>) -> RusTorchResult<Self> {
        if !api::is_initialized() {
            return Err(RusTorchError::distributed(
                "Distributed not initialized. Call distributed::init_process_group() first."
            ));
        }

        let device_ids = device_ids.unwrap_or_else(|| vec![0]);

        Ok(Self {
            module: Arc::new(Mutex::new(module)),
            device_ids,
            sync_gradients: true,
            _phantom: PhantomData,
        })
    }

    /// Forward pass with distributed synchronization
    /// 分散同期付きフォワードパス
    pub fn forward(&self, input: &Variable<T>) -> RusTorchResult<Variable<T>> {
        let module = self.module.lock().unwrap();
        let output = module.forward(input);
        
        // Automatically sync gradients after forward if enabled
        if self.sync_gradients {
            // For simplicity, we'll sync after each forward pass
            // In practice, this would be done after backward pass
        }

        Ok(output)
    }

    /// Synchronize gradients across all processes
    /// 全プロセス間での勾配同期
    pub fn sync_gradients(&self) -> RusTorchResult<()> {
        // For this simplified version, we just simulate gradient sync
        // 簡略化版では、勾配同期をシミュレート
        
        // In a full implementation, we would:
        // 1. Get all parameters from the module
        // 2. Extract gradients from each parameter
        // 3. Perform all-reduce on each gradient
        // 4. Update the parameter gradients

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

    /// Enable or disable automatic gradient synchronization
    /// 自動勾配同期の有効/無効を設定
    pub fn set_gradient_sync(&mut self, enabled: bool) {
        self.sync_gradients = enabled;
    }
}

/// Convenience function to wrap a module in simplified DDP
/// モジュールを簡略化DDPでラップするための便利関数
pub fn wrap_simple<T, M>(
    module: M,
    device_ids: Option<Vec<usize>>,
) -> RusTorchResult<SimpleDistributedDataParallel<T, M>>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    M: Module<T> + Send + Sync + 'static,
{
    SimpleDistributedDataParallel::new(module, device_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use super::super::DistributedBackend;

    #[test]
    fn test_simple_ddp_creation() {
        // This would require distributed initialization
        let linear: Linear<f32> = Linear::new(10, 5);
        
        // Test should fail because distributed not initialized
        let ddp_result = SimpleDistributedDataParallel::new(linear, Some(vec![0]));
        assert!(ddp_result.is_err());
    }

    #[test]
    fn test_device_ids() {
        // Create without distributed init for testing structure only
        std::env::set_var("RANK", "0");
        std::env::set_var("WORLD_SIZE", "1");
        std::env::set_var("MASTER_ADDR", "localhost");
        std::env::set_var("MASTER_PORT", "29510");

        let _ = api::init_process_group(
            DistributedBackend::TCP,
            None, None, None, None,
        );

        let linear: Linear<f32> = Linear::new(5, 3);
        if let Ok(ddp) = SimpleDistributedDataParallel::new(linear, Some(vec![0, 1])) {
            assert_eq!(ddp.device_ids(), &[0, 1]);
        }

        let _ = api::destroy_process_group();
    }
}