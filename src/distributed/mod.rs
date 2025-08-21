//! Distributed training support for RusTorch
//! RusTorchの分散学習サポート
//! 
//! This module provides comprehensive distributed training capabilities including:
//! - Data parallel training across multiple GPUs
//! - Model parallel training for large models
//! - Multi-machine cluster support
//! - Communication backends (NCCL, Gloo, MPI)
//! - Gradient synchronization and aggregation
//! 
//! このモジュールは包括的な分散学習機能を提供します：
//! - 複数GPU間でのデータ並列学習
//! - 大規模モデル向けのモデル並列学習
//! - 複数マシンクラスターサポート
//! - 通信バックエンド（NCCL、Gloo、MPI）
//! - 勾配同期と集約

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::tensor::Tensor;
use crate::gpu::{DeviceType, GpuError};
use num_traits::Float;

/// Distributed backend types
/// 分散バックエンドタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedBackend {
    /// NVIDIA Collective Communications Library
    /// NVIDIA集合通信ライブラリ
    NCCL,
    /// Facebook's collective communications library
    /// Facebookの集合通信ライブラリ
    Gloo,
    /// Message Passing Interface
    /// メッセージパッシングインターフェース
    MPI,
    /// Custom TCP backend
    /// カスタムTCPバックエンド
    TCP,
}

/// Process group for distributed training
/// 分散学習用プロセスグループ
#[derive(Debug, Clone)]
pub struct ProcessGroup {
    /// Rank of current process
    /// 現在のプロセスのランク
    pub rank: usize,
    /// Total number of processes
    /// プロセス総数
    pub world_size: usize,
    /// Backend used for communication
    /// 通信に使用するバックエンド
    pub backend: DistributedBackend,
    /// Master address for coordination
    /// 調整用マスターアドレス
    pub master_addr: String,
    /// Master port for coordination
    /// 調整用マスターポート
    pub master_port: u16,
}

impl ProcessGroup {
    /// Create a new process group
    /// 新しいプロセスグループを作成
    pub fn new(
        rank: usize,
        world_size: usize,
        backend: DistributedBackend,
        master_addr: String,
        master_port: u16,
    ) -> Self {
        Self {
            rank,
            world_size,
            backend,
            master_addr,
            master_port,
        }
    }
    
    /// Initialize the process group
    /// プロセスグループを初期化
    pub fn init(&self) -> Result<(), DistributedError> {
        match self.backend {
            DistributedBackend::NCCL => self.init_nccl(),
            DistributedBackend::Gloo => self.init_gloo(),
            DistributedBackend::MPI => self.init_mpi(),
            DistributedBackend::TCP => self.init_tcp(),
        }
    }
    
    fn init_nccl(&self) -> Result<(), DistributedError> {
        // NCCL initialization implementation
        // NCCL初期化実装
        #[cfg(feature = "nccl")]
        {
            // Initialize NCCL communicator
            // NCCL通信器を初期化
            Ok(())
        }
        #[cfg(not(feature = "nccl"))]
        {
            Err(DistributedError::BackendNotAvailable("NCCL not compiled".to_string()))
        }
    }
    
    fn init_gloo(&self) -> Result<(), DistributedError> {
        // Gloo initialization implementation
        // Gloo初期化実装
        Ok(())
    }
    
    fn init_mpi(&self) -> Result<(), DistributedError> {
        // MPI initialization implementation
        // MPI初期化実装
        #[cfg(feature = "mpi")]
        {
            Ok(())
        }
        #[cfg(not(feature = "mpi"))]
        {
            Err(DistributedError::BackendNotAvailable("MPI not compiled".to_string()))
        }
    }
    
    fn init_tcp(&self) -> Result<(), DistributedError> {
        // TCP backend initialization
        // TCPバックエンド初期化
        Ok(())
    }
}

/// Distributed training errors
/// 分散学習エラー
#[derive(Debug, Clone)]
pub enum DistributedError {
    /// Process group initialization failed
    /// プロセスグループ初期化失敗
    ProcessGroupError(String),
    /// Communication backend error
    /// 通信バックエンドエラー
    CommunicationError(String),
    /// Backend not available
    /// バックエンドが利用不可
    BackendNotAvailable(String),
    /// Configuration error
    /// 設定エラー
    ConfigurationError(String),
    /// Cluster management error
    /// クラスター管理エラー
    ClusterError(String),
    /// GPU error in distributed context
    /// 分散コンテキストでのGPUエラー
    GpuError(GpuError),
}

impl std::fmt::Display for DistributedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributedError::BackendNotAvailable(msg) => write!(f, "Backend not available: {}", msg),
            DistributedError::CommunicationError(msg) => write!(f, "Communication error: {}", msg),
            DistributedError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            DistributedError::ClusterError(msg) => write!(f, "Cluster error: {}", msg),
            DistributedError::ProcessGroupError(msg) => write!(f, "Process group error: {}", msg),
            DistributedError::GpuError(err) => write!(f, "GPU error in distributed context: {}", err),
        }
    }
}

impl std::error::Error for DistributedError {}

impl From<GpuError> for DistributedError {
    fn from(err: GpuError) -> Self {
        DistributedError::GpuError(err)
    }
}

/// Result type for distributed operations
/// 分散操作の結果タイプ
pub type DistributedResult<T> = Result<T, DistributedError>;

/// Communication operations for distributed training
/// 分散学習用通信操作
pub trait DistributedOps<T: Float> {
    /// All-reduce operation across all processes
    /// 全プロセス間でのall-reduce操作
    fn all_reduce(&self, tensor: &mut Tensor<T>, op: ReduceOp) -> DistributedResult<()>;
    
    /// All-gather operation across all processes
    /// 全プロセス間でのall-gather操作
    fn all_gather(&self, tensor: &Tensor<T>) -> DistributedResult<Vec<Tensor<T>>>;
    
    /// Broadcast operation from root process
    /// ルートプロセスからのブロードキャスト操作
    fn broadcast(&self, tensor: &mut Tensor<T>, root: usize) -> DistributedResult<()>;
    
    /// Reduce operation to root process
    /// ルートプロセスへのreduce操作
    fn reduce(&self, tensor: &mut Tensor<T>, root: usize, op: ReduceOp) -> DistributedResult<()>;
    
    /// Scatter operation from root process
    /// ルートプロセスからのscatter操作
    fn scatter(&self, tensors: &[Tensor<T>], root: usize) -> DistributedResult<Tensor<T>>;
    
    /// Gather operation to root process
    /// ルートプロセスへのgather操作
    fn gather(&self, tensor: &Tensor<T>, root: usize) -> DistributedResult<Vec<Tensor<T>>>;
}

/// Reduction operations for collective communications
/// 集合通信用リダクション操作
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum reduction
    /// 合計リダクション
    Sum,
    /// Product reduction
    /// 積リダクション
    Product,
    /// Minimum reduction
    /// 最小値リダクション
    Min,
    /// Maximum reduction
    /// 最大値リダクション
    Max,
    /// Average reduction
    /// 平均リダクション
    Average,
}

/// Global distributed state
/// グローバル分散状態
static mut DISTRIBUTED_STATE: Option<Arc<Mutex<DistributedState>>> = None;
static DISTRIBUTED_INIT: std::sync::Once = std::sync::Once::new();

/// Distributed state management
/// 分散状態管理
#[derive(Debug)]
pub struct DistributedState {
    /// Current process group
    /// 現在のプロセスグループ
    pub process_group: Option<ProcessGroup>,
    /// Available devices for distributed training
    /// 分散学習で利用可能なデバイス
    pub devices: Vec<DeviceType>,
    /// Device mapping for each rank
    /// 各ランクのデバイスマッピング
    pub device_map: HashMap<usize, Vec<DeviceType>>,
}

impl DistributedState {
    /// Create new distributed state
    /// 新しい分散状態を作成
    pub fn new() -> Self {
        Self {
            process_group: None,
            devices: Vec::new(),
            device_map: HashMap::new(),
        }
    }
    
    /// Set process group
    /// プロセスグループを設定
    pub fn set_process_group(&mut self, pg: ProcessGroup) {
        self.process_group = Some(pg);
    }
    
    /// Get current rank
    /// 現在のランクを取得
    pub fn rank(&self) -> Option<usize> {
        self.process_group.as_ref().map(|pg| pg.rank)
    }
    
    /// Get world size
    /// ワールドサイズを取得
    pub fn world_size(&self) -> Option<usize> {
        self.process_group.as_ref().map(|pg| pg.world_size)
    }
    
    /// Check if distributed training is initialized
    /// 分散学習が初期化されているかチェック
    pub fn is_initialized(&self) -> bool {
        self.process_group.is_some()
    }
}

/// Get global distributed state
/// グローバル分散状態を取得
pub fn get_distributed_state() -> &'static Arc<Mutex<DistributedState>> {
    unsafe {
        DISTRIBUTED_INIT.call_once(|| {
            DISTRIBUTED_STATE = Some(Arc::new(Mutex::new(DistributedState::new())));
        });
        DISTRIBUTED_STATE.as_ref().unwrap()
    }
}

/// Initialize distributed training
/// 分散学習を初期化
pub fn init_distributed(
    backend: DistributedBackend,
    rank: usize,
    world_size: usize,
    master_addr: String,
    master_port: u16,
) -> DistributedResult<()> {
    let process_group = ProcessGroup::new(rank, world_size, backend, master_addr, master_port);
    process_group.init()?;
    
    let state = get_distributed_state();
    let mut state_guard = state.lock().unwrap();
    state_guard.set_process_group(process_group);
    
    Ok(())
}

/// Check if distributed training is available
/// 分散学習が利用可能かチェック
pub fn is_available() -> bool {
    let state = get_distributed_state();
    let state_guard = state.lock().unwrap();
    state_guard.is_initialized()
}

/// Get current rank in distributed training
/// 分散学習での現在のランクを取得
pub fn get_rank() -> Option<usize> {
    let state = get_distributed_state();
    let state_guard = state.lock().unwrap();
    state_guard.rank()
}

/// Get world size in distributed training
/// 分散学習でのワールドサイズを取得
pub fn get_world_size() -> Option<usize> {
    let state = get_distributed_state();
    let state_guard = state.lock().unwrap();
    state_guard.world_size()
}

/// Finalize distributed training
/// 分散学習を終了
pub fn finalize() -> DistributedResult<()> {
    let state = get_distributed_state();
    let mut state_guard = state.lock().unwrap();
    state_guard.process_group = None;
    state_guard.devices.clear();
    state_guard.device_map.clear();
    
    Ok(())
}

/// Data parallel training module
/// データ並列学習モジュール
pub mod data_parallel;
pub mod model_parallel;
pub mod backends;
pub mod optimizer;
pub mod cluster;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_process_group_creation() {
        let pg = ProcessGroup::new(
            0,
            4,
            DistributedBackend::TCP,
            "localhost".to_string(),
            12345,
        );
        
        assert_eq!(pg.rank, 0);
        assert_eq!(pg.world_size, 4);
        assert_eq!(pg.backend, DistributedBackend::TCP);
        assert_eq!(pg.master_addr, "localhost");
        assert_eq!(pg.master_port, 12345);
    }
    
    #[test]
    fn test_distributed_state() {
        let mut state = DistributedState::new();
        assert!(!state.is_initialized());
        
        let pg = ProcessGroup::new(
            1,
            2,
            DistributedBackend::Gloo,
            "127.0.0.1".to_string(),
            29500,
        );
        
        state.set_process_group(pg);
        assert!(state.is_initialized());
        assert_eq!(state.rank(), Some(1));
        assert_eq!(state.world_size(), Some(2));
    }
    
    #[test]
    fn test_reduce_op_variants() {
        let ops = [
            ReduceOp::Sum,
            ReduceOp::Product,
            ReduceOp::Min,
            ReduceOp::Max,
            ReduceOp::Average,
        ];
        
        for op in &ops {
            assert!(matches!(op, ReduceOp::Sum | ReduceOp::Product | ReduceOp::Min | ReduceOp::Max | ReduceOp::Average));
        }
    }
}
