// 高度メモリ管理システム
// Advanced memory management system

pub mod tensor_pool;
pub mod compression;
pub mod garbage_collection;

// Re-exports
pub use tensor_pool::{TensorPool, PoolConfig, PooledTensor, PoolStats};
pub use compression::{CompressionEngine, CompressionConfig, CompressionFormat};
pub use garbage_collection::{GarbageCollector, GCConfig, GCStats};