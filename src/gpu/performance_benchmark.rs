//! GPU Performance Benchmark Suite (Legacy)
//! GPUパフォーマンスベンチマークスイート（レガシー）
//!
//! This module re-exports the new modular benchmark system for backward compatibility.
//! 後方互換性のために新しいモジュラーベンチマークシステムを再エクスポート。

// Re-export new benchmark module components
pub use crate::gpu::benchmark::*;