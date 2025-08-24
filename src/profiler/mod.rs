//! Performance profiler for RusTorch operations
//! RusTorch操作のパフォーマンスプロファイラー

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use std::fmt;

pub mod memory_profiler;
pub mod kernel_profiler;
pub mod timeline;

use self::memory_profiler::MemoryProfiler;
use self::kernel_profiler::KernelProfiler;
use self::timeline::Timeline;

lazy_static::lazy_static! {
    /// Global profiler instance
    /// グローバルプロファイラーインスタンス
    static ref PROFILER: Arc<Mutex<Profiler>> = Arc::new(Mutex::new(Profiler::new()));
}

/// Profile context manager
/// プロファイルコンテキストマネージャー
pub struct ProfileContext {
    /// Context name
    name: String,
    /// Start time
    start_time: Instant,
    /// Parent context
    parent: Option<String>,
}

impl ProfileContext {
    /// Create new profile context
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let start_time = Instant::now();
        
        // Register context start
        if let Ok(mut profiler) = PROFILER.lock() {
            profiler.start_operation(&name);
        }
        
        Self {
            name,
            start_time,
            parent: None,
        }
    }
}

impl Drop for ProfileContext {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        
        // Register context end
        if let Ok(mut profiler) = PROFILER.lock() {
            profiler.end_operation(&self.name, duration);
        }
    }
}

/// Main profiler structure
/// メインプロファイラー構造
pub struct Profiler {
    /// Operation records
    /// 操作記録
    operations: HashMap<String, OperationStats>,
    /// Memory profiler
    /// メモリプロファイラー
    memory_profiler: MemoryProfiler,
    /// Kernel profiler
    /// カーネルプロファイラー
    kernel_profiler: KernelProfiler,
    /// Timeline
    /// タイムライン
    timeline: Timeline,
    /// Profiling enabled flag
    /// プロファイリング有効フラグ
    enabled: bool,
    /// Current call stack
    /// 現在のコールスタック
    call_stack: Vec<String>,
    /// Thread-local storage for multi-threading
    /// マルチスレッディング用スレッドローカルストレージ
    thread_data: HashMap<thread::ThreadId, ThreadProfileData>,
}

/// Operation statistics
/// 操作統計
#[derive(Debug, Clone)]
pub struct OperationStats {
    /// Operation name
    /// 操作名
    pub name: String,
    /// Number of calls
    /// 呼び出し回数
    pub count: usize,
    /// Total time spent
    /// 総消費時間
    pub total_time: Duration,
    /// Average time per call
    /// 呼び出しごとの平均時間
    pub avg_time: Duration,
    /// Minimum time
    /// 最小時間
    pub min_time: Duration,
    /// Maximum time
    /// 最大時間
    pub max_time: Duration,
    /// Memory allocated (bytes)
    /// 割り当てメモリ（バイト）
    pub memory_allocated: usize,
    /// Memory freed (bytes)
    /// 解放メモリ（バイト）
    pub memory_freed: usize,
    /// CUDA time if applicable
    /// 該当する場合のCUDA時間
    pub cuda_time: Option<Duration>,
    /// Self CPU time (excluding children)
    /// 自己CPU時間（子を除く）
    pub self_cpu_time: Duration,
    /// Child operations
    /// 子操作
    pub children: Vec<String>,
}

impl Default for OperationStats {
    fn default() -> Self {
        Self {
            name: String::new(),
            count: 0,
            total_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            memory_allocated: 0,
            memory_freed: 0,
            cuda_time: None,
            self_cpu_time: Duration::ZERO,
            children: Vec::new(),
        }
    }
}

/// Thread-specific profile data
/// スレッド固有のプロファイルデータ
#[derive(Debug, Clone)]
struct ThreadProfileData {
    /// Thread ID
    thread_id: thread::ThreadId,
    /// Call stack
    call_stack: Vec<String>,
    /// Operation timings
    timings: HashMap<String, Vec<Duration>>,
}

impl Profiler {
    /// Create new profiler
    /// 新しいプロファイラーを作成
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            memory_profiler: MemoryProfiler::new(),
            kernel_profiler: KernelProfiler::new(),
            timeline: Timeline::new(),
            enabled: false,
            call_stack: Vec::new(),
            thread_data: HashMap::new(),
        }
    }

    /// Enable profiling
    /// プロファイリングを有効化
    pub fn enable(&mut self) {
        self.enabled = true;
        self.memory_profiler.start();
        self.kernel_profiler.start();
    }

    /// Disable profiling
    /// プロファイリングを無効化
    pub fn disable(&mut self) {
        self.enabled = false;
        self.memory_profiler.stop();
        self.kernel_profiler.stop();
    }

    /// Start an operation
    /// 操作を開始
    pub fn start_operation(&mut self, name: &str) {
        if !self.enabled {
            return;
        }

        let thread_id = thread::current().id();
        
        // Update call stack
        self.call_stack.push(name.to_string());
        
        // Record in timeline
        self.timeline.add_event(name, Instant::now(), None);
        
        // Initialize operation stats if needed
        self.operations.entry(name.to_string())
            .or_insert_with(|| {
                let mut stats = OperationStats::default();
                stats.name = name.to_string();
                stats
            });
        
        // Update thread data
        let thread_data = self.thread_data.entry(thread_id)
            .or_insert_with(|| ThreadProfileData {
                thread_id,
                call_stack: Vec::new(),
                timings: HashMap::new(),
            });
        thread_data.call_stack.push(name.to_string());
    }

    /// End an operation
    /// 操作を終了
    pub fn end_operation(&mut self, name: &str, duration: Duration) {
        if !self.enabled {
            return;
        }

        // Update operation stats
        if let Some(stats) = self.operations.get_mut(name) {
            stats.count += 1;
            stats.total_time += duration;
            stats.avg_time = stats.total_time / stats.count as u32;
            stats.min_time = stats.min_time.min(duration);
            stats.max_time = stats.max_time.max(duration);
            
            // Update memory stats
            let mem_stats = self.memory_profiler.get_current_stats();
            stats.memory_allocated = mem_stats.allocated;
            stats.memory_freed = mem_stats.freed;
            
            // Update CUDA time if applicable
            if let Some(cuda_time) = self.kernel_profiler.get_last_kernel_time() {
                stats.cuda_time = Some(cuda_time);
            }
        }
        
        // Update call stack
        if let Some(last) = self.call_stack.last() {
            if last == name {
                self.call_stack.pop();
            }
        }
        
        // Update timeline
        self.timeline.end_event(name, Instant::now());
    }

    /// Record memory allocation
    /// メモリ割り当てを記録
    pub fn record_allocation(&mut self, size: usize, name: &str) {
        if !self.enabled {
            return;
        }
        
        self.memory_profiler.record_allocation(size, name);
    }

    /// Record memory deallocation
    /// メモリ解放を記録
    pub fn record_deallocation(&mut self, size: usize, name: &str) {
        if !self.enabled {
            return;
        }
        
        self.memory_profiler.record_deallocation(size, name);
    }

    /// Get profiling summary
    /// プロファイリングサマリーを取得
    pub fn get_summary(&self) -> ProfileSummary {
        let mut operations: Vec<_> = self.operations.values().cloned().collect();
        operations.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        
        let total_time: Duration = operations.iter()
            .map(|op| op.total_time)
            .sum();
        
        let memory_stats = self.memory_profiler.get_summary();
        let kernel_stats = self.kernel_profiler.get_summary();
        
        ProfileSummary {
            operations,
            total_time,
            memory_stats,
            kernel_stats,
            timeline: self.timeline.clone(),
        }
    }

    /// Clear all profiling data
    /// すべてのプロファイリングデータをクリア
    pub fn clear(&mut self) {
        self.operations.clear();
        self.memory_profiler.clear();
        self.kernel_profiler.clear();
        self.timeline.clear();
        self.call_stack.clear();
        self.thread_data.clear();
    }

    /// Export profiling data to Chrome tracing format
    /// プロファイリングデータをChromeトレーシング形式にエクスポート
    pub fn export_chrome_trace(&self) -> String {
        self.timeline.export_chrome_trace()
    }

    /// Print formatted report
    /// フォーマットされたレポートを出力
    pub fn print_report(&self) {
        let summary = self.get_summary();
        println!("{}", summary);
    }
}

/// Profiling summary
/// プロファイリングサマリー
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    /// Operation statistics sorted by time
    /// 時間でソートされた操作統計
    pub operations: Vec<OperationStats>,
    /// Total profiling time
    /// 総プロファイリング時間
    pub total_time: Duration,
    /// Memory statistics
    /// メモリ統計
    pub memory_stats: memory_profiler::MemorySummary,
    /// Kernel statistics
    /// カーネル統計
    pub kernel_stats: kernel_profiler::KernelSummary,
    /// Timeline
    /// タイムライン
    pub timeline: Timeline,
}

impl fmt::Display for ProfileSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n================== Profiler Report ==================")?;
        writeln!(f, "Total Time: {:.3}ms", self.total_time.as_secs_f64() * 1000.0)?;
        writeln!(f)?;
        
        // Operation table
        writeln!(f, "Top Operations by Time:")?;
        writeln!(f, "{:<30} {:>10} {:>12} {:>12} {:>12}", 
            "Name", "Calls", "Total (ms)", "Avg (ms)", "Self (ms)")?;
        writeln!(f, "{}", "-".repeat(80))?;
        
        for op in self.operations.iter().take(20) {
            writeln!(f, "{:<30} {:>10} {:>12.3} {:>12.3} {:>12.3}",
                if op.name.len() > 29 { &op.name[..29] } else { &op.name },
                op.count,
                op.total_time.as_secs_f64() * 1000.0,
                op.avg_time.as_secs_f64() * 1000.0,
                op.self_cpu_time.as_secs_f64() * 1000.0)?;
        }
        
        writeln!(f)?;
        
        // Memory statistics
        writeln!(f, "Memory Statistics:")?;
        writeln!(f, "{}", self.memory_stats)?;
        
        // Kernel statistics
        if !self.kernel_stats.kernels.is_empty() {
            writeln!(f, "\nGPU Kernel Statistics:")?;
            writeln!(f, "{}", self.kernel_stats)?;
        }
        
        Ok(())
    }
}

/// Profile a code block
/// コードブロックをプロファイル
#[macro_export]
macro_rules! profile {
    ($name:expr, $body:expr) => {{
        let _ctx = $crate::profiler::ProfileContext::new($name);
        $body
    }};
}

/// Profile a function
/// 関数をプロファイル
#[macro_export]
macro_rules! profile_fn {
    ($name:expr) => {
        let _ctx = $crate::profiler::ProfileContext::new($name);
    };
}

/// Global profiler control functions
/// グローバルプロファイラー制御関数

/// Enable global profiler
/// グローバルプロファイラーを有効化
pub fn enable_profiler() {
    if let Ok(mut profiler) = PROFILER.lock() {
        profiler.enable();
    }
}

/// Disable global profiler
/// グローバルプロファイラーを無効化
pub fn disable_profiler() {
    if let Ok(mut profiler) = PROFILER.lock() {
        profiler.disable();
    }
}

/// Get profiler summary
/// プロファイラーサマリーを取得
pub fn get_profiler_summary() -> Option<ProfileSummary> {
    PROFILER.lock().ok().map(|p| p.get_summary())
}

/// Clear profiler data
/// プロファイラーデータをクリア
pub fn clear_profiler() {
    if let Ok(mut profiler) = PROFILER.lock() {
        profiler.clear();
    }
}

/// Print profiler report
/// プロファイラーレポートを出力
pub fn print_profiler_report() {
    if let Ok(profiler) = PROFILER.lock() {
        profiler.print_report();
    }
}

/// Export Chrome trace
/// Chromeトレースをエクスポート
pub fn export_chrome_trace() -> Option<String> {
    PROFILER.lock().ok().map(|p| p.export_chrome_trace())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_profiling() {
        clear_profiler();
        enable_profiler();
        
        {
            let _ctx = ProfileContext::new("test_operation");
            thread::sleep(Duration::from_millis(10));
        }
        
        let summary = get_profiler_summary().unwrap();
        assert!(summary.operations.len() > 0);
        let test_op = summary.operations.iter().find(|op| op.name == "test_operation");
        assert!(test_op.is_some());
        assert_eq!(test_op.unwrap().count, 1);
        
        disable_profiler();
    }

    #[test]
    fn test_nested_profiling() {
        clear_profiler();
        enable_profiler();
        
        {
            let _ctx1 = ProfileContext::new("outer");
            thread::sleep(Duration::from_millis(5));
            {
                let _ctx2 = ProfileContext::new("inner");
                thread::sleep(Duration::from_millis(5));
            }
        }
        
        let summary = get_profiler_summary().unwrap();
        assert!(summary.operations.len() > 0);
        let outer_op = summary.operations.iter().find(|op| op.name == "outer");
        let inner_op = summary.operations.iter().find(|op| op.name == "inner");
        assert!(outer_op.is_some());
        assert!(inner_op.is_some());
        
        disable_profiler();
    }

    #[test]
    fn test_profile_macro() {
        clear_profiler();
        enable_profiler();
        
        profile!("macro_test", {
            thread::sleep(Duration::from_millis(10));
        });
        
        let summary = get_profiler_summary().unwrap();
        assert!(summary.operations.iter().any(|op| op.name == "macro_test"));
        
        disable_profiler();
    }
}