// 高度ガベージコレクション機能
// Advanced garbage collection functionality

use crate::common::RusTorchResult;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// ガベージコレクション設定
/// Garbage collection configuration
#[derive(Debug, Clone)]
pub struct GCConfig {
    /// GC実行間隔
    /// GC execution interval
    pub gc_interval: Duration,

    /// メモリ閾値 (バイト単位)
    /// Memory threshold in bytes
    pub memory_threshold: usize,

    /// 使用率閾値 (0.0-1.0)
    /// Usage ratio threshold
    pub usage_threshold: f32,

    /// 強制GC後の待機時間
    /// Wait time after forced GC
    pub forced_gc_cooldown: Duration,

    /// 自動GCを有効にするか
    /// Whether to enable automatic GC
    pub enable_auto_gc: bool,

    /// 世代別GCを有効にするか
    /// Whether to enable generational GC
    pub enable_generational: bool,

    /// 若い世代の生存回数
    /// Survival count for young generation
    pub young_generation_limit: u32,
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            gc_interval: Duration::from_secs(30),
            memory_threshold: 1024 * 1024 * 1024, // 1GB
            usage_threshold: 0.8,
            forced_gc_cooldown: Duration::from_secs(5),
            enable_auto_gc: true,
            enable_generational: true,
            young_generation_limit: 3,
        }
    }
}

/// GC統計情報
/// GC statistics
#[derive(Debug, Clone, Default)]
pub struct GCStats {
    /// 総GC実行回数
    /// Total GC executions
    pub total_collections: u64,

    /// 最後のGC時刻
    /// Last GC time
    pub last_collection_time: Option<Instant>,

    /// 最後のGC所要時間
    /// Last GC duration
    pub last_collection_duration: Option<Duration>,

    /// 累計GC時間
    /// Total GC time
    pub total_gc_time: Duration,

    /// 回収されたメモリ量
    /// Memory reclaimed in bytes
    pub memory_reclaimed: usize,

    /// 回収されたオブジェクト数
    /// Objects reclaimed
    pub objects_reclaimed: u64,

    /// 若い世代GC回数
    /// Young generation GC count
    pub young_gc_count: u64,

    /// 古い世代GC回数
    /// Old generation GC count
    pub old_gc_count: u64,

    /// 平均GC時間
    /// Average GC time
    pub average_gc_time: Duration,
}

impl GCStats {
    /// 統計を更新
    /// Update statistics
    pub fn update_collection(
        &mut self,
        duration: Duration,
        memory_freed: usize,
        objects_freed: u64,
        is_young: bool,
    ) {
        self.total_collections += 1;
        self.last_collection_time = Some(Instant::now());
        self.last_collection_duration = Some(duration);
        self.total_gc_time += duration;
        self.memory_reclaimed += memory_freed;
        self.objects_reclaimed += objects_freed;

        if is_young {
            self.young_gc_count += 1;
        } else {
            self.old_gc_count += 1;
        }

        // 平均時間を計算
        // Calculate average time
        if self.total_collections > 0 {
            self.average_gc_time = self.total_gc_time / self.total_collections as u32;
        }
    }

    /// レポートを生成
    /// Generate report
    pub fn generate_report(&self) -> String {
        format!(
            "GC Statistics Report:\n\
             - Total Collections: {}\n\
             - Young Generation GCs: {}\n\
             - Old Generation GCs: {}\n\
             - Total Memory Reclaimed: {:.2} MB\n\
             - Total Objects Reclaimed: {}\n\
             - Total GC Time: {:.2}s\n\
             - Average GC Time: {:.2}ms\n\
             - Last GC Duration: {}\n",
            self.total_collections,
            self.young_gc_count,
            self.old_gc_count,
            self.memory_reclaimed as f64 / 1024.0 / 1024.0,
            self.objects_reclaimed,
            self.total_gc_time.as_secs_f64(),
            self.average_gc_time.as_millis(),
            self.last_collection_duration
                .map(|d| format!("{:.2}ms", d.as_millis()))
                .unwrap_or_else(|| "N/A".to_string())
        )
    }
}

/// GC可能オブジェクトのトレイト
/// Trait for garbage collectable objects
pub trait GCObject: Send + Sync + std::fmt::Debug {
    /// オブジェクトのサイズ（バイト単位）
    /// Object size in bytes
    fn size(&self) -> usize;

    /// 最後のアクセス時刻
    /// Last access time
    fn last_accessed(&self) -> Instant;

    /// 生存回数
    /// Survival count
    fn survival_count(&self) -> u32;

    /// 生存回数を増加
    /// Increment survival count
    fn increment_survival(&mut self);

    /// マークフェーズでマークされているか
    /// Whether marked in mark phase
    fn is_marked(&self) -> bool;

    /// マークを設定
    /// Set mark
    fn set_marked(&mut self, marked: bool);

    /// オブジェクトが参照しているオブジェクトのID
    /// IDs of objects this object references
    fn references(&self) -> Vec<u64>;
}

/// GCオブジェクトのエントリ
/// GC object entry
#[derive(Debug)]
struct GCEntry {
    object: Box<dyn GCObject>,
    id: u64,
    generation: u8, // 0: young, 1: old
    created_at: Instant,
}

/// ガベージコレクタ
/// Garbage collector
pub struct GarbageCollector {
    config: GCConfig,
    objects: Arc<Mutex<HashMap<u64, GCEntry>>>,
    stats: Arc<RwLock<GCStats>>,
    next_id: Arc<Mutex<u64>>,
    last_gc: Arc<Mutex<Instant>>,
    gc_thread: Option<thread::JoinHandle<()>>,
    shutdown_signal: Arc<Mutex<bool>>,
}

impl GarbageCollector {
    /// 新しいガベージコレクタを作成
    /// Create new garbage collector
    pub fn new(config: GCConfig) -> Self {
        let collector = Self {
            config: config.clone(),
            objects: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(GCStats::default())),
            next_id: Arc::new(Mutex::new(1)),
            last_gc: Arc::new(Mutex::new(Instant::now())),
            gc_thread: None,
            shutdown_signal: Arc::new(Mutex::new(false)),
        };

        collector
    }

    /// 自動GCスレッドを開始
    /// Start automatic GC thread
    pub fn start_auto_gc(&mut self) -> RusTorchResult<()> {
        if !self.config.enable_auto_gc {
            return Ok(());
        }

        let config = self.config.clone();
        let objects = Arc::clone(&self.objects);
        let stats = Arc::clone(&self.stats);
        let last_gc = Arc::clone(&self.last_gc);
        let shutdown = Arc::clone(&self.shutdown_signal);

        let handle = thread::spawn(move || {
            loop {
                // シャットダウンチェック
                // Check shutdown signal
                {
                    let shutdown_guard = shutdown.lock().unwrap();
                    if *shutdown_guard {
                        break;
                    }
                }

                // GC実行チェック
                // Check if GC should run
                let should_gc = {
                    let last = last_gc.lock().unwrap();
                    last.elapsed() >= config.gc_interval
                };

                if should_gc {
                    let _ = Self::perform_gc_internal(&config, &objects, &stats, &last_gc);
                }

                thread::sleep(Duration::from_millis(100));
            }
        });

        self.gc_thread = Some(handle);
        Ok(())
    }

    /// オブジェクトを登録
    /// Register object
    pub fn register_object(&self, object: Box<dyn GCObject>) -> u64 {
        let id = {
            let mut next_id = self.next_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let entry = GCEntry {
            object,
            id,
            generation: 0, // Start in young generation
            created_at: Instant::now(),
        };

        let mut objects = self.objects.lock().unwrap();
        objects.insert(id, entry);

        id
    }

    /// オブジェクトを削除
    /// Remove object
    pub fn unregister_object(&self, id: u64) -> bool {
        let mut objects = self.objects.lock().unwrap();
        objects.remove(&id).is_some()
    }

    /// 手動GCを実行
    /// Perform manual GC
    pub fn collect(&self) -> RusTorchResult<()> {
        Self::perform_gc_internal(&self.config, &self.objects, &self.stats, &self.last_gc)
    }

    /// 強制GCを実行
    /// Perform forced GC
    pub fn force_collect(&self) -> RusTorchResult<()> {
        self.collect()?;

        // クールダウン期間待機
        // Wait for cooldown period
        thread::sleep(self.config.forced_gc_cooldown);

        Ok(())
    }

    /// GC統計を取得
    /// Get GC statistics
    pub fn get_stats(&self) -> GCStats {
        self.stats.read().unwrap().clone()
    }

    /// メモリ使用量を取得
    /// Get memory usage
    pub fn get_memory_usage(&self) -> usize {
        let objects = self.objects.lock().unwrap();
        objects.values().map(|entry| entry.object.size()).sum()
    }

    /// オブジェクト数を取得
    /// Get object count
    pub fn get_object_count(&self) -> usize {
        let objects = self.objects.lock().unwrap();
        objects.len()
    }

    /// 世代別統計を取得
    /// Get generational statistics
    pub fn get_generational_stats(&self) -> (usize, usize) {
        let objects = self.objects.lock().unwrap();
        let mut young_count = 0;
        let mut old_count = 0;

        for entry in objects.values() {
            if entry.generation == 0 {
                young_count += 1;
            } else {
                old_count += 1;
            }
        }

        (young_count, old_count)
    }

    /// 内部GC実行関数
    /// Internal GC execution function
    fn perform_gc_internal(
        config: &GCConfig,
        objects: &Arc<Mutex<HashMap<u64, GCEntry>>>,
        stats: &Arc<RwLock<GCStats>>,
        last_gc: &Arc<Mutex<Instant>>,
    ) -> RusTorchResult<()> {
        let start_time = Instant::now();
        let mut memory_freed = 0;
        let mut objects_freed = 0;

        {
            let mut objects_guard = objects.lock().unwrap();
            let current_memory = objects_guard
                .values()
                .map(|e| e.object.size())
                .sum::<usize>();

            // メモリ閾値チェック
            // Check memory threshold
            if current_memory < config.memory_threshold {
                return Ok(());
            }

            // 世代別GCを実行
            // Perform generational GC
            if config.enable_generational {
                Self::perform_generational_gc(
                    &mut objects_guard,
                    config,
                    &mut memory_freed,
                    &mut objects_freed,
                )?;
            } else {
                Self::perform_full_gc(&mut objects_guard, &mut memory_freed, &mut objects_freed)?;
            }
        }

        let duration = start_time.elapsed();

        // 統計を更新
        // Update statistics
        {
            let mut stats_guard = stats.write().unwrap();
            stats_guard.update_collection(
                duration,
                memory_freed,
                objects_freed,
                config.enable_generational,
            );
        }

        // 最後のGC時刻を更新
        // Update last GC time
        {
            let mut last_gc_guard = last_gc.lock().unwrap();
            *last_gc_guard = Instant::now();
        }

        Ok(())
    }

    /// 世代別GCを実行
    /// Perform generational GC
    fn perform_generational_gc(
        objects: &mut HashMap<u64, GCEntry>,
        config: &GCConfig,
        memory_freed: &mut usize,
        objects_freed: &mut u64,
    ) -> RusTorchResult<()> {
        let now = Instant::now();
        let mut to_remove = Vec::new();
        let mut to_promote = Vec::new();

        // 若い世代のGC
        // Young generation GC
        for (id, entry) in objects.iter_mut() {
            if entry.generation == 0 {
                // アクセス時刻をチェック
                // Check access time
                if entry.object.last_accessed().elapsed() > Duration::from_secs(60) {
                    *memory_freed += entry.object.size();
                    *objects_freed += 1;
                    to_remove.push(*id);
                } else if entry.object.survival_count() >= config.young_generation_limit {
                    // 古い世代に昇格
                    // Promote to old generation
                    to_promote.push(*id);
                }

                entry.object.increment_survival();
            }
        }

        // 古い世代に昇格
        // Promote to old generation
        for id in to_promote {
            if let Some(entry) = objects.get_mut(&id) {
                entry.generation = 1;
            }
        }

        // オブジェクトを削除
        // Remove objects
        for id in to_remove {
            objects.remove(&id);
        }

        // 古い世代のGC（必要に応じて）
        // Old generation GC (if needed)
        let old_memory: usize = objects
            .values()
            .filter(|e| e.generation == 1)
            .map(|e| e.object.size())
            .sum();

        if old_memory > config.memory_threshold / 2 {
            Self::perform_old_generation_gc(objects, memory_freed, objects_freed)?;
        }

        Ok(())
    }

    /// 古い世代のGCを実行
    /// Perform old generation GC
    fn perform_old_generation_gc(
        objects: &mut HashMap<u64, GCEntry>,
        memory_freed: &mut usize,
        objects_freed: &mut u64,
    ) -> RusTorchResult<()> {
        let mut to_remove = Vec::new();

        for (id, entry) in objects.iter() {
            if entry.generation == 1 {
                // 長時間未使用のオブジェクトを削除
                // Remove long-unused objects
                if entry.object.last_accessed().elapsed() > Duration::from_secs(300) {
                    *memory_freed += entry.object.size();
                    *objects_freed += 1;
                    to_remove.push(*id);
                }
            }
        }

        for id in to_remove {
            objects.remove(&id);
        }

        Ok(())
    }

    /// フルGCを実行
    /// Perform full GC
    fn perform_full_gc(
        objects: &mut HashMap<u64, GCEntry>,
        memory_freed: &mut usize,
        objects_freed: &mut u64,
    ) -> RusTorchResult<()> {
        let mut to_remove = Vec::new();

        for (id, entry) in objects.iter() {
            // 未使用オブジェクトを削除
            // Remove unused objects
            if entry.object.last_accessed().elapsed() > Duration::from_secs(120) {
                *memory_freed += entry.object.size();
                *objects_freed += 1;
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            objects.remove(&id);
        }

        Ok(())
    }
}

impl Drop for GarbageCollector {
    fn drop(&mut self) {
        // シャットダウンシグナルを設定
        // Set shutdown signal
        {
            let mut shutdown = self.shutdown_signal.lock().unwrap();
            *shutdown = true;
        }

        // スレッドの終了を待つ
        // Wait for thread to finish
        if let Some(handle) = self.gc_thread.take() {
            let _ = handle.join();
        }
    }
}

/// グローバルガベージコレクタ
/// Global garbage collector
static GLOBAL_GC: OnceLock<Arc<Mutex<GarbageCollector>>> = OnceLock::new();

/// グローバルGCを初期化
/// Initialize global GC
pub fn init_global_gc(config: GCConfig) -> RusTorchResult<()> {
    let mut gc = GarbageCollector::new(config);
    let _ = gc.start_auto_gc();
    let gc_arc = Arc::new(Mutex::new(gc));

    let _ = GLOBAL_GC.set(gc_arc);
    Ok(())
}

/// グローバルGCを取得
/// Get global GC
pub fn get_global_gc() -> Option<Arc<Mutex<GarbageCollector>>> {
    GLOBAL_GC.get().cloned()
}

/// オブジェクトを登録
/// Register object with global GC
pub fn gc_register(object: Box<dyn GCObject>) -> Option<u64> {
    get_global_gc()?.lock().ok()?.register_object(object).into()
}

/// オブジェクトの登録を解除
/// Unregister object from global GC
pub fn gc_unregister(id: u64) -> bool {
    if let Some(gc_arc) = get_global_gc() {
        if let Ok(gc) = gc_arc.lock() {
            return gc.unregister_object(id);
        }
    }
    false
}

/// 手動GCを実行
/// Perform manual GC
pub fn gc_collect() -> RusTorchResult<()> {
    if let Some(gc) = get_global_gc() {
        if let Ok(gc) = gc.lock() {
            return gc.collect();
        }
    }
    Ok(())
}

/// GC統計を取得
/// Get GC statistics
pub fn gc_stats() -> Option<GCStats> {
    if let Some(gc_arc) = get_global_gc() {
        if let Ok(gc) = gc_arc.lock() {
            return Some(gc.get_stats());
        }
    }
    None
}
