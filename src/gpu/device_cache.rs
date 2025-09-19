//! Device Initialization Cache
//! デバイス初期化キャッシュ
//!
//! This module provides caching for expensive device initialization operations
//! 高コストなデバイス初期化操作のキャッシュを提供

use crate::gpu::DeviceType;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Device initialization result
/// デバイス初期化結果
#[derive(Debug, Clone)]
pub enum DeviceStatus {
    Available,
    Unavailable(String), // Error message
    Initializing,
}

/// Cached device information
/// キャッシュされたデバイス情報
#[derive(Debug, Clone)]
pub struct CachedDevice {
    pub status: DeviceStatus,
    pub last_checked: Instant,
    pub initialization_time: Option<Duration>,
}

impl CachedDevice {
    pub fn new(status: DeviceStatus) -> Self {
        Self {
            status,
            last_checked: Instant::now(),
            initialization_time: None,
        }
    }

    pub fn with_init_time(mut self, duration: Duration) -> Self {
        self.initialization_time = Some(duration);
        self
    }

    /// Check if cache entry is still valid (within 30 seconds)
    /// キャッシュエントリが有効かチェック（30秒以内）
    pub fn is_valid(&self) -> bool {
        self.last_checked.elapsed() < Duration::from_secs(30)
    }
}

/// Device availability cache with lazy initialization
/// 遅延初期化付きデバイス可用性キャッシュ
#[derive(Debug, Clone)]
pub struct DeviceCache {
    cache: Arc<Mutex<HashMap<DeviceType, CachedDevice>>>,
}

impl DeviceCache {
    /// Get global device cache instance
    /// グローバルデバイスキャッシュインスタンスを取得
    pub fn global() -> &'static DeviceCache {
        static CACHE: OnceLock<DeviceCache> = OnceLock::new();
        CACHE.get_or_init(|| DeviceCache::new())
    }

    /// Create new device cache
    /// 新しいデバイスキャッシュを作成
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Check if device is available (with caching)
    /// デバイスが利用可能かチェック（キャッシュ付き）
    pub fn is_device_available(&self, device: &DeviceType) -> bool {
        // Check cache first
        if let Some(cached) = self.get_cached_status(device) {
            if cached.is_valid() {
                return matches!(cached.status, DeviceStatus::Available);
            }
        }

        // Cache miss or expired - check device availability
        let start = Instant::now();
        let is_available = self.check_device_availability_impl(device);
        let check_duration = start.elapsed();

        // Update cache
        let status = if is_available {
            DeviceStatus::Available
        } else {
            DeviceStatus::Unavailable("Device check failed".to_string())
        };

        let cached_device = CachedDevice::new(status).with_init_time(check_duration);
        self.update_cache(device.clone(), cached_device);

        is_available
    }

    /// Get cached device status
    /// キャッシュされたデバイス状態を取得
    pub fn get_cached_status(&self, device: &DeviceType) -> Option<CachedDevice> {
        let cache = self.cache.lock().ok()?;
        cache.get(device).cloned()
    }

    /// Update device cache
    /// デバイスキャッシュを更新
    pub fn update_cache(&self, device: DeviceType, cached_device: CachedDevice) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(device, cached_device);
        }
    }

    /// Clear expired cache entries
    /// 期限切れキャッシュエントリをクリア
    pub fn cleanup_expired(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.retain(|_, cached| cached.is_valid());
        }
    }

    /// Get cache statistics
    /// キャッシュ統計を取得
    pub fn get_stats(&self) -> CacheStats {
        let cache = self.cache.lock().unwrap();
        let total_entries = cache.len();
        let valid_entries = cache.values().filter(|c| c.is_valid()).count();
        let available_devices = cache.values()
            .filter(|c| c.is_valid() && matches!(c.status, DeviceStatus::Available))
            .count();

        CacheStats {
            total_entries,
            valid_entries,
            available_devices,
            cache_hit_rate: if total_entries > 0 {
                valid_entries as f64 / total_entries as f64
            } else {
                0.0
            },
        }
    }

    /// Actual device availability check implementation
    /// 実際のデバイス可用性チェック実装
    fn check_device_availability_impl(&self, device: &DeviceType) -> bool {
        match device {
            DeviceType::Cpu => true, // CPU always available

            #[cfg(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback"))]
            DeviceType::CoreML(_) => {
                use crate::backends::DeviceManager;
                DeviceManager::is_coreml_available()
            }

            #[cfg(not(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback")))]
            DeviceType::CoreML(_) => false,

            #[cfg(feature = "metal")]
            DeviceType::Metal(_) => {
                use crate::backends::DeviceManager;
                DeviceManager::is_metal_available()
            }

            #[cfg(not(feature = "metal"))]
            DeviceType::Metal(_) => false,

            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                // TODO: Implement CUDA availability check
                false
            }

            #[cfg(not(feature = "cuda"))]
            DeviceType::Cuda(_) => false,

            #[cfg(feature = "opencl")]
            DeviceType::OpenCL(_) => {
                // TODO: Implement OpenCL availability check
                false
            }

            #[cfg(not(feature = "opencl"))]
            DeviceType::OpenCL(_) => false,

            DeviceType::Auto => true, // Auto always available (fallback logic)
        }
    }

    /// Warmup device cache by checking all common devices
    /// 一般的なデバイスをすべてチェックしてキャッシュをウォームアップ
    pub fn warmup(&self) {
        let devices_to_check = vec![
            DeviceType::Cpu,
            DeviceType::CoreML(0),
            DeviceType::Metal(0),
            DeviceType::Cuda(0),
        ];

        for device in devices_to_check {
            self.is_device_available(&device);
        }
    }
}

impl Default for DeviceCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache performance statistics
/// キャッシュパフォーマンス統計
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub valid_entries: usize,
    pub available_devices: usize,
    pub cache_hit_rate: f64,
}

/// CoreML-specific initialization cache
/// CoreML固有の初期化キャッシュ
#[derive(Debug)]
pub struct CoreMLCache {
    is_initialized: Arc<Mutex<bool>>,
    initialization_result: Arc<Mutex<Option<Result<(), String>>>>,
}

impl CoreMLCache {
    /// Get global CoreML cache instance
    /// グローバルCoreMLキャッシュインスタンスを取得
    pub fn global() -> &'static CoreMLCache {
        static COREML_CACHE: OnceLock<CoreMLCache> = OnceLock::new();
        COREML_CACHE.get_or_init(|| CoreMLCache::new())
    }

    pub fn new() -> Self {
        Self {
            is_initialized: Arc::new(Mutex::new(false)),
            initialization_result: Arc::new(Mutex::new(None)),
        }
    }

    /// Initialize CoreML with caching
    /// キャッシュ付きCoreML初期化
    pub fn ensure_initialized(&self) -> Result<(), String> {
        // Check if already initialized
        if let Ok(initialized) = self.is_initialized.lock() {
            if *initialized {
                // Return cached result
                if let Ok(result) = self.initialization_result.lock() {
                    if let Some(ref cached_result) = *result {
                        return cached_result.clone();
                    }
                }
            }
        }

        // Perform initialization
        let result = self.initialize_coreml();

        // Cache the result
        if let (Ok(mut initialized), Ok(mut cached_result)) =
            (self.is_initialized.lock(), self.initialization_result.lock()) {
            *initialized = true;
            *cached_result = Some(result.clone());
        }

        result
    }

    /// Actual CoreML initialization implementation
    /// 実際のCoreML初期化実装
    fn initialize_coreml(&self) -> Result<(), String> {
        #[cfg(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback"))]
        {
            // Check if CoreML is available on this platform
            if !cfg!(target_os = "macos") {
                return Err("CoreML is only available on macOS".to_string());
            }

            // Perform any necessary CoreML setup here
            // For now, just check availability
            use crate::backends::DeviceManager;
            if DeviceManager::is_coreml_available() {
                Ok(())
            } else {
                Err("CoreML not available on this system".to_string())
            }
        }

        #[cfg(not(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback")))]
        {
            Err("CoreML features not enabled".to_string())
        }
    }

    /// Reset initialization state (for testing)
    /// 初期化状態をリセット（テスト用）
    pub fn reset(&self) {
        if let (Ok(mut initialized), Ok(mut result)) =
            (self.is_initialized.lock(), self.initialization_result.lock()) {
            *initialized = false;
            *result = None;
        }
    }
}

impl Default for CoreMLCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_cache_basic() {
        let cache = DeviceCache::new();

        // CPU should always be available
        assert!(cache.is_device_available(&DeviceType::Cpu));

        // Should be cached now
        let cached = cache.get_cached_status(&DeviceType::Cpu);
        assert!(cached.is_some());
        assert!(matches!(cached.unwrap().status, DeviceStatus::Available));
    }

    #[test]
    fn test_cache_expiration() {
        let cache = DeviceCache::new();

        // Manually insert an expired entry
        let expired_device = CachedDevice {
            status: DeviceStatus::Available,
            last_checked: Instant::now() - Duration::from_secs(60), // 1 minute ago
            initialization_time: None,
        };

        cache.update_cache(DeviceType::Cpu, expired_device);

        // Should re-check because cache is expired
        assert!(cache.is_device_available(&DeviceType::Cpu));
    }

    #[test]
    fn test_coreml_cache() {
        let cache = CoreMLCache::new();

        // First call should initialize
        let result1 = cache.ensure_initialized();

        // Second call should use cached result
        let result2 = cache.ensure_initialized();

        // Results should be consistent
        match (result1, result2) {
            (Ok(()), Ok(())) => {},
            (Err(e1), Err(e2)) => assert_eq!(e1, e2),
            _ => panic!("Inconsistent cache results"),
        }
    }
}