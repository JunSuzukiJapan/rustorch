//! Gradient context management for automatic differentiation
//! 自動微分のための勾配コンテキスト管理

use std::cell::RefCell;
use std::sync::Mutex;

/// Global gradient context state
/// グローバル勾配コンテキスト状態
#[derive(Debug, Clone)]
pub struct GradientContext {
    pub grad_enabled: bool,
    pub anomaly_detection: bool,
}

impl Default for GradientContext {
    fn default() -> Self {
        Self {
            grad_enabled: true,
            anomaly_detection: false,
        }
    }
}

thread_local! {
    static GRADIENT_CONTEXT: RefCell<GradientContext> = RefCell::new(GradientContext::default());
}

static GLOBAL_GRADIENT_CONTEXT: Mutex<GradientContext> = Mutex::new(GradientContext {
    grad_enabled: true,
    anomaly_detection: false,
});

/// Check if gradient computation is currently enabled
/// 勾配計算が現在有効かどうかを確認
pub fn is_grad_enabled() -> bool {
    GRADIENT_CONTEXT.with(|ctx| ctx.borrow().grad_enabled)
}

/// Check if anomaly detection is enabled
/// 異常検出が有効かどうかを確認
pub fn is_anomaly_detection_enabled() -> bool {
    GRADIENT_CONTEXT.with(|ctx| ctx.borrow().anomaly_detection)
}

/// Set gradient computation state
/// 勾配計算状態を設定
pub fn set_grad_enabled(enabled: bool) {
    GRADIENT_CONTEXT.with(|ctx| {
        ctx.borrow_mut().grad_enabled = enabled;
    });
}

/// Set anomaly detection state
/// 異常検出状態を設定
pub fn set_anomaly_detection(enabled: bool) {
    GRADIENT_CONTEXT.with(|ctx| {
        ctx.borrow_mut().anomaly_detection = enabled;
    });
}

/// RAII guard for temporarily disabling gradient computation
/// 勾配計算を一時的に無効にするRAIIガード
pub struct NoGradGuard {
    previous_state: bool,
}

impl NoGradGuard {
    /// Create a new NoGradGuard, disabling gradient computation
    /// 新しいNoGradGuardを作成し、勾配計算を無効化
    pub fn new() -> Self {
        let previous_state = is_grad_enabled();
        set_grad_enabled(false);
        Self { previous_state }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.previous_state);
    }
}

/// RAII guard for temporarily enabling gradient computation
/// 勾配計算を一時的に有効にするRAIIガード
pub struct EnableGradGuard {
    previous_state: bool,
}

impl EnableGradGuard {
    /// Create a new EnableGradGuard, enabling gradient computation
    /// 新しいEnableGradGuardを作成し、勾配計算を有効化
    pub fn new() -> Self {
        let previous_state = is_grad_enabled();
        set_grad_enabled(true);
        Self { previous_state }
    }
}

impl Drop for EnableGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.previous_state);
    }
}

/// RAII guard for temporarily enabling anomaly detection
/// 異常検出を一時的に有効にするRAIIガード
pub struct AnomalyDetectionGuard {
    previous_state: bool,
}

impl AnomalyDetectionGuard {
    /// Create a new AnomalyDetectionGuard, enabling anomaly detection
    /// 新しいAnomalyDetectionGuardを作成し、異常検出を有効化
    pub fn new() -> Self {
        let previous_state = is_anomaly_detection_enabled();
        set_anomaly_detection(true);
        Self { previous_state }
    }
}

impl Drop for AnomalyDetectionGuard {
    fn drop(&mut self) {
        set_anomaly_detection(self.previous_state);
    }
}

/// Convenience function for no_grad context
/// no_gradコンテキスト用の便利関数
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}

/// Convenience function for enable_grad context
/// enable_gradコンテキスト用の便利関数
pub fn enable_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = EnableGradGuard::new();
    f()
}

/// Convenience function for anomaly detection context
/// 異常検出コンテキスト用の便利関数
pub fn detect_anomaly<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = AnomalyDetectionGuard::new();
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_grad_guard() {
        assert!(is_grad_enabled()); // Default state
        
        {
            let _guard = NoGradGuard::new();
            assert!(!is_grad_enabled()); // Disabled inside guard
        }
        
        assert!(is_grad_enabled()); // Restored after guard drop
    }

    #[test]
    fn test_enable_grad_guard() {
        set_grad_enabled(false);
        assert!(!is_grad_enabled());
        
        {
            let _guard = EnableGradGuard::new();
            assert!(is_grad_enabled()); // Enabled inside guard
        }
        
        assert!(!is_grad_enabled()); // Restored after guard drop
        set_grad_enabled(true); // Reset for other tests
    }

    #[test]
    fn test_anomaly_detection_guard() {
        assert!(!is_anomaly_detection_enabled()); // Default state
        
        {
            let _guard = AnomalyDetectionGuard::new();
            assert!(is_anomaly_detection_enabled()); // Enabled inside guard
        }
        
        assert!(!is_anomaly_detection_enabled()); // Restored after guard drop
    }

    #[test]
    fn test_convenience_functions() {
        let result = no_grad(|| {
            assert!(!is_grad_enabled());
            42
        });
        assert_eq!(result, 42);
        assert!(is_grad_enabled());

        set_grad_enabled(false);
        let result = enable_grad(|| {
            assert!(is_grad_enabled());
            24
        });
        assert_eq!(result, 24);
        assert!(!is_grad_enabled());
        set_grad_enabled(true);

        let result = detect_anomaly(|| {
            assert!(is_anomaly_detection_enabled());
            "test"
        });
        assert_eq!(result, "test");
        assert!(!is_anomaly_detection_enabled());
    }
}