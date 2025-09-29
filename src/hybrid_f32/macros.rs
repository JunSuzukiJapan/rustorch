//! hybrid_f32実験的マクロ定義
//! hybrid_f32 experimental macros

/// 実験的機能警告マクロ
/// Experimental feature warning macro
#[macro_export]
macro_rules! hybrid_f32_experimental {
    () => {
        #[cfg(feature = "hybrid-f32")]
        {
            // 実験的機能の使用を記録
            // Log experimental feature usage
            if std::env::var("RUSTORCH_DEBUG").unwrap_or_default() == "1" {
                eprintln!(
                    "[EXPERIMENTAL] hybrid_f32 feature used at {}:{}:{}",
                    file!(),
                    line!(),
                    column!()
                );
            }
        }
    };
    ($msg:expr) => {
        #[cfg(feature = "hybrid-f32")]
        {
            // カスタムメッセージ付き警告
            // Warning with custom message
            if std::env::var("RUSTORCH_DEBUG").unwrap_or_default() == "1" {
                eprintln!(
                    "[EXPERIMENTAL] hybrid_f32: {} at {}:{}:{}",
                    $msg,
                    file!(),
                    line!(),
                    column!()
                );
            }
        }
    };
}

/// 実験的機能の初期化チェック
/// Experimental feature initialization check
#[macro_export]
macro_rules! hybrid_f32_init_check {
    () => {
        #[cfg(feature = "hybrid-f32")]
        {
            static INIT_WARNING: std::sync::Once = std::sync::Once::new();
            INIT_WARNING.call_once(|| {
                if std::env::var("RUSTORCH_QUIET").unwrap_or_default() != "1" {
                    eprintln!("🧪 [EXPERIMENTAL] hybrid_f32 system initialized");
                    eprintln!("⚠️  This is experimental code - use with caution");
                    eprintln!("📝 Set RUSTORCH_QUIET=1 to suppress this message");
                }
            });
        }
    };
}

/// パフォーマンス測定マクロ
/// Performance measurement macro
#[macro_export]
macro_rules! hybrid_f32_perf_measure {
    ($name:expr, $code:block) => {
        #[cfg(feature = "hybrid-f32")]
        {
            let start = std::time::Instant::now();
            let result = $code;
            let duration = start.elapsed();
            
            if std::env::var("RUSTORCH_PERF").unwrap_or_default() == "1" {
                eprintln!("[PERF] {}: {:?}", $name, duration);
            }
            
            result
        }
        #[cfg(not(feature = "hybrid-f32"))]
        {
            $code
        }
    };
}

/// メモリ使用量チェックマクロ
/// Memory usage check macro
#[macro_export]
macro_rules! hybrid_f32_memory_check {
    ($tensor:expr, $operation:expr) => {
        #[cfg(feature = "hybrid-f32")]
        {
            if std::env::var("RUSTORCH_MEMORY_DEBUG").unwrap_or_default() == "1" {
                let memory_usage = $tensor.numel() * std::mem::size_of::<f32>();
                eprintln!(
                    "[MEMORY] {} - tensor size: {} elements, {} bytes",
                    $operation,
                    $tensor.numel(),
                    memory_usage
                );
            }
        }
    };
}

/// デバイス状態デバッグマクロ
/// Device state debug macro
#[macro_export]
macro_rules! hybrid_f32_device_debug {
    ($tensor:expr, $context:expr) => {
        #[cfg(feature = "hybrid-f32")]
        {
            if std::env::var("RUSTORCH_DEVICE_DEBUG").unwrap_or_default() == "1" {
                eprintln!(
                    "[DEVICE] {} - device: {:?}",
                    $context,
                    $tensor.device_state()
                );
            }
        }
    };
}

/// 条件付きコンパイル用ヘルパーマクロ
/// Conditional compilation helper macro
#[macro_export]
macro_rules! hybrid_f32_feature {
    ($code:block) => {
        #[cfg(feature = "hybrid-f32")]
        {
            $code
        }
    };
    (else $else_code:block) => {
        #[cfg(not(feature = "hybrid-f32"))]
        {
            $else_code
        }
    };
    ($code:block else $else_code:block) => {
        #[cfg(feature = "hybrid-f32")]
        {
            $code
        }
        #[cfg(not(feature = "hybrid-f32"))]
        {
            $else_code
        }
    };
}

/// 実験結果レポートマクロ
/// Experiment result report macro
#[macro_export]
macro_rules! hybrid_f32_report {
    ($results:expr) => {
        #[cfg(feature = "hybrid-f32")]
        {
            if std::env::var("RUSTORCH_REPORT").unwrap_or_default() == "1" {
                eprintln!("📊 [EXPERIMENT REPORT]");
                eprintln!("   Performance gain: {:.2}%", $results.performance_gain);
                eprintln!("   Memory efficiency: {:.2}%", $results.memory_efficiency);
                eprintln!("   Accuracy maintained: {}", $results.accuracy_maintained);
                eprintln!("   Compatible devices: {:?}", $results.device_compatibility);
            }
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_experimental_macro() {
        // 基本的なマクロテスト
        crate::hybrid_f32_experimental!();
        crate::hybrid_f32_experimental!("test message");
    }

    #[test]
    fn test_feature_macro() {
        let result = crate::hybrid_f32_feature!({
            "hybrid_f32 enabled"
        } else {
            "hybrid_f32 disabled"
        });

        #[cfg(feature = "hybrid-f32")]
        assert_eq!(result, "hybrid_f32 enabled");
        
        #[cfg(not(feature = "hybrid-f32"))]
        assert_eq!(result, "hybrid_f32 disabled");
    }
}