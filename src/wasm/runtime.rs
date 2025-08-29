//! WASM runtime utilities
//! WASMランタイムユーティリティ

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use web_sys::console;

/// Initialize WASM runtime
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn initialize_wasm_runtime() {
    // Set panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    console::log_1(&"RusTorch WASM runtime initialized".into());
}

/// Performance monitoring for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmPerformance {
    start_time: f64,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmPerformance {
    /// Create a new performance monitor
    /// 新しいパフォーマンスモニターを作成
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            start_time: js_sys::Date::now(),
        }
    }

    /// Start performance measurement
    #[wasm_bindgen]
    pub fn start(&mut self) {
        self.start_time = js_sys::Date::now();
    }

    /// Get elapsed time in milliseconds
    #[wasm_bindgen]
    pub fn elapsed(&self) -> f64 {
        js_sys::Date::now() - self.start_time
    }

    /// Log performance result
    #[wasm_bindgen]
    pub fn log(&self, operation_name: &str) {
        let elapsed = self.elapsed();
        let message = format!("{}: {:.2}ms", operation_name, elapsed);
        console::log_1(&message.into());
    }
}

/// Detect WASM runtime features
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn detect_wasm_features() -> js_sys::Object {
    let features = js_sys::Object::new();

    // Check for SIMD support (basic detection)
    let simd_supported =
        js_sys::Reflect::has(&js_sys::global(), &"WebAssembly".into()).unwrap_or(false);

    js_sys::Reflect::set(&features, &"simd".into(), &simd_supported.into()).unwrap();

    // Check for threads support
    let threads_supported =
        js_sys::Reflect::has(&js_sys::global(), &"SharedArrayBuffer".into()).unwrap_or(false);

    js_sys::Reflect::set(&features, &"threads".into(), &threads_supported.into()).unwrap();

    // Check available memory
    if let Ok(memory) = js_sys::Reflect::get(&js_sys::global(), &"WebAssembly".into()) {
        if let Ok(memory_obj) = js_sys::Reflect::get(&memory, &"Memory".into()) {
            js_sys::Reflect::set(&features, &"memory".into(), &true.into()).unwrap();
        }
    }

    features
}

/// WASM-specific logging utilities
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmLogger;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmLogger {
    /// Log info message
    #[wasm_bindgen]
    pub fn info(message: &str) {
        console::log_1(&format!("[INFO] {}", message).into());
    }

    /// Log warning message  
    #[wasm_bindgen]
    pub fn warn(message: &str) {
        console::warn_1(&format!("[WARN] {}", message).into());
    }

    /// Log error message
    #[wasm_bindgen]
    pub fn error(message: &str) {
        console::error_1(&format!("[ERROR] {}", message).into());
    }

    /// Log debug message
    #[wasm_bindgen]
    pub fn debug(message: &str) {
        console::debug_1(&format!("[DEBUG] {}", message).into());
    }
}
