// Direct Objective-C FFI for autoreleasepool management
// Metal APIが正しくautoreleasepoolと連携するように、直接Objective-Cを呼び出す

use std::ffi::c_void;

// Opaque pointer to NSAutoreleasePool
#[repr(C)]
pub struct NSAutoreleasePool {
    _private: [u8; 0],
}

extern "C" {
    // Direct Objective-C runtime calls
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

/// RAII wrapper for Objective-C autoreleasepool
/// Drop時に自動的にpoolをpopする
pub struct AutoreleasePool {
    pool: *mut c_void,
}

impl AutoreleasePool {
    /// Create a new autoreleasepool
    /// 新しいautoreleasepoolを作成
    pub fn new() -> Self {
        unsafe {
            let pool = objc_autoreleasePoolPush();
            AutoreleasePool { pool }
        }
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        unsafe {
            objc_autoreleasePoolPop(self.pool);
        }
    }
}

/// Execute a function inside a new autoreleasepool
/// 新しいautoreleasepoolの内側で関数を実行
pub fn with_autoreleasepool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _pool = AutoreleasePool::new();
    f()
    // pool is automatically popped when _pool goes out of scope
}
