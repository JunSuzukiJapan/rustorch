// Objective-C Bridge for Metal with proper RAII management
// MetalのためのObjective-Cブリッジ（適切なRAII管理付き）
//
// CRITICAL: All Metal objects MUST be wrapped in Rust structs with Drop trait
// 重要：すべてのMetalオブジェクトはDropトレイトを持つRust構造体でラップする必要がある
//
// This ensures proper cleanup of Objective-C objects when Rust values go out of scope
// これによりRust値がスコープを抜けたときにObjective-Cオブジェクトが確実にクリーンアップされる

use std::ffi::c_void;
use std::ptr::NonNull;

// ============================================================================
// Raw Objective-C FFI declarations
// ============================================================================

extern "C" {
    // Autoreleasepool management
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);

    // NSObject memory management
    fn objc_retain(obj: *mut c_void) -> *mut c_void;
    fn objc_release(obj: *mut c_void);

    // Metal device creation
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;

    // Metal device methods (called via objc_msgSend)
    // We'll use objc crate for message sending to avoid unsafe msgSend
}

// ============================================================================
// AutoreleasePool RAII wrapper
// ============================================================================

/// RAII wrapper for Objective-C autoreleasepool
/// Objective-Cのautoreleasepool用RAIIラッパー
pub struct AutoreleasePool {
    pool: *mut c_void,
}

impl AutoreleasePool {
    /// Create a new autoreleasepool
    /// 新しいautoreleasepoolを作成
    #[inline]
    pub fn new() -> Self {
        unsafe {
            let pool = objc_autoreleasePoolPush();
            AutoreleasePool { pool }
        }
    }

    /// Execute function inside autoreleasepool
    /// autoreleasepool内で関数を実行
    #[inline]
    pub fn with<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _pool = AutoreleasePool::new();
        f()
        // pool is automatically popped when _pool is dropped
    }
}

impl Drop for AutoreleasePool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            objc_autoreleasePoolPop(self.pool);
        }
    }
}

// ============================================================================
// MetalDevice RAII wrapper
// ============================================================================

/// RAII wrapper for MTLDevice
/// MTLDevice用RAIIラッパー
///
/// Holds a retained reference to MTLDevice
/// MTLDeviceへの保持参照を保持
pub struct MetalDevice {
    device: NonNull<c_void>,
}

impl MetalDevice {
    /// Create system default Metal device
    /// システムデフォルトのMetalデバイスを作成
    pub fn create_system_default() -> Option<Self> {
        unsafe {
            let device_ptr = MTLCreateSystemDefaultDevice();
            if device_ptr.is_null() {
                return None;
            }

            // MTLCreateSystemDefaultDevice returns a retained object, so we own it
            // MTLCreateSystemDefaultDeviceは保持されたオブジェクトを返すので、所有権を持つ
            Some(MetalDevice {
                device: NonNull::new_unchecked(device_ptr),
            })
        }
    }

    /// Get raw device pointer for passing to Metal APIs
    /// Metal APIに渡すための生のデバイスポインタを取得
    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.device.as_ptr()
    }

    /// Retain the device (increase reference count)
    /// デバイスを保持（参照カウントを増やす）
    #[inline]
    pub fn retain(&self) -> Self {
        unsafe {
            let retained_ptr = objc_retain(self.device.as_ptr());
            MetalDevice {
                device: NonNull::new_unchecked(retained_ptr),
            }
        }
    }
}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        unsafe {
            objc_release(self.device.as_ptr());
        }
    }
}

// Send + Sync: MetalDevice can be safely shared across threads
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

// ============================================================================
// MetalBuffer RAII wrapper
// ============================================================================

/// RAII wrapper for MTLBuffer
/// MTLBuffer用RAIIラッパー
///
/// Automatically releases buffer when dropped
/// ドロップ時に自動的にバッファを解放
pub struct MetalBuffer {
    buffer: NonNull<c_void>,
}

impl MetalBuffer {
    /// Create from raw pointer (takes ownership)
    /// 生ポインタから作成（所有権を取得）
    ///
    /// # Safety
    /// Pointer must be a valid MTLBuffer with +1 retain count
    /// ポインタは+1保持カウントを持つ有効なMTLBufferでなければならない
    #[inline]
    pub unsafe fn from_retained_ptr(ptr: *mut c_void) -> Option<Self> {
        NonNull::new(ptr).map(|buffer| MetalBuffer { buffer })
    }

    /// Get raw buffer pointer
    /// 生バッファポインタを取得
    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.buffer.as_ptr()
    }

    /// Retain the buffer (increase reference count)
    /// バッファを保持（参照カウントを増やす）
    #[inline]
    pub fn retain(&self) -> Self {
        unsafe {
            let retained_ptr = objc_retain(self.buffer.as_ptr());
            MetalBuffer {
                buffer: NonNull::new_unchecked(retained_ptr),
            }
        }
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        unsafe {
            objc_release(self.buffer.as_ptr());
        }
    }
}

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

// ============================================================================
// MetalCommandQueue RAII wrapper
// ============================================================================

/// RAII wrapper for MTLCommandQueue
/// MTLCommandQueue用RAIIラッパー
pub struct MetalCommandQueue {
    queue: NonNull<c_void>,
}

impl MetalCommandQueue {
    /// Create from raw pointer (takes ownership)
    /// 生ポインタから作成（所有権を取得）
    #[inline]
    pub unsafe fn from_retained_ptr(ptr: *mut c_void) -> Option<Self> {
        NonNull::new(ptr).map(|queue| MetalCommandQueue { queue })
    }

    /// Get raw queue pointer
    /// 生キューポインタを取得
    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.queue.as_ptr()
    }
}

impl Drop for MetalCommandQueue {
    fn drop(&mut self) {
        unsafe {
            objc_release(self.queue.as_ptr());
        }
    }
}

unsafe impl Send for MetalCommandQueue {}
unsafe impl Sync for MetalCommandQueue {}

// ============================================================================
// MetalCommandBuffer RAII wrapper
// ============================================================================

/// RAII wrapper for MTLCommandBuffer
/// MTLCommandBuffer用RAIIラッパー
///
/// CRITICAL: CommandBuffers are autoreleased by Metal, but we need to ensure
/// they complete before being deallocated
/// 重要：CommandBufferはMetalによって自動解放されるが、割り当て解除前に
/// 完了を確認する必要がある
pub struct MetalCommandBuffer {
    buffer: NonNull<c_void>,
    completed: bool,
}

impl MetalCommandBuffer {
    /// Create from raw pointer (does NOT take ownership - autoreleased)
    /// 生ポインタから作成（所有権を取らない - 自動解放される）
    ///
    /// # Safety
    /// Pointer must be a valid MTLCommandBuffer
    #[inline]
    pub unsafe fn from_ptr(ptr: *mut c_void) -> Option<Self> {
        NonNull::new(ptr).map(|buffer| MetalCommandBuffer {
            buffer,
            completed: false,
        })
    }

    /// Get raw buffer pointer
    /// 生バッファポインタを取得
    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.buffer.as_ptr()
    }

    /// Mark as completed (called after waitUntilCompleted)
    /// 完了としてマーク（waitUntilCompleted後に呼び出す）
    #[inline]
    pub fn mark_completed(&mut self) {
        self.completed = true;
    }
}

impl Drop for MetalCommandBuffer {
    fn drop(&mut self) {
        // CommandBuffer is autoreleased, we don't call release
        // But we can log if it wasn't completed
        if !self.completed {
            eprintln!("⚠️ [METAL] CommandBuffer dropped before completion!");
        }
    }
}

unsafe impl Send for MetalCommandBuffer {}

// ============================================================================
// MetalComputeEncoder RAII wrapper
// ============================================================================

/// RAII wrapper for MTLComputeCommandEncoder
/// MTLComputeCommandEncoder用RAIIラッパー
pub struct MetalComputeEncoder {
    encoder: NonNull<c_void>,
    ended: bool,
}

impl MetalComputeEncoder {
    /// Create from raw pointer (does NOT take ownership - autoreleased)
    /// 生ポインタから作成（所有権を取らない - 自動解放される）
    #[inline]
    pub unsafe fn from_ptr(ptr: *mut c_void) -> Option<Self> {
        NonNull::new(ptr).map(|encoder| MetalComputeEncoder {
            encoder,
            ended: false,
        })
    }

    /// Get raw encoder pointer
    /// 生エンコーダポインタを取得
    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.encoder.as_ptr()
    }

    /// Mark as ended (called after endEncoding)
    /// 終了としてマーク（endEncoding後に呼び出す）
    #[inline]
    pub fn mark_ended(&mut self) {
        self.ended = true;
    }
}

impl Drop for MetalComputeEncoder {
    fn drop(&mut self) {
        // Encoder is autoreleased, we don't call release
        // But ensure it was ended properly
        if !self.ended {
            eprintln!("⚠️ [METAL] ComputeEncoder dropped before endEncoding!");
        }
    }
}

unsafe impl Send for MetalComputeEncoder {}

// ============================================================================
// MetalLibrary RAII wrapper
// ============================================================================

/// RAII wrapper for MTLLibrary
/// MTLLibrary用RAIIラッパー
pub struct MetalLibrary {
    library: NonNull<c_void>,
}

impl MetalLibrary {
    /// Create from raw pointer (takes ownership)
    /// 生ポインタから作成（所有権を取得）
    #[inline]
    pub unsafe fn from_retained_ptr(ptr: *mut c_void) -> Option<Self> {
        NonNull::new(ptr).map(|library| MetalLibrary { library })
    }

    /// Get raw library pointer
    /// 生ライブラリポインタを取得
    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.library.as_ptr()
    }
}

impl Drop for MetalLibrary {
    fn drop(&mut self) {
        unsafe {
            objc_release(self.library.as_ptr());
        }
    }
}

unsafe impl Send for MetalLibrary {}
unsafe impl Sync for MetalLibrary {}

// ============================================================================
// MetalFunction RAII wrapper
// ============================================================================

/// RAII wrapper for MTLFunction
/// MTLFunction用RAIIラッパー
pub struct MetalFunction {
    function: NonNull<c_void>,
}

impl MetalFunction {
    /// Create from raw pointer (does NOT take ownership - retained by library)
    /// 生ポインタから作成（所有権を取らない - ライブラリによって保持される）
    ///
    /// We retain it to ensure it outlives the library if needed
    /// 必要に応じてライブラリより長く存在することを保証するために保持する
    #[inline]
    pub unsafe fn from_ptr(ptr: *mut c_void) -> Option<Self> {
        NonNull::new(ptr).map(|function| {
            // Retain the function to ensure proper lifecycle
            objc_retain(function.as_ptr());
            MetalFunction { function }
        })
    }

    /// Get raw function pointer
    /// 生関数ポインタを取得
    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.function.as_ptr()
    }
}

impl Drop for MetalFunction {
    fn drop(&mut self) {
        unsafe {
            objc_release(self.function.as_ptr());
        }
    }
}

unsafe impl Send for MetalFunction {}
unsafe impl Sync for MetalFunction {}

// ============================================================================
// MetalPipelineState RAII wrapper
// ============================================================================

/// RAII wrapper for MTLComputePipelineState
/// MTLComputePipelineState用RAIIラッパー
pub struct MetalPipelineState {
    state: NonNull<c_void>,
}

impl MetalPipelineState {
    /// Create from raw pointer (takes ownership)
    /// 生ポインタから作成（所有権を取得）
    #[inline]
    pub unsafe fn from_retained_ptr(ptr: *mut c_void) -> Option<Self> {
        NonNull::new(ptr).map(|state| MetalPipelineState { state })
    }

    /// Get raw state pointer
    /// 生ステートポインタを取得
    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.state.as_ptr()
    }
}

impl Drop for MetalPipelineState {
    fn drop(&mut self) {
        unsafe {
            objc_release(self.state.as_ptr());
        }
    }
}

unsafe impl Send for MetalPipelineState {}
unsafe impl Sync for MetalPipelineState {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoreleasepool_creation() {
        let pool = AutoreleasePool::new();
        drop(pool);
        // Should not crash
    }

    #[test]
    fn test_autoreleasepool_with() {
        let result = AutoreleasePool::with(|| {
            42
        });
        assert_eq!(result, 42);
    }

    #[test]
    #[ignore] // Only run on macOS with Metal support
    fn test_metal_device_creation() {
        if let Some(device) = MetalDevice::create_system_default() {
            assert!(!device.as_ptr().is_null());
        }
    }
}
