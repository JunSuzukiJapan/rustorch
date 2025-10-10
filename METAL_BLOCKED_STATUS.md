# Metal GPU Backend - BLOCKED Status

## Current Status: NOT FUNCTIONAL

The Metal GPU backend for RusTorch Llama inference is **blocked** due to fundamental Objective-C/Rust interop issues that cannot be resolved without major architectural changes.

## Problem Summary

Sequential Metal buffer allocations (Q, K, V projections) cause Objective-C memory corruption crashes after 12-18 buffer creations:

- **Error 1:** `objc: bad weak table` - Objective-C weak reference table corruption
- **Error 2:** `objc: Method cache corrupted` - Command buffer dealloc corruption
- **Error 3:** `Segmentation fault: 11` - Memory access violation

## Root Cause

The `metal-rs` crate (Rust bindings for Metal) does NOT properly integrate with Objective-C's autoreleasepool memory management system. Metal objects accumulate in an unknown autoreleasepool and are never released until the pool is drained by the OS (if ever).

### Technical Details

1. **Metal API creates autorelease objects**: `MTLDevice.newBuffer()` returns objects registered with thread's "current" autoreleasepool
2. **No current pool in Rust**: When calling from Rust, there may be no active autoreleasepool
3. **objc crate insufficient**: `objc::rc::autoreleasepool` creates a pool, but Metal objects don't register with it
4. **Direct FFI also fails**: Even `objc_autoreleasePoolPush/Pop` via FFI doesn't fix the issue
5. **Result**: 18+ Metal objects → Objective-C runtime corruption → crash

## Attempted Solutions (All Failed)

### 1. `objc::rc::autoreleasepool` wrapping
- Wrapped individual matmuls: **FAILED**
- Wrapped command buffer execution: **FAILED**
- Wrapped entire matmul_impl: **FAILED**

### 2. Direct Objective-C FFI
- Used `objc_autoreleasePoolPush/Pop` directly: **FAILED**
- RAII wrapper for autoreleasepool: **FAILED**

### 3. Explicit memory management
- `std::mem::drop()` of buffers: **Segmentation fault**
- Manual buffer cleanup: **Still crashed**

### 4. Synchronization strategies
- `commit()` + `wait_until_completed()`: **Insufficient**
- Force cleanup after layers: **Too late, crash happens within single layer**

## Why This is a Fundamental Problem

The issue is NOT a bug in our code. It's an **architectural incompatibility**:

- **metal-rs design**: Assumes Objective-C runtime manages object lifetimes
- **Rust execution**: No implicit autoreleasepool on threads
- **Gap**: No mechanism to bridge the two memory management systems properly

## Requirements for Solution

To fix this properly requires one of:

### Option A: Native Objective-C Bridge (Medium Effort, High Success)
Write Objective-C wrapper functions that:
```objective-c
// metal_bridge.m
void* matmul_with_autoreleasepool(/* params */) {
    @autoreleasepool {
        // Create Metal device
        // Create buffers
        // Execute compute shader
        // Copy results
        // Everything auto-released here
    }
    return result;
}
```

Expose via C ABI to Rust. This gives us FULL control over autoreleasepool lifecycle.

### Option B: Alternative GPU Backend (High Effort, High Success)
Replace Metal with cross-platform solution:
- **wgpu/WebGPU**: Modern, cross-platform, excellent Rust support
- **Vulkan**: via `vulkano` or `ash` crates
- **CUDA**: For NVIDIA GPUs (already have some support)

### Option C: Fix metal-rs Upstream (Low Control, Uncertain Timeline)
Contribute fixes to `metal-rs` crate to properly handle autoreleasepool. Requires:
- Deep Objective-C runtime knowledge
- Coordination with crate maintainers
- May take months to stabilize

## Recommendation

**Short-term (Now)**: Document Metal backend as non-functional, disable in production code

**Medium-term (1-2 months)**: Implement Option A (Objective-C bridge) for macOS-specific optimizations

**Long-term (3-6 months)**: Migrate to Option B (wgpu/WebGPU) for cross-platform GPU support

## Files Affected

- `src/gpu/metal_kernels.rs` - Core Metal implementation (broken)
- `src/gpu/objc_bridge.rs` - Attempted FFI solution (insufficient)
- `src/models/llama.rs` - Uses Metal executor (crashes on Q,K,V)

## Testing Evidence

All tests with Q8_0 quantization model:
- ✅ **1 token, 1 layer, Q projection only**: Success
- ✅ **1 token, 1 layer, Q+K projections**: Success
- ❌ **1 token, 1 layer, Q+K+V projections**: CRASH (12-18 buffers)
- ❌ **5 tokens, 22 layers**: Immediate crash

Pattern: Crashes when 12-18 Metal buffers exist simultaneously.

## Date: 2025-10-10
## Status: BLOCKED - Architectural Issue
## Priority: Medium (macOS GPU acceleration desirable but not critical)
