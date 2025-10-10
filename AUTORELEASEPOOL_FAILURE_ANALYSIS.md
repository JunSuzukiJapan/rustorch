# Autoreleasepool Failure Analysis

## Problem: objc::rc::autoreleasepool Not Working

### Symptoms
- Q projection: Success (6 buffers created)
- K projection: Success (6 buffers created)
- V projection: **CRASH** when creating 3rd buffer (18 buffers total accumulated)
- Error: `objc[9082]: bad weak table at 0x103069000`

### Root Cause
The `objc::rc::autoreleasepool(|| { ... })` macro from the `objc` crate is **NOT draining** Metal objects between matmul calls.

Debug output shows:
```
🆕 [BUFFER] Creating new buffer: 122880 bytes  // Q buffer 1
🆕 [BUFFER] Creating new buffer: 16777216 bytes  // Q buffer 2
🆕 [BUFFER] Creating new buffer: 122880 bytes  // Q buffer 3
🆕 [BUFFER] Creating new buffer: 4 bytes  // Q param 1
🆕 [BUFFER] Creating new buffer: 4 bytes  // Q param 2
🆕 [BUFFER] Creating new buffer: 4 bytes  // Q param 3
     ✓ Q projection complete  // <-- autoreleasepool should drain HERE
🆕 [BUFFER] Creating new buffer: 15360 bytes  // K buffer 1
...  // K creates 6 more buffers
     ✓ K projection complete  // <-- autoreleasepool should drain HERE
🆕 [BUFFER] Creating new buffer: 15360 bytes  // V buffer 1
🆕 [BUFFER] Creating new buffer: 2097152 bytes  // V buffer 2
🆕 [BUFFER] Creating new buffer: 122880 bytes  // V buffer 3 - CRASH!
```

**Total: 18 buffers created before any cleanup → Objective-C weak table corruption**

### Code Patterns Tested

#### Pattern 1: Inline autoreleasepool (FAILED)
```rust
objc::rc::autoreleasepool(|| executor.matmul_f32(&x_ln1, &q_weight_f32, &mut q_out, seq_len, d_model, q_out_dim))?;
```
**Result:** Still crashes - 18 buffers accumulate

#### Pattern 2: Explicit Result handling (FAILED)
```rust
let q_result = objc::rc::autoreleasepool(|| -> RusTorchResult<()> {
    executor.matmul_f32(&x_ln1, &q_weight_f32, &mut q_out, seq_len, d_model, q_out_dim)
});
q_result?;
```
**Result:** Still crashes - 18 buffers accumulate

### Why objc::rc::autoreleasepool Doesn't Work

The `objc` crate's `autoreleasepool` may not properly interact with Metal's memory management because:
1. Metal uses its own command buffer and command queue autorelease system
2. The Rust closure may not properly synchronize with Metal's async GPU operations
3. Metal buffers created in GPU code aren't properly tracked by Objective-C's autoreleasepool

### Solution Required

Need to directly control Metal's autoreleasepool lifecycle using:
1. `@autoreleasepool { }` blocks in Metal kernel code
2. Explicit Metal command buffer `commit()` and `waitUntilCompleted()`
3. Manual buffer cleanup in Rust after each matmul

## Attempted Solutions (All Failed)

### 1. Autoreleasepool after matmul completion
```rust
let result = self.matmul_f32_impl(a, b, c, m, n, k);
objc::rc::autoreleasepool(|| { });  // Drain pool
result
```
**Result:** Still crashes - 18 buffers accumulate

### 2. Autoreleasepool wrapping command buffer execution
```rust
objc::rc::autoreleasepool(|| {
    let command_buffer = self.command_queue.new_command_buffer();
    // ... execute ...
    command_buffer.wait_until_completed();
});
```
**Result:** Still crashes - buffers created outside pool

### 3. Explicit std::mem::drop() of buffers
```rust
std::mem::drop(a_buffer);
std::mem::drop(b_buffer);
// ...
```
**Result:** Segmentation fault - premature deallocation

### 4. Autoreleasepool wrapping entire matmul_impl
```rust
fn matmul_f32_impl(...) -> Result<()> {
    objc::rc::autoreleasepool(|| {
        self.matmul_f32_inner(...)
    })
}
```
**Result:** Still crashes - Metal objects not registered with pool

### 5. No autoreleasepool - rely on Rust Drop
```rust
fn matmul_f32_impl(...) -> Result<()> {
    // Create buffers, use them, let Rust drop them
}
```
**Result:** Segmentation fault after V projection - no cleanup happening

## Root Cause Analysis

The fundamental issue is that **`objc::rc::autoreleasepool` from Rust does NOT properly interact with Metal's Objective-C memory management**.

### Why objc::rc::autoreleasepool Doesn't Work

1. **Metal objects are registered with thread's "current" autoreleasepool** when created
2. When calling Metal APIs from Rust, there may be NO current autoreleasepool
3. `objc::rc::autoreleasepool` creates a new pool, but Metal APIs don't know about it
4. Metal buffers created inside `objc::rc::autoreleasepool` are registered with some OTHER pool (or none)
5. Result: Buffers accumulate in unknown pool → crash at 18+ buffers

### Evidence

- **With autoreleasepool:** "bad weak table" error (Objective-C memory corruption)
- **Without autoreleasepool:** Segmentation fault (complete lack of cleanup)
- **Explicit drop:** Segmentation fault (premature deallocation while Metal still using)
- **Pattern:** Always crashes after 12-18 Metal buffer allocations

## Conclusion

**The current architecture (Rust + metal-rs + objc crate) cannot reliably execute multiple sequential Metal matmul operations.**

The `metal-rs` crate and `objc` crate do not provide sufficient control over Objective-C autoreleasepool lifecycle to properly manage Metal object lifetimes.

## Recommended Solutions

### Short-term: Use hybrid-f32 backend
The hybrid-f32 backend (CPU-based) works reliably. Focus on that for production use.

### Medium-term: Objective-C bridge code
Write native Objective-C wrapper functions that:
1. Create `@autoreleasepool { }` blocks properly
2. Call Metal APIs from within those blocks
3. Expose simple C API to Rust
4. Manage all Metal object lifetimes correctly

### Long-term: Alternative GPU backend
Consider using:
- **Vulkan** via `vulkano` or `ash` crates (cross-platform)
- **CUDA** for NVIDIA GPUs
- **WebGPU** via `wgpu` crate (modern, cross-platform)

## Files Modified in Investigation

- `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/src/gpu/metal_kernels.rs`
  - Added buffer health checks
  - Added force_cleanup() method
  - Tried 5 different autoreleasepool strategies
  - All attempts failed

- `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/src/models/llama.rs`
  - Added/removed autoreleasepool wrappers around matmul calls
  - No solution found

## Status: BLOCKED

Metal GPU backend cannot proceed without fundamental architecture changes or native Objective-C bridge code.
