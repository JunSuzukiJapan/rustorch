// Metal Matrix Multiplication with proper RAII management
// 適切なRAII管理を持つMetal行列乗算
//
// This module implements matmul using our custom RAII wrappers
// to ensure proper cleanup of Metal objects
// このモジュールはカスタムRAIIラッパーを使用してmatmulを実装し、
// Metalオブジェクトの適切なクリーンアップを保証する

use crate::error::{RusTorchError, RusTorchResult};
use crate::gpu::metal_objc_bridge::{AutoreleasePool, MetalDevice};

// Use external objc crate for Objective-C interop
extern crate objc;
use objc::runtime::{Class, Object};
use objc::{msg_send, sel, sel_impl};

use std::ffi::c_void;
use std::ptr;

/// Metal matrix multiplication executor with RAII management
/// RAII管理を持つMetal行列乗算エグゼキュータ
pub struct MetalMatMulExecutor {
    device: MetalDevice,
    queue: *mut Object,
    pipeline: *mut Object,
}

impl MetalMatMulExecutor {
    /// Create new executor
    /// 新しいエグゼキュータを作成
    pub fn new() -> RusTorchResult<Self> {
        // Create device inside autoreleasepool
        let device = AutoreleasePool::with(|| {
            MetalDevice::create_system_default()
                .ok_or_else(|| RusTorchError::gpu("Metal device not available"))
        })?;

        // Create command queue
        let queue = unsafe {
            let device_obj = device.as_ptr() as *mut Object;
            let queue: *mut Object = msg_send![device_obj, newCommandQueue];
            if queue.is_null() {
                return Err(RusTorchError::gpu("Failed to create command queue"));
            }
            // Retain the queue since we're storing it
            let _: () = msg_send![queue, retain];
            queue
        };

        // Compile Metal shader
        let pipeline = Self::compile_matmul_shader(&device)?;

        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    /// Compile matmul shader
    /// matmulシェーダーをコンパイル
    fn compile_matmul_shader(device: &MetalDevice) -> RusTorchResult<*mut Object> {
        let shader_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* M [[buffer(3)]],
    device const uint* N [[buffer(4)]],
    device const uint* K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = *M;
    uint n = *N;
    uint k = *K;

    uint row = gid.y;
    uint col = gid.x;

    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (uint i = 0; i < k; i++) {
        sum += A[row * k + i] * B[i * n + col];
    }

    C[row * n + col] = sum;
}
"#;

        unsafe {
            AutoreleasePool::with(|| {
                let device_obj = device.as_ptr() as *mut Object;

                // Create NSString for shader source
                let ns_string_class = Class::get("NSString").unwrap();
                let shader_ns_string: *mut Object = msg_send![ns_string_class, alloc];
                let shader_ns_string: *mut Object = msg_send![
                    shader_ns_string,
                    initWithBytes: shader_source.as_ptr()
                    length: shader_source.len()
                    encoding: 4u64  // UTF8
                ];

                // Compile library
                let mut error: *mut Object = ptr::null_mut();
                let library: *mut Object = msg_send![
                    device_obj,
                    newLibraryWithSource: shader_ns_string
                    options: ptr::null::<Object>()
                    error: &mut error as *mut _
                ];

                if library.is_null() {
                    let error_desc: *mut Object = msg_send![error, localizedDescription];
                    let error_cstr: *const i8 = msg_send![error_desc, UTF8String];
                    let error_str = std::ffi::CStr::from_ptr(error_cstr).to_string_lossy();
                    return Err(RusTorchError::gpu(format!("Shader compilation failed: {}", error_str)));
                }

                // Get function
                let function_name_class = Class::get("NSString").unwrap();
                let function_name: *mut Object = msg_send![function_name_class, alloc];
                let function_name: *mut Object = msg_send![
                    function_name,
                    initWithBytes: b"matmul_f32\0".as_ptr()
                    length: 10u64
                    encoding: 4u64
                ];

                let function: *mut Object = msg_send![library, newFunctionWithName: function_name];
                if function.is_null() {
                    return Err(RusTorchError::gpu("Failed to get function from library"));
                }

                // Create pipeline
                let mut error: *mut Object = ptr::null_mut();
                let pipeline: *mut Object = msg_send![
                    device_obj,
                    newComputePipelineStateWithFunction: function
                    error: &mut error as *mut _
                ];

                if pipeline.is_null() {
                    return Err(RusTorchError::gpu("Failed to create pipeline state"));
                }

                // Retain pipeline since we're storing it
                let _: () = msg_send![pipeline, retain];

                Ok(pipeline)
            })
        }
    }

    /// Execute matrix multiplication
    /// 行列乗算を実行
    ///
    /// # Safety
    /// Slices must be valid and sized correctly: a[m*k], b[k*n], c[m*n]
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // Validate dimensions
        if a.len() != m * k {
            return Err(RusTorchError::InvalidOperation(format!(
                "Matrix A size mismatch: expected {}, got {}",
                m * k,
                a.len()
            )));
        }
        if b.len() != k * n {
            return Err(RusTorchError::InvalidOperation(format!(
                "Matrix B size mismatch: expected {}, got {}",
                k * n,
                b.len()
            )));
        }
        if c.len() != m * n {
            return Err(RusTorchError::InvalidOperation(format!(
                "Matrix C size mismatch: expected {}, got {}",
                m * n,
                c.len()
            )));
        }

        // Execute inside autoreleasepool to clean up Metal objects
        AutoreleasePool::with(|| {
            unsafe {
                let device_obj = self.device.as_ptr() as *mut Object;

                // Create buffers inside autoreleasepool
                let a_size = a.len() * std::mem::size_of::<f32>();
                let b_size = b.len() * std::mem::size_of::<f32>();
                let c_size = c.len() * std::mem::size_of::<f32>();
                let param_size = std::mem::size_of::<u32>();

                let options = 0u64; // MTLResourceStorageModeShared

                let a_buffer: *mut Object = msg_send![device_obj, newBufferWithLength:a_size as u64 options:options];
                let b_buffer: *mut Object = msg_send![device_obj, newBufferWithLength:b_size as u64 options:options];
                let c_buffer: *mut Object = msg_send![device_obj, newBufferWithLength:c_size as u64 options:options];
                let m_buffer: *mut Object = msg_send![device_obj, newBufferWithLength:param_size as u64 options:options];
                let n_buffer: *mut Object = msg_send![device_obj, newBufferWithLength:param_size as u64 options:options];
                let k_buffer: *mut Object = msg_send![device_obj, newBufferWithLength:param_size as u64 options:options];

                if a_buffer.is_null() || b_buffer.is_null() || c_buffer.is_null() ||
                   m_buffer.is_null() || n_buffer.is_null() || k_buffer.is_null() {
                    return Err(RusTorchError::gpu("Failed to create Metal buffers"));
                }

                // Copy data to buffers
                let a_contents: *mut c_void = msg_send![a_buffer, contents];
                let b_contents: *mut c_void = msg_send![b_buffer, contents];
                let m_contents: *mut c_void = msg_send![m_buffer, contents];
                let n_contents: *mut c_void = msg_send![n_buffer, contents];
                let k_contents: *mut c_void = msg_send![k_buffer, contents];

                ptr::copy_nonoverlapping(a.as_ptr(), a_contents as *mut f32, a.len());
                ptr::copy_nonoverlapping(b.as_ptr(), b_contents as *mut f32, b.len());

                let m_u32 = m as u32;
                let n_u32 = n as u32;
                let k_u32 = k as u32;
                ptr::copy_nonoverlapping(&m_u32 as *const u32, m_contents as *mut u32, 1);
                ptr::copy_nonoverlapping(&n_u32 as *const u32, n_contents as *mut u32, 1);
                ptr::copy_nonoverlapping(&k_u32 as *const u32, k_contents as *mut u32, 1);

                // Create command buffer and encoder inside autoreleasepool
                let command_buffer: *mut Object = msg_send![self.queue, commandBuffer];
                if command_buffer.is_null() {
                    return Err(RusTorchError::gpu("Failed to create command buffer"));
                }

                let compute_encoder: *mut Object = msg_send![command_buffer, computeCommandEncoder];
                if compute_encoder.is_null() {
                    return Err(RusTorchError::gpu("Failed to create compute encoder"));
                }

                // Set pipeline and buffers
                let _: () = msg_send![compute_encoder, setComputePipelineState: self.pipeline];
                let _: () = msg_send![compute_encoder, setBuffer:a_buffer offset:0u64 atIndex:0u64];
                let _: () = msg_send![compute_encoder, setBuffer:b_buffer offset:0u64 atIndex:1u64];
                let _: () = msg_send![compute_encoder, setBuffer:c_buffer offset:0u64 atIndex:2u64];
                let _: () = msg_send![compute_encoder, setBuffer:m_buffer offset:0u64 atIndex:3u64];
                let _: () = msg_send![compute_encoder, setBuffer:n_buffer offset:0u64 atIndex:4u64];
                let _: () = msg_send![compute_encoder, setBuffer:k_buffer offset:0u64 atIndex:5u64];

                // Dispatch
                let threads_per_threadgroup = (16u64, 16u64, 1u64);
                let threadgroups_per_grid = (
                    ((n as u64 + 15) / 16),
                    ((m as u64 + 15) / 16),
                    1u64
                );

                #[repr(C)]
                struct MTLSize {
                    width: u64,
                    height: u64,
                    depth: u64,
                }

                let tpt = MTLSize {
                    width: threads_per_threadgroup.0,
                    height: threads_per_threadgroup.1,
                    depth: threads_per_threadgroup.2,
                };

                let tpg = MTLSize {
                    width: threadgroups_per_grid.0,
                    height: threadgroups_per_grid.1,
                    depth: threadgroups_per_grid.2,
                };

                let _: () = msg_send![compute_encoder, dispatchThreadgroups:tpg threadsPerThreadgroup:tpt];
                let _: () = msg_send![compute_encoder, endEncoding];

                // Commit and wait
                let _: () = msg_send![command_buffer, commit];
                let _: () = msg_send![command_buffer, waitUntilCompleted];

                // Copy result
                let c_contents: *mut c_void = msg_send![c_buffer, contents];
                ptr::copy_nonoverlapping(c_contents as *const f32, c.as_mut_ptr(), c.len());

                // Buffers and command buffer will be auto-released when autoreleasepool exits
                Ok(())
            }
        })
    }
}

impl Drop for MetalMatMulExecutor {
    fn drop(&mut self) {
        unsafe {
            if !self.queue.is_null() {
                let _: () = msg_send![self.queue, release];
            }
            if !self.pipeline.is_null() {
                let _: () = msg_send![self.pipeline, release];
            }
        }
    }
}

unsafe impl Send for MetalMatMulExecutor {}
unsafe impl Sync for MetalMatMulExecutor {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run on macOS with Metal
    fn test_matmul_2x2() {
        let executor = MetalMatMulExecutor::new().unwrap();

        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2
        let mut c = vec![0.0f32; 4]; // 2x2

        executor.matmul_f32(&a, &b, &mut c, 2, 2, 2).unwrap();

        // Expected: [19, 22, 43, 50]
        assert!((c[0] - 19.0).abs() < 0.001);
        assert!((c[1] - 22.0).abs() < 0.001);
        assert!((c[2] - 43.0).abs() < 0.001);
        assert!((c[3] - 50.0).abs() < 0.001);
    }
}
