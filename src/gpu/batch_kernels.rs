//! Batch processing kernel wrappers for Metal GPU
//! Metal GPU用バッチ処理カーネルラッパー
//!
//! Phase 3 implementation of parallel batch processing using Metal compute shaders.
//! Metalコンピュートシェーダーを使用した並列バッチ処理のPhase 3実装。

use crate::error::{RusTorchError, RusTorchResult};

#[cfg(feature = "metal")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions,
    MTLSize,
};

/// RMS Normalization with batch support
/// バッチ対応RMS正規化
///
/// Applies RMS normalization across batch dimension:
/// output[b,s,h] = input[b,s,h] * weight[h] / RMS(input[b,s,:])
///
/// # Arguments
/// * `input` - Input tensor [batch_size, seq_len, hidden_dim]
/// * `weight` - RMS norm weights [hidden_dim]
/// * `output` - Output tensor [batch_size, seq_len, hidden_dim]
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_dim` - Hidden dimension
/// * `eps` - Epsilon for numerical stability (typically 1e-5 or 1e-6)
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(RusTorchError)` if Metal operations fail
#[cfg(feature = "metal")]
pub fn metal_rms_norm_batch_f32(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    eps: f32,
) -> RusTorchResult<()> {
    // Validate input sizes
    let expected_size = batch_size * seq_len * hidden_dim;
    if input.len() != expected_size || output.len() != expected_size {
        return Err(RusTorchError::tensor_op(&format!(
            "RMS Norm batch: size mismatch. Expected {}, got input={}, output={}",
            expected_size,
            input.len(),
            output.len()
        )));
    }

    if weight.len() != hidden_dim {
        return Err(RusTorchError::tensor_op(&format!(
            "RMS Norm batch: weight size mismatch. Expected {}, got {}",
            hidden_dim,
            weight.len()
        )));
    }

    // Get Metal device
    let device = Device::system_default()
        .ok_or_else(|| RusTorchError::gpu("Metal device not available"))?;

    // Create command queue
    let queue = device.new_command_queue();

    // Compile shader
    let shader_source = include_str!("metal_shaders.metal");
    let library = device
        .new_library_with_source(shader_source, &CompileOptions::new())
        .map_err(|e| RusTorchError::gpu(&format!("Failed to compile shader: {:?}", e)))?;

    // Get kernel function
    let kernel_name = "rms_norm_f32";
    let function = library
        .get_function(kernel_name, None)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to get kernel function: {:?}", e)))?;

    // Create pipeline state
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to create pipeline: {:?}", e)))?;

    // Create buffers
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        (input.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let weight_buffer = device.new_buffer_with_data(
        weight.as_ptr() as *const _,
        (weight.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (output.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create command buffer and encoder
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Set pipeline and buffers
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input_buffer), 0);
    encoder.set_buffer(1, Some(&weight_buffer), 0);
    encoder.set_buffer(2, Some(&output_buffer), 0);

    // Set parameters
    let batch_size_u32 = batch_size as u32;
    let seq_len_u32 = seq_len as u32;
    let hidden_dim_u32 = hidden_dim as u32;

    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &batch_size_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &seq_len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &hidden_dim_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        std::mem::size_of::<f32>() as u64,
        &eps as *const f32 as *const _,
    );

    // Configure thread groups (3D grid for batch parallelism)
    // Each threadgroup processes one (batch, seq) row with 256 threads for reduction
    let threads_per_threadgroup = MTLSize {
        width: 256,  // Fixed size for tree reduction
        height: 1,
        depth: 1,
    };

    let threadgroups_per_grid = MTLSize {
        width: 1,  // One threadgroup per row
        height: seq_len as u64,
        depth: batch_size as u64,
    };

    encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    encoder.end_encoding();

    // Execute and wait
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Copy results back
    let output_ptr = output_buffer.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(output_ptr, output.as_mut_ptr(), output.len());
    }

    Ok(())
}

#[cfg(not(feature = "metal"))]
pub fn metal_rms_norm_batch_f32(
    _input: &[f32],
    _weight: &[f32],
    _output: &mut [f32],
    _batch_size: usize,
    _seq_len: usize,
    _hidden_dim: usize,
    _eps: f32,
) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// RoPE (Rotary Position Embedding) with batch support
/// バッチ対応RoPE（回転位置埋め込み）
///
/// Applies rotary position embedding in-place to Q or K tensors.
/// Q/Kテンソルにin-placeで回転位置埋め込みを適用。
///
/// # Arguments
/// * `x` - Input/output tensor [batch_size, seq_len, num_heads, head_dim] (modified in-place)
/// * `batch_size` - Batch size
/// * `start_pos` - Starting position for RoPE (for KV caching)
/// * `seq_len` - Sequence length
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
/// * `rope_theta` - RoPE theta parameter (typically 10000.0)
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(RusTorchError)` if Metal operations fail
#[cfg(feature = "metal")]
pub fn metal_rope_batch_f32(
    x: &mut [f32],
    batch_size: usize,
    start_pos: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) -> RusTorchResult<()> {
    // Validate input size
    let expected_size = batch_size * seq_len * num_heads * head_dim;
    if x.len() != expected_size {
        return Err(RusTorchError::tensor_op(&format!(
            "RoPE batch: size mismatch. Expected {}, got {}",
            expected_size,
            x.len()
        )));
    }

    if head_dim % 2 != 0 {
        return Err(RusTorchError::tensor_op(&format!(
            "RoPE batch: head_dim must be even, got {}",
            head_dim
        )));
    }

    // Get Metal device
    let device = Device::system_default()
        .ok_or_else(|| RusTorchError::gpu("Metal device not available"))?;

    let queue = device.new_command_queue();

    // Compile shader
    let shader_source = include_str!("metal_shaders.metal");
    let library = device
        .new_library_with_source(shader_source, &CompileOptions::new())
        .map_err(|e| RusTorchError::gpu(&format!("Failed to compile shader: {:?}", e)))?;

    // Get kernel function
    let kernel_name = "apply_rope_f32";
    let function = library
        .get_function(kernel_name, None)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to get kernel function: {:?}", e)))?;

    // Create pipeline state
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to create pipeline: {:?}", e)))?;

    // Create buffer (in-place modification)
    let x_buffer = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create command buffer and encoder
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Set pipeline and buffer
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&x_buffer), 0);

    // Set parameters
    let batch_size_u32 = batch_size as u32;
    let start_pos_u32 = start_pos as u32;
    let seq_len_u32 = seq_len as u32;
    let num_heads_u32 = num_heads as u32;
    let head_dim_u32 = head_dim as u32;

    encoder.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &batch_size_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &start_pos_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &seq_len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &num_heads_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &head_dim_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        std::mem::size_of::<f32>() as u64,
        &rope_theta as *const f32 as *const _,
    );

    // Configure thread groups (4D grid for batch + position + head + dim_pair)
    // Note: Metal supports max 3D, so we flatten some dimensions
    let dim_pairs = head_dim / 2;

    let threads_per_threadgroup = MTLSize {
        width: 8,  // dim_pairs per threadgroup
        height: 4, // heads per threadgroup
        depth: 1,  // pos per threadgroup
    };

    let threadgroups_per_grid = MTLSize {
        width: ((dim_pairs as u64 + threads_per_threadgroup.width - 1)
            / threads_per_threadgroup.width)
            .max(1),
        height: ((num_heads as u64 + threads_per_threadgroup.height - 1)
            / threads_per_threadgroup.height)
            .max(1),
        depth: (batch_size * seq_len) as u64, // Flatten batch and seq into depth
    };

    encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    encoder.end_encoding();

    // Execute and wait
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Copy results back (in-place modification)
    let x_ptr = x_buffer.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(x_ptr, x.as_mut_ptr(), x.len());
    }

    Ok(())
}

#[cfg(not(feature = "metal"))]
pub fn metal_rope_batch_f32(
    _x: &mut [f32],
    _batch_size: usize,
    _start_pos: usize,
    _seq_len: usize,
    _num_heads: usize,
    _head_dim: usize,
    _rope_theta: f32,
) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "metal")]
    fn test_rms_norm_batch_basic() {
        let batch_size = 2;
        let seq_len = 3;
        let hidden_dim = 4;
        let eps = 1e-5;

        // Create test input: all ones
        let input = vec![1.0f32; batch_size * seq_len * hidden_dim];
        let weight = vec![1.0f32; hidden_dim];
        let mut output = vec![0.0f32; batch_size * seq_len * hidden_dim];

        let result =
            metal_rms_norm_batch_f32(&input, &weight, &mut output, batch_size, seq_len, hidden_dim, eps);

        assert!(result.is_ok(), "RMS Norm batch should succeed: {:?}", result.err());

        // For input of all ones, RMS = sqrt(sum(1^2) / N) = sqrt(N/N) = 1
        // So output should be input / 1 * weight = input * weight = 1.0
        for val in &output {
            assert!(
                (val - 1.0).abs() < 1e-4,
                "Expected ~1.0, got {}",
                val
            );
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_rope_batch_basic() {
        let batch_size = 1;
        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 4;
        let start_pos = 0;
        let rope_theta = 10000.0;

        // Create test input
        let mut x = vec![1.0f32; batch_size * seq_len * num_heads * head_dim];

        let result = metal_rope_batch_f32(
            &mut x,
            batch_size,
            start_pos,
            seq_len,
            num_heads,
            head_dim,
            rope_theta,
        );

        assert!(result.is_ok(), "RoPE batch should succeed: {:?}", result.err());

        // RoPE applies rotation, so output should differ from input
        let all_same = x.iter().all(|&v| (v - 1.0).abs() < 1e-6);
        assert!(!all_same, "RoPE should modify input values");
    }

    #[test]
    fn test_rms_norm_batch_size_validation() {
        let batch_size = 2;
        let seq_len = 3;
        let hidden_dim = 4;

        let input = vec![1.0f32; 10]; // Wrong size
        let weight = vec![1.0f32; hidden_dim];
        let mut output = vec![0.0f32; batch_size * seq_len * hidden_dim];

        let result =
            metal_rms_norm_batch_f32(&input, &weight, &mut output, batch_size, seq_len, hidden_dim, 1e-5);

        #[cfg(feature = "metal")]
        assert!(result.is_err(), "Should fail with size mismatch");

        #[cfg(not(feature = "metal"))]
        assert!(result.is_err(), "Should fail without Metal");
    }

    #[test]
    fn test_rope_batch_odd_head_dim() {
        let batch_size = 1;
        let seq_len = 1;
        let num_heads = 1;
        let head_dim = 3; // Odd number - should fail

        let mut x = vec![1.0f32; batch_size * seq_len * num_heads * head_dim];

        let result = metal_rope_batch_f32(&mut x, batch_size, 0, seq_len, num_heads, head_dim, 10000.0);

        #[cfg(feature = "metal")]
        assert!(result.is_err(), "Should fail with odd head_dim");

        #[cfg(not(feature = "metal"))]
        assert!(result.is_err(), "Should fail without Metal");
    }
}
