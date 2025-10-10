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

/// Compute attention scores: Q @ K^T / sqrt(head_dim) with batch support
///
/// # Arguments
/// * `q` - Query tensor [batch_size, q_len, num_heads, head_dim]
/// * `k` - Key tensor [batch_size, kv_len, num_heads, head_dim]
/// * `scores` - Output scores [batch_size, num_heads, q_len, kv_len]
/// * `batch_size` - Batch size
/// * `q_len` - Query sequence length
/// * `kv_len` - Key/Value sequence length
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
#[cfg(feature = "metal")]
pub fn metal_attention_scores_batch_f32(
    q: &[f32],
    k: &[f32],
    scores: &mut [f32],
    batch_size: usize,
    q_len: usize,
    kv_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> RusTorchResult<()> {
    // Input validation
    let q_expected = batch_size * q_len * num_heads * head_dim;
    let k_expected = batch_size * kv_len * num_heads * head_dim;
    let scores_expected = batch_size * num_heads * q_len * kv_len;

    if q.len() != q_expected {
        return Err(RusTorchError::tensor_op(&format!(
            "Attention scores: Q size mismatch. Expected {}, got {}",
            q_expected,
            q.len()
        )));
    }
    if k.len() != k_expected {
        return Err(RusTorchError::tensor_op(&format!(
            "Attention scores: K size mismatch. Expected {}, got {}",
            k_expected,
            k.len()
        )));
    }
    if scores.len() != scores_expected {
        return Err(RusTorchError::tensor_op(&format!(
            "Attention scores: output size mismatch. Expected {}, got {}",
            scores_expected,
            scores.len()
        )));
    }

    // Get Metal device and queue
    let device = Device::system_default()
        .ok_or_else(|| RusTorchError::gpu("Metal device not available"))?;
    let queue = device.new_command_queue();

    // Compile shader
    let shader_source = include_str!("metal_shaders.metal");
    let library = device
        .new_library_with_source(shader_source, &CompileOptions::new())
        .map_err(|e| RusTorchError::gpu(&format!("Failed to compile shader: {:?}", e)))?;

    let function = library
        .get_function("compute_attention_scores_batch_f32", None)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to get kernel function: {:?}", e)))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to create pipeline: {:?}", e)))?;

    // Create buffers
    let q_buffer = device.new_buffer_with_data(
        q.as_ptr() as *const _,
        (q.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let k_buffer = device.new_buffer_with_data(
        k.as_ptr() as *const _,
        (k.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scores_buffer = device.new_buffer(
        (scores.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Encode commands
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&q_buffer), 0);
    encoder.set_buffer(1, Some(&k_buffer), 0);
    encoder.set_buffer(2, Some(&scores_buffer), 0);

    // Set parameters
    let batch_size_u32 = batch_size as u32;
    let q_len_u32 = q_len as u32;
    let kv_len_u32 = kv_len as u32;
    let num_heads_u32 = num_heads as u32;
    let head_dim_u32 = head_dim as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &batch_size_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &q_len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &kv_len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &num_heads_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &head_dim_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(8, std::mem::size_of::<f32>() as u64, &scale as *const f32 as *const _);

    // Configure thread groups
    // Grid: (batch*q_len, kv_len, num_heads)
    let threads_per_threadgroup = MTLSize {
        width: 8,  // batch*q_len dimension
        height: 8, // kv_len dimension
        depth: 1,
    };

    let threadgroups_per_grid = MTLSize {
        width: ((batch_size * q_len) as u64 + 7) / 8,
        height: ((kv_len as u64 + 7) / 8).max(1),
        depth: num_heads as u64,
    };

    encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    encoder.end_encoding();

    // Execute
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Copy results
    let scores_ptr = scores_buffer.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(scores_ptr, scores.as_mut_ptr(), scores.len());
    }

    Ok(())
}

#[cfg(not(feature = "metal"))]
pub fn metal_attention_scores_batch_f32(
    _q: &[f32],
    _k: &[f32],
    _scores: &mut [f32],
    _batch_size: usize,
    _q_len: usize,
    _kv_len: usize,
    _num_heads: usize,
    _head_dim: usize,
) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// Apply softmax to attention scores with batch support
///
/// Computes: scores[i] = exp(scores[i]) / sum(exp(scores))
///
/// # Arguments
/// * `scores` - Attention scores [batch_size, num_heads, q_len, kv_len] (in-place)
/// * `batch_size` - Batch size
/// * `num_heads` - Number of attention heads
/// * `q_len` - Query sequence length
/// * `kv_len` - Key/Value sequence length
#[cfg(feature = "metal")]
pub fn metal_softmax_batch_f32(
    scores: &mut [f32],
    batch_size: usize,
    num_heads: usize,
    q_len: usize,
    kv_len: usize,
) -> RusTorchResult<()> {
    // Input validation
    let expected_size = batch_size * num_heads * q_len * kv_len;
    if scores.len() != expected_size {
        return Err(RusTorchError::tensor_op(&format!(
            "Softmax batch: size mismatch. Expected {}, got {}",
            expected_size,
            scores.len()
        )));
    }

    // Get Metal device and queue
    let device = Device::system_default()
        .ok_or_else(|| RusTorchError::gpu("Metal device not available"))?;
    let queue = device.new_command_queue();

    // Compile shader
    let shader_source = include_str!("metal_shaders.metal");
    let library = device
        .new_library_with_source(shader_source, &CompileOptions::new())
        .map_err(|e| RusTorchError::gpu(&format!("Failed to compile shader: {:?}", e)))?;

    // Create buffer
    let scores_buffer = device.new_buffer_with_data(
        scores.as_ptr() as *const _,
        (scores.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();

    // Step 1: Find max per row
    let max_function = library
        .get_function("softmax_max_batch_f32", None)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to get max function: {:?}", e)))?;

    let max_pipeline = device
        .new_compute_pipeline_state_with_function(&max_function)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to create max pipeline: {:?}", e)))?;

    // Create temp buffer for max values (one per row)
    let num_rows = batch_size * num_heads * q_len;
    let max_buffer = device.new_buffer(
        (num_rows * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&max_pipeline);
    encoder.set_buffer(0, Some(&scores_buffer), 0);
    encoder.set_buffer(1, Some(&max_buffer), 0);

    let batch_size_u32 = batch_size as u32;
    let num_heads_u32 = num_heads as u32;
    let q_len_u32 = q_len as u32;
    let kv_len_u32 = kv_len as u32;

    encoder.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &batch_size_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &q_len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &kv_len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &num_heads_u32 as *const u32 as *const _,
    );

    // Grid: (batch, q_len, num_heads)
    let threads_per_threadgroup = MTLSize {
        width: 1,
        height: 4,
        depth: 2,
    };
    let threadgroups_per_grid = MTLSize {
        width: batch_size as u64,
        height: ((q_len as u64 + 3) / 4).max(1),
        depth: ((num_heads as u64 + 1) / 2).max(1),
    };

    encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    encoder.end_encoding();

    // Step 2: Compute exp and sum
    // Use legacy kernel by treating [batch_size, num_heads, q_len, kv_len] as [(batch*num_heads), q_len, kv_len]
    let exp_function = library
        .get_function("softmax_exp_sum_f32", None)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to get exp function: {:?}", e)))?;

    let exp_pipeline = device
        .new_compute_pipeline_state_with_function(&exp_function)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to create exp pipeline: {:?}", e)))?;

    let sum_buffer = device.new_buffer(
        (num_rows * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder2 = command_buffer.new_compute_command_encoder();
    encoder2.set_compute_pipeline_state(&exp_pipeline);
    encoder2.set_buffer(0, Some(&scores_buffer), 0);
    encoder2.set_buffer(1, Some(&max_buffer), 0);
    encoder2.set_buffer(2, Some(&sum_buffer), 0);

    // Legacy kernel expects: q_len, kv_len, num_heads
    // We treat batch*num_heads as "num_heads"
    let effective_heads_u32 = (batch_size * num_heads) as u32;
    encoder2.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &q_len_u32 as *const u32 as *const _,
    );
    encoder2.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &kv_len_u32 as *const u32 as *const _,
    );
    encoder2.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &effective_heads_u32 as *const u32 as *const _,
    );

    // Grid: (q_len, batch*num_heads) for legacy 2D kernel
    let exp_threads_per_threadgroup = MTLSize {
        width: 4,
        height: 4,
        depth: 1,
    };
    let exp_threadgroups_per_grid = MTLSize {
        width: ((q_len as u64 + 3) / 4).max(1),
        height: ((batch_size * num_heads) as u64 + 3) / 4,
        depth: 1,
    };

    encoder2.dispatch_thread_groups(exp_threadgroups_per_grid, exp_threads_per_threadgroup);
    encoder2.end_encoding();

    // Step 3: Normalize
    // Use legacy kernel by treating [batch_size, num_heads, q_len, kv_len] as [(batch*num_heads), q_len, kv_len]
    let norm_function = library
        .get_function("softmax_normalize_f32", None)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to get normalize function: {:?}", e)))?;

    let norm_pipeline = device
        .new_compute_pipeline_state_with_function(&norm_function)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to create normalize pipeline: {:?}", e)))?;

    let encoder3 = command_buffer.new_compute_command_encoder();
    encoder3.set_compute_pipeline_state(&norm_pipeline);
    encoder3.set_buffer(0, Some(&scores_buffer), 0);
    encoder3.set_buffer(1, Some(&sum_buffer), 0);

    // Legacy kernel expects: q_len, kv_len, num_heads
    // We treat batch*num_heads as "num_heads"
    encoder3.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &q_len_u32 as *const u32 as *const _,
    );
    encoder3.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &kv_len_u32 as *const u32 as *const _,
    );
    encoder3.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &effective_heads_u32 as *const u32 as *const _,
    );

    // Grid: (q_len, batch*num_heads) for legacy 2D kernel
    encoder3.dispatch_thread_groups(exp_threadgroups_per_grid, exp_threads_per_threadgroup);
    encoder3.end_encoding();

    // Execute
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Copy results back
    let scores_ptr = scores_buffer.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(scores_ptr, scores.as_mut_ptr(), scores.len());
    }

    Ok(())
}

#[cfg(not(feature = "metal"))]
pub fn metal_softmax_batch_f32(
    _scores: &mut [f32],
    _batch_size: usize,
    _num_heads: usize,
    _q_len: usize,
    _kv_len: usize,
) -> RusTorchResult<()> {
    Err(RusTorchError::UnsupportedDevice(
        "Metal not available".to_string(),
    ))
}

/// Apply attention to values: output = scores @ V with batch support
///
/// # Arguments
/// * `scores` - Attention scores [batch_size, num_heads, q_len, kv_len]
/// * `v` - Value tensor [batch_size, kv_len, num_heads, head_dim]
/// * `output` - Output [batch_size, q_len, num_heads, head_dim]
/// * `batch_size` - Batch size
/// * `q_len` - Query sequence length
/// * `kv_len` - Key/Value sequence length
/// * `num_heads` - Number of heads
/// * `head_dim` - Dimension per head
#[cfg(feature = "metal")]
pub fn metal_apply_attention_batch_f32(
    scores: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch_size: usize,
    q_len: usize,
    kv_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> RusTorchResult<()> {
    // Input validation
    let scores_expected = batch_size * num_heads * q_len * kv_len;
    let v_expected = batch_size * kv_len * num_heads * head_dim;
    let output_expected = batch_size * q_len * num_heads * head_dim;

    if scores.len() != scores_expected {
        return Err(RusTorchError::tensor_op(&format!(
            "Apply attention: scores size mismatch. Expected {}, got {}",
            scores_expected,
            scores.len()
        )));
    }
    if v.len() != v_expected {
        return Err(RusTorchError::tensor_op(&format!(
            "Apply attention: V size mismatch. Expected {}, got {}",
            v_expected,
            v.len()
        )));
    }
    if output.len() != output_expected {
        return Err(RusTorchError::tensor_op(&format!(
            "Apply attention: output size mismatch. Expected {}, got {}",
            output_expected,
            output.len()
        )));
    }

    // Get Metal device and queue
    let device = Device::system_default()
        .ok_or_else(|| RusTorchError::gpu("Metal device not available"))?;
    let queue = device.new_command_queue();

    // Compile shader
    let shader_source = include_str!("metal_shaders.metal");
    let library = device
        .new_library_with_source(shader_source, &CompileOptions::new())
        .map_err(|e| RusTorchError::gpu(&format!("Failed to compile shader: {:?}", e)))?;

    let function = library
        .get_function("apply_attention_to_values_batch_f32", None)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to get kernel function: {:?}", e)))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| RusTorchError::gpu(&format!("Failed to create pipeline: {:?}", e)))?;

    // Create buffers
    let scores_buffer = device.new_buffer_with_data(
        scores.as_ptr() as *const _,
        (scores.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let v_buffer = device.new_buffer_with_data(
        v.as_ptr() as *const _,
        (v.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (output.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Encode commands
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&scores_buffer), 0);
    encoder.set_buffer(1, Some(&v_buffer), 0);
    encoder.set_buffer(2, Some(&output_buffer), 0);

    // Set parameters
    let batch_size_u32 = batch_size as u32;
    let q_len_u32 = q_len as u32;
    let kv_len_u32 = kv_len as u32;
    let num_heads_u32 = num_heads as u32;
    let head_dim_u32 = head_dim as u32;

    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &batch_size_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &q_len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &kv_len_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &num_heads_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &head_dim_u32 as *const u32 as *const _,
    );

    // Configure thread groups
    // Grid: (batch*q_len, num_heads, head_dim)
    let threads_per_threadgroup = MTLSize {
        width: 8,  // batch*q_len dimension
        height: 4, // num_heads dimension
        depth: 4,  // head_dim dimension
    };

    let threadgroups_per_grid = MTLSize {
        width: ((batch_size * q_len) as u64 + 7) / 8,
        height: ((num_heads as u64 + 3) / 4).max(1),
        depth: ((head_dim as u64 + 3) / 4).max(1),
    };

    encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    encoder.end_encoding();

    // Execute
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Copy results
    let output_ptr = output_buffer.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(output_ptr, output.as_mut_ptr(), output.len());
    }

    Ok(())
}

#[cfg(not(feature = "metal"))]
pub fn metal_apply_attention_batch_f32(
    _scores: &[f32],
    _v: &[f32],
    _output: &mut [f32],
    _batch_size: usize,
    _q_len: usize,
    _kv_len: usize,
    _num_heads: usize,
    _head_dim: usize,
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

    #[test]
    #[cfg(feature = "metal")]
    fn test_attention_scores_batch_basic() {
        let batch_size = 1;
        let q_len = 2;
        let kv_len = 3;
        let num_heads = 2;
        let head_dim = 4;

        // Create test Q and K tensors
        let q = vec![1.0f32; batch_size * q_len * num_heads * head_dim];
        let k = vec![1.0f32; batch_size * kv_len * num_heads * head_dim];
        let mut scores = vec![0.0f32; batch_size * num_heads * q_len * kv_len];

        let result = metal_attention_scores_batch_f32(
            &q,
            &k,
            &mut scores,
            batch_size,
            q_len,
            kv_len,
            num_heads,
            head_dim,
        );

        assert!(
            result.is_ok(),
            "Attention scores batch should succeed: {:?}",
            result.err()
        );

        // Q@K^T for all-ones should give head_dim (4) scaled by 1/sqrt(head_dim) = 1/2 = 2.0
        let expected_score = 2.0f32;
        for &score in &scores {
            assert!(
                (score - expected_score).abs() < 1e-4,
                "Expected ~{}, got {}",
                expected_score,
                score
            );
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_softmax_batch_basic() {
        let batch_size = 1;
        let num_heads = 1;
        let q_len = 2;
        let kv_len = 3;

        // Create test scores (all ones)
        let mut scores = vec![1.0f32; batch_size * num_heads * q_len * kv_len];

        let result = metal_softmax_batch_f32(&mut scores, batch_size, num_heads, q_len, kv_len);

        assert!(
            result.is_ok(),
            "Softmax batch should succeed: {:?}",
            result.err()
        );

        // Softmax of all ones should give 1/kv_len
        let expected = 1.0f32 / kv_len as f32;
        for &score in &scores {
            assert!(
                (score - expected).abs() < 1e-4,
                "Expected ~{}, got {}",
                expected,
                score
            );
        }

        // Verify sum per row is ~1.0
        for b in 0..batch_size {
            for h in 0..num_heads {
                for q in 0..q_len {
                    let row_offset = ((b * num_heads + h) * q_len + q) * kv_len;
                    let sum: f32 = scores[row_offset..row_offset + kv_len].iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-4,
                        "Row sum should be ~1.0, got {}",
                        sum
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_apply_attention_batch_basic() {
        let batch_size = 1;
        let q_len = 2;
        let kv_len = 3;
        let num_heads = 2;
        let head_dim = 4;

        // Create uniform attention scores (sum to 1 per row)
        let mut scores = vec![1.0f32 / kv_len as f32; batch_size * num_heads * q_len * kv_len];

        // Create test V tensor
        let v = vec![2.0f32; batch_size * kv_len * num_heads * head_dim];
        let mut output = vec![0.0f32; batch_size * q_len * num_heads * head_dim];

        let result = metal_apply_attention_batch_f32(
            &scores,
            &v,
            &mut output,
            batch_size,
            q_len,
            kv_len,
            num_heads,
            head_dim,
        );

        assert!(
            result.is_ok(),
            "Apply attention batch should succeed: {:?}",
            result.err()
        );

        // Uniform scores * constant V should give the same constant
        let expected = 2.0f32;
        for &val in &output {
            assert!(
                (val - expected).abs() < 1e-4,
                "Expected ~{}, got {}",
                expected,
                val
            );
        }
    }

    #[test]
    fn test_attention_scores_batch_size_validation() {
        let q = vec![1.0f32; 10]; // Wrong size
        let k = vec![1.0f32; 20];
        let mut scores = vec![0.0f32; 12];

        let result = metal_attention_scores_batch_f32(&q, &k, &mut scores, 1, 2, 3, 2, 4);

        #[cfg(feature = "metal")]
        assert!(result.is_err(), "Should fail with size mismatch");

        #[cfg(not(feature = "metal"))]
        assert!(result.is_err(), "Should fail without Metal");
    }

    /// Integration test: Full attention pipeline for a batch
    /// 統合テスト: バッチに対する完全なattentionパイプライン
    #[test]
    #[cfg(feature = "metal")]
    fn test_full_attention_pipeline_batch() {
        let batch_size = 2;
        let q_len = 3;
        let kv_len = 4;
        let num_heads = 2;
        let head_dim = 8;

        // Create test Q, K, V tensors
        let q = vec![0.5f32; batch_size * q_len * num_heads * head_dim];
        let k = vec![0.5f32; batch_size * kv_len * num_heads * head_dim];
        let v = vec![1.0f32; batch_size * kv_len * num_heads * head_dim];

        // Step 1: Compute attention scores
        let mut scores = vec![0.0f32; batch_size * num_heads * q_len * kv_len];
        let result = metal_attention_scores_batch_f32(
            &q,
            &k,
            &mut scores,
            batch_size,
            q_len,
            kv_len,
            num_heads,
            head_dim,
        );
        assert!(
            result.is_ok(),
            "Attention scores should succeed: {:?}",
            result.err()
        );

        // Step 2: Apply softmax
        let result = metal_softmax_batch_f32(&mut scores, batch_size, num_heads, q_len, kv_len);
        assert!(
            result.is_ok(),
            "Softmax should succeed: {:?}",
            result.err()
        );

        // Verify softmax normalization (each row sums to 1)
        for b in 0..batch_size {
            for h in 0..num_heads {
                for q in 0..q_len {
                    let row_offset = ((b * num_heads + h) * q_len + q) * kv_len;
                    let sum: f32 = scores[row_offset..row_offset + kv_len].iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-4,
                        "Row sum should be ~1.0, got {}",
                        sum
                    );
                }
            }
        }

        // Step 3: Apply attention to values
        let mut output = vec![0.0f32; batch_size * q_len * num_heads * head_dim];
        let result = metal_apply_attention_batch_f32(
            &scores,
            &v,
            &mut output,
            batch_size,
            q_len,
            kv_len,
            num_heads,
            head_dim,
        );
        assert!(
            result.is_ok(),
            "Apply attention should succeed: {:?}",
            result.err()
        );

        // Verify output is not all zeros (computation happened)
        let non_zero_count = output.iter().filter(|&&x| x.abs() > 1e-6).count();
        assert!(
            non_zero_count > 0,
            "Output should contain non-zero values"
        );

        eprintln!("✅ Full attention pipeline test passed!");
    }

    /// Integration test: Layer processing simulation
    /// 統合テスト: レイヤー処理シミュレーション
    #[test]
    #[cfg(feature = "metal")]
    fn test_layer_processing_simulation() {
        let batch_size = 2;
        let seq_len = 4;
        let hidden_dim = 64;
        let eps = 1e-5f32;

        // Simulate hidden states
        let mut hidden_states = vec![0.5f32; batch_size * seq_len * hidden_dim];
        let weight = vec![1.0f32; hidden_dim];

        // Step 1: RMS Norm
        let mut normed = vec![0.0f32; batch_size * seq_len * hidden_dim];
        let result = metal_rms_norm_batch_f32(
            &hidden_states,
            &weight,
            &mut normed,
            batch_size,
            seq_len,
            hidden_dim,
            eps,
        );
        assert!(result.is_ok(), "RMS Norm should succeed: {:?}", result.err());

        // Verify normalized values are different from input
        let norm_mean: f32 = normed.iter().sum::<f32>() / normed.len() as f32;
        let input_mean: f32 = hidden_states.iter().sum::<f32>() / hidden_states.len() as f32;
        assert!(
            (norm_mean - input_mean).abs() > 1e-6,
            "Normalized values should differ from input"
        );

        // Step 2: RoPE (simulate rotation)
        let num_heads = 8;
        let head_dim = hidden_dim / num_heads;
        let mut rotated = normed.clone();
        let result = metal_rope_batch_f32(
            &mut rotated,
            batch_size,
            0, // start_pos
            seq_len,
            num_heads,
            head_dim,
            10000.0, // theta
        );
        assert!(result.is_ok(), "RoPE should succeed: {:?}", result.err());

        // Verify rotation happened
        let diff_count = rotated
            .iter()
            .zip(normed.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();
        assert!(
            diff_count > 0,
            "RoPE should modify values (found {} differences)",
            diff_count
        );

        eprintln!("✅ Layer processing simulation test passed!");
    }

    /// Performance comparison test: Batch vs Sequential
    /// パフォーマンス比較テスト: バッチ vs 順次
    #[test]
    #[cfg(feature = "metal")]
    #[ignore] // Run with: cargo test --features metal -- --ignored --nocapture
    fn test_batch_performance_comparison() {
        use std::time::Instant;

        let batch_size = 4;
        let seq_len = 8;
        let hidden_dim = 512;
        let eps = 1e-5f32;

        let input = vec![0.5f32; batch_size * seq_len * hidden_dim];
        let weight = vec![1.0f32; hidden_dim];
        let mut output_batch = vec![0.0f32; batch_size * seq_len * hidden_dim];

        // Batch processing
        let start = Instant::now();
        let result = metal_rms_norm_batch_f32(
            &input,
            &weight,
            &mut output_batch,
            batch_size,
            seq_len,
            hidden_dim,
            eps,
        );
        let batch_duration = start.elapsed();
        assert!(result.is_ok(), "Batch RMS Norm failed");

        // Sequential processing (simulate)
        let mut output_sequential = vec![0.0f32; batch_size * seq_len * hidden_dim];
        let start = Instant::now();
        for b in 0..batch_size {
            let input_offset = b * seq_len * hidden_dim;
            let output_offset = b * seq_len * hidden_dim;
            let _ = metal_rms_norm_batch_f32(
                &input[input_offset..input_offset + seq_len * hidden_dim],
                &weight,
                &mut output_sequential[output_offset..output_offset + seq_len * hidden_dim],
                1, // batch_size = 1
                seq_len,
                hidden_dim,
                eps,
            );
        }
        let sequential_duration = start.elapsed();

        eprintln!("\n📊 Performance Comparison (batch_size={}):", batch_size);
        eprintln!("   Batch processing:      {:?}", batch_duration);
        eprintln!("   Sequential processing: {:?}", sequential_duration);
        eprintln!(
            "   Speedup: {:.2}x",
            sequential_duration.as_secs_f64() / batch_duration.as_secs_f64()
        );

        // Verify results are similar
        let max_diff = output_batch
            .iter()
            .zip(output_sequential.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "Batch and sequential results should match (max_diff={})",
            max_diff
        );
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_batch_memory_usage() {
        // Memory usage measurement for batch processing
        let batch_size = 4;
        let seq_len = 8;
        let hidden_dim = 64;
        let num_heads = 4;
        let head_dim = hidden_dim / num_heads;
        let vocab_size = 128;

        println!("\n📊 Memory Usage Analysis (batch_size={}):", batch_size);

        // Calculate expected memory allocations

        // 1. Input embeddings: batch_size × seq_len × hidden_dim × sizeof(f32)
        let embedding_bytes = batch_size * seq_len * hidden_dim * std::mem::size_of::<f32>();
        println!("   Input embeddings:        {:>10} bytes ({:.2} MB)",
                 embedding_bytes, embedding_bytes as f64 / 1_048_576.0);

        // 2. Q/K/V projections: 3 × (batch_size × seq_len × hidden_dim) × sizeof(f32)
        let qkv_bytes = 3 * batch_size * seq_len * hidden_dim * std::mem::size_of::<f32>();
        println!("   Q/K/V projections:       {:>10} bytes ({:.2} MB)",
                 qkv_bytes, qkv_bytes as f64 / 1_048_576.0);

        // 3. Attention scores: batch_size × num_heads × seq_len × seq_len × sizeof(f32)
        let scores_bytes = batch_size * num_heads * seq_len * seq_len * std::mem::size_of::<f32>();
        println!("   Attention scores:        {:>10} bytes ({:.2} MB)",
                 scores_bytes, scores_bytes as f64 / 1_048_576.0);

        // 4. Attention output: batch_size × seq_len × hidden_dim × sizeof(f32)
        let attn_out_bytes = batch_size * seq_len * hidden_dim * std::mem::size_of::<f32>();
        println!("   Attention output:        {:>10} bytes ({:.2} MB)",
                 attn_out_bytes, attn_out_bytes as f64 / 1_048_576.0);

        // 5. FFN intermediate (gate + up): 2 × (batch_size × seq_len × hidden_dim × 4) × sizeof(f32)
        let ffn_intermediate_dim = hidden_dim * 4;
        let ffn_bytes = 2 * batch_size * seq_len * ffn_intermediate_dim * std::mem::size_of::<f32>();
        println!("   FFN intermediate:        {:>10} bytes ({:.2} MB)",
                 ffn_bytes, ffn_bytes as f64 / 1_048_576.0);

        // 6. Output logits: batch_size × seq_len × vocab_size × sizeof(f32)
        let logits_bytes = batch_size * seq_len * vocab_size * std::mem::size_of::<f32>();
        println!("   Output logits:           {:>10} bytes ({:.2} MB)",
                 logits_bytes, logits_bytes as f64 / 1_048_576.0);

        // Total per-layer memory
        let per_layer_total = embedding_bytes + qkv_bytes + scores_bytes + attn_out_bytes + ffn_bytes;
        println!("   ─────────────────────────────────────────────");
        println!("   Total per layer:         {:>10} bytes ({:.2} MB)",
                 per_layer_total, per_layer_total as f64 / 1_048_576.0);

        // Estimate for full model (assuming 22 layers like TinyLlama)
        let num_layers = 22;
        let full_model_bytes = per_layer_total * num_layers + logits_bytes;
        println!("   Full model ({} layers): {:>10} bytes ({:.2} MB)",
                 num_layers, full_model_bytes, full_model_bytes as f64 / 1_048_576.0);

        // Compare with sequential processing (batch_size=1)
        let sequential_per_layer = (seq_len * hidden_dim * std::mem::size_of::<f32>())  // embeddings
                                  + (3 * seq_len * hidden_dim * std::mem::size_of::<f32>())  // Q/K/V
                                  + (num_heads * seq_len * seq_len * std::mem::size_of::<f32>())  // scores
                                  + (seq_len * hidden_dim * std::mem::size_of::<f32>())  // attn out
                                  + (2 * seq_len * ffn_intermediate_dim * std::mem::size_of::<f32>());  // FFN

        let memory_overhead_ratio = per_layer_total as f64 / sequential_per_layer as f64;
        println!("\n   Memory overhead vs sequential:");
        println!("   Sequential (batch=1):    {:>10} bytes ({:.2} MB)",
                 sequential_per_layer, sequential_per_layer as f64 / 1_048_576.0);
        println!("   Batch overhead ratio:    {:.2}x", memory_overhead_ratio);
        println!("   Expected ratio:          {:.2}x (linear scaling)", batch_size as f64);

        // Verify actual allocation works without OOM
        let test_data = vec![0.0f32; batch_size * seq_len * hidden_dim];
        let test_scores = vec![0.0f32; batch_size * num_heads * seq_len * seq_len];

        println!("\n   ✅ Allocations successful (no OOM)");
        println!("   Test data size:          {} elements", test_data.len());
        println!("   Test scores size:        {} elements", test_scores.len());

        // Test actual Metal kernel execution to measure GPU memory
        let mut q = vec![0.1f32; batch_size * seq_len * num_heads * head_dim];
        let k = vec![0.2f32; batch_size * seq_len * num_heads * head_dim];
        let v = vec![0.3f32; batch_size * seq_len * num_heads * head_dim];
        let mut scores = vec![0.0f32; batch_size * num_heads * seq_len * seq_len];
        let mut output = vec![0.0f32; batch_size * seq_len * num_heads * head_dim];

        // Execute kernels to ensure GPU memory allocations work
        let result = metal_attention_scores_batch_f32(
            &q, &k, &mut scores,
            batch_size, seq_len, seq_len, num_heads, head_dim
        );

        assert!(result.is_ok(), "GPU memory allocation failed: {:?}", result.err());
        println!("   ✅ GPU memory allocations successful");
    }
}
