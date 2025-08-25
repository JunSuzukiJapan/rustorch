//! Metal Compute Shaders for RusTorch
//! RusTorch用Metalコンピュートシェーダー

#include <metal_stdlib>
using namespace metal;

// Element-wise addition kernel
kernel void elementwise_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] + b[index];
}

// Element-wise subtraction kernel
kernel void elementwise_sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] - b[index];
}

// Element-wise multiplication kernel
kernel void elementwise_mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] * b[index];
}

// Element-wise division kernel
kernel void elementwise_div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] / b[index];
}

// Matrix multiplication kernel
kernel void matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= m || col >= n) return;
    
    float value = 0.0;
    for (uint i = 0; i < k; i++) {
        value += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = value;
}

// Batch normalization kernel
kernel void batch_normalize_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* mean [[buffer(2)]],
    device const float* variance [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    float norm = (input[index] - mean[0]) / sqrt(variance[0] + epsilon);
    output[index] = norm;
}

// ReLU activation kernel
kernel void relu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output[index] = max(0.0f, input[index]);
}

// Softmax kernel (simplified version)
kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* max_vals [[buffer(2)]],
    device float* sum_vals [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    // First pass: find max
    if (index == 0) {
        float max_val = input[0];
        for (uint i = 1; i < size; i++) {
            max_val = max(max_val, input[i]);
        }
        max_vals[0] = max_val;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Second pass: compute exp and sum
    float exp_val = exp(input[index] - max_vals[0]);
    output[index] = exp_val;
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Third pass: normalize
    if (index == 0) {
        float sum = 0.0;
        for (uint i = 0; i < size; i++) {
            sum += output[i];
        }
        sum_vals[0] = sum;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    output[index] = output[index] / sum_vals[0];
}

// Reduction sum kernel
kernel void reduce_sum_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float shared_data[256];
    
    // Load data into shared memory
    if (index < size) {
        shared_data[index % threads_per_group] = input[index];
    } else {
        shared_data[index % threads_per_group] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        uint local_index = index % threads_per_group;
        if (local_index < stride) {
            shared_data[local_index] += shared_data[local_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result for this threadgroup
    if (index % threads_per_group == 0) {
        uint group_id = index / threads_per_group;
        output[group_id] = shared_data[0];
    }
}

// Conv2D kernel (simplified)
kernel void conv2d_f32(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_height [[buffer(3)]],
    constant uint& input_width [[buffer(4)]],
    constant uint& kernel_height [[buffer(5)]],
    constant uint& kernel_width [[buffer(6)]],
    constant uint& output_height [[buffer(7)]],
    constant uint& output_width [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint out_y = gid.y;
    uint out_x = gid.x;
    
    if (out_y >= output_height || out_x >= output_width) return;
    
    float sum = 0.0;
    for (uint ky = 0; ky < kernel_height; ky++) {
        for (uint kx = 0; kx < kernel_width; kx++) {
            uint in_y = out_y + ky;
            uint in_x = out_x + kx;
            
            if (in_y < input_height && in_x < input_width) {
                sum += input[in_y * input_width + in_x] * 
                       kernel[ky * kernel_width + kx];
            }
        }
    }
    
    output[out_y * output_width + out_x] = sum;
}

// Attention mechanism kernel (simplified)
kernel void attention_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* scores [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& d_model [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    
    if (i >= seq_len || j >= seq_len) return;
    
    // Compute attention score: query[i] * key[j]
    float score = 0.0;
    for (uint k = 0; k < d_model; k++) {
        score += query[i * d_model + k] * key[j * d_model + k];
    }
    
    scores[i * seq_len + j] = score;
    
    // Note: This is a simplified version
    // Full attention would require softmax normalization
    // and weighted sum with values
}