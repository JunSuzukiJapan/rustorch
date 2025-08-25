//! OpenCL Kernels for RusTorch
//! RusTorch用OpenCLカーネル

// Element-wise addition kernel
__kernel void elementwise_add_f32(
    __global const float* a,
    __global const float* b,
    __global float* result,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

// Element-wise subtraction kernel
__kernel void elementwise_sub_f32(
    __global const float* a,
    __global const float* b,
    __global float* result,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

// Element-wise multiplication kernel
__kernel void elementwise_mul_f32(
    __global const float* a,
    __global const float* b,
    __global float* result,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

// Element-wise division kernel
__kernel void elementwise_div_f32(
    __global const float* a,
    __global const float* b,
    __global float* result,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        result[idx] = a[idx] / b[idx];
    }
}

// Matrix multiplication kernel
__kernel void matmul_f32(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int m,
    const int n,
    const int k
) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < m && col < n) {
        float value = 0.0f;
        for (int i = 0; i < k; i++) {
            value += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = value;
    }
}

// Convolution 2D kernel
__kernel void conv2d_f32(
    __global const float* input,
    __global const float* kernel,
    __global float* output,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w
) {
    int out_y = get_global_id(1);
    int out_x = get_global_id(0);
    
    if (out_y >= output_height || out_x >= output_width) return;
    
    float sum = 0.0f;
    
    for (int ky = 0; ky < kernel_height; ky++) {
        for (int kx = 0; kx < kernel_width; kx++) {
            int in_y = out_y * stride_h + ky - padding_h;
            int in_x = out_x * stride_w + kx - padding_w;
            
            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                sum += input[in_y * input_width + in_x] * kernel[ky * kernel_width + kx];
            }
        }
    }
    
    output[out_y * output_width + out_x] = sum;
}

// Max pooling 2D kernel
__kernel void max_pool2d_f32(
    __global const float* input,
    __global float* output,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int pool_height,
    const int pool_width,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w
) {
    int out_y = get_global_id(1);
    int out_x = get_global_id(0);
    
    if (out_y >= output_height || out_x >= output_width) return;
    
    float max_val = -INFINITY;
    
    for (int py = 0; py < pool_height; py++) {
        for (int px = 0; px < pool_width; px++) {
            int in_y = out_y * stride_h + py - padding_h;
            int in_x = out_x * stride_w + px - padding_w;
            
            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                float val = input[in_y * input_width + in_x];
                max_val = fmax(max_val, val);
            }
        }
    }
    
    output[out_y * output_width + out_x] = max_val;
}

// ReLU activation kernel
__kernel void relu_f32(
    __global const float* input,
    __global float* output,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        output[idx] = fmax(0.0f, input[idx]);
    }
}

// Batch normalization kernel
__kernel void batch_normalize_f32(
    __global const float* input,
    __global float* output,
    __global const float* mean,
    __global const float* variance,
    const float epsilon,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        float norm = (input[idx] - mean[0]) / sqrt(variance[0] + epsilon);
        output[idx] = norm;
    }
}

// Reduction sum kernel with local memory
__kernel void reduce_sum_f32(
    __global const float* input,
    __global float* output,
    __local float* local_sum,
    const int n
) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    
    // Load data into local memory
    if (global_id < n) {
        local_sum[local_id] = input[global_id];
    } else {
        local_sum[local_id] = 0.0f;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_sum[local_id] += local_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result for this work group
    if (local_id == 0) {
        int group_id = get_group_id(0);
        output[group_id] = local_sum[0];
    }
}

// Softmax kernel (simplified version)
__kernel void softmax_f32(
    __global const float* input,
    __global float* output,
    __global float* max_vals,
    __global float* sum_vals,
    const int size
) {
    int idx = get_global_id(0);
    
    // First pass: find max
    if (idx == 0) {
        float max_val = input[0];
        for (int i = 1; i < size; i++) {
            max_val = fmax(max_val, input[i]);
        }
        max_vals[0] = max_val;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // Second pass: compute exp and sum
    if (idx < size) {
        float exp_val = exp(input[idx] - max_vals[0]);
        output[idx] = exp_val;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // Third pass: compute sum and normalize
    if (idx == 0) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += output[i];
        }
        sum_vals[0] = sum;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    if (idx < size) {
        output[idx] = output[idx] / sum_vals[0];
    }
}

// GELU activation kernel
__kernel void gelu_f32(
    __global const float* input,
    __global float* output,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + erf(x / sqrt(2.0f)));
        output[idx] = x * cdf;
    }
}

// Transpose kernel
__kernel void transpose_f32(
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols
) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Element-wise power kernel
__kernel void pow_f32(
    __global const float* input,
    __global float* output,
    const float exponent,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        output[idx] = pow(input[idx], exponent);
    }
}