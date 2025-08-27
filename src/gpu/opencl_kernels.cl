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

// 3D Convolution kernel
__kernel void conv3d_f32(
    __global const float* input,     // [batch, in_channels, depth, height, width]
    __global const float* weight,    // [out_channels, in_channels_per_group, kd, kh, kw]
    __global float* output,          // [batch, out_channels, out_depth, out_height, out_width]
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    const int groups
) {
    // Calculate global thread indices
    int batch_idx = get_global_id(2);
    int out_ch_idx = get_global_id(1);
    int spatial_idx = get_global_id(0);
    
    if (batch_idx >= batch_size || out_ch_idx >= out_channels) return;
    
    // Calculate spatial coordinates
    int out_size = output_d * output_h * output_w;
    if (spatial_idx >= out_size) return;
    
    int od = spatial_idx / (output_h * output_w);
    int temp = spatial_idx % (output_h * output_w);
    int oh = temp / output_w;
    int ow = temp % output_w;
    
    // Calculate group parameters
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    int group_idx = out_ch_idx / out_channels_per_group;
    int in_start = group_idx * in_channels_per_group;
    
    float sum = 0.0f;
    
    // Perform 3D convolution
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // Calculate input coordinates with stride, padding, and dilation
                    int id = od * stride_d + kd * dilation_d - pad_d;
                    int ih = oh * stride_h + kh * dilation_h - pad_h;
                    int iw = ow * stride_w + kw * dilation_w - pad_w;
                    
                    // Check bounds
                    if (id >= 0 && id < input_d && ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                        // Calculate indices
                        int input_idx = ((batch_idx * in_channels + in_start + ic) * input_d + id) * input_h * input_w + ih * input_w + iw;
                        int weight_idx = ((out_ch_idx * in_channels_per_group + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Write output
    int output_idx = ((batch_idx * out_channels + out_ch_idx) * output_d + od) * output_h * output_w + oh * output_w + ow;
    output[output_idx] = sum;
}