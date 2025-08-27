//! CUDA Kernels for RusTorch
//! RusTorch用CUDAカーネル

extern "C" {

// Element-wise addition kernel
__global__ void elementwise_add_f32(
    const float* a, 
    const float* b, 
    float* result, 
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

// Element-wise subtraction kernel  
__global__ void elementwise_sub_f32(
    const float* a,
    const float* b, 
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

// Element-wise multiplication kernel
__global__ void elementwise_mul_f32(
    const float* a,
    const float* b,
    float* result, 
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

// Element-wise division kernel
__global__ void elementwise_div_f32(
    const float* a,
    const float* b,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] / b[idx];
    }
}

// Matrix multiplication kernel (simple version)
__global__ void matmul_f32(
    const float* a,
    const float* b, 
    float* c,
    int m, int n, int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float value = 0.0f;
        for (int i = 0; i < k; ++i) {
            value += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = value;
    }
}

// Batch normalization kernel
__global__ void batch_normalize_f32(
    const float* input,
    float* output,
    const float* mean,
    const float* variance,
    float epsilon,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float norm = (input[idx] - *mean) / sqrtf(*variance + epsilon);
        output[idx] = norm;
    }
}

// ReLU activation kernel
__global__ void relu_f32(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Reduction sum kernel (simple version)
__global__ void reduce_sum_f32(
    const float* input,
    float* output,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0 && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// 3D Convolution kernel
__global__ void conv3d_f32(
    const float* input,     // [batch, in_channels, depth, height, width]
    const float* weight,    // [out_channels, in_channels_per_group, kd, kh, kw]
    float* output,          // [batch, out_channels, out_depth, out_height, out_width]
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
    int batch_idx = blockIdx.z;
    int out_ch_idx = blockIdx.y;
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
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

} // extern "C"