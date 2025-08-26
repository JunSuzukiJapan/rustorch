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

} // extern "C"