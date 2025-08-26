//! WASM performance-optimized operations
//! WASMパフォーマンス最適化済み操作

#[cfg(feature = "wasm")]
use super::tensor::WasmTensor;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Optimized tensor operations for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct OptimizedOps;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl OptimizedOps {
    /// Create new optimized operations utility
    #[wasm_bindgen(constructor)]
    pub fn new() -> OptimizedOps {
        OptimizedOps
    }

    /// Fast matrix multiplication using blocking for cache efficiency
    #[wasm_bindgen]
    pub fn fast_matmul(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(JsValue::from_str("Only 2D matrices supported"));
        }

        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);

        if k != k2 {
            return Err(JsValue::from_str("Matrix dimensions don't match"));
        }

        let a_data = a.data();
        let b_data = b.data();
        let mut result = vec![0.0f32; m * n];

        // Block size for cache efficiency
        const BLOCK_SIZE: usize = 32;

        // Blocked matrix multiplication
        for ii in (0..m).step_by(BLOCK_SIZE) {
            for jj in (0..n).step_by(BLOCK_SIZE) {
                for kk in (0..k).step_by(BLOCK_SIZE) {
                    let i_end = (ii + BLOCK_SIZE).min(m);
                    let j_end = (jj + BLOCK_SIZE).min(n);
                    let k_end = (kk + BLOCK_SIZE).min(k);

                    for i in ii..i_end {
                        for j in jj..j_end {
                            let mut sum = result[i * n + j];
                            for p in kk..k_end {
                                sum += a_data[i * k + p] * b_data[p * n + j];
                            }
                            result[i * n + j] = sum;
                        }
                    }
                }
            }
        }

        Ok(WasmTensor::new(result, vec![m, n]))
    }

    /// Vectorized element-wise operations
    #[wasm_bindgen]
    pub fn vectorized_add(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if a.shape() != b.shape() {
            return Err(JsValue::from_str("Shape mismatch"));
        }

        let a_data = a.data();
        let b_data = b.data();
        let len = a_data.len();
        let mut result = Vec::with_capacity(len);

        // Process in chunks for better performance
        const CHUNK_SIZE: usize = 8;
        let chunks = len / CHUNK_SIZE;
        let remainder = len % CHUNK_SIZE;

        // Process chunks
        for i in 0..chunks {
            let base = i * CHUNK_SIZE;
            for j in 0..CHUNK_SIZE {
                let idx = base + j;
                result.push(a_data[idx] + b_data[idx]);
            }
        }

        // Process remainder
        let base = chunks * CHUNK_SIZE;
        for j in 0..remainder {
            let idx = base + j;
            result.push(a_data[idx] + b_data[idx]);
        }

        Ok(WasmTensor::new(result, a.shape().clone()))
    }

    /// Fast ReLU with fused operations
    #[wasm_bindgen]
    pub fn fused_relu_add(
        &self,
        input: &WasmTensor,
        bias: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        if input.shape() != bias.shape() {
            return Err(JsValue::from_str("Shape mismatch"));
        }

        let input_data = input.data();
        let bias_data = bias.data();
        let result: Vec<f32> = input_data
            .iter()
            .zip(bias_data.iter())
            .map(|(a, b)| (a + b).max(0.0))
            .collect();

        Ok(WasmTensor::new(result, input.shape().clone()))
    }

    /// Memory-efficient convolution-like operation (simplified 1D)
    #[wasm_bindgen]
    pub fn conv1d(
        &self,
        input: &WasmTensor,
        kernel: &WasmTensor,
        stride: usize,
    ) -> Result<WasmTensor, JsValue> {
        let input_shape = input.shape();
        let kernel_shape = kernel.shape();

        if input_shape.len() != 1 || kernel_shape.len() != 1 {
            return Err(JsValue::from_str("Only 1D tensors supported"));
        }

        let input_len = input_shape[0];
        let kernel_len = kernel_shape[0];

        if kernel_len > input_len {
            return Err(JsValue::from_str("Kernel larger than input"));
        }

        let output_len = (input_len - kernel_len) / stride + 1;
        let mut result = Vec::with_capacity(output_len);

        let input_data = input.data();
        let kernel_data = kernel.data();

        for i in (0..=(input_len - kernel_len)).step_by(stride) {
            let mut sum = 0.0f32;
            for j in 0..kernel_len {
                sum += input_data[i + j] * kernel_data[j];
            }
            result.push(sum);
        }

        Ok(WasmTensor::new(result, vec![output_len]))
    }

    /// Batch normalization-like operation
    #[wasm_bindgen]
    pub fn batch_normalize(&self, input: &WasmTensor, epsilon: f32) -> WasmTensor {
        let data = input.data();

        // Calculate mean
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;

        // Calculate variance
        let variance: f32 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

        // Normalize
        let std_inv = 1.0 / (variance + epsilon).sqrt();
        let result: Vec<f32> = data.iter().map(|&x| (x - mean) * std_inv).collect();

        WasmTensor::new(result, input.shape().clone())
    }
}

/// Memory pool for efficient tensor allocation
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmMemoryPool {
    pools: Vec<Vec<Vec<f32>>>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmMemoryPool {
    /// Create new memory pool for efficient buffer management
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmMemoryPool {
        // Create pools for common sizes (powers of 2)
        let mut pools = Vec::new();
        for _i in 0..20 {
            // Up to 2^20 elements
            pools.push(Vec::new());
        }

        WasmMemoryPool { pools }
    }

    /// Get a buffer from the pool or allocate new one
    #[wasm_bindgen]
    pub fn get_buffer(&mut self, size: usize) -> Vec<f32> {
        if size == 0 {
            return Vec::new();
        }

        // Find the appropriate pool (round up to next power of 2)
        let pool_idx = (size as f64).log2().ceil() as usize;

        if pool_idx < self.pools.len() {
            if let Some(buffer) = self.pools[pool_idx].pop() {
                return buffer;
            }
        }

        // Allocate new buffer
        let actual_size = 1 << pool_idx.min(19); // Cap at 2^19
        Vec::with_capacity(actual_size)
    }

    /// Return a buffer to the pool
    #[wasm_bindgen]
    pub fn return_buffer(&mut self, mut buffer: Vec<f32>) {
        let capacity = buffer.capacity();
        if capacity == 0 {
            return;
        }

        buffer.clear();

        // Find the appropriate pool
        let pool_idx = (capacity as f64).log2() as usize;

        if pool_idx < self.pools.len() && self.pools[pool_idx].len() < 100 {
            self.pools[pool_idx].push(buffer);
        }
    }

    /// Get pool statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> String {
        let total_buffers: usize = self.pools.iter().map(|p| p.len()).sum();
        format!("Total cached buffers: {}", total_buffers)
    }

    /// Clear all pools
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        for pool in &mut self.pools {
            pool.clear();
        }
    }
}

/// Parallel execution utilities (simulated for WASM single-threaded environment)
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct ParallelOps;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl ParallelOps {
    /// Parallel-style reduction (sequential in WASM)
    #[wasm_bindgen]
    pub fn parallel_sum(data: &[f32]) -> f32 {
        // In a multi-threaded environment, this would be parallelized
        // For WASM, we use efficient sequential processing
        const CHUNK_SIZE: usize = 64;

        let mut partial_sums = Vec::new();

        // Process in chunks
        for chunk in data.chunks(CHUNK_SIZE) {
            partial_sums.push(chunk.iter().sum::<f32>());
        }

        // Sum partial results
        partial_sums.iter().sum()
    }

    /// Parallel-style element-wise operation
    #[wasm_bindgen]
    pub fn parallel_map_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            return Vec::new();
        }

        const CHUNK_SIZE: usize = 64;
        let mut result = Vec::with_capacity(a.len());

        // Process in chunks for better cache locality
        for i in (0..a.len()).step_by(CHUNK_SIZE) {
            let end = (i + CHUNK_SIZE).min(a.len());
            for j in i..end {
                result.push(a[j] + b[j]);
            }
        }

        result
    }
}
