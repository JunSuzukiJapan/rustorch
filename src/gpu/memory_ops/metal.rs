//! Metal Memory Operations
//! Metalメモリ操作

#[cfg(feature = "metal")]
use crate::error::{RusTorchError, RusTorchResult};
#[cfg(feature = "metal")]
use metal::{Buffer, Device as MetalDeviceType};
#[cfg(feature = "metal")]
use num_traits::Float;
#[cfg(feature = "metal")]
use std::sync::Arc;

#[cfg(feature = "metal")]
use super::buffer::GpuBuffer;

#[cfg(feature = "metal")]
/// Metal-specific memory operations
/// Metal固有のメモリ操作
pub struct MetalOperations;

#[cfg(feature = "metal")]
impl MetalOperations {
    /// Transfer data to Metal device
    /// データをMetalデバイスに転送
    pub fn transfer_to_device<T>(data: Vec<T>) -> RusTorchResult<GpuBuffer<T>>
    where
        T: Float + 'static,
    {
        use metal::{Device, MTLResourceOptions};

        let device =
            Device::system_default().ok_or_else(|| RusTorchError::gpu("No Metal device found"))?;

        let device = Arc::new(device);

        // Calculate byte size - handle Float trait
        let byte_size = data.len() * std::mem::size_of::<T>();

        // Create Metal buffer with data
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(GpuBuffer::Metal {
            buffer: Arc::new(buffer),
            device,
        })
    }

    /// Transfer data from Metal device to CPU
    /// データをMetalデバイスからCPUに転送
    pub fn transfer_to_cpu<T>(metal_buffer: &Arc<Buffer>, shape: &[usize]) -> RusTorchResult<Vec<T>>
    where
        T: Float + 'static,
    {
        let total_elements: usize = shape.iter().product();
        let byte_size = total_elements * std::mem::size_of::<T>();

        if metal_buffer.length() as usize != byte_size {
            return Err(RusTorchError::gpu("Buffer size mismatch"));
        }

        // Copy data from Metal buffer
        let contents = metal_buffer.contents();
        let mut cpu_data = vec![T::zero(); total_elements];

        unsafe {
            std::ptr::copy_nonoverlapping(
                contents as *const T,
                cpu_data.as_mut_ptr(),
                total_elements,
            );
        }

        Ok(cpu_data)
    }

    /// Execute Metal element-wise operation
    /// Metal要素ごとの演算実行
    pub fn execute_elementwise<T, F>(
        lhs: &Arc<Buffer>,
        rhs: &Arc<Buffer>,
        device: &Arc<MetalDeviceType>,
        op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: Float + 'static,
    {
        use metal::{CommandQueue, ComputeCommandEncoder, MTLSize};

        let size = lhs.length() as usize / std::mem::size_of::<T>();

        // Create test values to determine operation type
        let test_a = T::from(2.0).unwrap();
        let test_b = T::from(3.0).unwrap();
        let test_result = op(test_a, test_b);

        // Determine shader function name based on operation
        let function_name = if test_result == T::from(5.0).unwrap() {
            "elementwise_add_f32"
        } else if test_result == T::from(6.0).unwrap() {
            "elementwise_mul_f32"
        } else if test_result == T::from(-1.0).unwrap() {
            "elementwise_sub_f32"
        } else if test_result == T::from(2.0 / 3.0).unwrap() {
            "elementwise_div_f32"
        } else {
            // Use generic shader for unknown operations
            return Self::execute_elementwise_fallback(lhs, rhs, device, op);
        };

        // For now, use CPU fallback as we don't have the metal shaders included
        // In a production implementation, this would use Metal compute shaders
        Self::execute_elementwise_fallback(lhs, rhs, device, op)
    }

    /// Fallback CPU implementation for Metal element-wise operation
    /// Metal要素ごと演算のCPUフォールバック実装
    fn execute_elementwise_fallback<T, F>(
        lhs: &Arc<Buffer>,
        rhs: &Arc<Buffer>,
        _device: &Arc<MetalDeviceType>,
        op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: Float + 'static,
    {
        let lhs_cpu =
            Self::transfer_to_cpu(lhs, &[lhs.length() as usize / std::mem::size_of::<T>()])?;
        let rhs_cpu =
            Self::transfer_to_cpu(rhs, &[rhs.length() as usize / std::mem::size_of::<T>()])?;

        let result: Vec<T> = lhs_cpu
            .iter()
            .zip(rhs_cpu.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        // Transfer result back to Metal
        Self::transfer_to_device(result)
    }

    /// Execute Metal batch normalization
    /// Metalバッチ正規化実行
    pub fn execute_batch_normalize<T>(
        buffer: &Arc<Buffer>,
        device: &Arc<MetalDeviceType>,
        epsilon: T,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: Float + 'static,
    {
        let size = buffer.length() as usize / std::mem::size_of::<T>();

        // For now, use CPU fallback for batch normalization
        // In a production implementation, this would use Metal compute shaders
        let cpu_data = Self::transfer_to_cpu(buffer, &[size])?;
        let n = cpu_data.len();

        if n == 0 {
            return Ok(GpuBuffer::Metal {
                buffer: buffer.clone(),
                device: device.clone(),
            });
        }

        // Calculate mean and variance on CPU
        let mean: T =
            cpu_data.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(size as f64).unwrap();
        let variance: T = cpu_data
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x)
            / T::from(size as f64).unwrap();

        // Normalize
        let std_dev = (variance + epsilon).sqrt();
        let normalized: Vec<T> = cpu_data.iter().map(|&x| (x - mean) / std_dev).collect();

        // Transfer result back to Metal
        Self::transfer_to_device(normalized)
    }

    /// Execute Metal attention mechanism
    /// Metalアテンション機構実行
    pub fn execute_metal_attention<T>(
        query: &Arc<Buffer>,
        key: &Arc<Buffer>,
        value: &Arc<Buffer>,
        _device: &Arc<MetalDeviceType>,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: Float + 'static,
    {
        // For now, fall back to CPU implementation
        // Note: actual Metal compute shader for attention not implemented in current version
        let query_cpu: Vec<T> =
            Self::transfer_to_cpu(query, &[query.length() as usize / std::mem::size_of::<T>()])?;
        let key_cpu: Vec<T> =
            Self::transfer_to_cpu(key, &[key.length() as usize / std::mem::size_of::<T>()])?;
        let value_cpu: Vec<T> =
            Self::transfer_to_cpu(value, &[value.length() as usize / std::mem::size_of::<T>()])?;

        // Simple attention computation on CPU
        let scores: Vec<T> = query_cpu
            .iter()
            .zip(key_cpu.iter())
            .map(|(&q, &k)| q * k)
            .collect();

        // Softmax
        let max_score = scores
            .iter()
            .fold(T::neg_infinity(), |max, &x| if x > max { x } else { max });

        let exp_scores: Vec<T> = scores.iter().map(|&x| (x - max_score).exp()).collect();
        let sum_exp = exp_scores.iter().fold(T::zero(), |acc, &x| acc + x);
        let attention_weights: Vec<T> = exp_scores.iter().map(|&x| x / sum_exp).collect();

        // Weighted sum with values
        let result: Vec<T> = attention_weights
            .iter()
            .zip(value_cpu.iter())
            .map(|(&w, &v)| w * v)
            .collect();

        // Transfer result back to Metal
        Self::transfer_to_device(result)
    }

    /// Execute Metal attention mechanism (alias for compatibility)
    /// Metalアテンション機構実行（互換性のためのエイリアス）
    pub fn execute_attention<T>(
        query: &Arc<Buffer>,
        key: &Arc<Buffer>,
        value: &Arc<Buffer>,
        device: &Arc<MetalDeviceType>,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: Float + 'static,
    {
        Self::execute_metal_attention(query, key, value, device)
    }
}

#[cfg(not(feature = "metal"))]
/// Stub for Metal operations when Metal is not available
/// Metal無効時のMetal操作スタブ
pub struct MetalOperations;

#[cfg(not(feature = "metal"))]
impl MetalOperations {
    // Stub implementations that return errors
}

#[cfg(test)]
#[cfg(feature = "metal")]
mod tests {
    use super::*;

    #[test]
    fn test_metal_operations_stub() {
        // This test will only run when Metal is enabled
        // Basic compilation test
    }
}
