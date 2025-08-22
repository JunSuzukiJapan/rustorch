//! Data type conversion utilities for mixed precision
//! 混合精度のためのデータ型変換ユーティリティ

use crate::tensor::Tensor;
use crate::dtype::DType;
use half::{f16, bf16};

/// Cast tensor to FP16 (simulated - converts to FP16 and back for precision loss)
pub fn cast_to_fp16(tensor: &Tensor<f32>) -> Tensor<f32> {
    let data = tensor.as_slice().unwrap();
    let fp16_converted: Vec<f32> = data.iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();
    
    Tensor::from_vec(fp16_converted, tensor.shape().to_vec())
}

/// Cast tensor to FP32 (passthrough for compatibility)
pub fn cast_to_fp32(tensor: &Tensor<f32>) -> Tensor<f32> {
    tensor.clone()
}

/// Cast tensor to BF16 (simulated - converts to BF16 and back for precision loss)
pub fn cast_to_bf16(tensor: &Tensor<f32>) -> Tensor<f32> {
    let data = tensor.as_slice().unwrap();
    let bf16_converted: Vec<f32> = data.iter()
        .map(|&x| bf16::from_f32(x).to_f32())
        .collect();
    
    Tensor::from_vec(bf16_converted, tensor.shape().to_vec())
}

/// Cast BF16 tensor to FP32 (passthrough for compatibility)
pub fn cast_bf16_to_fp32(tensor: &Tensor<f32>) -> Tensor<f32> {
    tensor.clone()
}


/// Generic tensor casting function
pub fn cast_tensor(tensor: &Tensor<f32>, target_dtype: DType) -> Tensor<f32> {
    match target_dtype {
        DType::Float16 => cast_to_fp16(tensor),
        DType::BFloat16 => cast_to_bf16(tensor),
        DType::Float32 => tensor.clone(),
        _ => tensor.clone(),
    }
}

/// Mixed precision tensor operations
pub trait MixedPrecisionTensor<T: num_traits::Float> {
    /// Cast tensor to target dtype
    fn cast_to(&self, dtype: DType) -> Tensor<f32>;
    
    /// Check if tensor can be safely cast to target dtype
    fn can_cast_to(&self, dtype: DType) -> bool;
    
    /// Get memory footprint in bytes
    fn memory_footprint(&self) -> usize;
    
    /// Get memory footprint for target dtype
    fn memory_footprint_for_dtype(&self, dtype: DType) -> usize;
}

impl MixedPrecisionTensor<f32> for Tensor<f32> {
    fn cast_to(&self, dtype: DType) -> Tensor<f32> {
        cast_tensor(self, dtype)
    }
    
    fn can_cast_to(&self, dtype: DType) -> bool {
        matches!(dtype, DType::Float16 | DType::BFloat16 | DType::Float32)
    }
    
    fn memory_footprint(&self) -> usize {
        self.numel() * std::mem::size_of::<f32>()
    }
    
    fn memory_footprint_for_dtype(&self, dtype: DType) -> usize {
        self.numel() * dtype.size()
    }
}

/// Utility functions for mixed precision
pub mod utils {
    use super::*;
    
    
    /// Clip gradients to prevent overflow
    pub fn clip_grad_norm(gradients: &mut [Tensor<f32>], max_norm: f32) -> f32 {
        // Calculate total norm
        let mut total_norm = 0.0f32;
        for grad in gradients.iter() {
            if let Some(data) = grad.as_slice() {
                for &value in data {
                    total_norm += value * value;
                }
            }
        }
        total_norm = total_norm.sqrt();
        
        // Clip if necessary
        if total_norm > max_norm {
            let clip_factor = max_norm / total_norm;
            for grad in gradients.iter_mut() {
                *grad = grad.clone() * clip_factor;
            }
        }
        
        total_norm
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    
    
    
    #[test]
    fn test_clip_grad_norm() {
        let mut grads = vec![
            Tensor::from_vec(vec![3.0, 4.0, 0.0], vec![3]),  // norm = 5
        ];
        
        let norm = utils::clip_grad_norm(&mut grads, 2.5);
        assert!(norm > 4.9 && norm < 5.1);  // Original norm ~5
        
        // After clipping, norm should be <= 2.5
        let clipped_data = grads[0].as_slice().unwrap();
        let clipped_norm = clipped_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(clipped_norm <= 2.51);  // Allow small numerical error
    }
    
}