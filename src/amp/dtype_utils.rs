//! Data type conversion utilities for mixed precision
//! 混合精度のためのデータ型変換ユーティリティ

#![allow(dead_code)]

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

/// Master weight management for mixed precision
pub struct MasterWeights {
    /// FP32 master weights
    fp32_weights: Vec<Tensor<f32>>,
    /// FP16/BF16 model weights (stored as f32 with reduced precision)
    model_weights: Vec<Tensor<f32>>,
}

impl MasterWeights {
    /// Create new master weights
    pub fn new() -> Self {
        Self {
            fp32_weights: Vec::new(),
            model_weights: Vec::new(),
        }
    }
    
    /// Register a parameter with master weights
    pub fn register(&mut self, param: &Tensor<f32>) {
        self.fp32_weights.push(param.clone());
        self.model_weights.push(cast_to_fp16(param));
    }
    
    /// Update model weights from master weights
    pub fn sync_to_model(&mut self) {
        for (master, model) in self.fp32_weights.iter().zip(self.model_weights.iter_mut()) {
            *model = cast_to_fp16(master);
        }
    }
    
    /// Update master weights from model weights
    pub fn sync_from_model(&mut self) {
        for (master, model) in self.fp32_weights.iter_mut().zip(self.model_weights.iter()) {
            *master = cast_to_fp32(model);
        }
    }
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
    
    /// Check if tensor contains inf or nan
    pub fn has_inf_or_nan(tensor: &Tensor<f32>) -> bool {
        if let Some(data) = tensor.as_slice() {
            data.iter().any(|&x| !x.is_finite())
        } else {
            false
        }
    }
    
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
    
    /// Convert dtype enum to string
    pub fn dtype_to_str(dtype: DType) -> &'static str {
        match dtype {
            DType::Float32 => "float32",
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            _ => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_master_weights() {
        let mut master = MasterWeights::new();
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        
        master.register(&param);
        assert_eq!(master.fp32_weights.len(), 1);
        assert_eq!(master.model_weights.len(), 1);
    }
    
    #[test]
    fn test_has_inf_or_nan() {
        let normal = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(!utils::has_inf_or_nan(&normal));
        
        let with_nan = Tensor::from_vec(vec![1.0, f32::NAN, 3.0], vec![3]);
        assert!(utils::has_inf_or_nan(&with_nan));
        
        let with_inf = Tensor::from_vec(vec![1.0, f32::INFINITY, 3.0], vec![3]);
        assert!(utils::has_inf_or_nan(&with_inf));
    }
    
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
    
    #[test]
    fn test_dtype_to_str() {
        assert_eq!(utils::dtype_to_str(DType::Float32), "float32");
        assert_eq!(utils::dtype_to_str(DType::Float16), "float16");
        assert_eq!(utils::dtype_to_str(DType::BFloat16), "bfloat16");
    }
}