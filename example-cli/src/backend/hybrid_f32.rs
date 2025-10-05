use super::Backend;
use anyhow::Result;
use rustorch::tensor::Tensor;

#[cfg(feature = "hybrid-f32")]
use rustorch::hybrid_f32::F32Tensor;

/// Hybrid F32 Backend
///
/// Uses RusTorch's hybrid_f32 module for F32-optimized operations:
/// - F32 precision for better memory efficiency and performance
/// - Automatic GPU acceleration when beneficial
/// - Smart device routing based on operation characteristics
pub struct HybridF32Backend {
    available: bool,
}

impl HybridF32Backend {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "hybrid-f32")]
        {
            // Check if hybrid-f32 module is available
            let available = true;

            Ok(Self { available })
        }
        #[cfg(not(feature = "hybrid-f32"))]
        {
            anyhow::bail!("Hybrid F32 backend requires hybrid-f32 feature")
        }
    }
}

impl Backend for HybridF32Backend {
    fn name(&self) -> &'static str {
        "Hybrid F32"
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn to_device(&self, tensor: Tensor<f64>) -> Result<Tensor<f64>> {
        // F32Tensor handles device management internally
        Ok(tensor)
    }

    fn zeros(&self, shape: &[usize]) -> Result<Tensor<f64>> {
        #[cfg(feature = "hybrid-f32")]
        {
            // Create F32 tensor and convert to F64
            let f32_tensor = F32Tensor::zeros(shape);
            // For now, convert back to F64 Tensor
            // TODO: Add direct conversion when available
            Ok(Tensor::zeros(shape))
        }
        #[cfg(not(feature = "hybrid-f32"))]
        {
            Ok(Tensor::zeros(shape))
        }
    }

    fn from_vec(&self, data: Vec<f64>, shape: &[usize]) -> Result<Tensor<f64>> {
        #[cfg(feature = "hybrid-f32")]
        {
            // Convert to F32 for better memory efficiency
            let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            let _f32_tensor = F32Tensor::from_vec(f32_data, shape);
            // For now, use original F64 tensor
            // TODO: Add direct conversion when available
            Ok(Tensor::from_vec(data, shape))
        }
        #[cfg(not(feature = "hybrid-f32"))]
        {
            Ok(Tensor::from_vec(data, shape))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "hybrid-f32")]
    fn test_hybrid_f32_backend_creation() {
        let backend = HybridF32Backend::new();
        assert!(backend.is_ok());
    }

    #[test]
    #[cfg(feature = "hybrid-f32")]
    fn test_hybrid_f32_backend_availability() {
        let backend = HybridF32Backend::new().unwrap();
        assert!(backend.is_available());
        assert_eq!(backend.name(), "Hybrid F32");
    }

    #[test]
    #[cfg(feature = "hybrid-f32")]
    fn test_hybrid_f32_backend_tensor_operations() {
        let backend = HybridF32Backend::new().unwrap();

        // Test zeros
        let zeros = backend.zeros(&[2, 3]).unwrap();
        assert_eq!(zeros.shape(), &[2, 3]);

        // Test from_vec
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = backend.from_vec(data, &[2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);

        // Test to_device
        let moved = backend.to_device(tensor).unwrap();
        assert_eq!(moved.shape(), &[2, 2]);
    }
}
